from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from emg_pretraining.config_utils import load_config
    from emg_pretraining.submitit_launch import (
        DEFAULT_CONFIG,
        _build_optimizer_sweep_jobs,
        _build_train_command,
        _parse_float_grid,
        _resolve_joint_rows,
    )
else:
    from .config_utils import load_config
    from .submitit_launch import (
        DEFAULT_CONFIG,
        _build_optimizer_sweep_jobs,
        _build_train_command,
        _parse_float_grid,
        _resolve_joint_rows,
    )


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--experiment-dir", type=str, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--run-index", type=int, default=None, help="Select a single shared tokenizer run by index after score sorting.")
    parser.add_argument("--batch-index", type=int, default=0, help="Zero-based batch index. With batch-size 4, batch-index 1 runs rows 4-7.")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of runs to launch concurrently in one Slurm allocation.")
    parser.add_argument("--gpu-ids", type=str, default="0,1,2,3", help="Comma-separated physical GPU ids to use for this batch.")
    parser.add_argument(
        "--lr-backbone-values",
        type=str,
        default=None,
        help="Comma-separated lr_backbone sweep values, e.g. '2e-6,2e-5,2e-4'.",
    )
    parser.add_argument(
        "--lr-head-values",
        type=str,
        default=None,
        help="Comma-separated lr_head sweep values, e.g. '2e-6,2e-5,2e-4'.",
    )
    parser.add_argument(
        "--layer-decay-values",
        type=str,
        default=None,
        help="Comma-separated layer_decay sweep values, e.g. '0,0.75,1.0'.",
    )
    parser.add_argument("--diagnostics", action="store_true", help="Enable diagnostic output during training")
    return parser.parse_args()


def _parse_gpu_ids(arg_value: str) -> list[str]:
    gpu_ids = [chunk.strip() for chunk in str(arg_value).split(",") if chunk.strip()]
    if not gpu_ids:
        raise ValueError("Expected at least one GPU id in --gpu-ids.")
    return gpu_ids


def _build_pending_specs(args) -> list[dict]:
    cfg = load_config(args.config)
    row_jobs = _resolve_joint_rows(cfg)

    if args.run_index is not None:
        run_index = int(args.run_index)
        if run_index < 0 or run_index >= len(row_jobs):
            raise IndexError(f"--run-index {run_index} is out of range for {len(row_jobs)} compatible shared runs.")
        row_jobs = [row_jobs[run_index]]
    if args.max_rows is not None:
        row_jobs = row_jobs[: int(args.max_rows)]

    if not row_jobs:
        raise ValueError("No compatible shared run_name rows found across left/right results.csv files.")

    lr_backbone_values = _parse_float_grid(args.lr_backbone_values, default=[float(cfg.optim.optimizer.lr_backbone)])
    lr_head_values = _parse_float_grid(args.lr_head_values, default=[float(cfg.optim.optimizer.lr_head)])
    layer_decay_values = _parse_float_grid(args.layer_decay_values, default=[float(cfg.optim.optimizer.layer_decay)])
    sweep_optimizer = (
        (args.lr_backbone_values is not None)
        or (args.lr_head_values is not None)
        or (args.layer_decay_values is not None)
    )

    pending_specs = []
    for run_name, cfg_path in row_jobs:
        if sweep_optimizer:
            cfg_path.unlink(missing_ok=True)
            sweep_jobs = _build_optimizer_sweep_jobs(
                cfg=cfg,
                run_name=run_name,
                lr_backbone_values=lr_backbone_values,
                lr_head_values=lr_head_values,
                layer_decay_values=layer_decay_values,
            )
        else:
            sweep_jobs = [
                (
                    run_name,
                    float(cfg.optim.optimizer.lr_backbone),
                    float(cfg.optim.optimizer.lr_head),
                    float(cfg.optim.optimizer.layer_decay),
                    cfg_path,
                )
            ]

        for job_tag, lr_backbone, lr_head, layer_decay, sweep_cfg_path in sweep_jobs:
            exp_dir = str(Path(args.experiment_dir) / job_tag) if args.experiment_dir else None
            pending_specs.append(
                {
                    "run_name": run_name,
                    "job_tag": job_tag,
                    "lr_backbone": lr_backbone,
                    "lr_head": lr_head,
                    "layer_decay": layer_decay,
                    "sweep_cfg_path": sweep_cfg_path,
                    "command": _build_train_command(sweep_cfg_path, exp_dir, args.diagnostics),
                }
            )

    return pending_specs


def main():
    args = parse_args()
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    pending_specs = _build_pending_specs(args)

    start = int(args.batch_index) * int(args.batch_size)
    end = start + int(args.batch_size)
    selected_specs = pending_specs[start:end]
    if not selected_specs:
        raise ValueError(
            f"Batch index {args.batch_index} with batch size {args.batch_size} selected no runs "
            f"out of {len(pending_specs)} total pending runs."
        )
    if len(selected_specs) > len(gpu_ids):
        raise ValueError(
            f"Selected {len(selected_specs)} runs but only {len(gpu_ids)} GPU ids were provided via --gpu-ids."
        )

    batch_log_dir = PROJECT_ROOT / "slurm_logs" / f"batch_{int(args.batch_index):03d}"
    batch_log_dir.mkdir(parents=True, exist_ok=True)

    active_entries: list[dict] = []

    def _terminate_children(signum, _frame):
        print(f"received signal {signum}; terminating {len(active_entries)} child training process(es)", flush=True)
        for entry in active_entries:
            proc = entry["proc"]
            if proc.poll() is None:
                proc.terminate()
        for entry in active_entries:
            proc = entry["proc"]
            if proc.poll() is None:
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, _terminate_children)
    signal.signal(signal.SIGTERM, _terminate_children)

    print(
        f"launching batch_index={args.batch_index} batch_size={args.batch_size} "
        f"selected_runs={len(selected_specs)} total_pending_runs={len(pending_specs)} "
        f"gpu_ids={','.join(gpu_ids)}",
        flush=True,
    )

    for local_slot, spec in enumerate(selected_specs):
        gpu_id = gpu_ids[local_slot]
        stdout_path = batch_log_dir / f"{spec['job_tag']}.out"
        stderr_path = batch_log_dir / f"{spec['job_tag']}.err"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONUNBUFFERED"] = "1"

        stdout_handle = open(stdout_path, "w")
        stderr_handle = open(stderr_path, "w")
        proc = subprocess.Popen(
            spec["command"],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        active_entries.append(
            {
                "spec": spec,
                "gpu_id": gpu_id,
                "proc": proc,
                "stdout_handle": stdout_handle,
                "stderr_handle": stderr_handle,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
            }
        )
        print(
            f"launched job_tag={spec['job_tag']} run_name={spec['run_name']} gpu={gpu_id} pid={proc.pid} "
            f"stdout={stdout_path} stderr={stderr_path}",
            flush=True,
        )

    failures = []
    for entry in active_entries:
        proc = entry["proc"]
        returncode = proc.wait()
        entry["stdout_handle"].close()
        entry["stderr_handle"].close()
        if returncode == 0:
            print(
                f"completed job_tag={entry['spec']['job_tag']} gpu={entry['gpu_id']} "
                f"stdout={entry['stdout_path']}",
                flush=True,
            )
        else:
            failures.append(
                f"job_tag={entry['spec']['job_tag']} gpu={entry['gpu_id']} returncode={returncode} "
                f"stdout={entry['stdout_path']} stderr={entry['stderr_path']}"
            )

    if failures:
        raise RuntimeError("One or more batch runs failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
