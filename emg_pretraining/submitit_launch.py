from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
import time

import pandas as pd
import yaml

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from emg_pretraining.config_utils import load_config, namespace_to_dict
    from emg_pretraining.vqvae_utils import load_joint_tokenizer_specs
else:
    from .config_utils import load_config, namespace_to_dict
    from .vqvae_utils import load_joint_tokenizer_specs


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
DEFAULT_CONFIG = THIS_DIR / "configs" / "default.yaml"


class SubmitJob:
    def __init__(self, command: list[str]):
        self.command = command

    def __call__(self):
        os.execvp(self.command[0], self.command)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--experiment-dir", type=str, default=None)
    parser.add_argument("--per-row", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--run-index", type=int, default=None, help="Select a single shared tokenizer run by index after score sorting.")
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


def _filtered_results_df(results_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    if "status" in df.columns:
        df = df[df["status"].fillna("").str.upper() != "FAILED"].copy()
    if "score" in df.columns and not df.empty:
        df = df.sort_values("score", ascending=False, na_position="last")
    return df


def _resolve_joint_rows(cfg) -> list[tuple[str, Path]]:
    cfg_dict = namespace_to_dict(cfg)
    left_csv = Path(cfg_dict["tokenizers"]["left_hand"]["results_csv"])
    right_csv = Path(cfg_dict["tokenizers"]["right_hand"]["results_csv"])

    left_df = _filtered_results_df(left_csv)
    right_df = _filtered_results_df(right_csv)
    common_run_names = [name for name in left_df["run_name"].tolist() if name in set(right_df["run_name"].tolist())]

    outputs = []
    for run_name in common_run_names:
        job_cfg = namespace_to_dict(cfg)
        job_cfg["tokenizers"]["left_hand"]["run_name"] = str(run_name)
        job_cfg["tokenizers"]["right_hand"]["run_name"] = str(run_name)

        (PROJECT_ROOT / "slurm_logs").mkdir(parents=True, exist_ok=True)
        tmp = NamedTemporaryFile(mode="w", suffix=".yaml", prefix=f"{run_name}_", delete=False, dir=str(PROJECT_ROOT / "slurm_logs"))
        with tmp:
            yaml.safe_dump(job_cfg, tmp, sort_keys=False)
        tmp_path = Path(tmp.name)

        try:
            load_joint_tokenizer_specs(load_config(tmp_path).tokenizers.__dict__)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            continue

        outputs.append((str(run_name), tmp_path))
    return outputs


def _parse_float_grid(arg_value: str | None, default: list[float]) -> list[float]:
    if arg_value is None:
        return list(default)
    values = []
    for chunk in str(arg_value).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise ValueError(f"Expected at least one float value, got: {arg_value!r}")
    return values


def _format_float_tag(value: float) -> str:
    text = f"{float(value):.12g}"
    return text.replace("-", "m").replace(".", "p")


def _write_temp_config(job_cfg: dict, prefix: str) -> Path:
    (PROJECT_ROOT / "slurm_logs").mkdir(parents=True, exist_ok=True)
    tmp = NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix=f"{prefix}_",
        delete=False,
        dir=str(PROJECT_ROOT / "slurm_logs"),
    )
    with tmp:
        yaml.safe_dump(job_cfg, tmp, sort_keys=False)
    return Path(tmp.name)


def _describe_tokenizers(config_path: str | Path) -> tuple[str, str, str, str]:
    cfg = load_config(str(config_path))
    cfg_dict = namespace_to_dict(cfg)
    left_cfg = cfg_dict["tokenizers"]["left_hand"]
    right_cfg = cfg_dict["tokenizers"]["right_hand"]
    return (
        str(left_cfg.get("run_name", "")),
        str(left_cfg.get("results_csv", "")),
        str(right_cfg.get("run_name", "")),
        str(right_cfg.get("results_csv", "")),
    )


def _build_optimizer_sweep_jobs(
    cfg,
    run_name: str,
    lr_backbone_values: list[float],
    lr_head_values: list[float],
    layer_decay_values: list[float],
) -> list[tuple[str, float, float, float, Path]]:
    jobs = []
    for lr_backbone in lr_backbone_values:
        for lr_head in lr_head_values:
            for layer_decay in layer_decay_values:
                job_cfg = namespace_to_dict(cfg)
                job_cfg["tokenizers"]["left_hand"]["run_name"] = str(run_name)
                job_cfg["tokenizers"]["right_hand"]["run_name"] = str(run_name)
                job_cfg["optim"]["optimizer"]["lr_backbone"] = float(lr_backbone)
                job_cfg["optim"]["optimizer"]["lr_head"] = float(lr_head)
                job_cfg["optim"]["optimizer"]["layer_decay"] = float(layer_decay)

                tag = (
                    f"{run_name}_lrb{_format_float_tag(lr_backbone)}"
                    f"_lrh{_format_float_tag(lr_head)}"
                    f"_ld{_format_float_tag(layer_decay)}"
                )
                cfg_path = _write_temp_config(job_cfg, prefix=tag)
                jobs.append((tag, float(lr_backbone), float(lr_head), float(layer_decay), cfg_path))
    return jobs


def _build_executor(submitit, cfg, log_root: Path):
    cfg_dict = namespace_to_dict(cfg)
    slurm_cfg = cfg_dict.get("slurm", {})
    distributed_cfg = cfg_dict.get("distributed", {})

    partition = str(slurm_cfg.get("partition", "scavenger"))
    qos = str(slurm_cfg.get("qos", partition))
    account = str(slurm_cfg.get("account", partition))
    timeout_hours = float(slurm_cfg.get("timeout_hours", 24.0))
    time_str = str(slurm_cfg.get("time", f"{int(timeout_hours):02d}:00:00"))
    mem = str(slurm_cfg.get("mem", "70G"))
    cpus_per_task = int(slurm_cfg.get("cpus_per_task", 2))
    nodes = int(slurm_cfg.get("nodes", 1))
    gpus_per_node = int(slurm_cfg.get("gpus_per_node", 1))
    gpu_type = str(slurm_cfg.get("gpu_type", "rtxa5000"))
    ntasks_per_node = int(slurm_cfg.get("ntasks_per_node", gpus_per_node if bool(distributed_cfg.get("enabled", False)) else 1))

    executor = submitit.SlurmExecutor(folder=str(log_root))
    additional_parameters = {"chdir": str(PROJECT_ROOT)}
    if "signal_delay_s" in slurm_cfg:
        additional_parameters["signal"] = f"USR2@{int(slurm_cfg['signal_delay_s'])}"
    if "wckey" in slurm_cfg:
        additional_parameters["wckey"] = str(slurm_cfg["wckey"])
    if "open_mode" in slurm_cfg:
        additional_parameters["open-mode"] = str(slurm_cfg["open_mode"])

    executor.update_parameters(
        job_name=str(cfg_dict.get("experiment", {}).get("name", "emg_pretrain_dual")),
        partition=partition,
        qos=qos,
        account=account,
        time=time_str,
        mem=mem,
        cpus_per_task=cpus_per_task,
        nodes=nodes,
        ntasks_per_node=ntasks_per_node,
        gres=f"gpu:{gpu_type}:{gpus_per_node}",
        additional_parameters=additional_parameters,
    )
    return executor


def _build_train_command(config_path: str | Path, experiment_dir: str | None, diagnostics: bool) -> list[str]:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = config_path.resolve()
    command = [
        "python",
        "-u",
        "-m",
        "emg_pretraining.train",
        "--config",
        str(config_path),
    ]
    if experiment_dir:
        command.extend(["--experiment-dir", str(experiment_dir)])
    if diagnostics:
        command.append("--diagnostics")
    return command


def _tail_text(text: str | None, max_lines: int = 40) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def _terminal_job_state(job) -> str:
    try:
        info = job.get_info(mode="force")
    except Exception:
        info = {}
    state = str(info.get("State", "") or getattr(job, "state", "") or "UNKNOWN")
    return state.upper()


def _submit_job(
    executor,
    command: list[str],
    run_name: str,
    job_tag: str,
    lr_backbone: float,
    lr_head: float,
    layer_decay: float,
    sweep_cfg_path: Path,
    attempt: int,
):
    left_run_name, left_results_csv, right_run_name, right_results_csv = _describe_tokenizers(sweep_cfg_path)
    print(
        f"tokenizers job_tag={job_tag} attempt={attempt} "
        f"left_run_name={left_run_name} left_results_csv={left_results_csv} "
        f"right_run_name={right_run_name} right_results_csv={right_results_csv}"
    )
    job = executor.submit(SubmitJob(command))
    print(
        f"submitted run_name={run_name} lr_backbone={lr_backbone:g} lr_head={lr_head:g} layer_decay={layer_decay:g} "
        f"attempt={attempt} job_id={job.job_id} config={sweep_cfg_path}"
    )
    return job


def main():
    args = parse_args()

    try:
        import submitit
    except ImportError as exc:
        raise ImportError(
            "submitit is required for submitit_launch.py. "
            "Use the cluster environment that has submitit installed, or install it there."
        ) from exc

    log_root = PROJECT_ROOT / "slurm_logs"
    log_root.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    executor = _build_executor(submitit, cfg, log_root)

    if not args.per_row:
        command = _build_train_command(args.config, args.experiment_dir, args.diagnostics)
        job = executor.submit(SubmitJob(command))
        print(f"submitted job_id={job.job_id} config={args.config}")
        return

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
            command = _build_train_command(sweep_cfg_path, exp_dir, args.diagnostics)
            pending_specs.append(
                {
                    "run_name": run_name,
                    "job_tag": job_tag,
                    "lr_backbone": lr_backbone,
                    "lr_head": lr_head,
                    "layer_decay": layer_decay,
                    "sweep_cfg_path": sweep_cfg_path,
                    "command": command,
                    "attempt": 0,
                }
            )

    active_jobs = {}
    for spec in pending_specs:
        spec["attempt"] += 1
        job = _submit_job(
            executor=executor,
            command=spec["command"],
            run_name=spec["run_name"],
            job_tag=spec["job_tag"],
            lr_backbone=spec["lr_backbone"],
            lr_head=spec["lr_head"],
            layer_decay=spec["layer_decay"],
            sweep_cfg_path=spec["sweep_cfg_path"],
            attempt=int(spec["attempt"]),
        )
        active_jobs[spec["job_tag"]] = {"job": job, "spec": spec}

    while active_jobs:
        for job_tag in list(active_jobs.keys()):
            entry = active_jobs[job_tag]
            job = entry["job"]
            spec = entry["spec"]
            if not job.done(force_check=True):
                continue

            state = _terminal_job_state(job)
            if "COMPLETED" in state:
                print(
                    f"completed run_name={spec['run_name']} job_tag={job_tag} "
                    f"attempt={spec['attempt']} job_id={job.job_id} state={state}"
                )
                del active_jobs[job_tag]
                continue

            stdout_tail = _tail_text(job.stdout())
            stderr_tail = _tail_text(job.stderr())
            raise RuntimeError(
                f"Job failed for run_name={spec['run_name']} job_tag={job_tag} job_id={job.job_id} "
                f"state={state} after {spec['attempt']} attempt(s).\n"
                f"stdout_path={job.paths.stdout}\n"
                f"stderr_path={job.paths.stderr}\n"
                f"--- stdout tail ---\n{stdout_tail}\n"
                f"--- stderr tail ---\n{stderr_tail}"
            )

        if active_jobs:
            time.sleep(5)


if __name__ == "__main__":
    main()
