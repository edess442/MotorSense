from __future__ import annotations

import argparse
import datetime
import io
import json
import math
import os
import random
import socket
import time
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from emg_pretraining.optim_factory import make_param_groups_layerwise
    from emg_pretraining.config_utils import apply_overrides, load_config, namespace_to_dict
    from emg_pretraining.data import build_dataloaders
    from emg_pretraining.models import HieraVideoMultiHeadCodePredictor
    from emg_pretraining.vqvae_utils import (
        codebook_embeddings_from_ids,
        code_histogram_from_ids,
        load_joint_tokenizer_specs,
        load_vqvae_model,
        token_steps_for_window,
    )
else:
    from .optim_factory import make_param_groups_layerwise
    from .config_utils import apply_overrides, load_config, namespace_to_dict
    from .data import build_dataloaders
    from .models import HieraVideoMultiHeadCodePredictor
    from .vqvae_utils import (
        codebook_embeddings_from_ids,
        code_histogram_from_ids,
        load_joint_tokenizer_specs,
        load_vqvae_model,
        token_steps_for_window,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).parent / "configs" / "default.yaml"))
    parser.add_argument("--results-csv", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--n-embed", type=int, default=None)
    parser.add_argument("--codes-per-second", type=float, default=None)
    parser.add_argument("--emg-window-size", type=int, default=None)
    parser.add_argument("--experiment-dir", type=str, default=None)
    parser.add_argument("--diagnostics", action="store_true", help="Enable diagnostic output during training")
    return parser.parse_args()


def diagnostic_print(message, diagnostics_enabled=False):
    if diagnostics_enabled:
        print(message, flush=True)


def diagnostic_print_timing(label: str, seconds: float, diagnostics_enabled=False):
    if diagnostics_enabled:
        print(f"DIAGNOSTIC_TIMING: {label} took {seconds:.3f}s", flush=True)


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def get_tokenizer_cfg_map(cfg):
    hand_mode = str(getattr(cfg.data, "hand_mode", "both"))
    if hasattr(cfg, "tokenizers"):
        tokenizers = vars(cfg.tokenizers)
        if hand_mode == "left_hand":
            return {"left_hand": tokenizers["left_hand"]}
        if hand_mode == "right_hand":
            return {"right_hand": tokenizers["right_hand"]}
        return tokenizers
    return {"left_hand": cfg.tokenizer}


def setup_distributed_env(cfg):
    distributed_cfg = getattr(cfg, "distributed", None)
    if "WORLD_SIZE" not in os.environ:
        if "SLURM_NTASKS" in os.environ:
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            if "SLURM_NODELIST" in os.environ:
                import subprocess

                result = subprocess.run(
                    ["scontrol", "show", "hostnames", os.environ["SLURM_NODELIST"]],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                hostnames = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                if hostnames:
                    os.environ["MASTER_ADDR"] = hostnames[0]
        else:
            os.environ["WORLD_SIZE"] = "1"
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", socket.gethostname())
    if distributed_cfg is not None and getattr(distributed_cfg, "master_addr", None) is not None:
        os.environ["MASTER_ADDR"] = str(distributed_cfg.master_addr)
    os.environ["MASTER_PORT"] = str(getattr(distributed_cfg, "master_port", 29500) if distributed_cfg is not None else 29500)


def init_distributed(cfg, diagnostics_enabled=False):
    setup_distributed_env(cfg)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed_cfg = getattr(cfg, "distributed", None)
    distributed_enabled = bool(getattr(distributed_cfg, "enabled", False))
    distributed = distributed_enabled and world_size > 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank if distributed else 0)
        device = torch.device("cuda", local_rank if distributed else 0)
    else:
        device = torch.device("cpu")

    if distributed and not is_dist_avail_and_initialized():
        # Match the known-good multi-node NCCL setup used elsewhere in this codebase.
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "bond0")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        os.environ.setdefault("NCCL_DEBUG", "INFO")
        dist.init_process_group(
            backend=str(getattr(distributed_cfg, "backend", "nccl")),
            init_method=str(getattr(distributed_cfg, "init_method", "env://")),
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=30),
        )

    diagnostic_print(
        f"DIAGNOSTIC: distributed={distributed} world_size={world_size} rank={rank} local_rank={local_rank} "
        f"MASTER_ADDR={os.environ.get('MASTER_ADDR')} MASTER_PORT={os.environ.get('MASTER_PORT')}",
        diagnostics_enabled,
    )
    return device, distributed, rank, world_size, local_rank


def maybe_init_wandb(cfg, tokenizer_specs, experiment_dir: Path):
    if not is_main_process():
        return None

    wandb_cfg = getattr(cfg, "wandb", None)
    if wandb_cfg is None:
        return None

    mode = str(getattr(wandb_cfg, "mode", "disabled")).lower()
    if mode == "disabled":
        return None

    try:
        import wandb
    except ImportError:
        print("wandb logging requested but wandb is not installed; continuing without wandb.")
        return None

    configured_run_name = getattr(wandb_cfg, "run_name", None)
    default_suffix = "__".join(f"{hand}:{spec.run_name}" for hand, spec in tokenizer_specs.items())
    run_name = str(configured_run_name or f"{cfg.experiment.name}_{default_suffix}")
    return wandb.init(
        mode=mode,
        project=str(wandb_cfg.project),
        entity=getattr(wandb_cfg, "entity", None),
        name=run_name,
        dir=str(experiment_dir),
        config=namespace_to_dict(cfg),
    )


def maybe_get_wandb_module():
    try:
        import wandb
    except ImportError:
        return None
    return wandb


def unnormalize_video_frame(frame_chw: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=frame_chw.dtype, device=frame_chw.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=frame_chw.dtype, device=frame_chw.device).view(3, 1, 1)
    frame = frame_chw * std + mean
    frame = frame.clamp(0.0, 1.0)
    return (frame.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)


def render_frames_strip(video_tchw: torch.Tensor) -> np.ndarray:
    frames = [unnormalize_video_frame(frame) for frame in video_tchw]
    return np.concatenate(frames, axis=1)


def render_signal_comparison_image(input_signal_ct: torch.Tensor, recon_signal_ct: torch.Tensor, title: str) -> np.ndarray:
    input_arr = input_signal_ct.detach().cpu().numpy()
    recon_arr = recon_signal_ct.detach().cpu().numpy()
    n_channels = input_arr.shape[0]
    fig, axes = plt.subplots(n_channels, 1, figsize=(14, max(8, 1.2 * n_channels)), sharex=True)
    axes = np.atleast_1d(axes)
    for ch, ax in enumerate(axes):
        ax.plot(input_arr[ch], color="tab:blue", linewidth=0.8, label="input")
        ax.plot(recon_arr[ch], color="tab:orange", linewidth=0.8, alpha=0.8, label="recon")
        ax.set_ylabel(f"ch{ch}")
        if ch == 0:
            ax.set_title(title)
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    import PIL.Image

    return np.array(PIL.Image.open(buf).convert("RGB"))


def render_histogram_comparison_image(target_hist: torch.Tensor, pred_hist: torch.Tensor, title: str) -> np.ndarray:
    target_arr = target_hist.detach().cpu().numpy()
    pred_arr = pred_hist.detach().cpu().numpy()
    xs = np.arange(target_arr.shape[0], dtype=np.int32)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(xs - 0.2, target_arr, width=0.4, label="gt", color="tab:blue", alpha=0.85)
    ax.bar(xs + 0.2, pred_arr, width=0.4, label="pred", color="tab:orange", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("code bin")
    ax.set_ylabel("probability")
    ax.legend(loc="upper right")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    import PIL.Image

    return np.array(PIL.Image.open(buf).convert("RGB"))


def render_sequence_comparison_image(target_seq: torch.Tensor, pred_seq: torch.Tensor, title: str) -> np.ndarray:
    target_arr = target_seq.detach().cpu().numpy().T
    pred_arr = pred_seq.detach().cpu().numpy().T
    target_abs_max = float(np.abs(target_arr).max()) if target_arr.size else 0.0
    pred_abs_max = float(np.abs(pred_arr).max()) if pred_arr.size else 0.0
    vabs = max(target_abs_max, pred_abs_max, 1e-6)
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].imshow(target_arr, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-vabs, vmax=vabs)
    axes[0].set_title(f"{title} | gt")
    axes[0].set_ylabel("embed dim")
    axes[1].imshow(pred_arr, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-vabs, vmax=vabs)
    axes[1].set_title(f"{title} | pred")
    axes[1].set_xlabel("token step")
    axes[1].set_ylabel("embed dim")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    import PIL.Image

    return np.array(PIL.Image.open(buf).convert("RGB"))


def render_sequence_code_comparison_image(target_ids: torch.Tensor, pred_ids: torch.Tensor, n_embed: int, title: str) -> np.ndarray:
    target_arr = target_ids.detach().cpu().numpy()[None, :]
    pred_arr = pred_ids.detach().cpu().numpy()[None, :]
    fig, axes = plt.subplots(2, 1, figsize=(14, 3.5), sharex=True)
    axes[0].imshow(target_arr, aspect="auto", interpolation="nearest", cmap="tab20", vmin=0, vmax=max(int(n_embed) - 1, 1))
    axes[0].set_title(f"{title} | gt")
    axes[0].set_ylabel("code")
    axes[0].set_yticks([])
    axes[1].imshow(pred_arr, aspect="auto", interpolation="nearest", cmap="tab20", vmin=0, vmax=max(int(n_embed) - 1, 1))
    axes[1].set_title(f"{title} | pred")
    axes[1].set_xlabel("token step")
    axes[1].set_ylabel("code")
    axes[1].set_yticks([])
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    import PIL.Image

    return np.array(PIL.Image.open(buf).convert("RGB"))


def get_target_mode(cfg) -> str:
    mode = str(getattr(cfg.train, "target_mode", "histogram")).strip().lower()
    aliases = {
        "hist": "histogram",
        "histogram": "histogram",
        "sequence": "sequence_codes",
        "sequence_code": "sequence_codes",
        "sequence_codes": "sequence_codes",
        "sequence_embedding": "sequence_embeddings",
        "sequence_embeddings": "sequence_embeddings",
    }
    resolved = aliases.get(mode)
    if resolved is None:
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unsupported train.target_mode={mode!r}. Expected one of: {valid}")
    return resolved


def build_media_samples(batch, tokenizers, tokenizer_specs, predictions, targets, device, max_samples: int, target_mode: str):
    if max_samples <= 0:
        return []
    samples = []
    video = batch["video"]
    batch_size = min(int(video.shape[0]), int(max_samples))
    with torch.no_grad():
        for i in range(batch_size):
            sample_payload = {
                "recording": batch["recording"][i],
                "participant": batch["participant"][i],
                "center_time": float(batch["center_time"][i].item()),
                "clip_sec": float(batch["clip_sec"][i].item()),
                "frames_image": render_frames_strip(video[i]),
                "hands": {},
            }
            for hand, tokenizer in tokenizers.items():
                vq_in = batch["vqvae_emg"][hand][i : i + 1].to(device, non_blocking=True)
                ids = tokenizer.encode_ids(vq_in)
                model_eval = tokenizer.module if hasattr(tokenizer, "module") else tokenizer
                quant = model_eval.codebook.embed_code(ids).transpose(1, 2).contiguous()
                recon = model_eval.decode(quant, output_length=vq_in.shape[-1])[0]
                hand_payload = {
                    "input": batch["vqvae_emg"][hand][i],
                    "recon": recon.detach().cpu(),
                }
                if target_mode == "histogram":
                    target_hist = code_histogram_from_ids(targets[hand][i], n_embed=tokenizer_specs[hand].n_embed, normalize=True)[0]
                    pred_hist = torch.softmax(predictions[hand][i], dim=-1)
                    hand_payload["target_hist"] = target_hist.detach().cpu()
                    hand_payload["pred_hist"] = pred_hist.detach().cpu()
                elif target_mode == "sequence_codes":
                    spec = tokenizer_specs[hand]
                    token_steps = token_steps_for_window(spec.fs, spec.codes_per_second, spec.window_sec)
                    pred_logits = predictions[hand][i].view(token_steps, spec.n_embed)
                    pred_ids = pred_logits.argmax(dim=-1)
                    hand_payload["n_embed"] = int(spec.n_embed)
                    hand_payload["target_ids"] = targets[hand][i].detach().cpu()
                    hand_payload["pred_ids"] = pred_ids.detach().cpu()
                    hand_payload["token_accuracy"] = float((pred_ids == targets[hand][i]).to(dtype=torch.float32).mean().item())
                else:
                    spec = tokenizer_specs[hand]
                    token_steps = token_steps_for_window(spec.fs, spec.codes_per_second, spec.window_sec)
                    pred_seq = predictions[hand][i].view(token_steps, spec.embed_dim)
                    target_seq = codebook_embeddings_from_ids(model_eval, targets[hand][i : i + 1])[0].contiguous()
                    hand_payload["target_seq"] = target_seq.detach().cpu()
                    hand_payload["pred_seq"] = pred_seq.detach().cpu()
                sample_payload["hands"][hand] = hand_payload
            samples.append(sample_payload)
    return samples


def log_vqvae_reconstruction_media(wandb_run, samples, prefix: str):
    wandb = maybe_get_wandb_module()
    if wandb is None or not samples:
        return
    payload = {}
    for idx, sample in enumerate(samples):
        caption = (
            f"sample={idx} participant={sample['participant']} "
            f"recording={sample['recording']} center_time={sample['center_time']:.3f}"
        )
        payload[f"{prefix}/sample_{idx}/frames_pre_norm"] = wandb.Image(sample["frames_image"], caption=caption)
        for hand, hand_payload in sample["hands"].items():
            signal_title = (
                f"{hand} VQ-VAE input vs reconstruction | "
                f"{sample['participant']} | {sample['recording']}"
            )
            signal_image = render_signal_comparison_image(hand_payload["input"], hand_payload["recon"], signal_title)
            payload[f"{prefix}/sample_{idx}/{hand}/input_vs_recon"] = wandb.Image(signal_image, caption=caption)
            if "target_hist" in hand_payload:
                hist_title = f"{hand} histogram gt vs pred | {sample['participant']} | {sample['recording']}"
                hist_image = render_histogram_comparison_image(hand_payload["target_hist"], hand_payload["pred_hist"], hist_title)
                payload[f"{prefix}/sample_{idx}/{hand}/hist_gt_vs_pred"] = wandb.Image(hist_image, caption=caption)
            if "target_ids" in hand_payload:
                seq_title = (
                    f"{hand} code sequence gt vs pred | {sample['participant']} | {sample['recording']} "
                    f"| acc={hand_payload['token_accuracy']:.3f}"
                )
                seq_image = render_sequence_code_comparison_image(
                    hand_payload["target_ids"],
                    hand_payload["pred_ids"],
                    n_embed=int(hand_payload["n_embed"]),
                    title=seq_title,
                )
                payload[f"{prefix}/sample_{idx}/{hand}/sequence_codes_gt_vs_pred"] = wandb.Image(seq_image, caption=caption)
            if "target_seq" in hand_payload:
                seq_title = f"{hand} sequence embedding gt vs pred | {sample['participant']} | {sample['recording']}"
                seq_image = render_sequence_comparison_image(hand_payload["target_seq"], hand_payload["pred_seq"], seq_title)
                payload[f"{prefix}/sample_{idx}/{hand}/sequence_gt_vs_pred"] = wandb.Image(seq_image, caption=caption)
    if payload:
        wandb_run.log(payload)


def build_optimizer(cfg, model):
    ocfg = cfg.optim.optimizer
    head_params = [p for p in model.heads.parameters() if p.requires_grad]
    backbone = model.backbone

    layer_decay = float(getattr(ocfg, "layer_decay", 1.0))
    if layer_decay not in (0.0, 1.0):
        backbone_groups = make_param_groups_layerwise(
            model=backbone,
            base_lr=float(ocfg.lr_backbone),
            weight_decay=float(ocfg.weight_decay),
            layer_decay=layer_decay,
        )
    else:
        backbone_params = [p for p in backbone.parameters() if p.requires_grad]
        backbone_groups = []
        if backbone_params:
            backbone_groups.append(
                {
                    "params": backbone_params,
                    "lr": float(ocfg.lr_backbone),
                    "weight_decay": float(ocfg.weight_decay),
                }
            )

    if head_params:
        backbone_groups.append(
            {
                "params": head_params,
                "lr": float(ocfg.lr_head),
                "weight_decay": float(ocfg.weight_decay),
            }
        )

    return torch.optim.AdamW(
        backbone_groups,
        betas=tuple(getattr(ocfg, "betas", [0.9, 0.999])),
        eps=float(getattr(ocfg, "eps", 1e-8)),
    )


def build_scheduler(cfg, optimizer, steps_per_epoch: int):
    scfg = cfg.optim.scheduler
    total_steps = int(cfg.train.epochs) * max(steps_per_epoch, 1)
    warmup_steps = int(getattr(scfg, "warmup_epochs", 0)) * max(steps_per_epoch, 1)
    min_lr = float(getattr(scfg, "min_lr", 1e-6))
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        progress = 0.0
        if total_steps > warmup_steps:
            progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        max_base_lr = max(base_lrs) if base_lrs else 1.0
        min_scale = min_lr / max(max_base_lr, 1e-12)
        return min_scale + (1.0 - min_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def format_progress(current: int, total: int, width: int = 24) -> str:
    total = max(int(total), 1)
    current = min(max(int(current), 0), total)
    filled = int(round(width * current / total))
    filled = min(max(filled, 0), width)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total}"


def run_epoch(
    loader,
    model,
    tokenizers,
    tokenizer_specs,
    optimizer,
    scheduler,
    device,
    train: bool,
    diagnostics_enabled=False,
    collect_media_samples: int = 0,
    epoch: int | None = None,
    total_epochs: int | None = None,
    target_mode: str = "histogram",
    recording_names: list[str] | None = None,
):
    epoch_start = time.perf_counter()
    diagnostic_print(f"DIAGNOSTIC: Starting run_epoch with train={train}, loader has {len(loader)} batches", diagnostics_enabled)
    model.train(train)
    total_loss_sum = 0.0
    total_examples = 0.0
    primary_metric_name = "l1"
    secondary_metric_name = "mse"
    if target_mode == "sequence_codes":
        primary_metric_name = "ce"
        secondary_metric_name = "acc"
    per_hand_primary_sum = {hand: 0.0 for hand in tokenizers}
    per_hand_secondary_sum = {hand: 0.0 for hand in tokenizers}
    recording_names = list(recording_names or [])
    recording_to_idx = {name: idx for idx, name in enumerate(recording_names)}
    per_recording_loss_sum = [0.0 for _ in recording_names]
    per_recording_secondary_sum = [0.0 for _ in recording_names]
    per_recording_count = [0.0 for _ in recording_names]
    media_samples = []
    data_wait_total = 0.0
    target_encode_total = 0.0
    forward_total = 0.0
    backward_total = 0.0
    media_total = 0.0
    batch_count = 0
    next_batch_start = time.perf_counter()
    for batch in loader:
        data_wait_total += time.perf_counter() - next_batch_start
        batch_count += 1
        if batch_count % 10 == 1 or batch_count == len(loader):
            diagnostic_print(f"DIAGNOSTIC: Processing batch {batch_count}/{len(loader)}", diagnostics_enabled)
        video = batch["video"].to(device, non_blocking=True)

        t_targets = time.perf_counter()
        with torch.no_grad():
            targets = {}
            for hand, tokenizer in tokenizers.items():
                vqvae_emg = batch["vqvae_emg"][hand].to(device, non_blocking=True)
                targets[hand] = tokenizer.encode_ids(vqvae_emg)
        target_encode_total += time.perf_counter() - t_targets

        t_forward = time.perf_counter()
        predictions = model(video)
        forward_total += time.perf_counter() - t_forward
        hand_losses = []
        batch_recordings = [str(name) for name in batch["recording"]]
        batch_size = int(video.shape[0])
        per_example_loss_terms = []
        per_example_secondary_terms = []
        for hand in tokenizers:
            spec = tokenizer_specs[hand]
            if target_mode == "histogram":
                pred_target = torch.softmax(predictions[hand], dim=-1)
                target_target = code_histogram_from_ids(targets[hand], n_embed=spec.n_embed, normalize=True)
                hand_primary = torch.abs(pred_target - target_target).mean()
                hand_secondary = ((pred_target - target_target) ** 2).mean()
                per_example_primary = torch.abs(pred_target - target_target).mean(dim=-1)
                per_example_secondary = ((pred_target - target_target) ** 2).mean(dim=-1)
            elif target_mode == "sequence_codes":
                token_steps = token_steps_for_window(spec.fs, spec.codes_per_second, spec.window_sec)
                pred_logits = predictions[hand].view(batch_size, token_steps, spec.n_embed)
                target_ids = targets[hand].long()
                hand_primary = F.cross_entropy(pred_logits.reshape(batch_size * token_steps, spec.n_embed), target_ids.reshape(batch_size * token_steps))
                hand_secondary = (pred_logits.argmax(dim=-1) == target_ids).to(dtype=torch.float32).mean()
                per_example_primary = F.cross_entropy(
                    pred_logits.reshape(batch_size * token_steps, spec.n_embed),
                    target_ids.reshape(batch_size * token_steps),
                    reduction="none",
                ).view(batch_size, token_steps).mean(dim=-1)
                per_example_secondary = (pred_logits.argmax(dim=-1) == target_ids).to(dtype=torch.float32).mean(dim=-1)
            else:
                token_steps = token_steps_for_window(spec.fs, spec.codes_per_second, spec.window_sec)
                pred_target = predictions[hand].view(batch_size, token_steps, spec.embed_dim)
                model_eval = tokenizers[hand].module if hasattr(tokenizers[hand], "module") else tokenizers[hand]
                target_target = codebook_embeddings_from_ids(model_eval, targets[hand]).contiguous()
                hand_primary = torch.abs(pred_target - target_target).mean()
                hand_secondary = ((pred_target - target_target) ** 2).mean()
                per_example_primary = torch.abs(pred_target - target_target).mean(dim=(1, 2))
                per_example_secondary = ((pred_target - target_target) ** 2).mean(dim=(1, 2))
            hand_losses.append(hand_primary)
            per_hand_primary_sum[hand] += float(hand_primary.detach().item()) * batch_size
            per_hand_secondary_sum[hand] += float(hand_secondary.detach().item()) * batch_size
            per_example_loss_terms.append(per_example_primary.detach())
            per_example_secondary_terms.append(per_example_secondary.detach())

        loss = sum(hand_losses) / max(len(hand_losses), 1)
        if per_example_loss_terms and recording_to_idx:
            batch_example_loss = torch.stack(per_example_loss_terms, dim=0).mean(dim=0)
            batch_example_secondary = torch.stack(per_example_secondary_terms, dim=0).mean(dim=0)
            for example_idx, recording_name in enumerate(batch_recordings):
                rec_idx = recording_to_idx.get(recording_name)
                if rec_idx is None:
                    continue
                per_recording_loss_sum[rec_idx] += float(batch_example_loss[example_idx].item())
                per_recording_secondary_sum[rec_idx] += float(batch_example_secondary[example_idx].item())
                per_recording_count[rec_idx] += 1.0

        if train:
            t_backward = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            backward_total += time.perf_counter() - t_backward

        if collect_media_samples > 0 and len(media_samples) < collect_media_samples:
            t_media = time.perf_counter()
            remaining = collect_media_samples - len(media_samples)
            media_samples.extend(
                build_media_samples(
                    batch=batch,
                    tokenizers=tokenizers,
                    tokenizer_specs=tokenizer_specs,
                    predictions={hand: predictions[hand].detach() for hand in tokenizers},
                    targets={hand: targets[hand].detach() for hand in tokenizers},
                    device=device,
                    max_samples=remaining,
                    target_mode=target_mode,
                )
            )
            media_total += time.perf_counter() - t_media

        total_loss_sum += float(loss.item()) * batch_size
        total_examples += batch_size
        if is_main_process():
            split = "train" if train else "val"
            progress = 100.0 * batch_count / max(len(loader), 1)
            prefix = f"epoch={epoch}/{total_epochs} " if epoch is not None and total_epochs is not None else ""
            print(
                f"{prefix}[{split}] iter_progress={format_progress(batch_count, len(loader))} ({progress:.1f}%) "
                f"loss={loss.item():.4f}",
                flush=True,
            )
        if diagnostics_enabled and (batch_count <= 5 or batch_count % 100 == 0 or batch_count == len(loader)):
            split = "train" if train else "val"
            diagnostic_print(
                "DIAGNOSTIC_TIMING: "
                f"{split} batch={batch_count}/{len(loader)} "
                f"data_wait_avg={data_wait_total/max(batch_count, 1):.3f}s "
                f"target_encode_avg={target_encode_total/max(batch_count, 1):.3f}s "
                f"forward_avg={forward_total/max(batch_count, 1):.3f}s "
                f"backward_avg={backward_total/max(batch_count, 1):.3f}s "
                f"media_avg={media_total/max(batch_count, 1):.3f}s",
                diagnostics_enabled,
            )
        if batch_count % 10 == 0 or batch_count == len(loader):
            diagnostic_print(f"DIAGNOSTIC: Batch {batch_count}/{len(loader)} completed, loss: {loss.item():.4f}", diagnostics_enabled)
        next_batch_start = time.perf_counter()

    stats_values = (
        [total_loss_sum, total_examples]
        + [per_hand_primary_sum[hand] for hand in tokenizers]
        + [per_hand_secondary_sum[hand] for hand in tokenizers]
        + per_recording_loss_sum
        + per_recording_secondary_sum
        + per_recording_count
    )
    stats_tensor = torch.tensor(
        stats_values,
        dtype=torch.float64,
        device=device,
    )
    if is_dist_avail_and_initialized():
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

    denom = max(float(stats_tensor[1].item()), 1.0)
    stats = {"loss": float(stats_tensor[0].item()) / denom}
    offset = 2
    for hand in tokenizers:
        stats[f"{hand}/{primary_metric_name}"] = float(stats_tensor[offset].item()) / denom
        offset += 1
    for hand in tokenizers:
        stats[f"{hand}/{secondary_metric_name}"] = float(stats_tensor[offset].item()) / denom
        offset += 1
    if recording_names:
        loss_offset = 2 + 2 * len(tokenizers)
        secondary_offset = loss_offset + len(recording_names)
        count_offset = secondary_offset + len(recording_names)
        recording_stats = {}
        for rec_idx, recording_name in enumerate(recording_names):
            rec_count = float(stats_tensor[count_offset + rec_idx].item())
            if rec_count <= 0:
                continue
            recording_stats[recording_name] = {
                "loss": float(stats_tensor[loss_offset + rec_idx].item()) / rec_count,
                secondary_metric_name: float(stats_tensor[secondary_offset + rec_idx].item()) / rec_count,
                "count": rec_count,
            }
    stats["per_recording"] = recording_stats
    diagnostic_print(f"DIAGNOSTIC: run_epoch completed, final loss: {stats['loss']:.4f}", diagnostics_enabled)
    diagnostic_print_timing(f"run_epoch({'train' if train else 'val'}) total", time.perf_counter() - epoch_start, diagnostics_enabled)
    return stats, media_samples


def save_checkpoint(path: Path, model, optimizer, epoch: int, tokenizer_specs, stats, cfg):
    checkpoint = {
        "epoch": epoch,
        "model": (model.module if hasattr(model, "module") else model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "tokenizer_specs": {hand: spec.__dict__ for hand, spec in tokenizer_specs.items()},
        "stats": stats,
        "config": namespace_to_dict(cfg),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def maybe_save_checkpoints(cfg, experiment_dir: Path, model, optimizer, epoch: int, stats, tokenizer_specs, best_value):
    if not is_main_process():
        return best_value

    ccfg = cfg.checkpoint
    if not bool(getattr(ccfg, "enabled", True)):
        return best_value

    save_dir = experiment_dir / "checkpoints"
    monitor_key = str(ccfg.monitor)
    monitor_value = stats["val"].get(monitor_key.replace("val/", ""), stats["val"]["loss"]) if "/" not in monitor_key else stats["val"].get(monitor_key[4:], stats["val"]["loss"])
    if monitor_key in ("val/loss", "loss"):
        monitor_value = stats["val"]["loss"]
    else:
        monitor_value = stats["val"].get(monitor_key.replace("val/", ""), stats["val"]["loss"])

    if bool(getattr(ccfg, "save_last", True)):
        save_checkpoint(save_dir / "last.pt", model, optimizer, epoch, tokenizer_specs, stats, cfg)

    improved = False
    if str(ccfg.mode).lower() == "min":
        if monitor_value < best_value:
            best_value = monitor_value
            improved = True
    else:
        if monitor_value > best_value:
            best_value = monitor_value
            improved = True

    if bool(getattr(ccfg, "save_best", True)) and improved:
        save_checkpoint(save_dir / "best.pt", model, optimizer, epoch, tokenizer_specs, stats, cfg)

    save_freq = int(getattr(ccfg, "save_freq", 0))
    if save_freq > 0 and epoch % save_freq == 0:
        save_checkpoint(
            save_dir / f"epoch_{epoch:03d}.pt",
            model,
            optimizer,
            epoch,
            tokenizer_specs,
            stats,
            cfg,
        )

    return best_value


def main():
    main_start = time.perf_counter()
    args = parse_args()
    diagnostics_enabled = args.diagnostics
    diagnostic_print("DIAGNOSTIC: Starting main()", diagnostics_enabled)
    t0 = time.perf_counter()
    cfg = load_config(args.config)
    diagnostic_print("DIAGNOSTIC: Config loaded", diagnostics_enabled)
    diagnostic_print_timing("main.load_config", time.perf_counter() - t0, diagnostics_enabled)
    cfg = apply_overrides(cfg, args)
    diagnostic_print("DIAGNOSTIC: Config overrides applied", diagnostics_enabled)

    device, distributed, rank, world_size, local_rank = init_distributed(cfg, diagnostics_enabled)
    diagnostic_print(f"DIAGNOSTIC: Using device: {device}", diagnostics_enabled)
    set_seed(int(cfg.experiment.seed) + rank)
    diagnostic_print("DIAGNOSTIC: Seed set", diagnostics_enabled)

    tokenizer_cfg_map = get_tokenizer_cfg_map(cfg)
    diagnostic_print("DIAGNOSTIC: Tokenizer config map created", diagnostics_enabled)
    tokenizer_specs = load_joint_tokenizer_specs(tokenizer_cfg_map)
    diagnostic_print("DIAGNOSTIC: Tokenizer specs loaded", diagnostics_enabled)
    if is_main_process():
        print("[emg_pretraining] resolved tokenizers:", flush=True)
        for hand, spec in tokenizer_specs.items():
            print(
                f"  hand={hand} run_name={spec.run_name} checkpoint_path={spec.checkpoint_path}",
                flush=True,
            )
    target_mode = get_target_mode(cfg)
    diagnostic_print(f"DIAGNOSTIC: target_mode={target_mode}", diagnostics_enabled)

    if args.experiment_dir is not None:
        experiment_dir = Path(args.experiment_dir)
    else:
        suffix = "__".join(f"{hand}_{spec.run_name}" for hand, spec in tokenizer_specs.items())
        experiment_dir = Path(cfg.experiment.root) / suffix
    experiment_dir.mkdir(parents=True, exist_ok=True)
    if is_dist_avail_and_initialized():
        barrier_device_ids = [local_rank] if device.type == "cuda" else None
        dist.barrier(device_ids=barrier_device_ids)
    diagnostic_print(f"DIAGNOSTIC: Experiment dir created: {experiment_dir}", diagnostics_enabled)

    diagnostic_print("DIAGNOSTIC: Loading tokenizers...", diagnostics_enabled)
    t0 = time.perf_counter()
    tokenizers = {
        hand: load_vqvae_model(spec, in_channels=int(spec.input_channels), device=device)
        for hand, spec in tokenizer_specs.items()
    }
    diagnostic_print("DIAGNOSTIC: Tokenizers loaded", diagnostics_enabled)
    diagnostic_print_timing("main.load_tokenizers", time.perf_counter() - t0, diagnostics_enabled)

    head_dims = {
        hand: {
            "num_queries": (
                1
                if target_mode == "histogram"
                else token_steps_for_window(spec.fs, spec.codes_per_second, spec.window_sec)
            ),
            "query_output_dim": (
                spec.n_embed
                if target_mode in {"histogram", "sequence_codes"}
                else spec.embed_dim
            ),
            "output_dim": (
                spec.n_embed
                if target_mode == "histogram"
                else token_steps_for_window(spec.fs, spec.codes_per_second, spec.window_sec) * (
                    spec.n_embed if target_mode == "sequence_codes" else spec.embed_dim
                )
            ),
        }
        for hand, spec in tokenizer_specs.items()
    }
    diagnostic_print("DIAGNOSTIC: Head dims calculated", diagnostics_enabled)

    diagnostic_print("DIAGNOSTIC: Building dataloaders...", diagnostics_enabled)
    t0 = time.perf_counter()
    train_loader, val_loader, train_ds, val_ds, train_sampler, val_sampler = build_dataloaders(cfg, tokenizer_specs, diagnostics_enabled)
    diagnostic_print("DIAGNOSTIC: Dataloaders built", diagnostics_enabled)
    diagnostic_print_timing("main.build_dataloaders", time.perf_counter() - t0, diagnostics_enabled)

    diagnostic_print("DIAGNOSTIC: Building model...", diagnostics_enabled)
    t0 = time.perf_counter()
    model = HieraVideoMultiHeadCodePredictor(
        head_dims=head_dims,
        variant=str(cfg.model.variant),
        checkpoint=str(cfg.model.checkpoint),
        pretrained=bool(cfg.model.pretrained),
        strict=bool(cfg.model.strict),
        dropout=float(cfg.model.dropout),
        drop_path=float(cfg.model.drop_path),
        freeze_backbone=bool(cfg.model.freeze_backbone),
        input_layout=str(cfg.model.input_layout),
        backbone_ckpt=getattr(cfg.model, "backbone_ckpt", None),
        finetune_last_layer_only=bool(getattr(cfg.model, "finetune_last_layer_only", False)),
        last_layer_blocks=int(getattr(cfg.model, "last_layer_blocks", 1)),
        mode=str(getattr(cfg.model, "mode", "video")),
    ).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=False)
    diagnostic_print("DIAGNOSTIC: Model built and moved to device", diagnostics_enabled)
    diagnostic_print_timing("main.build_model", time.perf_counter() - t0, diagnostics_enabled)
    if is_main_process():
        model_for_report = model.module if hasattr(model, "module") else model
        print(
            "[emg_pretraining] model load report:\n"
            + json.dumps(getattr(model_for_report, "pretrained_load_report", {}), indent=2),
            flush=True,
        )

    optimizer = build_optimizer(cfg, model.module if hasattr(model, "module") else model)
    diagnostic_print("DIAGNOSTIC: Optimizer built", diagnostics_enabled)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))
    diagnostic_print("DIAGNOSTIC: Scheduler built", diagnostics_enabled)
    wandb_run = maybe_init_wandb(cfg, tokenizer_specs, experiment_dir)
    diagnostic_print("DIAGNOSTIC: Wandb initialized", diagnostics_enabled)

    cfg_dict = namespace_to_dict(cfg)
    cfg_dict["resolved_tokenizers"] = {
        hand: {
            "run_name": spec.run_name,
            "checkpoint_path": spec.checkpoint_path,
            "embed_dim": spec.embed_dim,
            "n_embed": spec.n_embed,
            "target_mode": target_mode,
            "target_output_dim": head_dims[hand]["output_dim"],
            "target_embed_dim": spec.embed_dim if target_mode == "sequence_embeddings" else None,
            "target_token_steps": token_steps_for_window(spec.fs, spec.codes_per_second, float(spec.window_sec)),
            "target_codebook_size": spec.n_embed if target_mode == "sequence_codes" else None,
            "codes_per_second": spec.codes_per_second,
            "emg_window_size": spec.emg_window_size,
            "histogram_bins": spec.n_embed if target_mode == "histogram" else None,
            "model_token_steps": token_steps_for_window(spec.fs, spec.codes_per_second, float(cfg.data.model_window_sec)),
            "vqvae_token_steps": token_steps_for_window(spec.fs, spec.codes_per_second, float(spec.window_sec)),
        }
        for hand, spec in tokenizer_specs.items()
    }
    cfg_dict["train_size"] = len(train_ds)
    cfg_dict["val_size"] = len(val_ds)
    cfg_dict["n_train_recordings"] = len(train_ds.recording_states)
    cfg_dict["n_val_recordings"] = len(val_ds.recording_states)
    cfg_dict["n_train_participants"] = len({state.recording.participant for state in train_ds.recording_states})
    cfg_dict["n_val_participants"] = len({state.recording.participant for state in val_ds.recording_states})
    cfg_dict["distributed"] = {
        "enabled": distributed,
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
    }
    if is_main_process():
        save_json(experiment_dir / "config_resolved.json", cfg_dict)
    diagnostic_print("DIAGNOSTIC: Config saved", diagnostics_enabled)

    best_value = float("inf") if str(cfg.checkpoint.mode).lower() == "min" else -float("inf")
    history = []
    diagnostic_print("DIAGNOSTIC: Starting training loop", diagnostics_enabled)

    visualize_n = int(getattr(getattr(cfg, "wandb", None), "visualize_n_samples", 0) or 0)
    total_epochs = int(cfg.train.epochs)
    train_recording_names = sorted(getattr(train_ds, "center_indices_by_recording", {}).keys())
    val_recording_names = sorted(getattr(val_ds, "center_indices_by_recording", {}).keys())
    for epoch in range(1, total_epochs + 1):
        diagnostic_print(f"DIAGNOSTIC: Starting epoch {epoch}", diagnostics_enabled)
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if distributed and val_sampler is not None:
            val_sampler.set_epoch(epoch)

        train_stats, train_media_samples = run_epoch(
            train_loader,
            model,
            tokenizers,
            tokenizer_specs,
            optimizer,
            scheduler,
            device,
            train=True,
            diagnostics_enabled=diagnostics_enabled,
            collect_media_samples=visualize_n if is_main_process() else 0,
            epoch=epoch,
            total_epochs=total_epochs,
            target_mode=target_mode,
            recording_names=train_recording_names,
        )
        diagnostic_print(f"DIAGNOSTIC: Training epoch {epoch} completed", diagnostics_enabled)
        val_stats, val_media_samples = run_epoch(
            val_loader,
            model,
            tokenizers,
            tokenizer_specs,
            optimizer,
            scheduler=None,
            device=device,
            train=False,
            diagnostics_enabled=diagnostics_enabled,
            collect_media_samples=visualize_n if is_main_process() else 0,
            epoch=epoch,
            total_epochs=total_epochs,
            target_mode=target_mode,
            recording_names=val_recording_names,
        )
        diagnostic_print(f"DIAGNOSTIC: Validation epoch {epoch} completed", diagnostics_enabled)

        stats = {"train": train_stats, "val": val_stats}
        if is_main_process():
            history.append({"epoch": epoch, **stats})
            save_json(experiment_dir / "history.json", history)
        best_value = maybe_save_checkpoints(
            cfg,
            experiment_dir,
            model,
            optimizer,
            epoch,
            stats,
            tokenizer_specs,
            best_value,
        )

        if wandb_run is not None:
            payload = {"epoch": epoch, "train/loss": train_stats["loss"], "val/loss": val_stats["loss"]}
            for hand in tokenizer_specs:
                if target_mode == "sequence_codes":
                    payload[f"train/{hand}/ce"] = train_stats[f"{hand}/ce"]
                    payload[f"train/{hand}/acc"] = train_stats[f"{hand}/acc"]
                    payload[f"val/{hand}/ce"] = val_stats[f"{hand}/ce"]
                    payload[f"val/{hand}/acc"] = val_stats[f"{hand}/acc"]
                else:
                    payload[f"train/{hand}/l1"] = train_stats[f"{hand}/l1"]
                    payload[f"train/{hand}/mse"] = train_stats[f"{hand}/mse"]
                    payload[f"val/{hand}/l1"] = val_stats[f"{hand}/l1"]
                    payload[f"val/{hand}/mse"] = val_stats[f"{hand}/mse"]
            for recording_name, recording_stats in train_stats.get("per_recording", {}).items():
                payload[f"train_recordings/{recording_name}/loss"] = recording_stats["loss"]
                if target_mode == "sequence_codes":
                    payload[f"train_recordings/{recording_name}/code_acc"] = recording_stats["acc"]
            for recording_name, recording_stats in val_stats.get("per_recording", {}).items():
                payload[f"val_recordings/{recording_name}/loss"] = recording_stats["loss"]
                if target_mode == "sequence_codes":
                    payload[f"val_recordings/{recording_name}/code_acc"] = recording_stats["acc"]
            wandb_run.log(payload)
            log_vqvae_reconstruction_media(wandb_run, train_media_samples, prefix=f"train_media/epoch_{epoch}")
            log_vqvae_reconstruction_media(wandb_run, val_media_samples, prefix=f"val_media/epoch_{epoch}")

        if is_main_process():
            parts = [
                f"epoch={epoch}",
                f"train_loss={train_stats['loss']:.4f}",
                f"val_loss={val_stats['loss']:.4f}",
            ]
            for hand in tokenizer_specs:
                if target_mode == "sequence_codes":
                    parts.append(f"train_{hand}_ce={train_stats[f'{hand}/ce']:.4f}")
                    parts.append(f"val_{hand}_ce={val_stats[f'{hand}/ce']:.4f}")
                    parts.append(f"train_{hand}_acc={train_stats[f'{hand}/acc']:.4f}")
                    parts.append(f"val_{hand}_acc={val_stats[f'{hand}/acc']:.4f}")
                else:
                    parts.append(f"train_{hand}_l1={train_stats[f'{hand}/l1']:.4f}")
                    parts.append(f"val_{hand}_l1={val_stats[f'{hand}/l1']:.4f}")
            print(" ".join(parts), flush=True)

    if wandb_run is not None:
        wandb_run.finish()
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()
    diagnostic_print("DIAGNOSTIC: Training completed", diagnostics_enabled)
    diagnostic_print_timing("main.total", time.perf_counter() - main_start, diagnostics_enabled)


if __name__ == "__main__":
    main()
