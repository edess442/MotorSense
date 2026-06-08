from __future__ import annotations

import importlib.util
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Optional
import sys

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, iirnotch


REPO_ROOT = Path(__file__).resolve().parent
VQVAE_MODULE_PATH = REPO_ROOT / "vqvae_emg.py"
UTILS_MODULE_PATH = REPO_ROOT / "utils.py"
FIXED_CHECKPOINT_EPOCH = 20


def expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def _load_vqvae_module():
    spec = importlib.util.spec_from_file_location("emg_vqvae_module", VQVAE_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load VQ-VAE module from {VQVAE_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except TypeError as exc:
        if "unsupported operand type(s) for |" not in str(exc):
            raise

    source = VQVAE_MODULE_PATH.read_text()
    source = source.replace("from typing import Dict, Tuple, Optional", "from typing import Dict, Tuple, Optional, Union")
    source = source.replace(
        "def decode(self, encoded: Dict[str, torch.Tensor] | torch.Tensor, output_length: Optional[int] = None) -> torch.Tensor:",
        "def decode(self, encoded: Union[Dict[str, torch.Tensor], torch.Tensor], output_length: Optional[int] = None) -> torch.Tensor:",
    )
    fallback_module = ModuleType("emg_vqvae_module_py39_compat")
    fallback_module.__file__ = str(VQVAE_MODULE_PATH)
    sys.modules[fallback_module.__name__] = fallback_module
    exec(compile(source, str(VQVAE_MODULE_PATH), "exec"), fallback_module.__dict__)
    return fallback_module


def _load_utils_module():
    spec = importlib.util.spec_from_file_location("emg_vqvae_utils_module", UTILS_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load utils module from {UTILS_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class TokenizerSpec:
    run_name: str
    checkpoint_path: str
    embed_dim: int
    n_embed: int
    codes_per_second: float
    emg_window_size: int
    input_channels: int = 8
    window_sec: float = 1.5
    fs: int = 500
    channel: int = 256
    decay: float = 0.99
    tds_blocks: int = 2
    tds_channels: int = 8
    kernel_width: int = 5
    dropout: float = 0.0

    @property
    def checkpoint_stem(self) -> str:
        return Path(self.checkpoint_path).stem

    def compatibility_dict(self) -> dict[str, object]:
        return {
            "embed_dim": int(self.embed_dim),
            "n_embed": int(self.n_embed),
            "codes_per_second": float(self.codes_per_second),
            "emg_window_size": int(self.emg_window_size),
            "input_channels": int(self.input_channels),
            "window_sec": float(self.window_sec),
            "fs": int(self.fs),
            "channel": int(self.channel),
            "decay": float(self.decay),
            "tds_blocks": int(self.tds_blocks),
            "tds_channels": int(self.tds_channels),
            "kernel_width": int(self.kernel_width),
            "dropout": float(self.dropout),
        }


def nearest_power_of_two(x: float, min_value: int = 1) -> int:
    x = max(float(min_value), float(x))
    return int(2 ** round(math.log2(x)))


def make_compatible_window_size(n: int, base: int) -> int:
    return max(int(base), int(round(float(n) / float(base)) * base))


def highpass_only_emg(data: np.ndarray, fs: int = 500, cutoff: float = 10.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    high = cutoff / nyq
    b, a = butter(order, high, btype="high")
    return filtfilt(b, a, data, axis=0)


def zscore_per_channel(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (x - mean) / std


def _find_numeric_columns(df: pd.DataFrame):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def pick_signal_columns(df: pd.DataFrame, emg_only: bool) -> tuple[list[str], list[str], list[str]]:
    cols = list(df.columns)
    numeric_cols = _find_numeric_columns(df)

    time_like = {"unix_time_s", "unix_time", "timestamp", "time", "time_s"}
    numeric_signal_cols = [c for c in numeric_cols if c not in time_like]

    emg_cols = [c for c in numeric_signal_cols if ("raw" in c.lower() and "emg" in c.lower())]
    accel_cols = [c for c in numeric_signal_cols if any(k in c.lower() for k in ["accel", "acc_", "accx", "accy", "accz"])]
    gyro_cols = [c for c in numeric_signal_cols if any(k in c.lower() for k in ["gyro", "gyr_", "gyrox", "gyroy", "gyroz"])]

    if len(emg_cols) == 0:
        if emg_only:
            emg_cols = numeric_signal_cols
        else:
            emg_cols = numeric_signal_cols
            accel_cols = []
            gyro_cols = []

    emg_cols = [c for c in cols if c in emg_cols]
    accel_cols = [c for c in cols if c in accel_cols and c not in emg_cols]
    gyro_cols = [c for c in cols if c in gyro_cols and c not in emg_cols and c not in accel_cols]

    if emg_only:
        return emg_cols, [], []
    return emg_cols, accel_cols, gyro_cols


def get_vqvae_input(
    raw_emg: np.ndarray,
    window_size: int,
    fs: int = 500,
    cutoff: float = 8.0,
    notch_freq: float = 60.0,
    notch_q: float = 30.0,
    highpass_order: int = 4,
) -> np.ndarray:
    module = _load_utils_module()
    processed = module.get_emg_data(
        data=raw_emg,
        window_size=window_size,
        fs=fs,
        cutoff=cutoff,
        notch_freq=notch_freq,
        notch_q=notch_q,
        highpass_order=highpass_order,
    )
    return zscore_per_channel(np.asarray(processed, dtype=np.float32)).astype(np.float32)


def _checkpoint_path_for_run(sweep_root: Path, run_name: str, epoch: int = FIXED_CHECKPOINT_EPOCH) -> Path:
    return sweep_root / run_name / "checkpoint" / f"vqvae_{int(epoch):03d}.pt"


def _fixed_epoch_checkpoint_from_explicit_path(checkpoint_path: str, epoch: int = FIXED_CHECKPOINT_EPOCH) -> Path:
    path = expand_path(checkpoint_path)
    return path.with_name(f"vqvae_{int(epoch):03d}.pt")


def infer_checkpoint_metadata(checkpoint_path: str) -> dict[str, int]:
    state = torch.load(checkpoint_path, map_location="cpu")
    block_key_suffix = ".blocks."
    residual_stack_prefixes = sorted(
        {
            key.split(block_key_suffix, 1)[0] + block_key_suffix
            for key in state.keys()
            if "encoder.net." in key and block_key_suffix in key
        }
    )
    if not residual_stack_prefixes:
        raise ValueError(f"Could not infer residual-stack layout from checkpoint {checkpoint_path}")
    residual_stack_prefix = residual_stack_prefixes[-1]
    tds_block_ids = {
        int(key[len(residual_stack_prefix):].split(".", 1)[0])
        for key in state.keys()
        if key.startswith(residual_stack_prefix)
    }
    first_block_conv_key = f"{residual_stack_prefix}0.block.0.weight"
    encoder_conv_keys = []
    for key in state.keys():
        if not key.startswith("encoder.net.") or not key.endswith(".weight") or state[key].ndim != 3:
            continue
        if ".blocks." in key:
            continue
        encoder_conv_keys.append(key)
    encoder_conv_keys.sort(key=lambda key: int(key.split(".")[2]))
    return {
        "input_channels": int(state["encoder.net.0.weight"].shape[1]),
        "channel": int(state["codebook.proj.weight"].shape[1]),
        "embed_dim": int(state["codebook.embed"].shape[0]),
        "n_embed": int(state["codebook.embed"].shape[1]),
        "tds_blocks": max(len(tds_block_ids), 1),
        "tds_channels": int(state[first_block_conv_key].shape[0]),
        "kernel_width": int(state[encoder_conv_keys[-1]].shape[-1]),
    }


def load_tokenizer_spec_from_row(row: pd.Series, sweep_root: Path) -> TokenizerSpec:
    run_name = str(row["run_name"])
    checkpoint_path = _checkpoint_path_for_run(sweep_root=sweep_root, run_name=run_name, epoch=FIXED_CHECKPOINT_EPOCH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Expected checkpoint {checkpoint_path} for run {run_name}")
    meta = infer_checkpoint_metadata(str(checkpoint_path))
    return TokenizerSpec(
        run_name=run_name,
        checkpoint_path=str(checkpoint_path),
        embed_dim=int(meta["embed_dim"]),
        n_embed=int(meta["n_embed"]),
        codes_per_second=float(row["codes_per_second"]),
        emg_window_size=int(row["emg_window_size"]),
        input_channels=int(meta["input_channels"]),
        channel=int(meta["channel"]),
        tds_blocks=int(meta["tds_blocks"]),
        tds_channels=int(meta["tds_channels"]),
        kernel_width=int(meta["kernel_width"]),
    )


def load_tokenizer_spec_from_checkpoint_row(row: pd.Series, sweep_root: Path) -> TokenizerSpec | None:
    run_name = str(row["run_name"])
    parent = sweep_root / run_name
    if not parent.exists():
        return None
    checkpoint_path = _checkpoint_path_for_run(sweep_root=sweep_root, run_name=run_name, epoch=FIXED_CHECKPOINT_EPOCH)
    if not checkpoint_path.exists():
        return None
    meta = infer_checkpoint_metadata(str(checkpoint_path))
    return TokenizerSpec(
        run_name=run_name,
        checkpoint_path=str(checkpoint_path),
        embed_dim=int(meta["embed_dim"]),
        n_embed=int(meta["n_embed"]),
        codes_per_second=float(row["codes_per_second"]),
        emg_window_size=int(row["emg_window_size"]),
        input_channels=int(meta["input_channels"]),
        channel=int(meta["channel"]),
        tds_blocks=int(meta["tds_blocks"]),
        tds_channels=int(meta["tds_channels"]),
        kernel_width=int(meta["kernel_width"]),
    )


def _resolve_results_path(results_csv: str) -> Path:
    results_path = expand_path(results_csv)
    if not results_path.exists() and results_path.name == "results_sorted.csv":
        fallback = results_path.with_name("results.csv")
        if fallback.exists():
            results_path = fallback
    return results_path


def _filtered_results_df(results_path: Path) -> pd.DataFrame:
    df = pd.read_csv(results_path)
    if "status" in df.columns:
        df = df[df["status"].fillna("").str.upper() != "FAILED"].copy()
    if "score" in df.columns and not df.empty:
        df = df.sort_values("score", ascending=False, na_position="last")
    return df


def infer_model_kwargs_from_checkpoint(checkpoint_path: str, in_channels: int, spec: TokenizerSpec) -> dict[str, object]:
    return {
        "in_channel": int(in_channels),
        "channel": int(spec.channel),
        "embed_dim": int(spec.embed_dim),
        "n_embed": int(spec.n_embed),
        "decay": float(spec.decay),
        "tds_blocks": int(spec.tds_blocks),
        "tds_channels": int(spec.tds_channels),
        "kernel_width": int(spec.kernel_width),
        "dropout": float(spec.dropout),
        "fs": int(spec.fs),
        "codes_per_second": float(spec.codes_per_second),
    }


def _candidate_specs_from_results(tokenizer_cfg) -> list[TokenizerSpec]:
    results_csv = getattr(tokenizer_cfg, "results_csv", None)
    if not results_csv:
        raise ValueError("Tokenizer config must provide results_csv for candidate loading")

    results_path = _resolve_results_path(results_csv)
    sweep_root = results_path.parent
    df = _filtered_results_df(results_path)
    specs = []
    seen = set()

    if not df.empty:
        for _, row in df.iterrows():
            try:
                spec = load_tokenizer_spec_from_row(row, sweep_root=sweep_root)
            except FileNotFoundError:
                continue
            spec.window_sec = float(getattr(tokenizer_cfg, "window_sec", 1.5))
            spec.fs = int(getattr(tokenizer_cfg, "fs", 500))
            key = (spec.run_name, spec.checkpoint_path)
            if key not in seen:
                specs.append(spec)
                seen.add(key)

    raw_df = pd.read_csv(results_path)
    for _, row in raw_df.iterrows():
        spec = load_tokenizer_spec_from_checkpoint_row(row, sweep_root=sweep_root)
        if spec is None:
            continue
        spec.window_sec = float(getattr(tokenizer_cfg, "window_sec", 1.5))
        spec.fs = int(getattr(tokenizer_cfg, "fs", 500))
        key = (spec.run_name, spec.checkpoint_path)
        if key not in seen:
            specs.append(spec)
            seen.add(key)

    if not specs:
        raise ValueError(f"No tokenizer runs found in {results_path}")
    return specs


def load_tokenizer_spec_from_config(tokenizer_cfg) -> TokenizerSpec:
    checkpoint_path = getattr(tokenizer_cfg, "checkpoint_path", None)
    run_name = getattr(tokenizer_cfg, "run_name", None)
    results_csv = getattr(tokenizer_cfg, "results_csv", None)

    if checkpoint_path and run_name:
        resolved_checkpoint = _fixed_epoch_checkpoint_from_explicit_path(
            str(checkpoint_path),
            epoch=FIXED_CHECKPOINT_EPOCH,
        )
        if not resolved_checkpoint.exists():
            raise FileNotFoundError(
                f"Expected epoch-{FIXED_CHECKPOINT_EPOCH} checkpoint at {resolved_checkpoint} "
                f"for run {run_name}"
            )
        meta = infer_checkpoint_metadata(str(resolved_checkpoint))
        return TokenizerSpec(
            run_name=str(run_name),
            checkpoint_path=str(resolved_checkpoint),
            embed_dim=int(meta["embed_dim"]),
            n_embed=int(meta["n_embed"]),
            codes_per_second=float(tokenizer_cfg.codes_per_second),
            emg_window_size=int(tokenizer_cfg.emg_window_size),
            input_channels=int(meta["input_channels"]),
            window_sec=float(getattr(tokenizer_cfg, "window_sec", 1.5)),
            fs=int(getattr(tokenizer_cfg, "fs", 500)),
            channel=int(meta["channel"]),
            tds_blocks=int(meta["tds_blocks"]),
            tds_channels=int(meta["tds_channels"]),
            kernel_width=int(meta["kernel_width"]),
        )

    if not results_csv:
        raise ValueError("Tokenizer config must provide either explicit checkpoint fields or results_csv")

    results_path = _resolve_results_path(results_csv)
    sweep_root = results_path.parent
    df = _filtered_results_df(results_path)
    if run_name:
        row = df[df["run_name"] == run_name]
        if row.empty:
            raw_df = pd.read_csv(results_path)
            raw_row = raw_df[raw_df["run_name"] == run_name]
            if not raw_row.empty:
                spec = load_tokenizer_spec_from_checkpoint_row(raw_row.iloc[0], sweep_root=sweep_root)
                if spec is not None:
                    spec.window_sec = float(getattr(tokenizer_cfg, "window_sec", 1.5))
                    spec.fs = int(getattr(tokenizer_cfg, "fs", 500))
                    return spec
            raise ValueError(f"Could not find run_name={run_name} in {results_path}")
        selected = row.iloc[0]
    else:
        if df.empty:
            raw_df = pd.read_csv(results_path)
            for _, row in raw_df.iterrows():
                spec = load_tokenizer_spec_from_checkpoint_row(row, sweep_root=sweep_root)
                if spec is not None:
                    spec.window_sec = float(getattr(tokenizer_cfg, "window_sec", 1.5))
                    spec.fs = int(getattr(tokenizer_cfg, "fs", 500))
                    return spec
            raise ValueError(f"No tokenizer runs found in {results_path}")
        selected = df.iloc[0]

    spec = load_tokenizer_spec_from_row(selected, sweep_root=sweep_root)
    spec.window_sec = float(getattr(tokenizer_cfg, "window_sec", 1.5))
    spec.fs = int(getattr(tokenizer_cfg, "fs", 500))
    return spec


def load_vqvae_model(spec: TokenizerSpec, in_channels: int, device: torch.device) -> torch.nn.Module:
    module = _load_vqvae_module()
    model_kwargs = infer_model_kwargs_from_checkpoint(spec.checkpoint_path, in_channels=in_channels, spec=spec)
    model = module.VQVAE1D(**model_kwargs)
    state = torch.load(spec.checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model


def codebook_embeddings_from_ids(model: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
    model_eval = model.module if hasattr(model, "module") else model
    embeds = model_eval.codebook.embed_code(ids)
    return embeds


def code_histogram_from_ids(ids: torch.Tensor, n_embed: int, normalize: bool = True) -> torch.Tensor:
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2:
        raise ValueError(f"Expected ids to have shape [B, T] or [T], got {tuple(ids.shape)}")
    one_hot = torch.nn.functional.one_hot(ids.long(), num_classes=int(n_embed)).to(dtype=torch.float32)
    hist = one_hot.sum(dim=1)
    if normalize:
        denom = hist.sum(dim=1, keepdim=True).clamp_min(1.0)
        hist = hist / denom
    return hist


def infer_emg_channels(csv_path: str) -> int:
    df = pd.read_csv(csv_path, nrows=4)
    emg_cols, _, _ = pick_signal_columns(df, emg_only=True)
    if not emg_cols:
        raise ValueError(f"Could not infer EMG channels from {csv_path}")
    return len(emg_cols)


def token_steps_for_spec(spec: TokenizerSpec) -> int:
    return token_steps_for_window(spec.fs, spec.codes_per_second, spec.window_sec)


def token_steps_for_window(fs: int, codes_per_second: float, window_sec: float) -> int:
    downsample_factor = nearest_power_of_two(fs / codes_per_second, min_value=1)
    window_size = make_compatible_window_size(int(round(window_sec * fs)), base=downsample_factor)
    return window_size // downsample_factor


def validate_tokenizer_compatibility(tokenizer_specs: dict[str, TokenizerSpec]):
    if len(tokenizer_specs) <= 1:
        return

    items = list(tokenizer_specs.items())
    ref_hand, ref_spec = items[0]
    ref_cfg = ref_spec.compatibility_dict()
    mismatches = []
    for hand, spec in items[1:]:
        cfg = spec.compatibility_dict()
        for key, ref_value in ref_cfg.items():
            value = cfg[key]
            if value != ref_value:
                mismatches.append(f"{hand}.{key}={value} != {ref_hand}.{key}={ref_value}")

    if mismatches:
        joined = "; ".join(mismatches)
        raise ValueError(
            "Left/right tokenizer checkpoints must have matching shared VQ-VAE parameters. "
            f"Found mismatches: {joined}"
        )


def load_joint_tokenizer_specs(tokenizer_cfg_map: dict[str, object]) -> dict[str, TokenizerSpec]:
    if len(tokenizer_cfg_map) <= 1:
        specs = {hand: load_tokenizer_spec_from_config(cfg) for hand, cfg in tokenizer_cfg_map.items()}
        validate_tokenizer_compatibility(specs)
        return specs

    if any(getattr(cfg, "checkpoint_path", None) or getattr(cfg, "run_name", None) for cfg in tokenizer_cfg_map.values()):
        specs = {hand: load_tokenizer_spec_from_config(cfg) for hand, cfg in tokenizer_cfg_map.items()}
        validate_tokenizer_compatibility(specs)
        return specs

    candidate_lists = {hand: _candidate_specs_from_results(cfg) for hand, cfg in tokenizer_cfg_map.items()}
    best_specs = None
    best_rank = None
    hands = list(candidate_lists.keys())
    compatibility_to_candidates = {}
    for hand in hands:
        mapping = {}
        for rank, spec in enumerate(candidate_lists[hand]):
            mapping.setdefault(tuple(sorted(spec.compatibility_dict().items())), (rank, spec))
        compatibility_to_candidates[hand] = mapping

    common_keys = set.intersection(*(set(mapping.keys()) for mapping in compatibility_to_candidates.values()))
    if not common_keys:
        raise ValueError("Could not find any compatible left/right tokenizer pair in the provided results.csv files.")

    for compat_key in common_keys:
        rank_sum = 0
        selected = {}
        for hand in hands:
            rank, spec = compatibility_to_candidates[hand][compat_key]
            rank_sum += rank
            selected[hand] = spec
        if best_rank is None or rank_sum < best_rank:
            best_rank = rank_sum
            best_specs = selected

    if best_specs is None:
        raise ValueError("Failed to resolve a compatible tokenizer pair from the provided results.csv files.")

    validate_tokenizer_compatibility(best_specs)
    return best_specs
