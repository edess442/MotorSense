from __future__ import annotations

import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .utils import get_emg_data as vqvae_get_emg_data
from .video_transforms import build_video_transforms
from .vqvae_utils import (
    highpass_only_emg,
    make_compatible_window_size,
    nearest_power_of_two,
    pick_signal_columns,
    zscore_per_channel,
)


def _normalize_recording_names(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        raise ValueError(
            "val_recordings must be a string or a sequence of strings, "
            f"got {type(value).__name__}."
        )

    names: list[str] = []
    seen: set[str] = set()
    for item in items:
        name = str(item).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def diagnostic_print(message, diagnostics_enabled=False):
    if diagnostics_enabled:
        print(message, flush=True)


def diagnostic_print_timing(label: str, seconds: float, diagnostics_enabled=False):
    if diagnostics_enabled:
        print(f"DIAGNOSTIC_TIMING: {label} took {seconds:.3f}s", flush=True)


def drop_filtered_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        col for col in df.columns
        if ("filtered" in str(col).lower()) or ("battery" in str(col).lower())
    ]
    if not drop_cols:
        return df
    return df.drop(columns=drop_cols)


def _estimate_object_bytes(value) -> int:
    if value is None:
        return 0
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, pd.DataFrame):
        return int(value.memory_usage(deep=True).sum())
    if isinstance(value, pd.Series):
        return int(value.memory_usage(deep=True))
    if torch.is_tensor(value):
        return int(value.nelement() * value.element_size())
    if isinstance(value, dict):
        total = int(sys.getsizeof(value))
        for k, v in value.items():
            total += _estimate_object_bytes(k)
            total += _estimate_object_bytes(v)
        return total
    if hasattr(value, "__dict__"):
        total = int(sys.getsizeof(value))
        total += _estimate_object_bytes(vars(value))
        return total
    if isinstance(value, (list, tuple)):
        total = int(sys.getsizeof(value))
        for item in value:
            total += _estimate_object_bytes(item)
        return total
    if isinstance(value, Path):
        return int(sys.getsizeof(str(value)))
    if isinstance(value, str):
        return int(sys.getsizeof(value))
    return int(sys.getsizeof(value))


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def diagnostic_print_memory_map(label: str, values: dict[str, object], diagnostics_enabled=False):
    if not diagnostics_enabled:
        return
    entries = []
    total = 0
    for key, value in values.items():
        size_bytes = _estimate_object_bytes(value)
        total += size_bytes
        entries.append((size_bytes, key))
    entries.sort(reverse=True)
    parts = [f"{key}={_format_bytes(size_bytes)}" for size_bytes, key in entries]
    diagnostic_print(
        f"DIAGNOSTIC_MEMORY: {label} total={_format_bytes(total)} | " + " | ".join(parts),
        diagnostics_enabled,
    )


def filter_visible_center_indices_for_mode(
    state: "RecordingState",
    center_indices: np.ndarray,
    hand_mode: str,
    model_window_sec: float,
    num_frames: int,
) -> np.ndarray:
    if state.hand_bboxes_df is None:
        return center_indices
    if len(center_indices) == 0:
        return center_indices

    def _frame_is_visible(frame_idx: int) -> bool:
        rows = state.frame_df[state.frame_df["frame_idx"] == int(frame_idx)]
        if rows.empty:
            return False
        row = rows.iloc[0]
        if hand_mode == "left_hand":
            return bool(row["has_left"])
        if hand_mode == "right_hand":
            return bool(row["has_right"])
        return bool(row["has_left"]) or bool(row["has_right"])

    visible_indices = []
    half = 0.5 * float(model_window_sec)
    for center_idx in center_indices:
        center_time = float(state.frame_times[int(center_idx)])
        target_times = np.linspace(center_time - half, center_time + half, int(num_frames))
        frame_ids = [int(state.frame_indices[_nearest_index(state.frame_times, float(t))]) for t in target_times]
        if any(_frame_is_visible(frame_idx) for frame_idx in frame_ids):
            visible_indices.append(int(center_idx))

    return np.asarray(visible_indices, dtype=np.int64)


def filter_duplicate_timestamp_center_indices(
    state: "RecordingState",
    center_indices: np.ndarray,
    model_window_sec: float,
    num_frames: int,
) -> np.ndarray:
    if len(center_indices) == 0:
        return center_indices

    valid_indices = [
        int(center_idx)
        for center_idx in center_indices
        if not state.clip_contains_duplicate_timestamps(
            center_idx=int(center_idx),
            clip_window_sec=model_window_sec,
            num_frames=num_frames,
        )
    ]
    return np.asarray(valid_indices, dtype=np.int64)


def save_emg_hp_plot(emg_hp: np.ndarray, fs: int, output_path: Path):
    """Plot and save high-pass filtered EMG curves.
    
    Args:
        emg_hp: (time_steps, emg_channels) array
        fs: sampling frequency in Hz
        output_path: path to save PNG
    """
    time_steps, n_channels = emg_hp.shape
    time_axis = np.arange(time_steps) / float(fs)
    
    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 2 * n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]
    
    for ch_idx, ax in enumerate(axes):
        ax.plot(time_axis, emg_hp[:, ch_idx], linewidth=0.8, color='steelblue')
        ax.set_ylabel(f'Ch{ch_idx}', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (s)', fontsize=10)
    fig.suptitle('High-Pass Filtered EMG', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_multichannel_plot(x: np.ndarray, fs: int, output_path: Path, title: str, color: str = "steelblue"):
    """Plot and save multichannel timeseries.

    Args:
        x: (time_steps, channels) array
        fs: sampling frequency in Hz
        output_path: path to save PNG
        title: plot title
        color: line color
    """
    time_steps, n_channels = x.shape
    time_axis = np.arange(time_steps) / float(fs)

    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 2 * n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]

    for ch_idx, ax in enumerate(axes):
        ax.plot(time_axis, x[:, ch_idx], linewidth=0.8, color=color)
        ax.set_ylabel(f'Ch{ch_idx}', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)', fontsize=10)
    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


FIRST_RECORDING_SECONDS_TO_SKIP = 4 * 60.0


def _ns_to_builtin(value):
    if hasattr(value, "__dict__"):
        return {k: _ns_to_builtin(v) for k, v in vars(value).items()}
    if isinstance(value, list):
        return [_ns_to_builtin(v) for v in value]
    return value


def hand_key_to_csv_name(hand_key: str) -> str:
    return "left.csv" if "left" in str(hand_key).lower() else "right.csv"


def _frame_idx_from_name(path: Path) -> int | None:
    nums = re.findall(r"\d+", path.stem)
    if not nums:
        return None
    return int(nums[-1])


def _nearest_index(sorted_times: np.ndarray, t: float) -> int:
    idx = int(np.searchsorted(sorted_times, t))
    if idx <= 0:
        return 0
    if idx >= len(sorted_times):
        return len(sorted_times) - 1
    left = idx - 1
    right = idx
    if abs(sorted_times[right] - t) < abs(sorted_times[left] - t):
        return right
    return left


def _participant_name_from_recording_name(name: str) -> str:
    return str(name).split("_", 1)[0]


@dataclass
class Recording:
    name: str
    participant: str
    root: Path
    frames_dir: Path
    left_csv: Path
    right_csv: Path
    camera_poses_csv: Path
    hand_bboxes_csv: Path


def discover_recordings(emg_root: str) -> List[Recording]:
    root = Path(emg_root)
    out: List[Recording] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        needed = {
            "camera_poses.csv": child / "camera_poses.csv",
            "hand_bboxes.csv": child / "hand_bboxes.csv",
            "left.csv": child / "left.csv",
            "right.csv": child / "right.csv",
            "frames": child / "frames",
        }
        if all(path.exists() for path in needed.values()):
            out.append(
                Recording(
                    name=child.name,
                    participant=_participant_name_from_recording_name(child.name),
                    root=child,
                    frames_dir=needed["frames"],
                    left_csv=needed["left.csv"],
                    right_csv=needed["right.csv"],
                    camera_poses_csv=needed["camera_poses.csv"],
                    hand_bboxes_csv=needed["hand_bboxes.csv"],
                )
            )
    if not out:
        raise ValueError(f"No valid recordings found under {emg_root}")
    return out


def limit_participants(recordings: Sequence[Recording], max_participants: int | None, seed: int) -> list[Recording]:
    if max_participants is None:
        return list(recordings)

    max_participants = int(max_participants)
    if max_participants <= 0:
        raise ValueError(f"max_participants must be positive when set, got {max_participants}")

    participant_to_recordings: dict[str, list[Recording]] = {}
    for rec in recordings:
        participant_to_recordings.setdefault(rec.participant, []).append(rec)

    participants = sorted(participant_to_recordings.keys())
    rng = random.Random(int(seed))
    rng.shuffle(participants)
    keep = set(participants[:max_participants])
    return [rec for rec in recordings if rec.participant in keep]


class RecordingState:
    def __init__(self, recording: Recording, tokenizer_specs: dict[str, object], fs: int, diagnostics_enabled: bool = False):
        init_start = time.perf_counter()
        self.recording = recording
        self.tokenizer_specs = tokenizer_specs
        self.fs = int(fs)
        self.min_filter_len = 128
        self.diagnostics_enabled = bool(diagnostics_enabled)

        t0 = time.perf_counter()
        left_df = drop_filtered_columns(pd.read_csv(recording.left_csv)).sort_values("unix_time_s").reset_index(drop=True)
        right_df = drop_filtered_columns(pd.read_csv(recording.right_csv)).sort_values("unix_time_s").reset_index(drop=True)
        diagnostic_print_timing(f"RecordingState[{recording.name}] load_left_right_csv", time.perf_counter() - t0, self.diagnostics_enabled)
        t0 = time.perf_counter()
        self.camera_df = pd.read_csv(recording.camera_poses_csv).sort_values("frame_idx").reset_index(drop=True)
        self.hand_bboxes_df = pd.read_csv(recording.hand_bboxes_csv).sort_values("frame_idx").reset_index(drop=True)
        diagnostic_print_timing(f"RecordingState[{recording.name}] load_camera_bbox_csv", time.perf_counter() - t0, self.diagnostics_enabled)

        min_idx = int(self.camera_df["frame_idx"].min())
        max_idx = int(self.camera_df["frame_idx"].max())
        self.min_frame_idx = min_idx
        self.max_frame_idx = max_idx
        if self.max_frame_idx < self.min_frame_idx:
            raise ValueError(f"No frame indices found in {recording.camera_poses_csv}")

        t1 = time.perf_counter()
        self.frame_df = self._build_frame_table()
        self.frame_indices = self.frame_df["frame_idx"].to_numpy(dtype=np.int64)
        self.frame_times = self.frame_df["time_s"].to_numpy(dtype=np.float64)
        diagnostic_print_timing(f"RecordingState[{recording.name}] build_frame_table", time.perf_counter() - t1, self.diagnostics_enabled)

        t2 = time.perf_counter()
        (
            self.left_times,
            self.left_raw_emg,
            self.left_vqvae_array,
        ) = self._build_signal_cache(left_df)
        (
            self.right_times,
            self.right_raw_emg,
            self.right_vqvae_array,
        ) = self._build_signal_cache(right_df)
        diagnostic_print_timing(f"RecordingState[{recording.name}] build_signal_cache", time.perf_counter() - t2, self.diagnostics_enabled)

        self.vqvae_window_sec = max(float(getattr(spec, "window_sec", 1.5)) for spec in tokenizer_specs.values())
        self.min_center_time, self.max_center_time = self._compute_center_bounds()
        if self.max_center_time <= self.min_center_time:
            raise ValueError(f"Recording {recording.name} too short for configured windows")
        self.valid_center_indices = np.flatnonzero(
            (self.frame_times >= self.min_center_time) & (self.frame_times <= self.max_center_time)
        )
        if len(self.valid_center_indices) == 0:
            raise ValueError(f"Recording {recording.name} has no valid center frame indices")

        diagnostic_print_memory_map(
            f"RecordingState[{self.recording.name}]",
            {
                "camera_df": self.camera_df,
                "hand_bboxes_df": self.hand_bboxes_df,
                "frame_df": self.frame_df,
                "left_raw_emg": self.left_raw_emg,
                "right_raw_emg": self.right_raw_emg,
                "left_vqvae_array": self.left_vqvae_array,
                "right_vqvae_array": self.right_vqvae_array,
                "frame_indices": self.frame_indices,
                "frame_times": self.frame_times,
                "left_times": self.left_times,
                "right_times": self.right_times,
                "valid_center_indices": self.valid_center_indices,
            },
            diagnostics_enabled=self.diagnostics_enabled,
        )
        diagnostic_print_timing(f"RecordingState[{recording.name}] total_init", time.perf_counter() - init_start, self.diagnostics_enabled)

    def split_center_indices(self, split: str, is_validation_recording: bool) -> np.ndarray:
        split = str(split).lower()
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split}")
        if split == "val":
            if is_validation_recording:
                return self.valid_center_indices
            return np.empty(0, dtype=np.int64)
        if is_validation_recording:
            return np.empty(0, dtype=np.int64)
        return self.valid_center_indices

    def _build_frame_table(self) -> pd.DataFrame:
        df = self.camera_df.copy()
        if "utc_timestamp_ns" not in df.columns:
            raise ValueError(f"{self.recording.camera_poses_csv} must contain utc_timestamp_ns")
        df["time_s"] = df["utc_timestamp_ns"].astype(np.float64) / 1e9
        prev_same = df["utc_timestamp_ns"].eq(df["utc_timestamp_ns"].shift(1))
        next_same = df["utc_timestamp_ns"].eq(df["utc_timestamp_ns"].shift(-1))
        df["has_duplicate_timestamp_neighbor"] = (prev_same | next_same).fillna(False)

        df = df[(df["frame_idx"] >= self.min_frame_idx) & (df["frame_idx"] <= self.max_frame_idx)].copy()
        bbox_cols = [
            "frame_idx",
            "has_left",
            "left_xmin",
            "left_ymin",
            "left_xmax",
            "left_ymax",
            "has_right",
            "right_xmin",
            "right_ymin",
            "right_xmax",
            "right_ymax",
        ]
        missing = [c for c in bbox_cols if c not in self.hand_bboxes_df.columns]
        if missing:
            raise ValueError(f"{self.recording.hand_bboxes_csv} missing required bbox columns: {missing}")
        bbox_df = self.hand_bboxes_df[bbox_cols].copy()
        merged = df.merge(bbox_df, on="frame_idx", how="left")
        for col in ["has_left", "has_right"]:
            merged[col] = merged[col].where(merged[col].notna(), False).astype(bool)
        return (
            merged.dropna(subset=["time_s"])
            .drop_duplicates("frame_idx")
            .sort_values("frame_idx")
            .reset_index(drop=True)
        )

    def _compute_center_bounds(self) -> tuple[float, float]:
        half_vq = 0.5 * self.vqvae_window_sec
        frame_min = float(self.frame_times.min()) + FIRST_RECORDING_SECONDS_TO_SKIP + half_vq
        frame_max = float(self.frame_times.max()) - half_vq
        left_min = float(self.left_times.min()) + half_vq
        left_max = float(self.left_times.max()) - half_vq
        right_min = float(self.right_times.min()) + half_vq
        right_max = float(self.right_times.max()) - half_vq
        return max(frame_min, left_min, right_min), min(frame_max, left_max, right_max)

    def clip_contains_duplicate_timestamps(
        self,
        center_idx: int,
        clip_window_sec: float,
        num_frames: int,
    ) -> bool:
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        half = 0.5 * float(clip_window_sec)
        center_time = float(self.frame_times[int(center_idx)])
        target_times = np.linspace(center_time - half, center_time + half, int(num_frames))
        frame_table_idx = [_nearest_index(self.frame_times, float(t)) for t in target_times]
        duplicate_flags = self.frame_df.iloc[frame_table_idx]["has_duplicate_timestamp_neighbor"].to_numpy(dtype=bool)
        return bool(np.any(duplicate_flags))

    def sample_center_time(self) -> float:
        center_idx = int(self.valid_center_indices[random.randint(0, len(self.valid_center_indices) - 1)])
        return float(self.frame_times[center_idx])

    def load_frame(self, frame_idx: int) -> Image.Image:
        frame_idx = int(frame_idx)
        if frame_idx < self.min_frame_idx or frame_idx > self.max_frame_idx:
            raise ValueError(f"frame_idx={frame_idx} is outside valid range [{self.min_frame_idx}, {self.max_frame_idx}]")
        path = self.recording.frames_dir / f"frame_{frame_idx:06d}.jpg"
        return Image.open(path).convert("RGB").transpose(Image.Transpose.ROTATE_270)

    def _build_signal_cache(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        emg_cols, accel_cols, gyro_cols = pick_signal_columns(df, emg_only=False)
        if len(emg_cols) == 0:
            raise ValueError(f"No usable EMG columns found for recording {self.recording.name}")

        for cols in [emg_cols, accel_cols, gyro_cols]:
            for c in cols:
                if df[c].isna().any():
                    df[c] = df[c].fillna(df[c].mean())

        times = df["unix_time_s"].to_numpy(dtype=np.float64)
        emg = df[emg_cols].to_numpy(dtype=np.float32)
        if len(emg) >= self.min_filter_len:
            emg_proc = vqvae_get_emg_data(
                data=emg,
                window_size=int(next(iter(self.tokenizer_specs.values())).emg_window_size),
                fs=self.fs,
            ).astype(np.float32)
        else:
            emg_proc = emg.astype(np.float32)

        parts = [emg_proc]
        if len(accel_cols) > 0:
            parts.append(df[accel_cols].to_numpy(dtype=np.float32))
        if len(gyro_cols) > 0:
            parts.append(df[gyro_cols].to_numpy(dtype=np.float32))

        arr = np.concatenate(parts, axis=1).astype(np.float32)
        vqvae_array = zscore_per_channel(arr).astype(np.float32)
        raw_emg = emg.astype(np.float32) if self.diagnostics_enabled else None
        return times, raw_emg, vqvae_array


class RandomWindowEMGDataset(Dataset):
    def __init__(
        self,
        recording_states: Sequence[RecordingState],
        center_indices_by_recording: dict[str, np.ndarray],
        tokenizer_specs: dict[str, object],
        hand_mode: str,
        bbox_crop_image_scale: float,
        fs: int,
        vqvae_window_sec: float,
        model_window_sec: float,
        num_frames: int,
        samples_per_epoch: int,
        split: str,
        val_recordings: Sequence[str],
        model_mode: str = "video",
        transform=None,
        diagnostics_enabled: bool = False,
    ):
        super().__init__()
        self.tokenizer_specs = tokenizer_specs
        self.hand_mode = str(hand_mode)
        self.bbox_crop_image_scale = float(bbox_crop_image_scale)
        self.fs = int(fs)
        self.vqvae_window_sec = float(vqvae_window_sec)
        self.model_window_sec = float(model_window_sec)
        self.num_frames = int(num_frames)
        self.samples_per_epoch = int(samples_per_epoch)
        self.split = str(split).lower()
        self.val_recordings = tuple(_normalize_recording_names(val_recordings))
        self.model_mode = str(model_mode).lower()
        self.transform = transform
        self.diagnostics_enabled = bool(diagnostics_enabled)
        self.sample_counter = 0
        if not math.isclose(self.vqvae_window_sec, self.model_window_sec, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "RandomWindowEMGDataset requires matching VQ-VAE and video-model windows. "
                f"Got vqvae_window_sec={self.vqvae_window_sec} and model_window_sec={self.model_window_sec}."
            )
        if self.split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {self.split}")

        valid_modes = {"left_hand", "right_hand", "both"}
        if self.hand_mode not in valid_modes:
            raise ValueError(f"hand_mode must be one of {sorted(valid_modes)}, got {self.hand_mode}")
        if self.model_mode not in {"video", "image_pair"}:
            raise ValueError(f"model_mode must be one of ['image_pair', 'video'], got {self.model_mode}")
        if not (0.0 < self.bbox_crop_image_scale <= 1.0):
            raise ValueError(
                f"bbox_crop_image_scale must be in (0, 1], got {self.bbox_crop_image_scale}"
            )

        self.recording_states = list(recording_states)
        self.center_indices_by_recording = {
            str(name): np.asarray(indices, dtype=np.int64)
            for name, indices in center_indices_by_recording.items()
            if len(indices) > 0
        }

        self.recording_states = [
            state for state in self.recording_states
            if state.recording.name in self.center_indices_by_recording
        ]

        if not self.recording_states:
            raise ValueError(
                f"No recordings contain eligible center times for split={self.split} "
                f"with val_recordings={list(self.val_recordings)}."
            )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _num_video_frames_to_load(self) -> int:
        if self.model_mode == "image_pair":
            return 2
        return self.num_frames

    def _sample_center_time(self, state: RecordingState) -> float:
        center_indices = self.center_indices_by_recording[state.recording.name]
        center_idx = int(center_indices[random.randint(0, len(center_indices) - 1)])
        return float(state.frame_times[center_idx])

    def _frame_is_visible(self, state: RecordingState, frame_idx: int) -> bool:
        rows = state.frame_df[state.frame_df["frame_idx"] == int(frame_idx)]
        if rows.empty:
            return False
        row = rows.iloc[0]
        if self.hand_mode == "left_hand":
            return bool(row["has_left"])
        if self.hand_mode == "right_hand":
            return bool(row["has_right"])
        return bool(row["has_left"]) or bool(row["has_right"])

    def _filter_visible_center_indices(self, state: RecordingState, center_indices: np.ndarray) -> np.ndarray:
        if state.hand_bboxes_df is None:
            return center_indices
        if len(center_indices) == 0:
            return center_indices

        visible_indices = []
        half = 0.5 * self.model_window_sec
        num_video_frames = self._num_video_frames_to_load()
        for center_idx in center_indices:
            center_time = float(state.frame_times[int(center_idx)])
            target_times = np.linspace(center_time - half, center_time + half, num_video_frames)
            frame_ids = [int(state.frame_indices[_nearest_index(state.frame_times, float(t))]) for t in target_times]
            if any(self._frame_is_visible(state, frame_idx) for frame_idx in frame_ids):
                visible_indices.append(int(center_idx))

        return np.asarray(visible_indices, dtype=np.int64)

    def _extract_emg_window(self, utc: np.ndarray, raw_emg: np.ndarray, center_time: float, tokenizer_spec) -> np.ndarray:
        target_size = make_compatible_window_size(
            int(round(float(tokenizer_spec.window_sec) * self.fs)),
            base=nearest_power_of_two(self.fs / float(tokenizer_spec.codes_per_second), min_value=1),
        )

        half_window = 0.5 * float(tokenizer_spec.window_sec)
        start_time = center_time - half_window
        end_time = center_time + half_window

        start_idx = int(np.searchsorted(utc, start_time, side="left"))
        end_idx = int(np.searchsorted(utc, end_time, side="right"))
        window = raw_emg[start_idx:end_idx]

        if len(window) == 0:
            anchor = min(max(start_idx, 0), len(raw_emg) - 1)
            window = raw_emg[anchor : anchor + 1]

        if len(window) < target_size:
            pad = target_size - len(window)
            left = pad // 2
            right = pad - left
            window = np.pad(window, ((left, right), (0, 0)), mode="edge")
        elif len(window) > target_size:
            offset = max(0, (len(window) - target_size) // 2)
            window = window[offset : offset + target_size]

        return window

    def _extract_vqvae_window(self, utc: np.ndarray, full_arr: np.ndarray, center_time: float, tokenizer_spec) -> np.ndarray:
        target_size = make_compatible_window_size(
            int(round(float(tokenizer_spec.window_sec) * self.fs)),
            base=nearest_power_of_two(self.fs / float(tokenizer_spec.codes_per_second), min_value=1),
        )

        half_window = 0.5 * float(tokenizer_spec.window_sec)
        start_time = center_time - half_window
        end_time = center_time + half_window

        start_idx = int(np.searchsorted(utc, start_time, side="left"))
        end_idx = int(np.searchsorted(utc, end_time, side="right"))
        window = full_arr[start_idx:end_idx]

        if len(window) == 0:
            anchor = min(max(start_idx, 0), len(full_arr) - 1)
            window = full_arr[anchor : anchor + 1]

        if len(window) < target_size:
            pad = target_size - len(window)
            left = pad // 2
            right = pad - left
            window = np.pad(window, ((left, right), (0, 0)), mode="edge")
        elif len(window) > target_size:
            offset = max(0, (len(window) - target_size) // 2)
            window = window[offset : offset + target_size]

        expected_channels = int(getattr(tokenizer_spec, "input_channels", window.shape[1]))
        if window.shape[1] != expected_channels:
            raise ValueError(
                f"Constructed VQ-VAE input with {window.shape[1]} channels, but checkpoint expects {expected_channels}."
            )
        return window.astype(np.float32)

    def _candidate_bboxes_for_frame(self, state: RecordingState, frame_idx: int) -> list[tuple[float, float, float, float]]:
        rows = state.frame_df[state.frame_df["frame_idx"] == int(frame_idx)]
        if rows.empty:
            return []
        row = rows.iloc[0]
        candidates: list[tuple[float, float, float, float]] = []
        if self.hand_mode == "left_hand":
            if bool(row["has_left"]):
                coords = [row["left_xmin"], row["left_ymin"], row["left_xmax"], row["left_ymax"]]
                if not any(pd.isna(v) for v in coords):
                    candidates.append(tuple(float(v) for v in coords))
        elif self.hand_mode == "right_hand":
            if bool(row["has_right"]):
                coords = [row["right_xmin"], row["right_ymin"], row["right_xmax"], row["right_ymax"]]
                if not any(pd.isna(v) for v in coords):
                    candidates.append(tuple(float(v) for v in coords))
        else:
            if bool(row["has_left"]):
                left_coords = [row["left_xmin"], row["left_ymin"], row["left_xmax"], row["left_ymax"]]
                if not any(pd.isna(v) for v in left_coords):
                    candidates.append(tuple(float(v) for v in left_coords))
            if bool(row["has_right"]):
                right_coords = [row["right_xmin"], row["right_ymin"], row["right_xmax"], row["right_ymax"]]
                if not any(pd.isna(v) for v in right_coords):
                    candidates.append(tuple(float(v) for v in right_coords))
        return candidates

    def _sample_random_crop_for_bbox(
        self,
        image: Image.Image,
        bbox: tuple[float, float, float, float],
    ) -> tuple[int, int, int, int]:
        arr = np.asarray(image)
        h, w = arr.shape[:2]
        x1, y1, x2, y2 = bbox
        bbox_w = max(float(x2) - float(x1), 1.0)
        bbox_h = max(float(y2) - float(y1), 1.0)

        scale_w = random.uniform(self.bbox_crop_image_scale, 1.0)
        scale_h = random.uniform(self.bbox_crop_image_scale, 1.0)
        crop_w = max(1, int(round(w * scale_w)), int(math.ceil(bbox_w)))
        crop_h = max(1, int(round(h * scale_h)), int(math.ceil(bbox_h)))
        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)

        min_crop_x1 = max(0, int(math.ceil(x2 - crop_w)))
        max_crop_x1 = min(int(math.floor(x1)), max(0, w - crop_w))
        min_crop_y1 = max(0, int(math.ceil(y2 - crop_h)))
        max_crop_y1 = min(int(math.floor(y1)), max(0, h - crop_h))
        if min_crop_x1 > max_crop_x1 or min_crop_y1 > max_crop_y1:
            raise ValueError(
                f"Could not fit bbox {bbox} inside crop size {(crop_w, crop_h)} for image size {(w, h)}"
            )

        crop_x1 = random.randint(min_crop_x1, max_crop_x1)
        crop_y1 = random.randint(min_crop_y1, max_crop_y1)
        crop_x2 = crop_x1 + crop_w
        crop_y2 = crop_y1 + crop_h
        return crop_x1, crop_y1, crop_x2, crop_y2

    def _center_crop_for_bbox(
        self,
        image: Image.Image,
        bbox: tuple[float, float, float, float],
    ) -> tuple[int, int, int, int]:
        arr = np.asarray(image)
        h, w = arr.shape[:2]
        if self.split == "val":
            return 0, 0, w, h
        x1, y1, x2, y2 = bbox
        bbox_w = max(float(x2) - float(x1), 1.0)
        bbox_h = max(float(y2) - float(y1), 1.0)

        crop_w = max(1, int(round(w * self.bbox_crop_image_scale)), int(math.ceil(bbox_w)))
        crop_h = max(1, int(round(h * self.bbox_crop_image_scale)), int(math.ceil(bbox_h)))
        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)

        bbox_cx = 0.5 * (float(x1) + float(x2))
        bbox_cy = 0.5 * (float(y1) + float(y2))
        crop_x1 = int(round(bbox_cx - 0.5 * crop_w))
        crop_y1 = int(round(bbox_cy - 0.5 * crop_h))
        crop_x1 = min(max(crop_x1, 0), max(0, w - crop_w))
        crop_y1 = min(max(crop_y1, 0), max(0, h - crop_h))
        crop_x2 = crop_x1 + crop_w
        crop_y2 = crop_y1 + crop_h
        return crop_x1, crop_y1, crop_x2, crop_y2

    @staticmethod
    def _bbox_intersects_crop(
        bbox: tuple[float, float, float, float],
        crop_box: tuple[int, int, int, int],
    ) -> bool:
        bx1, by1, bx2, by2 = bbox
        cx1, cy1, cx2, cy2 = crop_box
        inter_w = min(float(bx2), float(cx2)) - max(float(bx1), float(cx1))
        inter_h = min(float(by2), float(cy2)) - max(float(by1), float(cy1))
        return (inter_w > 0.0) and (inter_h > 0.0)

    def _sample_random_crop_box(self, image: Image.Image) -> tuple[int, int, int, int]:
        arr = np.asarray(image)
        h, w = arr.shape[:2]
        scale_w = random.uniform(self.bbox_crop_image_scale, 1.0)
        scale_h = random.uniform(self.bbox_crop_image_scale, 1.0)
        crop_w = max(1, min(w, int(round(w * scale_w))))
        crop_h = max(1, min(h, int(round(h * scale_h))))
        max_crop_x1 = max(0, w - crop_w)
        max_crop_y1 = max(0, h - crop_h)
        crop_x1 = random.randint(0, max_crop_x1) if max_crop_x1 > 0 else 0
        crop_y1 = random.randint(0, max_crop_y1) if max_crop_y1 > 0 else 0
        return crop_x1, crop_y1, crop_x1 + crop_w, crop_y1 + crop_h

    @staticmethod
    def _average_bbox(bboxes: Sequence[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
        arr = np.asarray(bboxes, dtype=np.float32)
        if arr.ndim != 2 or arr.shape != (len(bboxes), 4) or len(bboxes) == 0:
            raise ValueError("average_bbox expects a non-empty sequence of 4D boxes")
        mean_box = arr.mean(axis=0)
        return tuple(float(v) for v in mean_box)

    def _sample_video_frames(self, state: RecordingState, center_time: float, clip_sec: float) -> tuple[List[Image.Image], List[int]]:
        half = 0.5 * clip_sec
        num_video_frames = self._num_video_frames_to_load()
        target_times = np.linspace(center_time - half, center_time + half, num_video_frames)
        frame_ids = [int(state.frame_indices[_nearest_index(state.frame_times, float(t))]) for t in target_times]

        if state.hand_bboxes_df is None:
            if self.split == "val":
                raise ValueError(f"Validation requires hand bboxes for recording {state.recording.name}")
            anchor_img = state.load_frame(int(frame_ids[random.randint(0, len(frame_ids) - 1)]))
            crop_box = self._sample_random_crop_box(anchor_img)
        else:
            visible_options = []
            for frame_idx in frame_ids:
                for bbox in self._candidate_bboxes_for_frame(state, frame_idx):
                    visible_options.append((frame_idx, bbox))
            if not visible_options:
                raise ValueError(f"No visible hand found in sampled clip for recording {state.recording.name}")

            if self.split == "val":
                avg_bbox = self._average_bbox([bbox for _, bbox in visible_options])
                anchor_frame_idx = frame_ids[len(frame_ids) // 2]
                anchor_img = state.load_frame(int(anchor_frame_idx))
                crop_box = self._center_crop_for_bbox(anchor_img, avg_bbox)
            else:
                anchor_frame_idx, anchor_bbox = visible_options[random.randint(0, len(visible_options) - 1)]
                anchor_img = state.load_frame(int(anchor_frame_idx))
                crop_box = self._sample_random_crop_for_bbox(anchor_img, anchor_bbox)

            intersects_any_frame = False
            for frame_idx in frame_ids:
                for bbox in self._candidate_bboxes_for_frame(state, frame_idx):
                    if self._bbox_intersects_crop(bbox, crop_box):
                        intersects_any_frame = True
                        break
                if intersects_any_frame:
                    break
            if not intersects_any_frame:
                raise ValueError(
                    f"No hand bbox intersects sampled crop for recording {state.recording.name}"
                )

        frames = []
        for frame_idx in frame_ids:
            img = state.load_frame(int(frame_idx))
            frames.append(img.crop(crop_box))
        return frames, frame_ids

    def _sample_visible_clip(self, state: RecordingState):
        max_tries = 128
        last_error = None
        for _ in range(max_tries):
            center_time = self._sample_center_time(state)
            clip_sec = self.model_window_sec
            try:
                frames, frame_ids = self._sample_video_frames(state, center_time=center_time, clip_sec=clip_sec)
                return center_time, clip_sec, frames, frame_ids
            except ValueError as exc:
                last_error = exc
                continue
        raise ValueError(
            f"Could not sample a visible {self.hand_mode} clip for recording {state.recording.name}"
        ) from last_error

    def __getitem__(self, idx: int):
        sample_start = time.perf_counter()
        state = self.recording_states[random.randint(0, len(self.recording_states) - 1)]
        t0 = time.perf_counter()
        center_time, clip_sec, frames, frame_ids = self._sample_visible_clip(state)
        sample_clip_time = time.perf_counter() - t0
        if self.transform is None:
            raise ValueError("RandomWindowEMGDataset expects a video transform callable")
        t1 = time.perf_counter()
        video = torch.stack(self.transform(frames), dim=0)
        transform_time = time.perf_counter() - t1

        vqvae_emg = {}
        emg_extract_time = 0.0
        vq_window_time = 0.0
        raw_window_time = 0.0
        hp_filter_time = 0.0
        tensor_wrap_time = 0.0
        for hand_key, spec in self.tokenizer_specs.items():
            is_left = "left" in hand_key.lower()
            utc = state.left_times if is_left else state.right_times
            raw_emg = state.left_raw_emg if is_left else state.right_raw_emg
            full_arr = state.left_vqvae_array if is_left else state.right_vqvae_array
            hand_t0 = time.perf_counter()
            t_vq = time.perf_counter()
            vq_in = self._extract_vqvae_window(utc, full_arr, center_time=center_time, tokenizer_spec=spec)
            vq_window_time += time.perf_counter() - t_vq
            if self.diagnostics_enabled:
                if raw_emg is None:
                    raise ValueError("Diagnostics require raw EMG arrays to be available")
                t_raw = time.perf_counter()
                raw_window = self._extract_emg_window(utc, raw_emg, center_time=center_time, tokenizer_spec=spec)
                raw_window_time += time.perf_counter() - t_raw
                t_hp = time.perf_counter()
                emg_hp = highpass_only_emg(raw_window, fs=self.fs)
                emg_hp = zscore_per_channel(emg_hp).astype(np.float32)
                hp_filter_time += time.perf_counter() - t_hp
            t_tensor = time.perf_counter()
            vqvae_emg[hand_key] = torch.from_numpy(vq_in.T.copy())
            tensor_wrap_time += time.perf_counter() - t_tensor
            emg_extract_time += time.perf_counter() - hand_t0

        self.sample_counter += 1
        total_sample_time = time.perf_counter() - sample_start
        if self.diagnostics_enabled and (self.sample_counter <= 5 or self.sample_counter % 200 == 0):
            diagnostic_print(
                "DIAGNOSTIC_TIMING: "
                f"sample={self.sample_counter} participant={state.recording.participant} recording={state.recording.name} "
                f"clip_sample={sample_clip_time:.3f}s transform={transform_time:.3f}s "
                f"emg_extract={emg_extract_time:.3f}s "
                f"vq_window={vq_window_time:.3f}s raw_window={raw_window_time:.3f}s "
                f"hp_filter={hp_filter_time:.3f}s tensor_wrap={tensor_wrap_time:.3f}s "
                f"total={total_sample_time:.3f}s",
                self.diagnostics_enabled,
            )

        return {
            "video": video,
            "vqvae_emg": vqvae_emg,
            "recording": state.recording.name,
            "participant": state.recording.participant,
            "clip_sec": torch.tensor(clip_sec, dtype=torch.float32),
            "center_time": torch.tensor(center_time, dtype=torch.float32),
        }


def build_dataloaders(cfg, tokenizer_specs, diagnostics_enabled=False):
    build_start = time.perf_counter()
    diagnostic_print("DIAGNOSTIC: Starting build_dataloaders", diagnostics_enabled)
    tokenizer_window_secs = sorted({float(spec.window_sec) for spec in tokenizer_specs.values()})
    if len(tokenizer_window_secs) != 1:
        raise ValueError(
            "All tokenizer specs must use the same window_sec for aligned training. "
            f"Found: {tokenizer_window_secs}"
        )
    shared_window_sec = float(tokenizer_window_secs[0])
    model_mode = str(getattr(cfg.model, "mode", "video")).lower()
    num_video_frames = 2 if model_mode == "image_pair" else int(cfg.data.num_frames)
    cfg_vqvae_window_sec = float(getattr(cfg.data, "vqvae_window_sec", shared_window_sec))
    cfg_model_window_sec = float(getattr(cfg.data, "model_window_sec", shared_window_sec))
    if not math.isclose(cfg_vqvae_window_sec, shared_window_sec, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            "Config/data.vqvae_window_sec must match tokenizer window_sec. "
            f"Got data.vqvae_window_sec={cfg_vqvae_window_sec} and tokenizer window_sec={shared_window_sec}."
        )
    if not math.isclose(cfg_model_window_sec, shared_window_sec, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            "Config/data.model_window_sec must match tokenizer window_sec so video predictions and EMG targets span the same duration. "
            f"Got data.model_window_sec={cfg_model_window_sec} and tokenizer window_sec={shared_window_sec}."
        )
    val_recordings = _normalize_recording_names(getattr(cfg.data, "val_recordings", ["Merawi_3"]))
    if not val_recordings:
        raise ValueError("Config/data.val_recordings must contain at least one recording name.")
    recordings = discover_recordings(cfg.data.emg_root)
    diagnostic_print_timing("build_dataloaders.discover_recordings", time.perf_counter() - build_start, diagnostics_enabled)
    diagnostic_print(f"DIAGNOSTIC: Discovered {len(recordings)} recordings", diagnostics_enabled)
    recordings = limit_participants(
        recordings,
        max_participants=getattr(cfg.data, "max_participants", None),
        seed=cfg.experiment.seed,
    )
    available_recording_names = {rec.name for rec in recordings}
    missing_val_recordings = [name for name in val_recordings if name not in available_recording_names]
    if missing_val_recordings:
        raise ValueError(
            f"Config/data.val_recordings includes unknown recordings: {missing_val_recordings}. "
            f"Available recordings: {sorted(available_recording_names)}"
        )
    val_recording_set = set(val_recordings)
    train_recordings = [rec for rec in recordings if rec.name not in val_recording_set]
    held_out_val_recordings = [rec for rec in recordings if rec.name in val_recording_set]
    if not train_recordings:
        raise ValueError("All recordings were assigned to validation; at least one train recording is required.")
    if not held_out_val_recordings:
        raise ValueError("No validation recordings remain after applying data.val_recordings.")
    diagnostic_print(
        f"DIAGNOSTIC: Using recording-based split with train_recordings={len(train_recordings)} "
        f"val_recordings={val_recordings}",
        diagnostics_enabled,
    )

    train_tf = build_video_transforms(_ns_to_builtin(cfg.data.transforms.train))
    val_tf = build_video_transforms(_ns_to_builtin(cfg.data.transforms.val))

    hand_mode = str(getattr(cfg.data, "hand_mode", "both"))
    bbox_crop_image_scale = float(getattr(cfg.data, "bbox_crop_image_scale", 0.5))
    states_start = time.perf_counter()
    all_states = [
        RecordingState(rec, tokenizer_specs, fs=cfg.data.fs, diagnostics_enabled=diagnostics_enabled)
        for rec in recordings
    ]
    diagnostic_print_timing("build_dataloaders.build_recording_states", time.perf_counter() - states_start, diagnostics_enabled)
    diagnostic_print_memory_map(
        "AllRecordingStates",
        {state.recording.name: state for state in all_states},
        diagnostics_enabled=diagnostics_enabled,
    )

    train_center_indices_by_recording: dict[str, np.ndarray] = {}
    val_center_indices_by_recording: dict[str, np.ndarray] = {}
    for state in all_states:
        is_validation_recording = state.recording.name in val_recording_set
        train_split_indices = state.split_center_indices(
            split="train",
            is_validation_recording=is_validation_recording,
        )
        val_split_indices = state.split_center_indices(
            split="val",
            is_validation_recording=is_validation_recording,
        )
        train_split_indices = filter_duplicate_timestamp_center_indices(
            state=state,
            center_indices=train_split_indices,
            model_window_sec=shared_window_sec,
            num_frames=num_video_frames,
        )
        val_split_indices = filter_duplicate_timestamp_center_indices(
            state=state,
            center_indices=val_split_indices,
            model_window_sec=shared_window_sec,
            num_frames=num_video_frames,
        )

        train_visible = filter_visible_center_indices_for_mode(
            state=state,
            center_indices=train_split_indices,
            hand_mode=hand_mode,
            model_window_sec=shared_window_sec,
            num_frames=num_video_frames,
        )
        val_visible = filter_visible_center_indices_for_mode(
            state=state,
            center_indices=val_split_indices,
            hand_mode=hand_mode,
            model_window_sec=shared_window_sec,
            num_frames=num_video_frames,
        )
        if len(train_visible) > 0:
            train_center_indices_by_recording[state.recording.name] = train_visible
        if len(val_visible) > 0:
            val_center_indices_by_recording[state.recording.name] = val_visible

    train_ds = RandomWindowEMGDataset(
        recording_states=all_states,
        center_indices_by_recording=train_center_indices_by_recording,
        tokenizer_specs=tokenizer_specs,
        hand_mode=hand_mode,
        bbox_crop_image_scale=bbox_crop_image_scale,
        fs=cfg.data.fs,
        vqvae_window_sec=shared_window_sec,
        model_window_sec=shared_window_sec,
        num_frames=cfg.data.num_frames,
        samples_per_epoch=cfg.data.train_samples_per_epoch,
        split="train",
        val_recordings=val_recordings,
        model_mode=model_mode,
        transform=train_tf,
        diagnostics_enabled=diagnostics_enabled,
    )
    val_ds = RandomWindowEMGDataset(
        recording_states=all_states,
        center_indices_by_recording=val_center_indices_by_recording,
        tokenizer_specs=tokenizer_specs,
        hand_mode=hand_mode,
        bbox_crop_image_scale=bbox_crop_image_scale,
        fs=cfg.data.fs,
        vqvae_window_sec=shared_window_sec,
        model_window_sec=shared_window_sec,
        num_frames=cfg.data.num_frames,
        samples_per_epoch=cfg.data.val_samples_per_epoch,
        split="val",
        val_recordings=val_recordings,
        model_mode=model_mode,
        transform=val_tf,
        diagnostics_enabled=diagnostics_enabled,
    )

    distributed = bool(getattr(getattr(cfg, "distributed", object()), "enabled", False))
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None

    def _worker_init_fn(worker_id: int):
        seed = int(getattr(cfg.experiment, "seed", 0))
        rank = int(getattr(train_sampler, "rank", 0) if distributed else 0)
        worker_seed = seed + 1000 * rank + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    diagnostic_print("DIAGNOSTIC: build_dataloaders completed", diagnostics_enabled)
    diagnostic_print_timing("build_dataloaders.total", time.perf_counter() - build_start, diagnostics_enabled)
    return train_loader, val_loader, train_ds, val_ds, train_sampler, val_sampler
