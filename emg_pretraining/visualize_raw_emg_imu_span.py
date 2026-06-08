from __future__ import annotations

import argparse
import random
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from emg_pretraining.vqvae_utils import (
        get_vqvae_input,
        make_compatible_window_size,
        nearest_power_of_two,
        pick_signal_columns,
        zscore_per_channel,
    )
else:
    from .vqvae_utils import (
        get_vqvae_input,
        make_compatible_window_size,
        nearest_power_of_two,
        pick_signal_columns,
        zscore_per_channel,
    )


DEFAULT_EMG_ROOT = Path("./data/EMG")


def parse_npy_filename(filename: str) -> tuple[str, str, float] | None:
    """Parse filename like 'Eadom_1_left_hand_123.456.npy' to (recording_name, hand_key, center_time)"""
    match = re.match(r"(.+?)_(left_hand|right_hand)_([\d.]+)\.npy$", filename)
    if match:
        recording_name = match.group(1)
        hand_key = "left_hand" if "left" in match.group(2) else "right_hand"
        center_time = float(match.group(3))
        return recording_name, hand_key, center_time
    return None


def discover_recordings(emg_root: Path) -> dict[str, Path]:
    """Return mapping from recording name to recording path"""
    recordings = {}
    for child in sorted(emg_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "left.csv").exists() and (child / "right.csv").exists():
            recordings[child.name] = child
    if not recordings:
        raise ValueError(f"No recordings with left.csv and right.csv found under {emg_root}")
    return recordings


def extract_vqvae_window(
    df: pd.DataFrame,
    center_time: float,
    window_sec: float,
    codes_per_second: float,
    emg_window_size: int,
    fs: int,
) -> np.ndarray:
    """Recreate the VQ-VAE input extraction logic from data.py"""
    target_size = make_compatible_window_size(
        int(round(float(window_sec) * fs)),
        base=nearest_power_of_two(fs / float(codes_per_second), min_value=1),
    )

    half_window = 0.5 * float(window_sec)
    start_time = center_time - half_window
    end_time = center_time + half_window

    utc = df["unix_time_s"].to_numpy(dtype=np.float64)
    emg_cols, accel_cols, gyro_cols = pick_signal_columns(df, emg_only=False)
    selected_cols = list(emg_cols)
    selected_cols.extend(accel_cols)
    selected_cols.extend(gyro_cols)

    raw = df[selected_cols].to_numpy(dtype=np.float32)

    start_idx = int(np.searchsorted(utc, start_time, side="left"))
    end_idx = int(np.searchsorted(utc, end_time, side="right"))
    window = raw[start_idx:end_idx]

    if len(window) == 0:
        anchor = min(max(start_idx, 0), len(raw) - 1)
        window = raw[anchor : anchor + 1]

    if len(window) < target_size:
        pad = target_size - len(window)
        left = pad // 2
        right = pad - left
        window = np.pad(window, ((left, right), (0, 0)), mode="edge")
    elif len(window) > target_size:
        offset = max(0, (len(window) - target_size) // 2)
        window = window[offset : offset + target_size]

    emg_count = len(emg_cols)
    emg_raw = window[:, :emg_count]
    emg_proc = get_vqvae_input(emg_raw, window_size=int(emg_window_size), fs=fs)

    parts = [emg_proc]
    if window.shape[1] > emg_count:
        parts.append(window[:, emg_count:])
    out = np.concatenate(parts, axis=1).astype(np.float32)

    return zscore_per_channel(out).astype(np.float32)


def plot_span(time_axis: np.ndarray, data: np.ndarray, signal_cols: list[str], title: str, output_path: Path):
    # data is (channels, time_steps)
    fig, axes = plt.subplots(len(signal_cols), 1, figsize=(16, 22), sharex=True)
    for ax, (col, signal_data) in zip(axes, zip(signal_cols, data)):
        ax.plot(time_axis, signal_data, linewidth=0.7)
        ax.set_ylabel(col, fontsize=8)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Seconds")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emg-root", type=str, default=str(DEFAULT_EMG_ROOT))
    parser.add_argument("--npy-path", type=str, required=True, help="Path to .npy file from diagnostics (e.g., 'Eadom_1_left_hand_123.456.npy')")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--window-sec", type=float, default=1.5, help="VQ-VAE window duration in seconds")
    parser.add_argument("--codes-per-second", type=float, default=12.0, help="VQ-VAE codes per second")
    parser.add_argument("--emg-window-size", type=int, default=512, help="EMG window size for get_vqvae_input")
    parser.add_argument("--fs", type=int, default=100, help="Sampling frequency in Hz")
    return parser.parse_args()


def main():
    args = parse_args()
    npy_path = Path(args.npy_path)

    # Parse filename to extract recording name, hand, and center_time
    parsed = parse_npy_filename(npy_path.name)
    if not parsed:
        raise ValueError(f"Could not parse npy filename: {npy_path.name}. Expected format: 'recording_name_left_hand_center_time.npy'")
    recording_name, hand_key, center_time = parsed

    # Discover recordings and find the one matching the filename
    emg_root = Path(args.emg_root)
    recordings = discover_recordings(emg_root)
    if recording_name not in recordings:
        raise ValueError(f"Recording '{recording_name}' not found in {emg_root}")
    recording_path = recordings[recording_name]

    # Load the CSV for the specified hand
    hand_csv = "left.csv" if "left" in hand_key else "right.csv"
    csv_path = recording_path / hand_csv
    if not csv_path.exists():
        raise ValueError(f"{csv_path} does not exist")

    df = pd.read_csv(csv_path).sort_values("unix_time_s").reset_index(drop=True)

    # Recreate the VQ-VAE input using the same logic as data.py
    vqvae_data = extract_vqvae_window(
        df,
        center_time=center_time,
        window_sec=float(args.window_sec),
        codes_per_second=float(args.codes_per_second),
        emg_window_size=int(args.emg_window_size),
        fs=int(args.fs),
    )

    # Load saved .npy for comparison if it exists
    saved_data = np.load(npy_path) if npy_path.exists() else None

    # vqvae_data is (time_steps, channels)
    signal_cols = [f"ch{i}" for i in range(vqvae_data.shape[1])]
    time_axis = np.arange(vqvae_data.shape[0], dtype=np.float32) / float(args.fs)

    title = f"{recording_name} | {hand_key} | center_time={center_time:.3f}s | recreated VQ-VAE input"
    output_path = Path(args.output) if args.output else npy_path.with_stem(f"{npy_path.stem}_recreated").with_suffix(".png")

    plot_span(time_axis, vqvae_data.T, signal_cols, title=title, output_path=output_path)
    print(f"Plotted: {output_path}")

    if saved_data is not None:
        diff = np.abs(vqvae_data - saved_data).max()
        print(f"Max difference from saved .npy: {diff}")


if __name__ == "__main__":
    main()
