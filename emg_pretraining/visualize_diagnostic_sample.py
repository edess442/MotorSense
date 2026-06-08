from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_diagnostic_sample(diagnostic_dir: Path, sample_name: str) -> dict:
    """Load all diagnostic files for a sample."""
    metadata_path = diagnostic_dir / f"{sample_name}_metadata.json"
    emg_hp_path = diagnostic_dir / f"{sample_name}_emg_hp.npy"
    vqvae_path = diagnostic_dir / f"{sample_name}.npy"
    frames_dir = diagnostic_dir / f"{sample_name}_frames"

    if not metadata_path.exists():
        raise ValueError(f"Metadata not found: {metadata_path}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    emg_hp = np.load(emg_hp_path) if emg_hp_path.exists() else None
    vqvae_data = np.load(vqvae_path) if vqvae_path.exists() else None

    frames = []
    if frames_dir.exists():
        for frame_path in sorted(frames_dir.glob("frame_*.jpg")):
            frames.append(Image.open(frame_path))

    return {
        "metadata": metadata,
        "emg_hp": emg_hp,
        "vqvae_data": vqvae_data,
        "frames": frames,
    }


def plot_emg_hp(sample_data: dict, output_path: Path):
    """Plot high-pass filtered EMG curves."""
    metadata = sample_data["metadata"]
    emg_hp = sample_data["emg_hp"]

    if emg_hp is None:
        raise ValueError("No EMG high-pass data available")

    # time_steps, channels
    time_steps, num_channels = emg_hp.shape
    fs = metadata["fs"]
    time_axis = np.arange(time_steps, dtype=np.float32) / fs

    fig, axes = plt.subplots(num_channels, 1, figsize=(14, 2 * num_channels), sharex=True)
    if num_channels == 1:
        axes = [axes]

    for ch, ax in enumerate(axes):
        ax.plot(time_axis, emg_hp[:, ch], linewidth=0.8, color="steelblue")
        ax.set_ylabel(f"Ch {ch}", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    title = f"{metadata['recording_name']} | {metadata['hand_key']} | EMG High-Pass Filtered"
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved EMG plot: {output_path}")


def plot_frames_grid(sample_data: dict, output_path: Path, cols: int = 4):
    """Plot frames in a grid."""
    frames = sample_data["frames"]
    if not frames:
        raise ValueError("No frames available")

    num_frames = len(frames)
    rows = (num_frames + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes.flatten()

    for idx, frame in enumerate(frames):
        axes[idx].imshow(frame)
        axes[idx].set_title(f"Frame {idx}", fontsize=9)
        axes[idx].axis("off")

    for idx in range(num_frames, len(axes)):
        axes[idx].axis("off")

    metadata = sample_data["metadata"]
    title = f"{metadata['recording_name']} | {metadata['hand_key']} | Frames"
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved frames plot: {output_path}")


def plot_emg_and_frames(sample_data: dict, output_path: Path):
    """Plot EMG on top half and first frame on bottom (for quick inspection)."""
    metadata = sample_data["metadata"]
    emg_hp = sample_data["emg_hp"]
    frames = sample_data["frames"]

    if emg_hp is None:
        raise ValueError("No EMG high-pass data available")
    if not frames:
        raise ValueError("No frames available")

    time_steps, num_channels = emg_hp.shape
    fs = metadata["fs"]
    time_axis = np.arange(time_steps, dtype=np.float32) / fs

    # Create figure with EMG on top, first frame on bottom
    fig = plt.figure(figsize=(16, 3 + 4))
    gs = fig.add_gridspec(num_channels + 2, 2, height_ratios=[1] * num_channels + [3, 0.5])

    # EMG subplots
    for ch in range(num_channels):
        ax = fig.add_subplot(gs[ch, :])
        ax.plot(time_axis, emg_hp[:, ch], linewidth=0.8, color="steelblue")
        ax.set_ylabel(f"Ch {ch}", fontsize=8)
        ax.grid(True, alpha=0.3)
        if ch < num_channels - 1:
            ax.set_xticklabels([])

    # First and last frames
    ax_first = fig.add_subplot(gs[num_channels, 0])
    ax_first.imshow(frames[0])
    ax_first.set_title("First Frame", fontsize=10)
    ax_first.axis("off")

    ax_last = fig.add_subplot(gs[num_channels, 1])
    ax_last.imshow(frames[-1])
    ax_last.set_title("Last Frame", fontsize=10)
    ax_last.axis("off")

    # Summary
    ax_info = fig.add_subplot(gs[num_channels + 1, :])
    ax_info.axis("off")
    info_text = f"Recording: {metadata['recording_name']} | Hand: {metadata['hand_key']}\n"
    info_text += f"Center Time: {metadata['center_time']:.3f}s | Clip Size: {metadata['clip_sec']:.1f}s\n"
    info_text += f"EMG Shape: {emg_hp.shape} | VQ-VAE Shape: {metadata['vqvae_emg_shape']} | Frames: {len(frames)}"
    ax_info.text(0.5, 0.5, info_text, ha="center", va="center", fontsize=10, family="monospace")

    title = f"{metadata['recording_name']} | {metadata['hand_key']} | Diagnostic Sample"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined plot: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-dir", type=str, default=str(Path(__file__).parent / "diagnostics"))
    parser.add_argument("--sample-name", type=str, required=True, help="Sample name without extension (e.g., 'Eadom_1_left_hand_123.456')")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for plots (default: diagnostic_dir)")
    parser.add_argument("--emg-only", action="store_true", help="Only plot EMG, skip frames")
    parser.add_argument("--frames-only", action="store_true", help="Only plot frames")
    parser.add_argument("--combined", action="store_true", help="Plot combined EMG and frames (default)")
    return parser.parse_args()


def main():
    args = parse_args()
    diagnostic_dir = Path(args.diagnostic_dir)
    output_dir = Path(args.output_dir) if args.output_dir else diagnostic_dir

    if not diagnostic_dir.exists():
        raise ValueError(f"Diagnostic directory not found: {diagnostic_dir}")

    sample_data = load_diagnostic_sample(diagnostic_dir, args.sample_name)

    output_dir.mkdir(exist_ok=True, parents=True)

    # Default to combined if no specific option
    if not args.emg_only and not args.frames_only:
        args.combined = True

    if args.emg_only or args.combined:
        emg_output = output_dir / f"{args.sample_name}_emg_hp.png"
        plot_emg_hp(sample_data, emg_output)

    if args.frames_only:
        frames_output = output_dir / f"{args.sample_name}_frames_grid.png"
        plot_frames_grid(sample_data, frames_output)

    if args.combined:
        combined_output = output_dir / f"{args.sample_name}_combined.png"
        plot_emg_and_frames(sample_data, combined_output)

    print(f"Metadata: {json.dumps(sample_data['metadata'], indent=2)}")


if __name__ == "__main__":
    main()
