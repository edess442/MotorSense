import os
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def moving_window_rms(signal, window_size):
    signal = np.asarray(signal, dtype=np.float64)
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    return np.sqrt(np.convolve(signal ** 2, kernel, mode="same"))


def get_emg_data(
    data,
    window_size,
    fs=500,
    cutoff=8.0,
    notch_freq=60.0,
    notch_q=30.0,
    highpass_order=4,
):
    """
    Apply EMG processing pipeline:

      highpass -> notch -> rectify -> moving RMS

    Args:
        data: numpy array of shape (n_samples, n_channels)
        window_size: RMS window size in samples
        fs: sampling rate
        cutoff: high-pass cutoff frequency
        notch_freq: notch frequency
        notch_q: notch Q
        highpass_order: Butterworth high-pass order

    Returns:
        numpy array of shape (n_samples, n_channels)
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2:
        raise ValueError(
            f"get_emg_data expects shape (n_samples, n_channels), got {data.shape}"
        )

    nyq = 0.5 * fs
    if cutoff >= nyq:
        raise ValueError(
            f"cutoff={cutoff} must be < Nyquist={nyq}."
        )

    high = cutoff / nyq

    # High-pass filter
    b_high, a_high = butter(highpass_order, high, btype="high")

    # Notch filter
    b_notch, a_notch = iirnotch(notch_freq, notch_q, fs)

    out = np.zeros_like(data, dtype=np.float32)

    for ch in range(data.shape[1]):
        x = data[:, ch]

        x = filtfilt(b_high, a_high, x)
        x = filtfilt(b_notch, a_notch, x)
        x = np.abs(x)
        x = moving_window_rms(x, window_size)

        out[:, ch] = x.astype(np.float32)

    return out

def save_file_name(basename, outdir, ext):
    idx = 0
    os.makedirs(outdir, exist_ok=True)
    while os.path.exists(os.path.join(outdir, f"{basename}_{idx}.{ext}")):
        idx += 1
    return os.path.join(outdir, f"{basename}_{idx}.{ext}")