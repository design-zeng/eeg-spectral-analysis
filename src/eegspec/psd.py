import numpy as np
from scipy.signal import welch, get_window
from typing import Tuple, Optional

def compute_psd_welch(data: np.ndarray, sfreq: float, nperseg: int = 1000, noverlap: Optional[int] = None, window: str = "hann") -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel PSD via Welch. data: (n_times, n_channels)."""
    if data.ndim != 2:
        raise ValueError("data must be (n_times, n_channels)")
    n_times, n_channels = data.shape
    if n_times <= 0:
        raise ValueError("data must contain at least one time sample")
    nperseg = min(int(nperseg), n_times)
    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = min(int(noverlap), max(0, nperseg - 1))
    win = get_window(window, nperseg, fftbins=True)
    psds = []
    for ch in range(n_channels):
        f, Pxx = welch(data[:, ch], fs=sfreq, nperseg=nperseg, noverlap=noverlap, window=win)
        psds.append(Pxx)
    psd = np.stack(psds, axis=0)  # (n_channels, n_freqs)
    return f, psd
