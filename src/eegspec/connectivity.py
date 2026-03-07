import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import os
from scipy import signal
from scipy.signal import butter, filtfilt, hilbert
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


def spectral_connectivity_matrix(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    method: List[str],
    epoch_sec: float = 2.0,
    overlap: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    Compute spectral connectivity matrix using MNE's spectral_connectivity.
    
    This function supports various connectivity methods including wPLI (weighted Phase Lag Index).
    
    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_times, n_channels) or (n_epochs, n_channels, n_times)
    sfreq : float
        Sampling frequency
    fmin : float
        Minimum frequency
    fmax : float
        Maximum frequency
    method : List[str]
        List of connectivity methods (e.g., ['wpli', 'coh', 'plv'])
    epoch_sec : float
        Epoch length in seconds (default: 2.0)
    overlap : float
        Overlap fraction between epochs (default: 0.5)
    
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary mapping method names to connectivity matrices (n_channels, n_channels)
    """
    if data.ndim == 2:
        n_times, n_ch = data.shape
        win = int(round(epoch_sec * sfreq))
        hop = max(1, int(round(win * (1.0 - overlap))))
        epochs = []
        for start in range(0, max(0, n_times - win + 1), hop):
            sl = slice(start, start + win)
            epochs.append(data[sl, :].T)
        if not epochs:
            raise ValueError("Not enough data for one epoch")
        X = np.stack(epochs, axis=0)
    elif data.ndim == 3:
        X = data
    else:
        raise ValueError("data must be (n_times, n_channels) or (n_epochs, n_channels, n_times)")

    con, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(
        X, method=method, mode="multitaper", sfreq=sfreq, fmin=fmin, fmax=fmax, faverage=True, verbose=False
    )
    out = {}
    for i, m in enumerate(method):
        M = con[i, :, :, 0]
        if m in ("coh", "plv"):
            np.fill_diagonal(M, 1.0)
        out[m] = M
    return out


def compute_wpli_pair(
    eeg1: np.ndarray,
    eeg2: np.ndarray,
    fs: float,
    freq_range: Tuple[float, float],
    *,
    padtype: Optional[str] = None,
    padlen: Optional[int] = None,
    detrend: bool = False,
    export_intermediates: Optional[str] = None,
    logger: Optional[Any] = None,
    ch1_idx: Optional[int] = None,
    ch2_idx: Optional[int] = None,
) -> float:
    """
    Compute weighted Phase Lag Index (wPLI) between two EEG channels.
    
    This function exactly matches MATLAB compute_wPLI.m logic:
    - Uses 2nd order Butterworth bandpass filter
    - Zero-phase filtering with filtfilt
    - Hilbert transform for analytic signal
    - wPLI = |mean(imag)| / mean(|imag|)
    - All calculations use 64-bit precision (float64) to match MATLAB double
    
    Parameters:
    -----------
    eeg1 : np.ndarray
        First EEG signal (1D array)
    eeg2 : np.ndarray
        Second EEG signal (1D array)
    fs : float
        Sampling frequency (Hz)
    freq_range : Tuple[float, float]
        Frequency range for filtering [low_freq, high_freq] (Hz)
    
    Returns:
    --------
    float
        Weighted Phase Lag Index (wPLI) value (64-bit precision)
    """
    # Handle diagonal elements (same channel): wPLI should be exactly 0
    if ch1_idx is not None and ch2_idx is not None and ch1_idx == ch2_idx:
        if logger:
            logger.debug(f"[wPLI Ch{ch1_idx}-Ch{ch2_idx}] Diagonal element: returning wPLI=0.0 (same channel)")
        return np.float64(0.0)
    
    # Ensure 1D arrays with 64-bit precision (matching MATLAB double)
    eeg1 = np.asarray(eeg1, dtype=np.float64).flatten()
    eeg2 = np.asarray(eeg2, dtype=np.float64).flatten()
    
    if len(eeg1) != len(eeg2):
        raise ValueError("Signals must have the same length")
    
    ch_label = f"Ch{ch1_idx}-Ch{ch2_idx}" if ch1_idx is not None and ch2_idx is not None else "pair"
    if logger:
        logger.debug(f"[wPLI {ch_label}] Input signals: length={len(eeg1)}, dtype={eeg1.dtype}, "
                    f"eeg1_range=[{np.min(eeg1):.6f}, {np.max(eeg1):.6f}], "
                    f"eeg2_range=[{np.min(eeg2):.6f}, {np.max(eeg2):.6f}], "
                    f"eeg1_mean={np.mean(eeg1):.6f}, eeg2_mean={np.mean(eeg2):.6f}, "
                    f"eeg1_std={np.std(eeg1):.6f}, eeg2_std={np.std(eeg2):.6f}")
    
    # 1. Apply Bandpass Filter (2nd order Butterworth, matching MATLAB)
    # MATLAB: [b, a] = butter(2, freq_range / (fs / 2), 'bandpass');
    nyquist = np.float64(fs) / np.float64(2.0)
    low = np.float64(freq_range[0]) / nyquist
    high = np.float64(freq_range[1]) / nyquist
    b, a = butter(2, [low, high], btype='bandpass')
    # Ensure filter coefficients are float64
    b = np.asarray(b, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    
    if logger:
        logger.debug(f"[wPLI {ch_label}] Filter design: fs={fs:.1f} Hz, nyquist={nyquist:.1f} Hz, "
                    f"freq_range=[{freq_range[0]:.2f}, {freq_range[1]:.2f}] Hz, "
                    f"normalized=[{low:.6f}, {high:.6f}], filter_order=2, btype=bandpass")

    # Optional detrending (MATLAB code does not detrend by default,
    # but some datasets require removing a linear trend before filtering)
    if detrend:
        eeg1 = signal.detrend(eeg1)
        eeg2 = signal.detrend(eeg2)

    # Zero-phase filtering (matching MATLAB filtfilt behavior as closely as possible).
    # MATLAB uses padlen = 3*(max(len(a),len(b)) - 1) and mirror/odd-like padding.
    # SciPy defaults are padtype='odd' and padlen=3*max(len(a), len(b)). We'll set
    # defaults that mimic MATLAB more closely unless overridden.
    filtfilt_kwargs = {}
    if padtype is None:
        filtfilt_kwargs['padtype'] = 'odd'
    else:
        filtfilt_kwargs['padtype'] = padtype

    # Compute MATLAB-like padlen default if not provided
    if padlen is None:
        # Use SciPy's filtfilt default padlen: 3 * max(len(a), len(b))
        default_padlen = 3 * max(len(a), len(b))
        # Ensure padlen is less than signal length - 1 (SciPy requirement)
        max_padlen = max(0, len(eeg1) - 1)
        filtfilt_kwargs['padlen'] = int(min(default_padlen, max_padlen))
    else:
        filtfilt_kwargs['padlen'] = int(padlen)

    eeg1_filt = filtfilt(b, a, eeg1, **filtfilt_kwargs)
    eeg2_filt = filtfilt(b, a, eeg2, **filtfilt_kwargs)
    
    if logger:
        logger.debug(f"[wPLI {ch_label}] After filtering: padtype={filtfilt_kwargs.get('padtype')}, "
                    f"padlen={filtfilt_kwargs.get('padlen')}, "
                    f"eeg1_filt_range=[{np.min(eeg1_filt):.6f}, {np.max(eeg1_filt):.6f}], "
                    f"eeg2_filt_range=[{np.min(eeg2_filt):.6f}, {np.max(eeg2_filt):.6f}], "
                    f"eeg1_filt_std={np.std(eeg1_filt):.6f}, eeg2_filt_std={np.std(eeg2_filt):.6f}")
    
    # 2. Compute Analytic Signal using Hilbert Transform
    # MATLAB: eeg1_hilbert = hilbert(eeg1_filt);
    # hilbert returns complex128 (128-bit complex = 64-bit real + 64-bit imag)
    eeg1_hilbert = hilbert(eeg1_filt)
    eeg2_hilbert = hilbert(eeg2_filt)
    
    if logger:
        logger.debug(f"[wPLI {ch_label}] After Hilbert transform: "
                    f"eeg1_hilbert_abs_range=[{np.min(np.abs(eeg1_hilbert)):.6f}, {np.max(np.abs(eeg1_hilbert)):.6f}], "
                    f"eeg2_hilbert_abs_range=[{np.min(np.abs(eeg2_hilbert)):.6f}, {np.max(np.abs(eeg2_hilbert)):.6f}], "
                    f"eeg1_phase_range=[{np.min(np.angle(eeg1_hilbert)):.6f}, {np.max(np.angle(eeg1_hilbert)):.6f}], "
                    f"eeg2_phase_range=[{np.min(np.angle(eeg2_hilbert)):.6f}, {np.max(np.angle(eeg2_hilbert)):.6f}]")
    
    # 3. Extract Instantaneous Phase Differences
    # MATLAB: phase_diff = angle(eeg1_hilbert) - angle(eeg2_hilbert);
    # (Not used in wPLI calculation, but computed in MATLAB)
    
    # 4. Compute Cross-Spectral Density (Imaginary Component)
    # MATLAB: imag_phase_diff = imag(eeg1_hilbert .* conj(eeg2_hilbert));
    # Ensure complex multiplication uses 64-bit precision
    imag_phase_diff = np.imag(eeg1_hilbert * np.conj(eeg2_hilbert))
    # Ensure result is float64
    imag_phase_diff = np.asarray(imag_phase_diff, dtype=np.float64)
    
    if logger:
        logger.debug(f"[wPLI {ch_label}] Cross-spectral density (imaginary): "
                    f"imag_phase_diff_range=[{np.min(imag_phase_diff):.6f}, {np.max(imag_phase_diff):.6f}], "
                    f"mean={np.mean(imag_phase_diff):.6f}, std={np.std(imag_phase_diff):.6f}, "
                    f"abs_mean={np.mean(np.abs(imag_phase_diff)):.6f}")
    
    # 5. Compute Weighted PLI
    # MATLAB: num = abs(mean(imag_phase_diff));
    #         denom = mean(abs(imag_phase_diff));
    num = np.abs(np.mean(imag_phase_diff, dtype=np.float64))
    denom = np.mean(np.abs(imag_phase_diff), dtype=np.float64)
    
    # For diagonal elements (same channel), imag_phase_diff should be exactly 0
    # Handle numerical precision issues with a small threshold
    EPS_WPLI = np.float64(1e-15)  # Threshold for numerical zero
    
    # Prevent division by zero (matching MATLAB)
    # Use threshold instead of exact zero check to handle numerical precision
    if denom < EPS_WPLI:
        wpli_value = np.float64(0.0)
        if logger:
            logger.debug(f"[wPLI {ch_label}] Division by near-zero: denom={denom:.2e} < {EPS_WPLI:.2e}, setting wPLI=0.0")
    else:
        wpli_value = num / denom
    
    # Log with scientific notation for very small values
    if logger:
        # Use scientific notation if value is very small
        num_str = f"{num:.10f}" if num >= 1e-10 else f"{num:.2e}"
        denom_str = f"{denom:.10f}" if denom >= 1e-10 else f"{denom:.2e}"
        wpli_str = f"{wpli_value:.10f}" if wpli_value >= 1e-10 else f"{wpli_value:.2e}"
        logger.info(f"[wPLI {ch_label}] Result: num=|mean(imag)|={num_str}, "
                   f"denom=mean(|imag|)={denom_str}, wPLI={wpli_str}")

    # Optionally export intermediates for cross-language comparison
    if export_intermediates:
        try:
            import os, numpy as _np
            os.makedirs(os.path.dirname(export_intermediates), exist_ok=True)
            _np.savez_compressed(
                export_intermediates,
                eeg1_filt=eeg1_filt.astype(_np.float64),
                eeg2_filt=eeg2_filt.astype(_np.float64),
                eeg1_hilbert=eeg1_hilbert.astype(_np.complex128),
                eeg2_hilbert=eeg2_hilbert.astype(_np.complex128),
                imag_phase_diff=imag_phase_diff.astype(_np.float64),
                wpli_value=_np.float64(wpli_value),
            )
        except Exception:
            # Don't raise on export failures; keep main computation robust
            pass

    return float(wpli_value)


def compute_wpli(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    use_full_length: bool = True,
    *,
    export_intermediates_dir: Optional[str] = None,
    logger: Optional[Any] = None,
) -> np.ndarray:
    """
    Compute weighted Phase Lag Index (wPLI) connectivity matrix.
    
    This function matches MATLAB Connectivity_Analysis.m logic:
    - Computes wPLI for all channel pairs
    - Uses full data length (L=round(n/1) = n) if use_full_length=True
    - Frequency range [fmin, fmax] for bandpass filtering
    
    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_channels, n_times) - MATLAB format
        or (n_times, n_channels) - will be transposed if needed
    sfreq : float
        Sampling frequency (Hz)
    fmin : float
        Minimum frequency (Hz)
    fmax : float
        Maximum frequency (Hz)
    use_full_length : bool
        If True, use full data length (matching MATLAB L=round(n/1))
        If False, use all available data
    
    Returns:
    --------
    np.ndarray
        wPLI connectivity matrix of shape (n_channels, n_channels)
    """
    # Ensure correct shape: (n_channels, n_times) - MATLAB format
    if data.ndim != 2:
        raise ValueError("Data must be 2D array")
    
    # Ensure 64-bit precision (matching MATLAB double)
    data = np.asarray(data, dtype=np.float64)
    
    # Check if needs transpose (if timepoints first)
    if data.shape[0] > data.shape[1] and data.shape[0] > 200:
        # Likely (n_times, n_channels), transpose to (n_channels, n_times)
        data = data.T
    
    n_channels, n_times = data.shape
    
    if logger:
        logger.info(f"[compute_wpli] Input data: shape={data.shape}, dtype={data.dtype}, "
                   f"sfreq={sfreq:.1f} Hz, freq_range=[{fmin:.2f}, {fmax:.2f}] Hz, "
                   f"data_range=[{np.min(data):.6f}, {np.max(data):.6f}], "
                   f"data_mean={np.mean(data):.6f}, data_std={np.std(data):.6f}")
    
    # Determine data length to use (matching MATLAB L=round(n/1) = n)
    if use_full_length:
        L = n_times
    else:
        L = n_times
    
    if logger:
        logger.info(f"[compute_wpli] Using data length: L={L} (full length), "
                   f"total pairs={n_channels * n_channels}")
    
    # Initialize wPLI matrix with 64-bit precision (matching MATLAB double)
    wpli_matrix = np.zeros((n_channels, n_channels), dtype=np.float64)
    
    # Compute wPLI for all channel pairs (matching MATLAB nested loops)
    freq_range = (fmin, fmax)
    total_pairs = n_channels * n_channels
    computed_pairs = 0
    for ii in range(n_channels):
        for jj in range(n_channels):
            # MATLAB: wPLI_value(ii,jj) = compute_wPLI(EEG_Signal(ii,1:L), EEG_Signal(jj,1:L), fs, freq_range);
            export_path = None
            if export_intermediates_dir is not None:
                try:
                    os.makedirs(export_intermediates_dir, exist_ok=True)
                except Exception:
                    pass
                export_path = os.path.join(export_intermediates_dir, f'pair_{ii}_{jj}.npz')

            # For diagonal elements, compute_wpli_pair will return 0 directly
            # For first pair (0,0), log it; for other diagonal elements, use debug level
            log_level = None
            if ii == 0 and jj == 0:
                log_level = logger  # Log first pair at INFO level
            elif ii == jj and logger:
                # Diagonal elements are handled directly in compute_wpli_pair, so no need to log
                log_level = None
            elif ii != jj and (ii < 3 and jj < 3):  # Log first few off-diagonal pairs for debugging
                log_level = logger if logger else None
            
            wpli_matrix[ii, jj] = compute_wpli_pair(
                data[ii, :L],
                data[jj, :L],
                sfreq,
                freq_range,
                export_intermediates=export_path,
                logger=log_level,
                ch1_idx=ii,
                ch2_idx=jj,
            )
            computed_pairs += 1
            if logger and (computed_pairs % max(1, total_pairs // 10) == 0 or computed_pairs == total_pairs):
                logger.debug(f"[compute_wpli] Progress: {computed_pairs}/{total_pairs} pairs computed "
                           f"({100.0 * computed_pairs / total_pairs:.1f}%)")
    
    # Explicitly set diagonal to zero (same channel pairs should have wPLI = 0)
    # This handles numerical precision issues where diagonal elements might have small non-zero values
    if logger:
        diagonal_mean_before = np.mean(np.diag(wpli_matrix))
        if diagonal_mean_before > 1e-10:
            logger.debug(f"[compute_wpli] Diagonal had non-zero mean before correction: {diagonal_mean_before:.2e}")
    
    np.fill_diagonal(wpli_matrix, np.float64(0.0))
    
    if logger:
        logger.info(f"[compute_wpli] Completed: wPLI matrix shape={wpli_matrix.shape}, "
                   f"wPLI_range=[{np.min(wpli_matrix):.10f}, {np.max(wpli_matrix):.10f}], "
                   f"wPLI_mean={np.mean(wpli_matrix):.10f}, wPLI_std={np.std(wpli_matrix):.10f}, "
                   f"diagonal_mean={np.mean(np.diag(wpli_matrix)):.10f} (set to 0.0), "
                   f"off_diagonal_mean={np.mean(wpli_matrix[~np.eye(n_channels, dtype=bool)]):.10f}")
    
    return wpli_matrix


def compute_strength(wpli_matrix: np.ndarray, logger: Optional[Any] = None) -> np.ndarray:
    """
    Compute node strength (weighted degree centrality) from wPLI matrix.
    
    This function exactly matches MATLAB compute_strength.m:
    - strength_values = sum(wpli_matrix, 2)  % Sum over columns
    - Returns column vector (will be transposed in Connectivity_Analysis)
    - All calculations use 64-bit precision (float64) to match MATLAB double
    
    Parameters:
    -----------
    wpli_matrix : np.ndarray
        NxN weighted adjacency matrix (wPLI values)
    logger : Optional[Any]
        Logger instance for detailed logging
    
    Returns:
    --------
    np.ndarray
        Column vector of node strength values (n_channels,) with dtype float64
    """
    # Ensure 64-bit precision (matching MATLAB double)
    wpli_matrix = np.asarray(wpli_matrix, dtype=np.float64)
    
    if logger:
        logger.info(f"[compute_strength] Input matrix: shape={wpli_matrix.shape}, "
                   f"matrix_range=[{np.min(wpli_matrix):.10f}, {np.max(wpli_matrix):.10f}], "
                   f"matrix_mean={np.mean(wpli_matrix):.10f}, matrix_std={np.std(wpli_matrix):.10f}")
    
    # MATLAB: strength_values = sum(wpli_matrix, 2);  % Sum over columns (axis=1)
    # Use dtype=np.float64 to ensure 64-bit accumulation
    strength_values = np.sum(wpli_matrix, axis=1, dtype=np.float64)
    
    if logger:
        logger.info(f"[compute_strength] Output: shape={strength_values.shape}, "
                   f"strength_range=[{np.min(strength_values):.10f}, {np.max(strength_values):.10f}], "
                   f"strength_mean={np.mean(strength_values):.10f}, strength_std={np.std(strength_values):.10f}, "
                   f"top5_channels={np.argsort(strength_values)[-5:][::-1].tolist()}, "
                   f"top5_values={np.sort(strength_values)[-5:][::-1].tolist()}")
    
    return strength_values


def compute_betweenness(wpli_matrix: np.ndarray, normalized: bool = False, return_unnormalized: bool = False, logger: Optional[Any] = None):
    """
    Compute Betweenness Centrality for a weighted adjacency matrix.
    
    This function exactly matches MATLAB compute_betweenness.m:
    - G = graph(wpli_matrix, 'upper')  % Upper triangular
    - betweenness = centrality(G, 'betweenness')  % Default: unnormalized
    
    Note: MATLAB uses thresholded matrix before calling this function.
    The thresholding should be done before calling this function.
    
    Parameters:
    -----------
    wpli_matrix : np.ndarray
        NxN weighted adjacency matrix (wPLI values)
        Should be thresholded before calling (threshold < 0.2 set to 0)
    
    Returns:
    --------
    np.ndarray
        Column vector of betweenness centrality values (n_channels,)
    """
    if not HAS_NETWORKX:
        raise ImportError(
            "networkx is required for betweenness centrality computation. "
            "Install it with: pip install networkx"
        )
    
    # Ensure 64-bit precision (matching MATLAB double)
    wpli_matrix = np.asarray(wpli_matrix, dtype=np.float64)
    
    if logger:
        logger.info(f"[compute_betweenness] Input matrix: shape={wpli_matrix.shape}, "
                   f"matrix_range=[{np.min(wpli_matrix):.10f}, {np.max(wpli_matrix):.10f}], "
                   f"matrix_mean={np.mean(wpli_matrix):.10f}, matrix_std={np.std(wpli_matrix):.10f}, "
                   f"non_zero_count={np.count_nonzero(wpli_matrix)}, "
                   f"non_zero_ratio={np.count_nonzero(wpli_matrix) / wpli_matrix.size:.4f}")
    
    # MATLAB: G = graph(wpli_matrix, 'upper');
    # 'upper' means use upper triangular part (including diagonal)
    # MATLAB automatically creates undirected graph from upper triangular.
    #
    # CRITICAL: In MATLAB, graph-based shortest-path algorithms (used by
    # centrality 'betweenness') interpret edge weight as STRENGTH (larger =
    # stronger connection = shorter effective distance). NetworkX instead
    # interprets weight as DISTANCE (larger = farther). To match MATLAB we
    # must invert the weights: distance = 1/weight. Zero-weight edges (from
    # thresholding) are excluded from the graph entirely.
    
    # Extract upper triangular part (k=1 to exclude diagonal self-loops)
    upper_tri = np.triu(wpli_matrix, k=1)
    # Make symmetric
    wpli_symmetric = upper_tri + upper_tri.T
    wpli_symmetric = np.asarray(wpli_symmetric, dtype=np.float64)
    
    if logger:
        logger.debug(f"[compute_betweenness] Symmetric matrix (upper-tri mirrored): "
                    f"shape={wpli_symmetric.shape}, "
                    f"symmetric_range=[{np.min(wpli_symmetric):.10f}, {np.max(wpli_symmetric):.10f}], "
                    f"non_zero_count={np.count_nonzero(wpli_symmetric)}")
    
    # Build graph with inverted weights (strength → distance = 1/strength)
    # Only add edges with positive weight (zero edges = not connected after threshold)
    G = nx.Graph()
    G.add_nodes_from(range(wpli_matrix.shape[0]))
    n_channels = wpli_matrix.shape[0]
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            w = wpli_symmetric[i, j]
            if w > np.float64(0.0):
                # distance = 1/weight so that stronger connections are "closer"
                G.add_edge(i, j, weight=np.float64(1.0) / w)
    
    if logger:
        connected = nx.is_connected(G)
        edge_weights = [d['weight'] for u, v, d in G.edges(data=True)]
        logger.info(f"[compute_betweenness] Graph created (weight=1/wPLI for MATLAB-matching): "
                   f"nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, "
                   f"density={nx.density(G):.6f}, is_connected={connected}, "
                   f"distance_range=[{min(edge_weights):.10f}, {max(edge_weights):.10f}] "
                   f"(≈ wPLI_range [{1/max(edge_weights):.6f}, {1/min(edge_weights):.6f}])")
    
    # MATLAB: betweenness = centrality(G, 'betweenness');
    # Use weight='weight' (now 1/wPLI = distance), normalized=False matches MATLAB default
    betweenness_norm = nx.betweenness_centrality(G, weight='weight', normalized=bool(normalized))

    # Convert to array in channel order (column vector) with 64-bit precision
    betweenness_list = [np.float64(betweenness_norm[i]) for i in range(n_channels)]
    betweenness_array = np.array(betweenness_list, dtype=np.float64)
    
    if logger:
        logger.info(f"[compute_betweenness] Output: shape={betweenness_array.shape}, normalized={normalized}, "
                   f"betweenness_range=[{np.min(betweenness_array):.10f}, {np.max(betweenness_array):.10f}], "
                   f"betweenness_mean={np.mean(betweenness_array):.10f}, betweenness_std={np.std(betweenness_array):.10f}, "
                   f"top5_channels={np.argsort(betweenness_array)[-5:][::-1].tolist()}, "
                   f"top5_values={np.sort(betweenness_array)[-5:][::-1].tolist()}")
        logger.info(f"[compute_betweenness] NOTE: weight was inverted (1/wPLI) to match MATLAB "
                   f"centrality() behaviour where larger weight = shorter path.")

    if return_unnormalized:
        # Also compute unnormalized version for comparison if requested
        betweenness_unnorm_dict = nx.betweenness_centrality(G, weight='weight', normalized=False)
        betweenness_list_unn = [np.float64(betweenness_unnorm_dict[i]) for i in range(n_channels)]
        betweenness_array_unn = np.array(betweenness_list_unn, dtype=np.float64)
        return betweenness_array, betweenness_array_unn

    return betweenness_array


def connectivity_analysis(
    eeg_signal: np.ndarray,
    fs: float = 500.0,
    freq_range: Tuple[float, float] = (8.0, 13.0),
    threshold: float = 0.2,
    *,
    export_intermediates_dir: Optional[str] = None,
    logger: Optional[Any] = None,
) -> Dict[str, np.ndarray]:
    """
    Complete connectivity analysis matching MATLAB Connectivity_Analysis.m.
    
    This function implements the exact logic from Connectivity_Analysis.m:
    - fs = 500 Hz
    - freq_range = [8, 13] (alpha band)
    - L = round(n/1) = n (use full data length)
    - threshold = 0.2 for betweenness computation
    - Strength uses original wPLI matrix
    - Betweenness uses thresholded wPLI matrix
    - Output: [betweenness, strength_values] as row vectors (transposed)
    
    Parameters:
    -----------
    eeg_signal : np.ndarray
        EEG signal of shape (n_channels, n_times) - MATLAB format
    fs : float
        Sampling frequency (Hz), default: 500.0
    freq_range : Tuple[float, float]
        Frequency range for filtering [low, high] (Hz), default: (8.0, 13.0)
    threshold : float
        Threshold for filtering weak connections, default: 0.2
    
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'wpli_matrix': Full wPLI connectivity matrix
        - 'thresholded_wpli': Thresholded wPLI matrix
        - 'strength': Strength values (row vector, transposed)
        - 'betweenness': Betweenness values (row vector, transposed)
        - 'features': Combined features [betweenness, strength] as row vector
    """
    # Ensure correct shape: (n_channels, n_times)
    if eeg_signal.ndim != 2:
        raise ValueError("EEG signal must be 2D array")
    
    if logger:
        logger.info(f"[connectivity_analysis] Starting analysis: input_shape={eeg_signal.shape}, "
                   f"fs={fs:.1f} Hz, freq_range=[{freq_range[0]:.2f}, {freq_range[1]:.2f}] Hz, "
                   f"threshold={threshold:.2f}")
    
    # Ensure 64-bit precision (matching MATLAB double)
    eeg_signal = np.asarray(eeg_signal, dtype=np.float64)
    
    # Check if needs transpose
    original_shape = eeg_signal.shape
    if eeg_signal.shape[0] > eeg_signal.shape[1] and eeg_signal.shape[0] > 200:
        eeg_signal = eeg_signal.T
        if logger:
            logger.debug(f"[connectivity_analysis] Transposed input: {original_shape} -> {eeg_signal.shape}")
    
    n_channels, n_times = eeg_signal.shape
    
    if logger:
        logger.info(f"[connectivity_analysis] Data shape: (n_channels={n_channels}, n_times={n_times}), "
                   f"data_range=[{np.min(eeg_signal):.6f}, {np.max(eeg_signal):.6f}], "
                   f"data_mean={np.mean(eeg_signal):.6f}, data_std={np.std(eeg_signal):.6f}")
    
    # MATLAB: L = round(n/1);  % Use full length
    L = n_times
    
    if logger:
        logger.info(f"[connectivity_analysis] Step 1/5: Computing wPLI matrix (using full length L={L})...")
    
    # Compute wPLI matrix for all channel pairs
    # MATLAB: wPLI_value(ii,jj) = compute_wPLI(EEG_Signal(ii,1:L), EEG_Signal(jj,1:L), fs, freq_range);
    wpli_matrix = compute_wpli(
        eeg_signal,
        sfreq=fs,
        fmin=freq_range[0],
        fmax=freq_range[1],
        use_full_length=True,
        export_intermediates_dir=export_intermediates_dir,
        logger=logger,
    )
    # Ensure wpli_matrix is float64
    wpli_matrix = np.asarray(wpli_matrix, dtype=np.float64)
    
    # MATLAB: threshold = 0.2;
    #         thresholded_wpli = wpli_matrix;
    #         thresholded_wpli(thresholded_wpli < threshold) = 0;
    if logger:
        logger.info(f"[connectivity_analysis] Step 2/5: Applying threshold={threshold:.2f} to wPLI matrix...")
        before_threshold_count = np.count_nonzero(wpli_matrix)
        before_threshold_ratio = before_threshold_count / wpli_matrix.size
    
    thresholded_wpli = wpli_matrix.copy()
    threshold = np.float64(threshold)  # Ensure threshold is float64
    thresholded_wpli[thresholded_wpli < threshold] = np.float64(0.0)
    
    if logger:
        after_threshold_count = np.count_nonzero(thresholded_wpli)
        after_threshold_ratio = after_threshold_count / thresholded_wpli.size
        logger.info(f"[connectivity_analysis] Threshold applied: "
                   f"non_zero_before={before_threshold_count} ({100*before_threshold_ratio:.2f}%), "
                   f"non_zero_after={after_threshold_count} ({100*after_threshold_ratio:.2f}%), "
                   f"removed={before_threshold_count - after_threshold_count} connections")
    
    # MATLAB: strength_values = compute_strength(wpli_matrix);
    #         strength_values = strength_values';  % Transpose to row vector
    if logger:
        logger.info(f"[connectivity_analysis] Step 3/5: Computing Strength features from original wPLI matrix...")
    strength_values = compute_strength(wpli_matrix, logger=logger)
    # Note: strength_values is already column vector (n_channels,), transpose to row vector
    if strength_values.ndim == 1:
        strength_values = strength_values.reshape(1, -1)  # Make row vector (1, n_channels)
    # Ensure float64
    strength_values = np.asarray(strength_values, dtype=np.float64)
    
    # MATLAB: betweenness = compute_betweenness(thresholded_wpli);
    #         betweenness = betweenness';  % Transpose to row vector
    if logger:
        logger.info(f"[connectivity_analysis] Step 4/5: Computing Betweenness features from thresholded wPLI matrix...")
    betweenness = compute_betweenness(thresholded_wpli, logger=logger)
    
    # Note: betweenness is already column vector (n_channels,), transpose to row vector
    if betweenness.ndim == 1:
        betweenness = betweenness.reshape(1, -1)  # Make row vector (1, n_channels)
    
    # Ensure float64
    betweenness = np.asarray(betweenness, dtype=np.float64)
    
    # MATLAB: Features = [betweenness strength_values];
    #         Conn_Feats = Features;
    # Concatenate horizontally: [betweenness(1, n) strength_values(1, n)] -> (1, 2*n)
    if logger:
        logger.info(f"[connectivity_analysis] Step 5/5: Combining features [betweenness, strength]...")
    features = np.concatenate([betweenness, strength_values], axis=1)
    # Ensure float64
    features = np.asarray(features, dtype=np.float64)
    
    if logger:
        logger.info(f"[connectivity_analysis] Analysis complete: "
                   f"wpli_matrix shape={wpli_matrix.shape}, "
                   f"thresholded_wpli shape={thresholded_wpli.shape}, "
                   f"strength shape={strength_values.shape}, "
                   f"betweenness shape={betweenness.shape}, "
                   f"features shape={features.shape}, "
                   f"features_range=[{np.min(features):.10f}, {np.max(features):.10f}]")
    
    return {
        'wpli_matrix': wpli_matrix,
        'thresholded_wpli': thresholded_wpli,
        'strength': strength_values,
        'betweenness': betweenness,
        'features': features,
    }


def extract_graph_features(
    connectivity_matrix: np.ndarray,
    threshold: Optional[float] = 0.2,
    logger: Optional[Any] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract graph theory features (Strength and Betweenness) from connectivity matrix.
    
    This function is a wrapper around connectivity_analysis for compatibility.
    It matches the feature extraction from Connectivity_Analysis.m.
    
    Parameters:
    -----------
    connectivity_matrix : np.ndarray
        Connectivity matrix of shape (n_channels, n_channels)
    threshold : Optional[float]
        Threshold for binarizing connectivity matrix in betweenness computation (default: 0.2)
    
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing 'strength' and 'betweenness' features
    """
    # Create dummy signal to use connectivity_analysis
    # We'll extract features directly from the matrix
    n_channels = connectivity_matrix.shape[0]
    
    # Ensure 64-bit precision
    connectivity_matrix = np.asarray(connectivity_matrix, dtype=np.float64)
    
    # Strength uses original matrix
    strength_values = compute_strength(connectivity_matrix, logger=logger)
    strength_values = strength_values.T  # Transpose to row vector
    strength_values = np.asarray(strength_values, dtype=np.float64)
    
    # Betweenness uses thresholded matrix
    thresholded_matrix = connectivity_matrix.copy()
    threshold = np.float64(threshold)  # Ensure threshold is float64
    thresholded_matrix[thresholded_matrix < threshold] = np.float64(0.0)
    betweenness = compute_betweenness(thresholded_matrix, logger=logger)
    betweenness = betweenness.T  # Transpose to row vector
    betweenness = np.asarray(betweenness, dtype=np.float64)
    
    return {
        'strength': strength_values,
        'betweenness': betweenness,
    }


def compute_mutual_information_pair(
    signal1: np.ndarray,
    signal2: np.ndarray,
    *,
    num_bins: Optional[int] = None,
    logger: Optional[Any] = None,
    ch1_idx: Optional[int] = None,
    ch2_idx: Optional[int] = None,
) -> float:
    """
    Compute Mutual Information (MI) between two signals.
    
    This function exactly matches MATLAB mutual_info_function.m logic:
    - Uses Freedman-Diaconis rule for binning: num_bins = round(sqrt(length(signal1)))
    - Computes marginal and joint probability distributions
    - MI = H(signal1) + H(signal2) - H(signal1, signal2)
    - All calculations use 64-bit precision (float64) to match MATLAB double
    
    Parameters:
    -----------
    signal1 : np.ndarray
        First signal (1D array)
    signal2 : np.ndarray
        Second signal (1D array)
    num_bins : Optional[int]
        Number of bins for histogram. If None, uses sqrt(length(signal1)) (Freedman-Diaconis rule)
    logger : Optional[Any]
        Logger instance for detailed logging
    ch1_idx : Optional[int]
        Channel 1 index (for logging)
    ch2_idx : Optional[int]
        Channel 2 index (for logging)
    
    Returns:
    --------
    float
        Mutual Information value (64-bit precision)
    """
    # Ensure 1D arrays with 64-bit precision (matching MATLAB double)
    signal1 = np.asarray(signal1, dtype=np.float64).flatten()
    signal2 = np.asarray(signal2, dtype=np.float64).flatten()
    
    if len(signal1) != len(signal2):
        raise ValueError("Signals must have the same length")
    
    # Handle diagonal elements (same channel): MI should be high (self-information)
    # But for consistency with wPLI (where diagonal=0), we can set MI diagonal to a fixed value
    # However, MATLAB doesn't explicitly set diagonal, so we compute it normally
    # The diagonal will have high MI (self-information), which is mathematically correct
    
    ch_label = f"Ch{ch1_idx}-Ch{ch2_idx}" if ch1_idx is not None and ch2_idx is not None else "pair"
    if logger:
        logger.debug(f"[MI {ch_label}] Input signals: length={len(signal1)}, dtype={signal1.dtype}, "
                    f"signal1_range=[{np.min(signal1):.6f}, {np.max(signal1):.6f}], "
                    f"signal2_range=[{np.min(signal2):.6f}, {np.max(signal2):.6f}]")
    
    # MATLAB: num_bins = round(sqrt(length(signal1))); % Adaptive binning
    if num_bins is None:
        num_bins = int(round(np.sqrt(len(signal1))))
    
    if logger:
        logger.debug(f"[MI {ch_label}] Using {num_bins} bins (Freedman-Diaconis rule: sqrt({len(signal1)})={np.sqrt(len(signal1)):.2f})")
    
    # MATLAB: [p1, edges1] = histcounts(signal1, num_bins, 'Normalization', 'probability');
    # MATLAB: [p2, edges2] = histcounts(signal2, num_bins, 'Normalization', 'probability');
    p1, edges1 = np.histogram(signal1, bins=num_bins, density=False)
    p2, edges2 = np.histogram(signal2, bins=num_bins, density=False)
    
    # Normalize to probabilities
    p1 = p1.astype(np.float64) / np.float64(len(signal1))
    p2 = p2.astype(np.float64) / np.float64(len(signal2))
    
    # MATLAB: [~, bin1] = histc(signal1, edges1);
    # MATLAB: [~, bin2] = histc(signal2, edges2);
    # Note: histc is deprecated, use digitize instead
    # digitize returns indices 1..num_bins for values in bins, 0 for values < edges[0], num_bins+1 for values >= edges[-1]
    bin1 = np.digitize(signal1, edges1) - 1  # Convert to 0-indexed
    bin2 = np.digitize(signal2, edges2) - 1
    
    # Clamp to valid range [0, num_bins-1]
    bin1 = np.clip(bin1, 0, num_bins - 1)
    bin2 = np.clip(bin2, 0, num_bins - 1)
    
    # MATLAB: joint_hist = accumarray([bin1, bin2], 1, [num_bins, num_bins]) / length(signal1);
    # Use bincount2d equivalent
    joint_hist = np.zeros((num_bins, num_bins), dtype=np.float64)
    for i in range(len(signal1)):
        joint_hist[bin1[i], bin2[i]] += np.float64(1.0)
    joint_hist = joint_hist / np.float64(len(signal1))
    
    if logger:
        logger.debug(f"[MI {ch_label}] Histograms: p1_range=[{np.min(p1):.6f}, {np.max(p1):.6f}], "
                    f"p2_range=[{np.min(p2):.6f}, {np.max(p2):.6f}], "
                    f"joint_hist_range=[{np.min(joint_hist):.6f}, {np.max(joint_hist):.6f}], "
                    f"joint_hist_nonzero={np.count_nonzero(joint_hist)}/{joint_hist.size}")
    
    # MATLAB: Remove zero probabilities for entropy calculations
    # p1(p1 == 0) = [];
    # p2(p2 == 0) = [];
    # joint_hist(joint_hist == 0) = [];
    p1_nonzero = p1[p1 > 0]
    p2_nonzero = p2[p2 > 0]
    joint_hist_nonzero = joint_hist[joint_hist > 0]
    
    # MATLAB: H1 = -sum(p1 .* log2(p1)); % Entropy of signal1
    # MATLAB: H2 = -sum(p2 .* log2(p2)); % Entropy of signal2
    # MATLAB: H_joint = -sum(joint_hist(:) .* log2(joint_hist(:))); % Joint entropy
    # Use np.float64 for all calculations to match MATLAB double precision
    H1 = -np.sum(p1_nonzero * np.log2(p1_nonzero, dtype=np.float64), dtype=np.float64)
    H2 = -np.sum(p2_nonzero * np.log2(p2_nonzero, dtype=np.float64), dtype=np.float64)
    H_joint = -np.sum(joint_hist_nonzero * np.log2(joint_hist_nonzero, dtype=np.float64), dtype=np.float64)
    
    # MATLAB: mi = H1 + H2 - H_joint;
    mi = H1 + H2 - H_joint
    mi = np.float64(mi)
    
    if logger:
        logger.info(f"[MI {ch_label}] Result: H1={H1:.6f}, H2={H2:.6f}, H_joint={H_joint:.6f}, MI={mi:.6f}")
    
    return float(mi)


def compute_mutual_information_matrix(
    data: np.ndarray,
    use_full_length: bool = True,
    *,
    num_bins: Optional[int] = None,
    export_intermediates_dir: Optional[str] = None,
    logger: Optional[Any] = None,
) -> np.ndarray:
    """
    Compute Mutual Information (MI) connectivity matrix.
    
    This function matches MATLAB Connectivity_Analysis.m logic for MI:
    - Computes MI for all channel pairs
    - Uses full data length (L=round(n/1) = n) if use_full_length=True
    - All calculations use 64-bit precision (float64) to match MATLAB double
    
    Parameters:
    -----------
    data : np.ndarray
        Input data of shape (n_channels, n_times) - MATLAB format
        or (n_times, n_channels) - will be transposed if needed
    use_full_length : bool
        If True, use full data length (matching MATLAB L=round(n/1))
        If False, use all available data
    num_bins : Optional[int]
        Number of bins for histogram. If None, uses sqrt(length(signal)) (Freedman-Diaconis rule)
    export_intermediates_dir : Optional[str]
        Directory to export intermediate results (not used for MI, kept for API consistency)
    logger : Optional[Any]
        Logger instance for detailed logging
    
    Returns:
    --------
    np.ndarray
        MI connectivity matrix of shape (n_channels, n_channels)
    """
    # Ensure correct shape: (n_channels, n_times) - MATLAB format
    if data.ndim != 2:
        raise ValueError("Data must be 2D array")
    
    # Ensure 64-bit precision (matching MATLAB double)
    data = np.asarray(data, dtype=np.float64)
    
    # Check if needs transpose (if timepoints first)
    if data.shape[0] > data.shape[1] and data.shape[0] > 200:
        # Likely (n_times, n_channels), transpose to (n_channels, n_times)
        data = data.T
    
    n_channels, n_times = data.shape
    
    if logger:
        logger.info(f"[compute_mi_matrix] Input data: shape={data.shape}, dtype={data.dtype}, "
                   f"data_range=[{np.min(data):.6f}, {np.max(data):.6f}], "
                   f"data_mean={np.mean(data):.6f}, data_std={np.std(data):.6f}")
    
    # Determine data length to use (matching MATLAB L=round(n/1) = n)
    if use_full_length:
        L = n_times
    else:
        L = n_times
    
    if logger:
        logger.info(f"[compute_mi_matrix] Using data length: L={L} (full length), "
                   f"total pairs={n_channels * n_channels}")
    
    # Initialize MI matrix with 64-bit precision (matching MATLAB double)
    mi_matrix = np.zeros((n_channels, n_channels), dtype=np.float64)
    
    # Compute MI for all channel pairs (matching MATLAB nested loops)
    total_pairs = n_channels * n_channels
    computed_pairs = 0
    for ii in range(n_channels):
        for jj in range(n_channels):
            # MATLAB: Mu_Info_EEG_1(ii,jj) = mutual_info_function(EEG_Signal(ii,1:L), EEG_Signal(jj,1:L));
            log_level = None
            if ii == 0 and jj == 0:
                log_level = logger  # Log first pair at INFO level
            elif ii < 3 and jj < 3:  # Log first few pairs for debugging
                log_level = logger if logger else None
            
            mi_matrix[ii, jj] = compute_mutual_information_pair(
                data[ii, :L],
                data[jj, :L],
                num_bins=num_bins,
                logger=log_level,
                ch1_idx=ii,
                ch2_idx=jj,
            )
            computed_pairs += 1
            if logger and (computed_pairs % max(1, total_pairs // 10) == 0 or computed_pairs == total_pairs):
                logger.debug(f"[compute_mi_matrix] Progress: {computed_pairs}/{total_pairs} pairs computed "
                           f"({100.0 * computed_pairs / total_pairs:.1f}%)")
    
    if logger:
        logger.info(f"[compute_mi_matrix] Completed: MI matrix shape={mi_matrix.shape}, "
                   f"MI_range=[{np.min(mi_matrix):.10f}, {np.max(mi_matrix):.10f}], "
                   f"MI_mean={np.mean(mi_matrix):.10f}, MI_std={np.std(mi_matrix):.10f}, "
                   f"diagonal_mean={np.mean(np.diag(mi_matrix)):.10f}, "
                   f"off_diagonal_mean={np.mean(mi_matrix[~np.eye(n_channels, dtype=bool)]):.10f}")
    
    return mi_matrix


def connectivity_analysis_mi(
    eeg_signal: np.ndarray,
    threshold: float = 0.2,
    *,
    num_bins: Optional[int] = None,
    export_intermediates_dir: Optional[str] = None,
    logger: Optional[Any] = None,
) -> Dict[str, np.ndarray]:
    """
    Complete connectivity analysis using Mutual Information, matching MATLAB Connectivity_Analysis.m for MI.
    
    This function implements the logic for MI-based connectivity:
    - Computes MI matrix for all channel pairs
    - Uses full data length (L=round(n/1) = n)
    - threshold = 0.2 for betweenness computation
    - Strength uses original MI matrix
    - Betweenness uses thresholded MI matrix
    - Output: [betweenness, strength_values] as row vectors (transposed)
    
    Parameters:
    -----------
    eeg_signal : np.ndarray
        EEG signal of shape (n_channels, n_times) - MATLAB format
    threshold : float
        Threshold for filtering weak connections, default: 0.2
    num_bins : Optional[int]
        Number of bins for histogram. If None, uses sqrt(length(signal)) (Freedman-Diaconis rule)
    export_intermediates_dir : Optional[str]
        Directory to export intermediate results (not used for MI, kept for API consistency)
    logger : Optional[Any]
        Logger instance for detailed logging
    
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'mi_matrix': Full MI connectivity matrix
        - 'thresholded_mi': Thresholded MI matrix
        - 'strength': Strength values (row vector, transposed)
        - 'betweenness': Betweenness values (row vector, transposed)
        - 'features': Combined features [betweenness, strength] as row vector
    """
    # Ensure correct shape: (n_channels, n_times)
    if eeg_signal.ndim != 2:
        raise ValueError("EEG signal must be 2D array")
    
    if logger:
        logger.info(f"[connectivity_analysis_mi] Starting analysis: input_shape={eeg_signal.shape}, "
                   f"threshold={threshold:.2f}")
    
    # Ensure 64-bit precision (matching MATLAB double)
    eeg_signal = np.asarray(eeg_signal, dtype=np.float64)
    
    # Check if needs transpose
    original_shape = eeg_signal.shape
    if eeg_signal.shape[0] > eeg_signal.shape[1] and eeg_signal.shape[0] > 200:
        eeg_signal = eeg_signal.T
        if logger:
            logger.debug(f"[connectivity_analysis_mi] Transposed input: {original_shape} -> {eeg_signal.shape}")
    
    n_channels, n_times = eeg_signal.shape
    
    if logger:
        logger.info(f"[connectivity_analysis_mi] Data shape: (n_channels={n_channels}, n_times={n_times}), "
                   f"data_range=[{np.min(eeg_signal):.6f}, {np.max(eeg_signal):.6f}], "
                   f"data_mean={np.mean(eeg_signal):.6f}, data_std={np.std(eeg_signal):.6f}")
    
    # MATLAB: L = round(n/1);  % Use full length
    L = n_times
    
    if logger:
        logger.info(f"[connectivity_analysis_mi] Step 1/5: Computing MI matrix (using full length L={L})...")
    
    # Compute MI matrix for all channel pairs
    mi_matrix = compute_mutual_information_matrix(
        eeg_signal,
        use_full_length=True,
        num_bins=num_bins,
        export_intermediates_dir=export_intermediates_dir,
        logger=logger,
    )
    # Ensure mi_matrix is float64
    mi_matrix = np.asarray(mi_matrix, dtype=np.float64)
    
    # Apply threshold (matching wPLI logic)
    if logger:
        logger.info(f"[connectivity_analysis_mi] Step 2/5: Applying threshold={threshold:.2f} to MI matrix...")
        before_threshold_count = np.count_nonzero(mi_matrix)
        before_threshold_ratio = before_threshold_count / mi_matrix.size
    
    thresholded_mi = mi_matrix.copy()
    threshold = np.float64(threshold)  # Ensure threshold is float64
    thresholded_mi[thresholded_mi < threshold] = np.float64(0.0)
    
    if logger:
        after_threshold_count = np.count_nonzero(thresholded_mi)
        after_threshold_ratio = after_threshold_count / thresholded_mi.size
        logger.info(f"[connectivity_analysis_mi] Threshold applied: "
                   f"non_zero_before={before_threshold_count} ({100*before_threshold_ratio:.2f}%), "
                   f"non_zero_after={after_threshold_count} ({100*after_threshold_ratio:.2f}%), "
                   f"removed={before_threshold_count - after_threshold_count} connections")
    
    # Strength uses original MI matrix
    if logger:
        logger.info(f"[connectivity_analysis_mi] Step 3/5: Computing Strength features from original MI matrix...")
    strength_values = compute_strength(mi_matrix, logger=logger)
    # Note: strength_values is already column vector (n_channels,), transpose to row vector
    if strength_values.ndim == 1:
        strength_values = strength_values.reshape(1, -1)  # Make row vector (1, n_channels)
    # Ensure float64
    strength_values = np.asarray(strength_values, dtype=np.float64)
    
    # Betweenness uses thresholded MI matrix
    if logger:
        logger.info(f"[connectivity_analysis_mi] Step 4/5: Computing Betweenness features from thresholded MI matrix...")
    betweenness = compute_betweenness(thresholded_mi, logger=logger)
    
    # Note: betweenness is already column vector (n_channels,), transpose to row vector
    if betweenness.ndim == 1:
        betweenness = betweenness.reshape(1, -1)  # Make row vector (1, n_channels)
    
    # Ensure float64
    betweenness = np.asarray(betweenness, dtype=np.float64)
    
    # MATLAB: Features = [betweenness strength_values];
    # Concatenate horizontally: [betweenness(1, n) strength_values(1, n)] -> (1, 2*n)
    if logger:
        logger.info(f"[connectivity_analysis_mi] Step 5/5: Combining features [betweenness, strength]...")
    features = np.concatenate([betweenness, strength_values], axis=1)
    # Ensure float64
    features = np.asarray(features, dtype=np.float64)
    
    if logger:
        logger.info(f"[connectivity_analysis_mi] Analysis complete: "
                   f"mi_matrix shape={mi_matrix.shape}, "
                   f"thresholded_mi shape={thresholded_mi.shape}, "
                   f"strength shape={strength_values.shape}, "
                   f"betweenness shape={betweenness.shape}, "
                   f"features shape={features.shape}, "
                   f"features_range=[{np.min(features):.10f}, {np.max(features):.10f}]")
    
    return {
        'mi_matrix': mi_matrix,
        'thresholded_mi': thresholded_mi,
        'strength': strength_values,
        'betweenness': betweenness,
        'features': features,
    }