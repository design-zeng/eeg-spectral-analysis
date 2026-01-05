import numpy as np
from typing import List, Dict, Optional, Tuple
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
    # Ensure 1D arrays with 64-bit precision (matching MATLAB double)
    eeg1 = np.asarray(eeg1, dtype=np.float64).flatten()
    eeg2 = np.asarray(eeg2, dtype=np.float64).flatten()
    
    if len(eeg1) != len(eeg2):
        raise ValueError("Signals must have the same length")
    
    # 1. Apply Bandpass Filter (2nd order Butterworth, matching MATLAB)
    # MATLAB: [b, a] = butter(2, freq_range / (fs / 2), 'bandpass');
    nyquist = np.float64(fs) / np.float64(2.0)
    low = np.float64(freq_range[0]) / nyquist
    high = np.float64(freq_range[1]) / nyquist
    b, a = butter(2, [low, high], btype='bandpass')
    # Ensure filter coefficients are float64
    b = np.asarray(b, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    
    # Zero-phase filtering (matching MATLAB filtfilt)
    # filtfilt preserves dtype, so eeg1_filt and eeg2_filt will be float64
    eeg1_filt = filtfilt(b, a, eeg1)
    eeg2_filt = filtfilt(b, a, eeg2)
    
    # 2. Compute Analytic Signal using Hilbert Transform
    # MATLAB: eeg1_hilbert = hilbert(eeg1_filt);
    # hilbert returns complex128 (128-bit complex = 64-bit real + 64-bit imag)
    eeg1_hilbert = hilbert(eeg1_filt)
    eeg2_hilbert = hilbert(eeg2_filt)
    
    # 3. Extract Instantaneous Phase Differences
    # MATLAB: phase_diff = angle(eeg1_hilbert) - angle(eeg2_hilbert);
    # (Not used in wPLI calculation, but computed in MATLAB)
    
    # 4. Compute Cross-Spectral Density (Imaginary Component)
    # MATLAB: imag_phase_diff = imag(eeg1_hilbert .* conj(eeg2_hilbert));
    # Ensure complex multiplication uses 64-bit precision
    imag_phase_diff = np.imag(eeg1_hilbert * np.conj(eeg2_hilbert))
    # Ensure result is float64
    imag_phase_diff = np.asarray(imag_phase_diff, dtype=np.float64)
    
    # 5. Compute Weighted PLI
    # MATLAB: num = abs(mean(imag_phase_diff));
    #         denom = mean(abs(imag_phase_diff));
    num = np.abs(np.mean(imag_phase_diff, dtype=np.float64))
    denom = np.mean(np.abs(imag_phase_diff), dtype=np.float64)
    
    # Prevent division by zero (matching MATLAB)
    if denom == 0.0:
        wpli_value = np.float64(0.0)
    else:
        wpli_value = num / denom
    
    return float(wpli_value)


def compute_wpli(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    use_full_length: bool = True,
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
    
    # Determine data length to use (matching MATLAB L=round(n/1) = n)
    if use_full_length:
        L = n_times
    else:
        L = n_times
    
    # Initialize wPLI matrix with 64-bit precision (matching MATLAB double)
    wpli_matrix = np.zeros((n_channels, n_channels), dtype=np.float64)
    
    # Compute wPLI for all channel pairs (matching MATLAB nested loops)
    freq_range = (fmin, fmax)
    for ii in range(n_channels):
        for jj in range(n_channels):
            # MATLAB: wPLI_value(ii,jj) = compute_wPLI(EEG_Signal(ii,1:L), EEG_Signal(jj,1:L), fs, freq_range);
            wpli_matrix[ii, jj] = compute_wpli_pair(
                data[ii, :L],
                data[jj, :L],
                sfreq,
                freq_range,
            )
    
    return wpli_matrix


def compute_strength(wpli_matrix: np.ndarray) -> np.ndarray:
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
    
    Returns:
    --------
    np.ndarray
        Column vector of node strength values (n_channels,) with dtype float64
    """
    # Ensure 64-bit precision (matching MATLAB double)
    wpli_matrix = np.asarray(wpli_matrix, dtype=np.float64)
    
    # MATLAB: strength_values = sum(wpli_matrix, 2);  % Sum over columns (axis=1)
    # Use dtype=np.float64 to ensure 64-bit accumulation
    strength_values = np.sum(wpli_matrix, axis=1, dtype=np.float64)
    return strength_values


def compute_betweenness(wpli_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Betweenness Centrality for a weighted adjacency matrix.
    
    This function exactly matches MATLAB compute_betweenness.m:
    - G = graph(wpli_matrix, 'upper')  % Upper triangular
    - betweenness = centrality(G, 'betweenness')
    
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
    
    # MATLAB: G = graph(wpli_matrix, 'upper');
    # 'upper' means use upper triangular part (including diagonal)
    # MATLAB automatically creates undirected graph from upper triangular
    
    # Extract upper triangular part (including diagonal)
    # MATLAB graph('upper') uses upper triangular, then creates symmetric undirected graph
    upper_tri = np.triu(wpli_matrix, k=0)  # Include diagonal (k=0)
    # Make symmetric by copying upper to lower (MATLAB does this automatically)
    # Ensure all operations use float64
    wpli_symmetric = upper_tri + upper_tri.T - np.diag(np.diag(upper_tri))  # Avoid double diagonal
    wpli_symmetric = np.asarray(wpli_symmetric, dtype=np.float64)
    
    # Create NetworkX graph from numpy array
    # NetworkX automatically handles symmetric matrices
    # Ensure the array passed to NetworkX is explicitly float64 to maintain precision
    # NetworkX will use the array's dtype for edge weights
    G = nx.from_numpy_array(wpli_symmetric, create_using=nx.Graph)
    
    # MATLAB: betweenness = centrality(G, 'betweenness');
    # NetworkX betweenness_centrality with default normalization
    betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)
    
    # Convert to array in channel order (column vector) with 64-bit precision
    # CRITICAL: Explicitly convert each value to float64 to ensure full precision
    # NetworkX returns Python float, but we need to ensure 64-bit precision is maintained
    n_channels = wpli_matrix.shape[0]
    betweenness_list = [np.float64(betweenness[i]) for i in range(n_channels)]
    
    # Create array with explicit float64 dtype and ensure all values are float64
    betweenness_array = np.array(betweenness_list, dtype=np.float64)
    # Double-check: ensure all values are actually float64 (not float32)
    betweenness_array = np.asarray(betweenness_array, dtype=np.float64)
    
    return betweenness_array


def connectivity_analysis(
    eeg_signal: np.ndarray,
    fs: float = 500.0,
    freq_range: Tuple[float, float] = (8.0, 13.0),
    threshold: float = 0.2,
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
    
    # Ensure 64-bit precision (matching MATLAB double)
    eeg_signal = np.asarray(eeg_signal, dtype=np.float64)
    
    # Check if needs transpose
    if eeg_signal.shape[0] > eeg_signal.shape[1] and eeg_signal.shape[0] > 200:
        eeg_signal = eeg_signal.T
    
    n_channels, n_times = eeg_signal.shape
    
    # MATLAB: L = round(n/1);  % Use full length
    L = n_times
    
    # Compute wPLI matrix for all channel pairs
    # MATLAB: wPLI_value(ii,jj) = compute_wPLI(EEG_Signal(ii,1:L), EEG_Signal(jj,1:L), fs, freq_range);
    wpli_matrix = compute_wpli(
        eeg_signal,
        sfreq=fs,
        fmin=freq_range[0],
        fmax=freq_range[1],
        use_full_length=True,
    )
    # Ensure wpli_matrix is float64
    wpli_matrix = np.asarray(wpli_matrix, dtype=np.float64)
    
    # MATLAB: threshold = 0.2;
    #         thresholded_wpli = wpli_matrix;
    #         thresholded_wpli(thresholded_wpli < threshold) = 0;
    thresholded_wpli = wpli_matrix.copy()
    threshold = np.float64(threshold)  # Ensure threshold is float64
    thresholded_wpli[thresholded_wpli < threshold] = np.float64(0.0)
    
    # MATLAB: strength_values = compute_strength(wpli_matrix);
    #         strength_values = strength_values';  % Transpose to row vector
    strength_values = compute_strength(wpli_matrix)
    # Note: strength_values is already column vector (n_channels,), transpose to row vector
    if strength_values.ndim == 1:
        strength_values = strength_values.reshape(1, -1)  # Make row vector (1, n_channels)
    # Ensure float64
    strength_values = np.asarray(strength_values, dtype=np.float64)
    
    # MATLAB: betweenness = compute_betweenness(thresholded_wpli);
    #         betweenness = betweenness';  % Transpose to row vector
    betweenness = compute_betweenness(thresholded_wpli)
    
    # Note: betweenness is already column vector (n_channels,), transpose to row vector
    if betweenness.ndim == 1:
        betweenness = betweenness.reshape(1, -1)  # Make row vector (1, n_channels)
    
    # Ensure float64
    betweenness = np.asarray(betweenness, dtype=np.float64)
    
    # MATLAB: Features = [betweenness strength_values];
    #         Conn_Feats = Features;
    # Concatenate horizontally: [betweenness(1, n) strength_values(1, n)] -> (1, 2*n)
    features = np.concatenate([betweenness, strength_values], axis=1)
    # Ensure float64
    features = np.asarray(features, dtype=np.float64)
    
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
    strength_values = compute_strength(connectivity_matrix)
    strength_values = strength_values.T  # Transpose to row vector
    strength_values = np.asarray(strength_values, dtype=np.float64)
    
    # Betweenness uses thresholded matrix
    thresholded_matrix = connectivity_matrix.copy()
    threshold = np.float64(threshold)  # Ensure threshold is float64
    thresholded_matrix[thresholded_matrix < threshold] = np.float64(0.0)
    betweenness = compute_betweenness(thresholded_matrix)
    betweenness = betweenness.T  # Transpose to row vector
    betweenness = np.asarray(betweenness, dtype=np.float64)
    
    return {
        'strength': strength_values,
        'betweenness': betweenness,
    }
