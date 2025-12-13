import numpy as np
from typing import List, Dict, Optional, Tuple
import mne
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


def compute_wpli(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    epoch_sec: float = 2.0,
    overlap: float = 0.5,
) -> np.ndarray:
    """
    Compute weighted Phase Lag Index (wPLI) connectivity matrix.
    
    wPLI is a measure of phase synchronization that is robust to volume conduction
    and noise. It is used in the paper for functional connectivity analysis.
    
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
    epoch_sec : float
        Epoch length in seconds (default: 2.0)
    overlap : float
        Overlap fraction between epochs (default: 0.5)
    
    Returns:
    --------
    np.ndarray
        wPLI connectivity matrix of shape (n_channels, n_channels)
    """
    # MNE uses 'wpli' as the method name
    # Try 'wpli' first, fallback to 'w_pli' if needed
    method_name = 'wpli'
    try:
        result = spectral_connectivity_matrix(
            data=data,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            method=[method_name],
            epoch_sec=epoch_sec,
            overlap=overlap,
        )
        return result[method_name]
    except (ValueError, KeyError) as e:
        # Try alternative method name
        method_name = 'w_pli'
        try:
            result = spectral_connectivity_matrix(
                data=data,
                sfreq=sfreq,
                fmin=fmin,
                fmax=fmax,
                method=[method_name],
                epoch_sec=epoch_sec,
                overlap=overlap,
            )
            return result[method_name]
        except (ValueError, KeyError):
            raise ValueError(
                f"wPLI method not available in MNE. "
                f"Tried 'wpli' and 'w_pli'. Error: {e}. "
                f"Please check MNE version and available connectivity methods."
            )


def compute_strength(connectivity_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Strength (node strength) from connectivity matrix.
    
    Strength is the sum of all edge weights connected to a node.
    In the paper, Strength features in the left hemisphere (central, parietal, 
    and temporal lobes) act as key hubs.
    
    Parameters:
    -----------
    connectivity_matrix : np.ndarray
        Connectivity matrix of shape (n_channels, n_channels)
    
    Returns:
    --------
    np.ndarray
        Strength values for each channel, shape (n_channels,)
    """
    # Strength is the sum of all connections to/from each node
    # For undirected graphs, we sum either row or column (they should be symmetric)
    # For directed graphs, we sum both
    strength = np.sum(np.abs(connectivity_matrix), axis=1)
    return strength


def compute_betweenness(
    connectivity_matrix: np.ndarray,
    threshold: Optional[float] = None,
    normalized: bool = True,
) -> np.ndarray:
    """
    Compute Betweenness centrality from connectivity matrix.
    
    Betweenness centrality measures the number of shortest paths that pass through
    a node. In the paper, Betweenness in the right hemisphere (frontal and central
    lobes) indicates network information flow.
    
    Parameters:
    -----------
    connectivity_matrix : np.ndarray
        Connectivity matrix of shape (n_channels, n_channels)
    threshold : Optional[float]
        Threshold to binarize the connectivity matrix. If None, uses weighted graph.
    normalized : bool
        Whether to normalize betweenness by (n-1)(n-2) for undirected graphs
    
    Returns:
    --------
    np.ndarray
        Betweenness centrality values for each channel, shape (n_channels,)
    """
    if not HAS_NETWORKX:
        raise ImportError(
            "networkx is required for betweenness centrality computation. "
            "Install it with: pip install networkx"
        )
    
    n_channels = connectivity_matrix.shape[0]
    
    # Create graph from connectivity matrix
    if threshold is not None:
        # Binarize: keep only connections above threshold
        adj_matrix = (np.abs(connectivity_matrix) > threshold).astype(float)
    else:
        # Use weighted graph
        adj_matrix = np.abs(connectivity_matrix)
    
    # Remove self-connections (diagonal)
    np.fill_diagonal(adj_matrix, 0.0)
    
    # Create NetworkX graph
    G = nx.from_numpy_array(adj_matrix)
    
    # Compute betweenness centrality
    betweenness = nx.betweenness_centrality(G, weight='weight', normalized=normalized)
    
    # Convert to array in channel order
    betweenness_array = np.array([betweenness[i] for i in range(n_channels)])
    
    return betweenness_array


def extract_graph_features(
    connectivity_matrix: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract graph theory features (Strength and Betweenness) from connectivity matrix.
    
    This function implements the feature extraction described in the paper:
    - Strength: sum of connection weights for each node
    - Betweenness: centrality measure indicating information flow
    
    Parameters:
    -----------
    connectivity_matrix : np.ndarray
        Connectivity matrix of shape (n_channels, n_channels)
    threshold : Optional[float]
        Threshold for binarizing connectivity matrix in betweenness computation
    
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing 'strength' and 'betweenness' features
    """
    strength = compute_strength(connectivity_matrix)
    betweenness = compute_betweenness(connectivity_matrix, threshold=threshold)
    
    return {
        'strength': strength,
        'betweenness': betweenness,
    }
