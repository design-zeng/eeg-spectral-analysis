"""
Data Conversion Module

This module handles data format conversion and channel dimension transformation
for EEG data processing. It supports:
1. 63-channel to 64-channel conversion (Cz channel interpolation)
2. MATLAB .mat file loading
3. Dimension transformation for different input formats

Based on MATLAB code: Num_Ch_Corr.m and Read_Signals.m
"""

import numpy as np
from typing import Dict, Tuple, Optional
import warnings

try:
    import scipy.io as sio
    HAS_SCIPY_IO = True
except ImportError:
    HAS_SCIPY_IO = False


def convert_63_to_64_channels(signal_63ch: np.ndarray) -> np.ndarray:
    """
    Convert 63-channel EEG data to 64-channel by interpolating Cz channel.
    
    This function implements the same logic as MATLAB's Num_Ch_Corr.m:
    - Preserves first 23 channels (indices 0-22)
    - Interpolates Cz channel at position 24 (index 23) using 8 adjacent channels
    - Maps original channels 24-63 to positions 25-64 (indices 24-63)
    
    Parameters:
    -----------
    signal_63ch : np.ndarray
        Input EEG data of shape (63, n_timepoints)
        Missing Cz channel (originally at position 24)
    
    Returns:
    --------
    signal_64ch : np.ndarray
        Output EEG data of shape (64, n_timepoints)
        With Cz channel interpolated at position 24
    
    Notes:
    ------
    The Cz channel is computed as the average of 8 adjacent channels:
    - Ch7 (FC1, index 6)
    - Ch39 (FCz, index 38)
    - Ch28 (FC2, index 27)
    - Ch40 (C1, index 39)
    - Ch57 (C2, index 56)
    - Ch12 (CP1, index 11)
    - Ch53 (CPz, index 52)
    - Ch23 (CP2, index 22)
    
    All indices are 0-based (Python convention).
    """
    if signal_63ch.ndim != 2:
        raise ValueError(
            f"Input signal must be 2D array (channels, timepoints), "
            f"got shape: {signal_63ch.shape}"
        )
    
    num_channels, num_timepoints = signal_63ch.shape
    
    if num_channels != 63:
        raise ValueError(
            f"Input must have 63 channels, got {num_channels} channels"
        )
    
    # Ensure float64 precision (consistent with MATLAB double)
    signal_63ch = signal_63ch.astype(np.float64)
    
    # Step 1: Compute Cz channel from 8 adjacent channels
    # Extract the 8 channels used for interpolation
    ch7  = signal_63ch[6, :]   # FC1
    ch39 = signal_63ch[38, :]  # FCz
    ch28 = signal_63ch[27, :]  # FC2
    ch40 = signal_63ch[39, :]  # C1
    ch57 = signal_63ch[56, :]  # C2
    ch12 = signal_63ch[11, :]  # CP1
    ch53 = signal_63ch[52, :]  # CPz
    ch23 = signal_63ch[22, :]  # CP2
    
    # Compute Cz as average of 8 channels
    cz_channel = (ch7 + ch39 + ch28 + ch40 + ch57 + ch12 + ch53 + ch23) / 8.0
    
    # Step 2: Reconstruct 64-channel array
    signal_64ch = np.zeros((64, num_timepoints), dtype=np.float64)
    
    # 1. Keep first 23 channels (indices 0-22)
    signal_64ch[0:23, :] = signal_63ch[0:23, :]
    
    # 2. Insert interpolated Cz at position 24 (index 23)
    signal_64ch[23, :] = cz_channel
    
    # 3. Map original channels 24-63 to positions 25-64 (indices 24-63)
    signal_64ch[24:64, :] = signal_63ch[23:63, :]
    
    return signal_64ch


def load_matlab_data(
    filepath: str,
    subject_id: Optional[int] = None,
    auto_convert_63_to_64: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Load EEG data from MATLAB .mat file.
    
    This function loads data in the format used by the creativity dataset:
    - Variable naming: Creativity_{subject_id}_{trial}_{state} or Creativity_{subject_id}_{RST}
    - States: IDG (Idea Generation), IDE (Idea Evolution), IDR (Idea Rating)
    - Rest states: RST1, RST2
    - Trials: 1, 2, 3
    
    Parameters:
    -----------
    filepath : str
        Path to MATLAB .mat file
    subject_id : Optional[int]
        Subject ID (used for variable name matching). If None, tries to infer from filename.
    auto_convert_63_to_64 : bool
        If True, automatically converts 63-channel data to 64-channel
    
    Returns:
    --------
    data_dict : Dict[str, np.ndarray]
        Dictionary mapping task names to EEG data arrays
        - Keys: task names (e.g., 'IDG_1', 'IDE_2', 'RST1')
        - Values: EEG data arrays of shape (n_channels, n_timepoints)
        - If auto_convert_63_to_64=True and input is 63 channels, output will be 64 channels
    
    Examples:
    ---------
    >>> data = load_matlab_data('Data_Creativity_Sub_1.mat')
    >>> data['IDG_1'].shape  # (64, n_timepoints) if converted
    """
    if not HAS_SCIPY_IO:
        raise ImportError(
            "scipy is required for MATLAB file loading. "
            "Install it with: pip install scipy"
        )
    
    import os
    
    # Try to infer subject_id from filename if not provided
    if subject_id is None:
        filename = os.path.basename(filepath)
        # Try to extract subject ID from filename like "Data_Creativity_Sub_1.mat"
        try:
            parts = filename.replace('.mat', '').split('_')
            if 'Sub' in parts:
                idx = parts.index('Sub')
                subject_id = int(parts[idx + 1])
        except (ValueError, IndexError):
            pass
    
    # Load MATLAB file
    mat_data = sio.loadmat(filepath)
    
    data_dict = {}
    
    # Extract task state data (3 trials × 3 states)
    for trial in [1, 2, 3]:
        for state in ['IDG', 'IDE', 'IDR']:
            # Try different variable name formats
            var_names = [
                f'Creativity_{subject_id}_{trial}_{state}' if subject_id else None,
                f'Creativity_{subject_id}_{state}_{trial}' if subject_id else None,
                f'{state}_{trial}',
            ]
            
            for var_name in var_names:
                if var_name and var_name in mat_data:
                    if not var_name.startswith('__'):  # Skip MATLAB metadata
                        data = mat_data[var_name]
                        # Handle MATLAB's 2D array format
                        if data.ndim == 2:
                            # Check if needs conversion
                            if auto_convert_63_to_64 and data.shape[0] == 63:
                                data = convert_63_to_64_channels(data)
                            data_dict[f'{state}_{trial}'] = data
                            break
    
    # Extract rest state data
    for rst in ['RST1', 'RST2']:
        var_names = [
            f'Creativity_{subject_id}_{rst}' if subject_id else None,
            rst,
        ]
        
        for var_name in var_names:
            if var_name and var_name in mat_data:
                if not var_name.startswith('__'):
                    data = mat_data[var_name]
                    if data.ndim == 2:
                        if auto_convert_63_to_64 and data.shape[0] == 63:
                            data = convert_63_to_64_channels(data)
                        data_dict[rst] = data
                        break
    
    if not data_dict:
        # Fallback: try to load all non-metadata variables
        warnings.warn(
            f"No standard variable names found in {filepath}. "
            f"Attempting to load all variables."
        )
        for key, value in mat_data.items():
            if not key.startswith('__') and isinstance(value, np.ndarray):
                if value.ndim == 2:
                    if auto_convert_63_to_64 and value.shape[0] == 63:
                        value = convert_63_to_64_channels(value)
                    data_dict[key] = value
    
    return data_dict


def ensure_correct_shape(
    data: np.ndarray,
    expected_format: str = "channels_first",
    auto_transpose: bool = True,
) -> Tuple[np.ndarray, bool]:
    """
    Ensure data has correct shape for processing.
    
    The pipeline expects data in format (n_timepoints, n_channels).
    This function checks and optionally transposes if needed.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data array
    expected_format : str
        Expected format: "channels_first" (n_channels, n_timepoints) or
        "timepoints_first" (n_timepoints, n_channels)
    auto_transpose : bool
        If True, automatically transpose if shape doesn't match expected format
    
    Returns:
    --------
    data : np.ndarray
        Data in correct format (n_timepoints, n_channels)
    was_transposed : bool
        Whether the data was transposed
    """
    if data.ndim != 2:
        raise ValueError(
            f"Data must be 2D array, got shape: {data.shape}"
        )
    
    n_dim1, n_dim2 = data.shape
    
    # Determine if channels are first or timepoints are first
    # Heuristic: compare dimensions - channels are typically 8-256, 
    # timepoints are typically much larger (even for short recordings)
    # If first dimension is significantly smaller than second, likely channels_first
    # Use ratio-based approach: if dim1/dim2 < 0.5, likely channels_first
    # Also check absolute values: channels typically 8-256, timepoints typically >100
    ratio = n_dim1 / n_dim2 if n_dim2 > 0 else float('inf')
    is_channels_first = (
        (ratio < 0.5) or  # First dimension is less than half of second
        (n_dim1 <= 256 and n_dim2 > n_dim1 * 2)  # Channels in reasonable range and timepoints much larger
    )
    
    was_transposed = False
    
    if expected_format == "timepoints_first":
        # We want (n_timepoints, n_channels)
        if is_channels_first:
            if auto_transpose:
                data = data.T
                was_transposed = True
            else:
                warnings.warn(
                    f"Data appears to be channels_first ({data.shape}), "
                    f"but expected timepoints_first. Set auto_transpose=True to fix."
                )
    elif expected_format == "channels_first":
        # We want (n_channels, n_timepoints)
        if not is_channels_first:
            if auto_transpose:
                data = data.T
                was_transposed = True
            else:
                warnings.warn(
                    f"Data appears to be timepoints_first ({data.shape}), "
                    f"but expected channels_first. Set auto_transpose=True to fix."
                )
    
    return data, was_transposed


def convert_matlab_to_json_format(
    mat_filepath: str,
    output_filepath: Optional[str] = None,
    subject_id: Optional[int] = None,
    auto_convert_63_to_64: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Convert MATLAB .mat file to JSON format compatible with the pipeline.
    
    This function loads MATLAB data and converts it to the task-centric JSON format:
    {
        "task_name": [[ch1_t1, ch1_t2, ...], [ch2_t1, ch2_t2, ...], ...]
    }
    
    The output format is (n_channels, n_timepoints) which will be transposed
    internally by the pipeline to (n_timepoints, n_channels).
    
    Parameters:
    -----------
    mat_filepath : str
        Path to input MATLAB .mat file
    output_filepath : Optional[str]
        Path to output JSON file. If None, returns dict without saving.
    subject_id : Optional[int]
        Subject ID for variable name matching
    auto_convert_63_to_64 : bool
        Automatically convert 63-channel to 64-channel
    
    Returns:
    --------
    json_data : Dict[str, np.ndarray]
        Dictionary in JSON-compatible format
        Keys: task names
        Values: 2D arrays of shape (n_channels, n_timepoints)
    """
    import json
    import os
    
    # Load MATLAB data
    data_dict = load_matlab_data(
        mat_filepath,
        subject_id=subject_id,
        auto_convert_63_to_64=auto_convert_63_to_64,
    )
    
    # Convert to JSON-compatible format
    # JSON format expects (n_channels, n_timepoints)
    json_data = {}
    for task_name, data in data_dict.items():
        # Ensure channels_first format for JSON
        data, _ = ensure_correct_shape(
            data,
            expected_format="channels_first",
            auto_transpose=True,
        )
        json_data[task_name] = data.tolist()
    
    # Save to JSON if output path provided
    if output_filepath:
        os.makedirs(os.path.dirname(output_filepath) if os.path.dirname(output_filepath) else '.', exist_ok=True)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    return json_data

