"""
EEG 63-to-64 channel conversion module.

Fully consistent with MATLAB logic for reproducible experiments.
Based on MATLAB code Num_Ch_Corr.m and Read_Signals.m.
"""

import numpy as np
import scipy.io as sio
import os
from typing import Dict, Tuple, Optional
import warnings


def num_ch_corr(signal_63ch: np.ndarray) -> np.ndarray:
    """
    Convert 63-channel EEG data to 64 channels (insert Cz channel).

    Logic matches MATLAB Num_Ch_Corr.m exactly.

    Parameters:
        signal_63ch: numpy array, shape (63, time_points)
                    63-channel EEG data with Cz missing.

    Returns:
        signal_64ch: numpy array, shape (64, time_points)
                    64-channel EEG data with Cz restored by interpolation.

    Conversion logic:
        1. Compute Cz as mean of 8 neighboring channels
        2. Keep first 23 channels unchanged
        3. Insert interpolated Cz at position 24
        4. Map original 24-63 to positions 25-64
    """
    if signal_63ch.ndim != 2:
        raise ValueError(f"Input must be 2D (channels, time_points), got ndim={signal_63ch.ndim}")
    
    num_channels, num_timepoints = signal_63ch.shape
    
    if num_channels != 63:
        raise ValueError(f"Input must have 63 channels, got {num_channels}")
    
    # Use float64 to match MATLAB double
    signal_63ch = signal_63ch.astype(np.float64)
    
    # Step 1: Compute Cz channel (matches MATLAB)
    # MATLAB 1-based -> Python 0-based index mapping:
    # Ch7  -> index 6  (FC1)
    # Ch39 -> index 38 (FCz)
    # Ch28 -> index 27 (FC2)
    # Ch40 -> index 39 (C1)
    # Ch57 -> index 56 (C2)
    # Ch12 -> index 11 (CP1)
    # Ch53 -> index 52 (CPz)
    # Ch23 -> index 22 (CP2)
    
    # Extract 8 neighboring channels
    ch7  = signal_63ch[6, :]   # FC1
    ch39 = signal_63ch[38, :]  # FCz
    ch28 = signal_63ch[27, :]  # FC2
    ch40 = signal_63ch[39, :]  # C1
    ch57 = signal_63ch[56, :]  # C2
    ch12 = signal_63ch[11, :]  # CP1
    ch53 = signal_63ch[52, :]  # CPz
    ch23 = signal_63ch[22, :]  # CP2
    
    # Compute Cz as mean of 8 channels (matches MATLAB)
    cz_channel = (ch7 + ch39 + ch28 + ch40 + ch57 + ch12 + ch53 + ch23) / 8.0
    
    # Step 2: Reorder channels (matches MATLAB)
    # Allocate 64-channel output
    signal_64ch = np.zeros((64, num_timepoints), dtype=np.float64)
    
    # 1. Keep first 23 channels
    signal_64ch[0:23, :] = signal_63ch[0:23, :]
    
    # 2. Insert Cz at position 24
    signal_64ch[23, :] = cz_channel
    
    # 3. Map original 24-63 to 25-64
    signal_64ch[24:64, :] = signal_63ch[23:63, :]
    
    return signal_64ch


def load_creativity_data(data_dir: str, subject_id: int) -> Dict[str, np.ndarray]:
    """
    Load EEG data for a given subject.

    Parameters:
        data_dir: Path to data directory (contains Creativity_EEG_Dataset)
        subject_id: Subject ID (1-28)

    Returns:
        data_dict: Dict of state -> (63, time_points) arrays
    """
    # Build file path
    filename = os.path.join(data_dir, 'Creativity_EEG_Dataset', 
                           f'Data_Creativity_Sub_{subject_id}.mat')
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Data file not found: {filename}")
    
    # Load MAT file
    mat_data = sio.loadmat(filename)
    
    # Extract relevant variables
    data_dict = {}
    
    # Task data: 3 trials x 3 states
    for trial in [1, 2, 3]:
        for state in ['IDG', 'IDE', 'IDR']:
            var_name = f'Creativity_{subject_id}_{trial}_{state}'
            if var_name in mat_data:
                # Skip MATLAB metadata keys
                if not var_name.startswith('__'):
                    data_dict[f'{state}_{trial}'] = mat_data[var_name]
    
    # Rest state data
    for rst in ['RST1', 'RST2']:
        var_name = f'Creativity_{subject_id}_{rst}'
        if var_name in mat_data:
            if not var_name.startswith('__'):
                data_dict[rst] = mat_data[var_name]
    
    return data_dict


def convert_all_channels(data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Convert all 63-channel arrays in the dict to 64 channels.
    """
    converted_dict = {}
    
    for key, signal_63ch in data_dict.items():
        # Validate shape
        if signal_63ch.ndim == 2 and signal_63ch.shape[0] == 63:
            converted_dict[key] = num_ch_corr(signal_63ch)
        else:
            warnings.warn(f"Skipping {key}: invalid shape {signal_63ch.shape}")
    
    return converted_dict


def process_all_subjects(data_dir: str, subject_ids: Optional[list] = None, 
                        save_output: bool = False, output_dir: Optional[str] = None) -> Dict:
    """
    Process data for all subjects.

    Parameters:
        data_dir: Data directory path
        subject_ids: Subject ID list; None means 1-28
        save_output: Whether to save converted data
        output_dir: Output directory when save_output=True

    Returns:
        all_data: Nested dict {subject_id: {state: 64ch_data}}
    """
    if subject_ids is None:
        subject_ids = list(range(1, 29))  # 1-28
    
    all_data = {}
    
    for subject_id in subject_ids:
        print(f"Processing subject {subject_id}...")
        
        try:
            # Load data
            data_63ch = load_creativity_data(data_dir, subject_id)
            
            # Convert to 64 channels
            data_64ch = convert_all_channels(data_63ch)
            
            all_data[subject_id] = data_64ch
            
            # Save if requested
            if save_output and output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, 
                                         f'Data_Creativity_Sub_{subject_id}_64ch.mat')
                
                # Prepare save dict with original variable names
                save_dict = {}
                for key, value in data_64ch.items():
                    # Reconstruct variable name
                    if key in ['RST1', 'RST2']:
                        var_name = f'Creativity_{subject_id}_{key}'
                    else:
                        state, trial = key.split('_')
                        var_name = f'Creativity_{subject_id}_{trial}_{state}'
                    save_dict[var_name] = value
                
                sio.savemat(output_file, save_dict)
                print(f"  Saved to: {output_file}")
            
        except Exception as e:
            print(f"  Error processing subject {subject_id}: {e}")
            continue
    
    return all_data


def verify_conversion(original_63ch: np.ndarray, converted_64ch: np.ndarray, 
                     tolerance: float = 1e-10) -> Tuple[bool, Dict]:
    """
    Verify conversion correctness.

    Parameters:
        original_63ch: Original 63-channel data
        converted_64ch: Converted 64-channel data
        tolerance: Numerical tolerance

    Returns:
        is_valid: Whether verification passed
        report: Verification report dict
    """
    report = {}
    
    # 1. Shape check
    expected_shape = (64, original_63ch.shape[1])
    shape_valid = converted_64ch.shape == expected_shape
    report['shape_check'] = {
        'passed': shape_valid,
        'expected': expected_shape,
        'actual': converted_64ch.shape
    }
    
    # 2. Channels 1-23 check
    channels_1_23_valid = np.allclose(
        converted_64ch[0:23, :], 
        original_63ch[0:23, :], 
        atol=tolerance
    )
    max_diff_1_23 = np.max(np.abs(converted_64ch[0:23, :] - original_63ch[0:23, :]))
    report['channels_1_23_check'] = {
        'passed': channels_1_23_valid,
        'max_diff': max_diff_1_23
    }
    
    # 3. Cz channel check (recompute Cz)
    ch7  = original_63ch[6, :]
    ch39 = original_63ch[38, :]
    ch28 = original_63ch[27, :]
    ch40 = original_63ch[39, :]
    ch57 = original_63ch[56, :]
    ch12 = original_63ch[11, :]
    ch53 = original_63ch[52, :]
    ch23 = original_63ch[22, :]
    expected_cz = (ch7 + ch39 + ch28 + ch40 + ch57 + ch12 + ch53 + ch23) / 8.0
    
    cz_valid = np.allclose(converted_64ch[23, :], expected_cz, atol=tolerance)
    max_diff_cz = np.max(np.abs(converted_64ch[23, :] - expected_cz))
    report['cz_channel_check'] = {
        'passed': cz_valid,
        'max_diff': max_diff_cz
    }
    
    # 4. Channels 25-64 check (original 24-63 mapped to 25-64)
    channels_25_64_valid = np.allclose(
        converted_64ch[24:64, :], 
        original_63ch[23:63, :], 
        atol=tolerance
    )
    max_diff_25_64 = np.max(np.abs(converted_64ch[24:64, :] - original_63ch[23:63, :]))
    report['channels_25_64_check'] = {
        'passed': channels_25_64_valid,
        'max_diff': max_diff_25_64
    }
    
    # Overall verification
    is_valid = (shape_valid and channels_1_23_valid and 
                cz_valid and channels_25_64_valid)
    
    report['overall'] = {
        'passed': is_valid,
        'tolerance': tolerance
    }
    
    return is_valid, report


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # Example 1: single signal conversion
    print("=" * 60)
    print("Example 1: Single signal conversion")
    print("=" * 60)
    
    # Create test data
    np.random.seed(42)
    test_signal_63ch = np.random.randn(63, 1000).astype(np.float64)
    
    test_signal_64ch = num_ch_corr(test_signal_63ch)
    
    print(f"Input shape: {test_signal_63ch.shape}")
    print(f"Output shape: {test_signal_64ch.shape}")
    
    # Verify conversion
    is_valid, report = verify_conversion(test_signal_63ch, test_signal_64ch)
    print(f"\nVerification: {'PASSED' if is_valid else 'FAILED'}")
    for check_name, check_result in report.items():
        print(f"  {check_name}: {check_result}")
    
    # Example 2: load and process real data
    print("\n" + "=" * 60)
    print("Example 2: Load and process real data")
    print("=" * 60)
    
    data_directory = "."  # Set to your data directory
    
    try:
        # Load subject 1 data
        data_63ch = load_creativity_data(data_directory, subject_id=1)
        print(f"\nLoaded states: {list(data_63ch.keys())}")
        
        # Show dimensions
        for state, signal in data_63ch.items():
            print(f"  {state}: {signal.shape}")
        
        data_64ch = convert_all_channels(data_63ch)
        print(f"\nConverted states: {list(data_64ch.keys())}")
        
        # Show output dimensions
        for state, signal in data_64ch.items():
            print(f"  {state}: {signal.shape}")
        
        # Verify first state
        first_state = list(data_63ch.keys())[0]
        is_valid, report = verify_conversion(
            data_63ch[first_state], 
            data_64ch[first_state]
        )
        print(f"\n{first_state} verification: {'PASSED' if is_valid else 'FAILED'}")
        
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        print("Ensure the data path is correct.")
    
    # Example 3: batch processing
    print("\n" + "=" * 60)
    print("Example 3: Batch processing (commented out)")
    print("=" * 60)
    print("# Uncomment below to process all subjects:")
    print("# all_data = process_all_subjects(data_directory, save_output=True, output_dir='./output_64ch')")

