"""
EEG channel conversion usage examples.

Demonstrates how to use the eeg_channel_conversion module.
"""

import numpy as np
from eeg_channel_conversion import (
    num_ch_corr, 
    load_creativity_data, 
    convert_all_channels,
    process_all_subjects,
    verify_conversion
)


def example_1_single_signal():
    """Example 1: Convert a single 63-channel signal to 64 channels."""
    print("=" * 60)
    print("Example 1: Convert 63-channel signal to 64 channels")
    print("=" * 60)
    
    # Create simulated 63-channel data
    np.random.seed(42)
    signal_63ch = np.random.randn(63, 1000).astype(np.float64)
    
    print(f"Input shape: {signal_63ch.shape}")
    
    # Convert
    signal_64ch = num_ch_corr(signal_63ch)
    
    print(f"Output shape: {signal_64ch.shape}")
    print("✓ Conversion complete\n")


def example_2_load_and_convert():
    """Example 2: Load and convert real data."""
    print("=" * 60)
    print("Example 2: Load and convert real data")
    print("=" * 60)
    
    # Data directory (adjust as needed)
    data_dir = "."
    subject_id = 1
    
    try:
        # Load data
        print(f"Loading subject {subject_id} data...")
        data_63ch = load_creativity_data(data_dir, subject_id)
        
        print(f"✓ Loaded successfully. States: {list(data_63ch.keys())}")
        
        print("\nOriginal data (63 channels):")
        for state, signal in data_63ch.items():
            print(f"  {state:10s}: {signal.shape}")
        
        print("\nConverting to 64 channels...")
        data_64ch = convert_all_channels(data_63ch)
        
        print("✓ Conversion complete")
        print("\nConverted data (64 channels):")
        for state, signal in data_64ch.items():
            print(f"  {state:10s}: {signal.shape}")
        
        # Verify conversion
        print("\nVerifying conversion...")
        first_state = list(data_63ch.keys())[0]
        is_valid, report = verify_conversion(
            data_63ch[first_state], 
            data_64ch[first_state]
        )
        
        if is_valid:
            print(f"✓ {first_state} verified")
        else:
            print(f"✗ {first_state} verification failed")
            print("  Report:", report)
        
    except FileNotFoundError as e:
        print(f"✗ Data file not found: {e}")
        print("  Ensure the data path is correct.")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()


def example_3_batch_processing():
    """Example 3: Batch process multiple subjects."""
    print("=" * 60)
    print("Example 3: Batch process multiple subjects")
    print("=" * 60)
    
    data_dir = "."
    
    # Process first 3 subjects (example)
    subject_ids = [1, 2, 3]
    
    print(f"Processing subjects: {subject_ids}")
    print("(Processes all states for each subject)\n")
    
    try:
        all_data = process_all_subjects(
            data_dir=data_dir,
            subject_ids=subject_ids,
            save_output=False  # Set True to save results
        )
        
        print(f"\n✓ Successfully processed {len(all_data)} subjects")
        
        for subject_id, data in all_data.items():
            print(f"\nSubject {subject_id}:")
            for state, signal in data.items():
                print(f"  {state:10s}: {signal.shape}")
        
    except Exception as e:
        print(f"✗ Batch processing error: {e}")
    
    print()


def example_4_verify_specific_channel():
    """Example 4: Verify specific channel conversion."""
    print("=" * 60)
    print("Example 4: Verify specific channel conversion")
    print("=" * 60)
    
    # Create test data
    np.random.seed(123)
    signal_63ch = np.random.randn(63, 500).astype(np.float64)
    
    signal_64ch = num_ch_corr(signal_63ch)
    
    # Verify channels 1-23
    print("Verify channels 1-23 (should match exactly):")
    diff_1_23 = np.max(np.abs(signal_64ch[0:23, :] - signal_63ch[0:23, :]))
    print(f"  Max diff: {diff_1_23:.2e}")
    print(f"  {'✓ Passed' if diff_1_23 < 1e-10 else '✗ Failed'}")
    
    # Verify Cz channel
    print("\nVerify Cz channel (ch 24, interpolated):")
    expected_cz = (signal_63ch[6] + signal_63ch[38] + signal_63ch[27] + 
                   signal_63ch[39] + signal_63ch[56] + signal_63ch[11] + 
                   signal_63ch[52] + signal_63ch[22]) / 8.0
    diff_cz = np.max(np.abs(signal_64ch[23, :] - expected_cz))
    print(f"  Max diff: {diff_cz:.2e}")
    print(f"  {'✓ Passed' if diff_cz < 1e-10 else '✗ Failed'}")
    
    # Verify channels 25-64
    print("\nVerify channels 25-64 (original 24-63 mapped to 25-64):")
    diff_25_64 = np.max(np.abs(signal_64ch[24:64, :] - signal_63ch[23:63, :]))
    print(f"  Max diff: {diff_25_64:.2e}")
    print(f"  {'✓ Passed' if diff_25_64 < 1e-10 else '✗ Failed'}")
    
    print()


def example_5_save_converted_data():
    """Example 5: Save converted data."""
    print("=" * 60)
    print("Example 5: Save converted data")
    print("=" * 60)
    
    import scipy.io as sio
    import os
    
    data_dir = "."
    subject_id = 1
    output_dir = "./output_64ch"
    
    try:
        # Load and convert
        data_63ch = load_creativity_data(data_dir, subject_id)
        data_64ch = convert_all_channels(data_63ch)
        
        # Create output dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare save dict with original variable names
        save_dict = {}
        for key, value in data_64ch.items():
            if key in ['RST1', 'RST2']:
                var_name = f'Creativity_{subject_id}_{key}'
            else:
                state, trial = key.split('_')
                var_name = f'Creativity_{subject_id}_{trial}_{state}'
            save_dict[var_name] = value
        
        # Save as MAT file
        output_file = os.path.join(output_dir, f'Data_Creativity_Sub_{subject_id}_64ch.mat')
        sio.savemat(output_file, save_dict)
        
        print(f"✓ Data saved to: {output_file}")
        print(f"  Contains {len(save_dict)} variables")
        
    except Exception as e:
        print(f"✗ Error saving data: {e}")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("EEG Channel Conversion Examples")
    print("=" * 60 + "\n")
    
    # Run all examples
    example_1_single_signal()
    example_2_load_and_convert()
    example_3_batch_processing()
    example_4_verify_specific_channel()
    example_5_save_converted_data()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60 + "\n")

