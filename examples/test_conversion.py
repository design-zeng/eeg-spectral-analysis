"""
Test script: verify Python implementation matches MATLAB logic.

This script:
1. Verifies conversion correctness
2. Compares with MATLAB results (if available)
3. Checks numerical precision
"""

import numpy as np
import scipy.io as sio
from eeg_channel_conversion import num_ch_corr, verify_conversion, load_creativity_data, convert_all_channels
import os


def test_basic_conversion():
    """Test basic conversion."""
    print("=" * 60)
    print("Test 1: Basic conversion")
    print("=" * 60)
    
    # Create test data
    np.random.seed(42)
    test_data = np.random.randn(63, 1000).astype(np.float64)
    
    # Convert
    result = num_ch_corr(test_data)
    
    # Check shape
    assert result.shape == (64, 1000), f"Shape error: expected (64, 1000), got {result.shape}"
    print("✓ Shape check passed")
    
    # Check channels 1-23
    assert np.allclose(result[0:23, :], test_data[0:23, :]), "Channels 1-23 mismatch"
    print("✓ Channels 1-23 check passed")
    
    # Check Cz channel
    expected_cz = (test_data[6] + test_data[38] + test_data[27] + test_data[39] + 
                   test_data[56] + test_data[11] + test_data[52] + test_data[22]) / 8.0
    assert np.allclose(result[23, :], expected_cz), "Cz channel mismatch"
    print("✓ Cz channel check passed")
    
    # Check channels 25-64
    assert np.allclose(result[24:64, :], test_data[23:63, :]), "Channels 25-64 mismatch"
    print("✓ Channels 25-64 check passed")
    
    print("\nAll basic tests passed!\n")


def test_verify_function():
    """Test verification function."""
    print("=" * 60)
    print("Test 2: Verification function")
    print("=" * 60)
    
    # Create test data
    np.random.seed(123)
    test_data = np.random.randn(63, 500).astype(np.float64)
    
    # Convert
    converted = num_ch_corr(test_data)
    
    # Use verification function
    is_valid, report = verify_conversion(test_data, converted)
    
    assert is_valid, "Verification failed"
    print("✓ Verification function test passed")
    print(f"  Report summary:")
    for check_name, check_result in report.items():
        if isinstance(check_result, dict) and 'passed' in check_result:
            status = "✓" if check_result['passed'] else "✗"
            print(f"    {status} {check_name}")
    
    print("\nVerification function test passed!\n")


def test_edge_cases():
    """Test edge cases."""
    print("=" * 60)
    print("Test 3: Edge cases")
    print("=" * 60)
    
    # Test 1: minimal time points
    test_data = np.random.randn(63, 1).astype(np.float64)
    result = num_ch_corr(test_data)
    assert result.shape == (64, 1), "Minimal time points test failed"
    print("✓ Minimal time points test passed")
    
    # Test 2: large time points
    test_data = np.random.randn(63, 100000).astype(np.float64)
    result = num_ch_corr(test_data)
    assert result.shape == (64, 100000), "Large time points test failed"
    print("✓ Large time points test passed")
    
    # Test 3: all zeros
    test_data = np.zeros((63, 100), dtype=np.float64)
    result = num_ch_corr(test_data)
    assert np.allclose(result, np.zeros((64, 100))), "All zeros test failed"
    print("✓ All zeros test passed")
    
    # Test 4: all ones
    test_data = np.ones((63, 100), dtype=np.float64)
    result = num_ch_corr(test_data)
    assert np.allclose(result, np.ones((64, 100))), "All ones test failed"
    print("✓ All ones test passed")
    
    # Test 5: invalid input (already 64 channels)
    try:
        wrong_data = np.random.randn(64, 100)
        num_ch_corr(wrong_data)
        assert False, "Should have raised an error"
    except ValueError as e:
        print(f"✓ Invalid input detection passed: {e}")
    
    print("\nEdge case tests passed!\n")


def test_numerical_precision():
    """Test numerical precision."""
    print("=" * 60)
    print("Test 4: Numerical precision")
    print("=" * 60)
    
    # Create test data
    np.random.seed(42)
    test_data = np.random.randn(63, 1000).astype(np.float64)
    
    # Convert
    result = num_ch_corr(test_data)
    
    # Verify precision
    is_valid, report = verify_conversion(test_data, result, tolerance=1e-10)
    
    if is_valid:
        print("✓ Numerical precision test passed (tolerance: 1e-10)")
    else:
        print("✗ Numerical precision test failed")
        print("  Detailed report:")
        for check_name, check_result in report.items():
            print(f"    {check_name}: {check_result}")
    
    # Check max diffs
    max_diff_1_23 = report['channels_1_23_check']['max_diff']
    max_diff_cz = report['cz_channel_check']['max_diff']
    max_diff_25_64 = report['channels_25_64_check']['max_diff']
    
    print(f"  Channels 1-23 max diff: {max_diff_1_23:.2e}")
    print(f"  Cz channel max diff: {max_diff_cz:.2e}")
    print(f"  Channels 25-64 max diff: {max_diff_25_64:.2e}")
    
    print("\nNumerical precision test done!\n")


def test_real_data_if_available():
    """Test with real data if available."""
    print("=" * 60)
    print("Test 5: Real data (if available)")
    print("=" * 60)
    
    data_directory = "."
    subject_id = 1
    
    try:
        # Try to load real data
        data_63ch = load_creativity_data(data_directory, subject_id)
        
        print(f"✓ Loaded subject {subject_id} data")
        print(f"  States: {list(data_63ch.keys())}")
        
        # Convert all states
        data_64ch = convert_all_channels(data_63ch)
        
        print(f"✓ Successfully converted all states")
        
        # Verify each state
        for state in data_63ch.keys():
            is_valid, report = verify_conversion(data_63ch[state], data_64ch[state])
            status = "✓" if is_valid else "✗"
            print(f"  {status} {state}: {data_63ch[state].shape} -> {data_64ch[state].shape}")
        
        print("\nReal data test done!\n")
        
    except FileNotFoundError:
        print(f"⚠ Data file not found, skipping real data test")
        print(f"  Expected: {os.path.join(data_directory, 'Creativity_EEG_Dataset', f'Data_Creativity_Sub_{subject_id}.mat')}")
        print()
    except Exception as e:
        print(f"✗ Real data test error: {e}")
        print()


def compare_with_matlab_result():
    """Compare with MATLAB results if available."""
    print("=" * 60)
    print("Test 6: MATLAB comparison (if available)")
    print("=" * 60)
    
    # Path to MATLAB result
    matlab_result_file = "./matlab_results/Data_Creativity_Sub_1_64ch.mat"
    
    if not os.path.exists(matlab_result_file):
        print("⚠ MATLAB result file not found, skipping comparison")
        print(f"  Expected: {matlab_result_file}")
        print()
        return
    
    try:
        # Load MATLAB result
        matlab_data = sio.loadmat(matlab_result_file)
        
        # Load Python-processed data
        data_directory = "."
        python_data_63ch = load_creativity_data(data_directory, 1)
        python_data_64ch = convert_all_channels(python_data_63ch)
        
        # Compare results
        print("Comparison results:")
        for key in python_data_64ch.keys():
            # Build MATLAB variable name
            if key in ['RST1', 'RST2']:
                matlab_key = f'Creativity_1_{key}'
            else:
                state, trial = key.split('_')
                matlab_key = f'Creativity_1_{trial}_{state}'
            
            if matlab_key in matlab_data:
                python_result = python_data_64ch[key]
                matlab_result = matlab_data[matlab_key]
                
                # Check shape
                if python_result.shape != matlab_result.shape:
                    print(f"  ✗ {key}: Shape mismatch")
                    print(f"    Python: {python_result.shape}, MATLAB: {matlab_result.shape}")
                    continue
                
                # Compare values
                max_diff = np.max(np.abs(python_result - matlab_result))
                is_close = np.allclose(python_result, matlab_result, atol=1e-10)
                
                status = "✓" if is_close else "✗"
                print(f"  {status} {key}: max diff = {max_diff:.2e}")
            else:
                print(f"  ⚠ {key}: Variable not found in MATLAB result")
        
        print("\nMATLAB comparison test done!\n")
        
    except Exception as e:
        print(f"✗ MATLAB comparison error: {e}")
        print()


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running all tests")
    print("=" * 60 + "\n")
    
    test_basic_conversion()
    test_verify_function()
    test_edge_cases()
    test_numerical_precision()
    test_real_data_if_available()
    compare_with_matlab_result()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_tests()

