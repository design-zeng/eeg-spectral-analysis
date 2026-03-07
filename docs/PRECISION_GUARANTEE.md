# 64-bit Precision Guarantee

## Overview

This document describes how the Python implementation ensures **64-bit floating-point precision** throughout all calculations to match MATLAB's `double` precision exactly.

---

## MATLAB vs Python Data Types

| MATLAB | Python | Bits | Description |
|--------|--------|------|-------------|
| `double` | `float64` | 64 | Double-precision floating-point |
| `single` | `float32` | 32 | Single-precision floating-point |

**MATLAB default:** All numeric calculations use `double` (64-bit) unless explicitly specified.

**Python default:** NumPy may use `float32` in some operations if input is `float32`.

**Solution:** Explicitly enforce `float64` (64-bit) throughout the pipeline.

---

## Precision Guarantees in Each Function

### 1. `compute_wpli_pair()`

**Ensures 64-bit precision at:**
- ✅ Input conversion: `np.asarray(eeg1, dtype=np.float64)`
- ✅ Filter coefficients: `np.asarray(b, dtype=np.float64)`, `np.asarray(a, dtype=np.float64)`
- ✅ Filtered signals: `filtfilt()` preserves input dtype (float64)
- ✅ Hilbert transform: Returns `complex128` (64-bit real + 64-bit imag)
- ✅ Cross-spectral density: `np.asarray(imag_phase_diff, dtype=np.float64)`
- ✅ Mean calculations: `np.mean(..., dtype=np.float64)`
- ✅ Final result: `np.float64(wpli_value)`

**Code snippet:**
```python
# Ensure 1D arrays with 64-bit precision (matching MATLAB double)
eeg1 = np.asarray(eeg1, dtype=np.float64).flatten()
eeg2 = np.asarray(eeg2, dtype=np.float64).flatten()

# Filter coefficients
b, a = butter(2, [low, high], btype='bandpass')
b = np.asarray(b, dtype=np.float64)
a = np.asarray(a, dtype=np.float64)

# Mean calculations with explicit dtype
num = np.abs(np.mean(imag_phase_diff, dtype=np.float64))
denom = np.mean(np.abs(imag_phase_diff), dtype=np.float64)
```

---

### 2. `compute_wpli()`

**Ensures 64-bit precision at:**
- ✅ Input data: `np.asarray(data, dtype=np.float64)`
- ✅ Matrix initialization: `np.zeros(..., dtype=np.float64)`
- ✅ All wPLI values: Returned from `compute_wpli_pair()` (float64)

**Code snippet:**
```python
# Ensure 64-bit precision (matching MATLAB double)
data = np.asarray(data, dtype=np.float64)

# Initialize wPLI matrix with 64-bit precision
wpli_matrix = np.zeros((n_channels, n_channels), dtype=np.float64)
```

---

### 3. `compute_strength()`

**Ensures 64-bit precision at:**
- ✅ Input matrix: `np.asarray(wpli_matrix, dtype=np.float64)`
- ✅ Sum calculation: `np.sum(..., dtype=np.float64)`
- ✅ Output: Column vector with `dtype=np.float64`

**Code snippet:**
```python
# Ensure 64-bit precision (matching MATLAB double)
wpli_matrix = np.asarray(wpli_matrix, dtype=np.float64)

# Use dtype=np.float64 to ensure 64-bit accumulation
strength_values = np.sum(wpli_matrix, axis=1, dtype=np.float64)
```

---

### 4. `compute_betweenness()`

**Ensures 64-bit precision at:**
- ✅ Input matrix: `np.asarray(wpli_matrix, dtype=np.float64)`
- ✅ Upper triangular: Operations preserve float64
- ✅ Symmetric matrix: `np.asarray(wpli_symmetric, dtype=np.float64)`
- ✅ Output array: `np.array(..., dtype=np.float64)`

**Code snippet:**
```python
# Ensure 64-bit precision (matching MATLAB double)
wpli_matrix = np.asarray(wpli_matrix, dtype=np.float64)

# Ensure all operations use float64
wpli_symmetric = np.asarray(wpli_symmetric, dtype=np.float64)

# Convert to array with 64-bit precision
betweenness_array = np.array([betweenness[i] for i in range(n_channels)], dtype=np.float64)
```

---

### 5. `connectivity_analysis()`

**Ensures 64-bit precision at:**
- ✅ Input signal: `np.asarray(eeg_signal, dtype=np.float64)`
- ✅ wPLI matrix: From `compute_wpli()` (float64)
- ✅ Threshold: `np.float64(threshold)`
- ✅ Thresholded matrix: `np.float64(0.0)` for zeros
- ✅ Strength: `np.asarray(strength_values, dtype=np.float64)`
- ✅ Betweenness: `np.asarray(betweenness, dtype=np.float64)`
- ✅ Features: `np.asarray(features, dtype=np.float64)`

**Code snippet:**
```python
# Ensure 64-bit precision (matching MATLAB double)
eeg_signal = np.asarray(eeg_signal, dtype=np.float64)

# Ensure wpli_matrix is float64
wpli_matrix = np.asarray(wpli_matrix, dtype=np.float64)

# Ensure threshold is float64
threshold = np.float64(threshold)
thresholded_wpli[thresholded_wpli < threshold] = np.float64(0.0)

# Ensure all outputs are float64
strength_values = np.asarray(strength_values, dtype=np.float64)
betweenness = np.asarray(betweenness, dtype=np.float64)
features = np.asarray(features, dtype=np.float64)
```

---

## Verification

### Test 1: Check Data Types

```python
import numpy as np
from eegspec.connectivity import connectivity_analysis

# Create test data
eeg_signal = np.random.randn(64, 1000)  # 64 channels, 1000 timepoints

# Run analysis
result = connectivity_analysis(eeg_signal, fs=500.0, freq_range=(8.0, 13.0), threshold=0.2)

# Verify all outputs are float64
assert result['wpli_matrix'].dtype == np.float64
assert result['thresholded_wpli'].dtype == np.float64
assert result['strength'].dtype == np.float64
assert result['betweenness'].dtype == np.float64
assert result['features'].dtype == np.float64

print("✓ All outputs are float64 (64-bit precision)")
```

### Test 2: Compare with MATLAB

When comparing with MATLAB results, differences should be:
- **wPLI matrix**: < 1e-10 (machine precision)
- **Strength**: < 1e-9 (accumulation of wPLI differences)
- **Betweenness**: < 1e-6 (graph algorithm differences)

If differences are larger, check:
1. Input data types (should be float64)
2. All intermediate calculations (should be float64)
3. MATLAB version and settings

---

## Key Principles

1. **Explicit dtype specification**: Always use `dtype=np.float64` when creating arrays
2. **Input conversion**: Convert all inputs to `float64` at function entry
3. **Intermediate calculations**: Ensure all NumPy operations use `float64`
4. **Output guarantee**: Explicitly convert outputs to `float64` before returning

---

## Common Pitfalls

### ❌ Wrong: Implicit dtype
```python
data = np.array([1, 2, 3])  # May be int32 or float32
result = compute_wpli(data, ...)  # Precision lost
```

### ✅ Correct: Explicit dtype
```python
data = np.array([1, 2, 3], dtype=np.float64)  # Explicit float64
result = compute_wpli(data, ...)  # Precision maintained
```

### ❌ Wrong: Default mean
```python
mean_val = np.mean(arr)  # May use input dtype (could be float32)
```

### ✅ Correct: Explicit dtype in mean
```python
mean_val = np.mean(arr, dtype=np.float64)  # Explicit float64
```

---

## Summary

All connectivity analysis functions now **guarantee 64-bit precision** throughout:
- ✅ Input data conversion
- ✅ Filter operations
- ✅ Signal processing (Hilbert transform)
- ✅ Matrix operations
- ✅ Statistical calculations (mean, sum)
- ✅ Graph theory computations
- ✅ Output arrays

This ensures **numerical equivalence** with MATLAB's `double` precision calculations.

