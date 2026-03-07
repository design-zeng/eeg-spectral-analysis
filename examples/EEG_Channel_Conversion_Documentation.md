# EEG 63-to-64 Channel Conversion Documentation

## 1. Background

### 1.1 Data Source

- **Original format**: 63 channels × time points
- **Reason**: Cz (channel 24) was removed during EEG recording/preprocessing
- **Goal**: Restore standard 64-channel format for analysis and visualization

### 1.2 File Structure

- **Path**: `./Creativity_EEG_Dataset/Data_Creativity_Sub_{subject_id}.mat`
- **Variable naming**:
  - Task states: `Creativity_{subject_id}_{trial}_{state}`
    - `state`: IDG (Idea Generation), IDE (Idea Evolution), IDR (Idea Rating)
    - `trial`: 1, 2, 3
  - Rest states: `Creativity_{subject_id}_RST1`, `Creativity_{subject_id}_RST2`
- **Dimensions**: All variables are 63 × time_points double arrays

## 2. Conversion Logic

### 2.1 Standard 64-Channel Mapping (BrainVision)

| Position | Channels | Notes |
|----------|----------|-------|
| 1-23 | Fp1...CP2 | Unchanged |
| **24** | **Cz** | **Interpolated** |
| 25-64 | C4...AF8 | Original 24-63 shifted |

### 2.2 Cz Interpolation

**Formula** (matches MATLAB):

```
CZ = (Ch7 + Ch39 + Ch28 + Ch40 + Ch57 + Ch12 + Ch53 + Ch23) / 8
```

**8 neighboring channels** (positions in 63-channel data):

- Ch7: FC1
- Ch39: FCz
- Ch28: FC2
- Ch40: C1
- Ch57: C2
- Ch12: CP1
- Ch53: CPz
- Ch23: CP2

### 2.3 Steps (matches MATLAB `Num_Ch_Corr.m`)

```python
# Step 1: Compute Cz
CZ = (Signal2[6] + Signal2[38] + Signal2[27] + Signal2[39] +
      Signal2[56] + Signal2[11] + Signal2[52] + Signal2[22]) / 8.0

# Step 2: Reorder
output[0:23, :] = input[0:23, :]
output[23, :] = CZ
output[24:64, :] = input[23:63, :]
```

## 3. Verification

- **Input**: 63 × time_points
- **Output**: 64 × time_points
- Channels 1-23: identical to input
- Cz: mean of 8 specified channels
- Channels 25-64: identical to input 24-63
