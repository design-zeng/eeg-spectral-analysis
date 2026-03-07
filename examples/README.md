# EEG 63-to-64 Channel Conversion Tool

This tool converts 63-channel EEG data to 64 channels (restoring the missing Cz channel) with logic fully consistent with the reference MATLAB implementation. **Guarantees MATLAB compatibility for reproducible experiments.**

## Files

- `eeg_channel_conversion.py`: Core conversion module
- `test_conversion.py`: Test script
- `EEG_Channel_Conversion_Documentation.md`: Technical documentation
- `example_usage.py`: Usage examples

## Quick Start

### 1. Install Dependencies

```bash
pip install numpy scipy
```

### 2. Basic Usage

```python
from eeg_channel_conversion import num_ch_corr
import numpy as np

# Load 63-channel data (example)
# signal_63ch should be numpy array of shape (63, time_points)
signal_63ch = np.random.randn(63, 1000).astype(np.float64)

# Convert to 64 channels
signal_64ch = num_ch_corr(signal_63ch)

print(f"Input shape: {signal_63ch.shape}")  # (63, 1000)
print(f"Output shape: {signal_64ch.shape}")  # (64, 1000)
```

### 3. Load and Process Real Data

```python
from eeg_channel_conversion import load_creativity_data, convert_all_channels

# Load subject 1 data
data_63ch = load_creativity_data(".", subject_id=1)

# Convert to 64 channels
data_64ch = convert_all_channels(data_63ch)

for state, signal in data_64ch.items():
    print(f"{state}: {signal.shape}")
```

### 4. Batch Process All Subjects

```python
from eeg_channel_conversion import process_all_subjects

all_data = process_all_subjects(
    data_dir=".",
    save_output=True,
    output_dir="./output_64ch"
)
```

## Conversion Logic

1. **Compute Cz channel**: Mean of 8 neighboring channels
   ```
   Cz = (FC1 + FCz + FC2 + C1 + C2 + CP1 + CPz + CP2) / 8
   ```

2. **Reorder channels**:
   - First 23 channels unchanged
   - Insert interpolated Cz at position 24
   - Original 24-63 mapped to 25-64

## Verification

Run the test script:

```bash
python test_conversion.py
```

## Data Format

- **Input**: `./Creativity_EEG_Dataset/Data_Creativity_Sub_{subject_id}.mat`
- **Output**: 64 × time_points per variable
