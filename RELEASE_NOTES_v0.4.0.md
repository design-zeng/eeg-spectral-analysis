# Release Notes v0.4.0

## Overview

Version 0.4.0 adds microstate-driven protocol segmentation and dynamic-window spectral analysis to `eegspec`, while preserving the existing Welch PSD, band power, TRP, design-creativity, and statistical workflows.

This release also moves the core protocol segmentation implementation into the sibling `microstate_analysis` package and keeps `eegspec` as the spectral-analysis caller and compatibility layer.

## Major Changes

### Protocol Segmentation Delegation

- Added `eegspec segment-protocol` as a thin CLI caller around `microstate_analysis.protocol_segmentation`.
- Added a backward-compatible `eegspec.segmentation` re-export package for existing imports.
- Declared `microstate_analysis>=0.4.0` as the protocol segmentation dependency.
- Supports Line A / Line B presets:
  - Line A: high-sensitivity macro windows (`macro_window_size=1`, `stride=1`).
  - Line B: coarse macro windows (`macro_window_size=5`, `stride=5` by default).

### Dynamic Microstate-Window Spectral Analysis

- Added `eegspec dynamic-window` with two independent windowing modes:
  - `--mode run`: regularizes microstate labels and uses retained microstate runs as spectral windows.
  - `--mode subtask`: runs protocol segmentation and uses merged subtasks as spectral windows.
- If `--labels-npy` is omitted, the command computes labels directly from EEG using `Microstate.opt_microstate` plus `fit_back`, then writes `computed_labels.npy` into `--out-dir`.
- A and B modes can run independently from the same EEG input; B does not need to reuse A's intermediate labels.
- Outputs:
  - `dynamic_windows.json`
  - `dynamic_metrics.json`
  - `dynamic_summary.json`
  - optional `dynamic_psd.json` with `--save-psd`

### Local Real-Data Examples

- README examples now use the local real dataset:
  - `D:\Documents\GitHub\microstate-analysis\storage\clean_data\sub_01.json`
  - task `1_rest`
- Commands demonstrate independent A/B dynamic-window runs and follow-up inspection of windows and spectral metrics.

### Compatibility and Robustness

- Added NumPy 2.x compatibility for FAA integration through `trapz_compat`.
- Made Welch PSD robust to short segments by clamping `nperseg` and `noverlap` to valid per-window values.
- Added automatic sibling-checkout discovery for `microstate_analysis` when it is not installed but exists at `D:\Documents\GitHub\microstate-analysis`.
- Improved JSON output for dynamic-window results by converting non-finite metric values to JSON-safe `null`.

## Testing

- Added tests for dynamic-window run mode, subtask mode, and optional label computation.
- Existing PSD/FAA and segmentation tests continue to pass.

Verified command:

```powershell
python -m pytest tests/test_dynamic_window.py tests/test_basic.py tests/test_segmentation.py -q
```

## Notes

- `cupy` remains optional. If it is not installed, `microstate_analysis` reports the missing GPU backend and continues on CPU.
- Generated local outputs under `output/` are ignored by git.
