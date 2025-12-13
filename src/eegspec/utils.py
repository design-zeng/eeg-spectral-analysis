import os, json, re
import warnings

import numpy as np
from typing import List, Tuple, Dict, Any

EPS = 1e-20

def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_channels(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [ln.strip() for ln in f if ln.strip() and ln.strip()[0] not in "#;%"]

def parse_channels_arg(chs: str) -> List[str]:
    return [c.strip() for c in chs.split(",") if c.strip()]

def builtin_montage_path() -> str:
    import importlib.resources as ir
    return str(ir.files("eegspec").joinpath("data/montages/caps63.locs"))

def parse_locs_file(path: str) -> List[str]:
    """Robust .locs parser: accept EEGLAB-like multi-column or one-label-per-line."""
    ch = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line[0] in "#;%":
                continue
            toks = re.split(r"[\s,]+", line)
            label = None
            for tok in toks:
                if re.search(r"[A-Za-z]", tok):
                    label = tok
                    break
            if label is None and len(toks) >= 2:
                label = toks[1]
            if label:
                ch.append(label)
    if not ch:
        raise ValueError(f"No channel names parsed from {path}")
    return ch

def resolve_channels(channels_file: str = None,  n_channels: int = None) -> List[str]:
    """Resolve channel names from CSV string, plaintext file, or .locs file; fallback to built-in montage.
    If n_channels is provided and names mismatch, generate fallback names Ch1..ChN to keep pipeline running."""
    if channels_file:
        if os.path.splitext(channels_file)[1].lower() in (".locs", ".eloc", ".sfp"):
            ch = parse_locs_file(channels_file)
        elif os.path.splitext(channels_file)[1].lower() in ".csv":
            ch = parse_channels_arg(channels_file)
        else:
            ch = load_channels(channels_file)
    else:
        warnings.warn("Missing montage input, using built-in caps63 montage.")
        try:
            ch = parse_locs_file(builtin_montage_path())
        except Exception as e:
            raise f"Error parse built-in montage channel file: {e}"
    if n_channels is not None and len(ch) != n_channels:
        warnings.warn(f"Input data channels amount {len(ch)} doesn't match montage file channels amount: {n_channels}")
        return [f"Ch{i+1}" for i in range(n_channels)]
    return ch

def list_subject_jsons(input_path: str) -> List[str]:
    """
    List subject files (JSON or MATLAB .mat) from input path.
    
    Parameters:
    -----------
    input_path : str
        Path to directory or single file
    
    Returns:
    --------
    List[str]
        List of file paths (JSON or .mat files)
    """
    if os.path.isdir(input_path):
        files = []
        for x in os.listdir(input_path):
            if x.lower().endswith((".json", ".mat")):
                files.append(os.path.join(input_path, x))
        return sorted(files)
    else:
        # Single file
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        return [input_path]

def subject_id_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def load_subject_tasks_json(path: str, auto_convert_63_to_64: bool = True) -> Dict[str, np.ndarray]:
    """
    Load a subject JSON in 'task-centric' format:
    { "task_A": [[ch1_series ...], [ch2_series ...], ...], ... }
    
    Also supports MATLAB .mat files (automatically detected by extension).
    
    Parameters:
    -----------
    path : str
        Path to JSON or MATLAB .mat file
    auto_convert_63_to_64 : bool
        If True, automatically converts 63-channel data to 64-channel
    
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary mapping task_name -> data (n_times, n_channels)
    """
    import os
    
    # Check file extension
    file_ext = os.path.splitext(path)[1].lower()
    
    # Handle MATLAB .mat files
    if file_ext == '.mat':
        from eegspec.data_conversion import load_matlab_data, ensure_correct_shape
        
        # Try to infer subject ID from filename
        subject_id = None
        filename = os.path.basename(path)
        try:
            parts = filename.replace('.mat', '').split('_')
            if 'Sub' in parts:
                idx = parts.index('Sub')
                subject_id = int(parts[idx + 1])
        except (ValueError, IndexError):
            pass
        
        # Load MATLAB data
        data_dict = load_matlab_data(
            path,
            subject_id=subject_id,
            auto_convert_63_to_64=auto_convert_63_to_64,
        )
        
        # Convert to (n_times, n_channels) format
        out = {}
        for task, data in data_dict.items():
            # Ensure correct shape: (n_timepoints, n_channels)
            data, _ = ensure_correct_shape(
                data,
                expected_format="timepoints_first",
                auto_transpose=True,
            )
            out[str(task)] = data
        
        return out
    
    # Handle JSON files
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("Top-level JSON must be an object mapping task_name -> channel_lists")
    
    out = {}
    for task, v in obj.items():
        arr = np.array(v, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"Task '{task}' is not a 2D matrix (n_channels x n_times)")
        
        # Check if 63 channels and conversion is enabled
        if auto_convert_63_to_64 and arr.shape[0] == 63:
            from eegspec.data_conversion import convert_63_to_64_channels
            arr = convert_63_to_64_channels(arr)
        
        out[str(task)] = arr.T  # to (n_times, n_channels)
    return out

if __name__ == "__main__":
    print(parse_locs_file("./data/montages/caps63.locs"))
