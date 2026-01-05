import os, json, re
import warnings

import numpy as np
from typing import List, Tuple, Dict, Any

EPS = 1e-20


class Float64JSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that preserves full 64-bit float precision.
    
    Overrides the default float serialization to use repr() which provides
    enough digits to exactly represent the float value without loss of precision.
    """
    def iterencode(self, obj, _one_shot=False):
        if isinstance(obj, float):
            # Use repr() to preserve full precision (enough digits to exactly represent float64)
            yield repr(obj)
        elif isinstance(obj, np.floating):
            yield repr(float(obj))
        elif isinstance(obj, np.integer):
            yield str(int(obj))
        elif isinstance(obj, np.ndarray):
            # Convert to list and recursively encode
            yield from self.iterencode(obj.tolist(), _one_shot)
        else:
            # Use default encoding for other types
            yield from super().iterencode(obj, _one_shot)


def save_json(obj: Any, path: str, preserve_precision: bool = True):
    """
    Save object to JSON file with 64-bit precision preservation.
    
    JSON standard supports IEEE 754 double precision (64-bit), and Python's
    json module preserves this precision. This function ensures numpy float64
    arrays are properly converted to Python lists before serialization.
    
    Parameters:
    -----------
    obj : Any
        Object to save (will be JSON-serialized)
    path : str
        Output file path
    preserve_precision : bool
        If True, ensure all numeric values maintain 64-bit precision (default: True)
        This recursively converts numpy arrays to lists, ensuring float64 values
        are preserved as Python float (which is 64-bit)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if preserve_precision:
            # Convert numpy arrays to lists, ensuring float64 precision is maintained
            obj_serialized = _convert_to_json_safe(obj)
            # Use custom encoder with manual formatting to preserve precision
            # We'll serialize with custom float handling
            json_str = _serialize_with_precision(obj_serialized, indent=2)
            f.write(json_str)
        else:
            json.dump(obj, f, ensure_ascii=False, indent=2)


def _serialize_with_precision(obj: Any, indent: int = 2, current_indent: int = 0) -> str:
    """
    Serialize object to JSON string with exactly 64 decimal places for floats.
    
    This function manually formats JSON to ensure all floats are displayed with
    exactly 64 digits after the decimal point, as requested for full precision.
    
    Parameters:
    -----------
    obj : Any
        Object to serialize
    indent : int
        Indentation level (spaces per level)
    current_indent : int
        Current indentation level
    
    Returns:
    --------
    str
        JSON string with 64 decimal places for all floats
    """
    indent_str = ' ' * current_indent
    next_indent_str = ' ' * (current_indent + indent)
    
    if isinstance(obj, float):
        # Format with exactly 64 decimal places as requested by user
        # Always show 64 digits after decimal point
        if obj == 0.0:
            return "0." + "0" * 64
        else:
            return f"{obj:.64f}"
    elif isinstance(obj, np.floating):
        # Convert numpy float to Python float, then format with exactly 64 decimal places
        val = float(obj)
        if val == 0.0:
            return "0." + "0" * 64
        else:
            return f"{val:.64f}"
    elif isinstance(obj, np.integer):
        return str(int(obj))
    elif isinstance(obj, bool):
        return 'true' if obj else 'false'
    elif obj is None:
        return 'null'
    elif isinstance(obj, str):
        return json.dumps(obj, ensure_ascii=False)
    elif isinstance(obj, int):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        # Convert numpy array to list and recursively serialize
        # Ensure float64 arrays are properly converted with full precision
        if obj.dtype == np.float64:
            # For 1D arrays, convert each element to float (preserves precision)
            if obj.ndim == 1:
                arr_list = [float(x) for x in obj]
            else:
                # For multi-dimensional arrays, recursively convert each element
                # This ensures each float uses repr() for full precision
                arr_list = obj.tolist()  # Convert to nested lists
                # Then recursively process to ensure all floats use repr()
                arr_list = _convert_nested_list_to_precise(arr_list)
        else:
            # Convert to float64 first, then process
            arr_float64 = np.asarray(obj, dtype=np.float64)
            if arr_float64.ndim == 1:
                arr_list = [float(x) for x in arr_float64]
            else:
                arr_list = arr_float64.tolist()
                arr_list = _convert_nested_list_to_precise(arr_list)
        return _serialize_with_precision(arr_list, indent, current_indent)
    elif isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return '[]'
        items = []
        for item in obj:
            item_str = _serialize_with_precision(item, indent, current_indent + indent)
            items.append(f'\n{next_indent_str}{item_str}')
        return '[' + ','.join(items) + f'\n{indent_str}]'
    elif isinstance(obj, dict):
        if len(obj) == 0:
            return '{}'
        items = []
        for k, v in obj.items():
            # Ensure key is properly serialized
            if isinstance(k, str):
                key_str = json.dumps(k, ensure_ascii=False)
            else:
                key_str = json.dumps(str(k), ensure_ascii=False)
            value_str = _serialize_with_precision(v, indent, current_indent + indent)
            items.append(f'\n{next_indent_str}{key_str}: {value_str}')
        return '{' + ','.join(items) + f'\n{indent_str}}}'
    else:
        # For unknown types, try to convert to JSON-compatible format
        try:
            return json.dumps(obj, ensure_ascii=False)
        except (TypeError, ValueError):
            # If that fails, convert to string
            return json.dumps(str(obj), ensure_ascii=False)


def _convert_nested_list_to_precise(obj: Any) -> Any:
    """
    Recursively convert nested lists to ensure all floats maintain precision.
    
    This function ensures that when we convert numpy arrays to lists,
    all float values are properly handled to preserve 64-bit precision.
    
    Parameters:
    -----------
    obj : Any
        Object (list, float, or other) to convert
    
    Returns:
    --------
    Any
        Object with floats preserved as Python float (64-bit)
    """
    if isinstance(obj, list):
        return [_convert_nested_list_to_precise(item) for item in obj]
    elif isinstance(obj, (float, np.floating)):
        # Convert to Python float (64-bit) to preserve precision
        # Explicitly convert through numpy float64 to ensure full precision
        return float(np.float64(obj))
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    else:
        return obj


def _convert_to_json_safe(obj: Any) -> Any:
    """
    Recursively convert numpy arrays to JSON-safe types while preserving 64-bit precision.
    
    Python's float type is already 64-bit (IEEE 754 double), so converting
    numpy float64 to Python float preserves precision. JSON standard also
    supports 64-bit floating-point numbers.
    
    Parameters:
    -----------
    obj : Any
        Object to convert
    
    Returns:
    --------
    Any
        JSON-safe object with 64-bit precision preserved
    """
    if isinstance(obj, np.ndarray):
        # Convert numpy array to list
        # If dtype is float64, values are converted to Python float (64-bit)
        # This preserves full precision
        if obj.dtype == np.float64:
            # For multi-dimensional arrays, convert to nested lists
            # Then recursively process to ensure all floats are Python float (64-bit)
            arr_list = obj.tolist()
            return _convert_nested_list_to_precise(arr_list)
        else:
            # For other dtypes, convert to float64 first, then process
            arr_float64 = np.asarray(obj, dtype=np.float64)
            arr_list = arr_float64.tolist()
            return _convert_nested_list_to_precise(arr_list)
    elif isinstance(obj, np.floating):
        # Convert numpy scalar to Python float (64-bit)
        return float(obj)
    elif isinstance(obj, np.integer):
        # Convert numpy integer to Python int
        return int(obj)
    elif isinstance(obj, dict):
        # Recursively convert dictionaries
        return {k: _convert_to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Recursively convert lists and tuples
        return [_convert_to_json_safe(item) for item in obj]
    else:
        # Return as-is for other types (strings, None, etc.)
        return obj

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
        # Ensure float64 precision when loading from JSON
        arr = np.array(v, dtype=np.float64)
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


def convert_channels_if_needed(
    tasks: Dict[str, np.ndarray],
    expected_n_channels=None,
    logger=None,
) -> Dict[str, np.ndarray]:
    """
    Convert channel dimensions if user specified expected channels and data differs.
    
    This function checks if the user specified an expected number of channels,
    and if the actual data has a different number, performs conversion.
    Currently supports: 63→64 conversion (using convert_63_to_64_channels).
    
    Parameters:
    -----------
    tasks : Dict[str, np.ndarray]
        Dictionary mapping task_name -> data (n_times, n_channels)
    expected_n_channels : Optional[int]
        Expected number of channels. If None, no conversion is performed.
        If specified and differs from data, conversion will be attempted.
    logger : Optional
        Logger instance for logging conversion messages
    
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary with potentially converted data (same format: n_times, n_channels)
    """
    if expected_n_channels is None:
        return tasks
    
    # Get actual channel count from first task
    if not tasks:
        return tasks
    
    sample_task = next(iter(tasks.values()))
    actual_n_channels = sample_task.shape[1]  # n_channels (second dimension after transpose)
    
    if actual_n_channels == expected_n_channels:
        # Already correct, no conversion needed
        if logger:
            logger.debug(f"Channel count matches expected: {expected_n_channels}")
        return tasks
    
    # Need conversion
    if logger:
        logger.info(f"Converting channels: {actual_n_channels} → {expected_n_channels}")
    
    # Currently only support 63→64 conversion
    if actual_n_channels == 63 and expected_n_channels == 64:
        from eegspec.data_conversion import convert_63_to_64_channels
        
        converted_tasks = {}
        for task_name, data in tasks.items():
            # Data is (n_times, n_channels), need to transpose to (n_channels, n_times) for conversion
            data_channels_first = data.T  # (n_channels, n_times)
            # Convert 63→64
            data_converted = convert_63_to_64_channels(data_channels_first)  # (64, n_times)
            # Transpose back to (n_times, n_channels)
            converted_tasks[task_name] = data_converted.T  # (n_times, 64)
        
        if logger:
            logger.info(f"Successfully converted {len(converted_tasks)} task(s) from 63 to 64 channels")
        return converted_tasks
    else:
        # Unsupported conversion
        if logger:
            logger.warning(
                f"Unsupported channel conversion: {actual_n_channels} → {expected_n_channels}. "
                f"Only 63→64 conversion is currently supported. "
                f"Data will be used as-is."
            )
        return tasks

if __name__ == "__main__":
    print(parse_locs_file("./data/montages/caps63.locs"))
