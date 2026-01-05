"""
Design Creativity Analysis Module

This module implements the complete pipeline described in the paper:
"EEG FUNCTIONAL CONNECTIVITY REVEALS NEURAL MECHANISMS OF DESIGN CREATIVITY 
IN ENGINEERING STUDENTS"

The pipeline includes:
1. wPLI (weighted Phase Lag Index) connectivity computation
2. Graph theory feature extraction (Strength and Betweenness)
3. Classification using SVM, MLP, and KNN

Four cognitive states are analyzed:
- IDG: Idea Generation
- IDE: Idea Evolution
- IDR: Idea Rating
- RST: Rest
"""

import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional, Any
from eegspec.base import BaseApp
from eegspec.utils import load_subject_tasks_json, list_subject_jsons, subject_id_from_path, resolve_channels, save_json
from eegspec.connectivity import connectivity_analysis, compute_wpli, extract_graph_features
from eegspec.classification import train_classifiers, get_classification_summary, print_classification_report


def _ensure_float64_in_dict(obj: Any) -> Any:
    """
    Recursively convert numeric values in a dictionary/list to maintain 64-bit precision.
    
    When loading from JSON, Python's json.load() returns numbers as Python float
    (which is 64-bit), but we need to ensure they're explicitly handled as float64
    for consistency with our calculations.
    
    Parameters:
    -----------
    obj : Any
        Object (dict, list, or scalar) to convert
    
    Returns:
    --------
    Any
        Object with numeric arrays converted to maintain 64-bit precision
    """
    if isinstance(obj, dict):
        return {k: _ensure_float64_in_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Check if it's a list of numbers (array-like)
        if len(obj) > 0:
            # Try to convert to numpy array with float64 to check if numeric
            try:
                arr = np.array(obj, dtype=np.float64)
                # If conversion successful and it's numeric, return as list
                # Python float is 64-bit, so this preserves precision
                if np.issubdtype(arr.dtype, np.number):
                    return arr.tolist()  # Returns list of Python floats (64-bit)
                else:
                    # Recursively process nested lists
                    return [_ensure_float64_in_dict(item) for item in obj]
            except (ValueError, TypeError):
                # Not a numeric array, recursively process
                return [_ensure_float64_in_dict(item) for item in obj]
        else:
            return obj
    elif isinstance(obj, (int, float)):
        # Python float is already 64-bit, int can be arbitrary precision
        return float(obj) if isinstance(obj, (int, float)) else obj
    else:
        return obj


def compute_design_creativity_features(
    data: np.ndarray,
    sfreq: float = 500.0,
    freq_range: Tuple[float, float] = (8.0, 13.0),
    threshold: float = 0.2,
) -> Dict[str, np.ndarray]:
    """
    Compute design creativity features from EEG data.
    
    This function matches MATLAB Connectivity_Analysis.m exactly:
    - fs = 500 Hz (default)
    - freq_range = [8, 13] (alpha band, default)
    - threshold = 0.2 (default)
    - Uses full data length (L=round(n/1)=n)
    - Strength from original wPLI matrix
    - Betweenness from thresholded wPLI matrix
    
    Parameters:
    -----------
    data : np.ndarray
        EEG data of shape (n_channels, n_times) - MATLAB format
        or (n_times, n_channels) - will be auto-transposed
    sfreq : float
        Sampling frequency (Hz), default: 500.0
    freq_range : Tuple[float, float]
        Frequency range for filtering [low, high] (Hz), default: (8.0, 13.0)
    threshold : float
        Threshold for filtering weak connections, default: 0.2
    
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'wpli_matrix': Full wPLI connectivity matrix (n_channels, n_channels)
        - 'thresholded_wpli': Thresholded wPLI matrix (n_channels, n_channels)
        - 'strength': Strength features (row vector: 1, n_channels)
        - 'betweenness': Betweenness features (row vector: 1, n_channels)
        - 'features': Combined features [betweenness, strength] (row vector: 1, 2*n_channels)
    """
    # Use connectivity_analysis which matches MATLAB exactly
    result = connectivity_analysis(
        eeg_signal=data,
        fs=sfreq,
        freq_range=freq_range,
        threshold=threshold,
    )
    
    return result


def run_design_creativity_analysis(
    subject_id: str,
    task_name: str,
    data_txc: np.ndarray,
    sfreq: float = 500.0,
    ch_names: Optional[List[str]] = None,
    freq_range: Tuple[float, float] = (8.0, 13.0),
    threshold: float = 0.2,
    out_dir: str = ".",
    log_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run complete design creativity analysis for a single task.
    
    This function matches MATLAB Connectivity_Analysis.m logic exactly.
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier
    task_name : str
        Task name (e.g., 'IDG_1', 'IDE_2', 'IDR_3', 'RST1')
    data_txc : np.ndarray
        EEG data of shape (n_times, n_channels) or (n_channels, n_times)
    sfreq : float
        Sampling frequency (Hz), default: 500.0
    ch_names : Optional[List[str]]
        Channel names (optional)
    freq_range : Tuple[float, float]
        Frequency range for filtering [low, high] (Hz), default: (8.0, 13.0)
    threshold : float
        Threshold for filtering weak connections, default: 0.2
    out_dir : str
        Output directory
    log_kwargs : Optional[Dict[str, Any]]
        Logging parameters
    
    Returns:
    --------
    Dict[str, Any]
        Analysis results dictionary
    """
    if log_kwargs is None:
        log_kwargs = {}
    app = BaseApp(**log_kwargs)
    
    try:
        app.logger.info(f"[Design Creativity] subject={subject_id} task={task_name} shape={data_txc.shape}")
        
        # Ensure data is in (n_channels, n_times) format for connectivity_analysis
        # data_txc comes as (n_times, n_channels) from load_subject_tasks_json
        # connectivity_analysis expects (n_channels, n_times) - MATLAB format
        # Ensure 64-bit precision (matching MATLAB double)
        data_txc = np.asarray(data_txc, dtype=np.float64)
        if data_txc.shape[0] > data_txc.shape[1] and data_txc.shape[0] > 200:
            # Likely (n_times, n_channels), transpose to (n_channels, n_times)
            data_for_connectivity = data_txc.T
        else:
            # Already (n_channels, n_times)
            data_for_connectivity = data_txc
        
        app.logger.debug(f"Data shape for connectivity: {data_for_connectivity.shape} (should be n_channels × n_times)")
        
        # Compute features using MATLAB-exact logic
        features = compute_design_creativity_features(
            data=data_for_connectivity,
            sfreq=sfreq,
            freq_range=freq_range,
            threshold=threshold,
        )
        
        # Save results
        subj_dir = os.path.join(out_dir, "subjects", subject_id)
        os.makedirs(subj_dir, exist_ok=True)
        
        output_path = os.path.join(subj_dir, f"design_creativity_{task_name}.json")
        
        # Ensure all arrays are float64 before converting to list
        # This preserves 64-bit precision when saving to JSON
        output_data = {
            "subject": subject_id,
            "task": task_name,
            "wpli_matrix": np.asarray(features['wpli_matrix'], dtype=np.float64).tolist(),
            "thresholded_wpli": np.asarray(features['thresholded_wpli'], dtype=np.float64).tolist(),
            "strength": np.asarray(features['strength'], dtype=np.float64).tolist(),
            "betweenness": np.asarray(features['betweenness'], dtype=np.float64).tolist(),
            "features": np.asarray(features['features'], dtype=np.float64).tolist(),  # Combined [betweenness, strength]
            "channels": ch_names if ch_names else [],
            "sfreq": float(sfreq),
            "freq_range": [float(freq_range[0]), float(freq_range[1])],
            "threshold": float(threshold),
        }
        
        # Save with precision preservation
        save_json(output_data, output_path, preserve_precision=True)
        
        app.logger.info(f"[Design Creativity done] subject={subject_id} task={task_name} -> {output_path}")
        
        return {
            "ok": True,
            "subject": subject_id,
            "task": task_name,
            "output_path": output_path,
            "features": features,
        }
    except Exception as e:
        app.logger.error(f"[Design Creativity error] subject={subject_id} task={task_name}: {e}")
        import traceback
        app.logger.debug(traceback.format_exc())
        return {
            "ok": False,
            "subject": subject_id,
            "task": task_name,
            "error": str(e),
        }


def prepare_classification_data(
    summary_dir: str,
    task_mapping: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare feature matrix and labels for classification.
    
    This function loads all design creativity features and prepares them
    for classification across the four cognitive states.
    
    Parameters:
    -----------
    summary_dir : str
        Directory containing subject results
    task_mapping : Optional[Dict[str, str]]
        Mapping from task names to class labels (e.g., {'1_idea generation': 'IDG'})
        If None, uses heuristics to map common task names
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, List[str]]
        (X, y, feature_names) where:
        - X: Feature matrix (n_samples, n_features)
        - y: Labels (n_samples,)
        - feature_names: List of feature names
    """
    if task_mapping is None:
        # Default mapping based on paper
        task_mapping = {
            'idea generation': 'IDG',
            'idea evolution': 'IDE',
            'idea rating': 'IDR',
            'rest': 'RST',
        }
    
    subjects_dir = os.path.join(summary_dir, "subjects")
    if not os.path.exists(subjects_dir):
        raise FileNotFoundError(f"Subjects directory not found: {subjects_dir}")
    
    X_list = []
    y_list = []
    
    # Collect all features
    for subject_id in os.listdir(subjects_dir):
        subj_dir = os.path.join(subjects_dir, subject_id)
        if not os.path.isdir(subj_dir):
            continue
        
        for filename in os.listdir(subj_dir):
            if not filename.startswith("design_creativity_") or not filename.endswith(".json"):
                continue
            
            filepath = os.path.join(subj_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Ensure all numeric values are loaded with 64-bit precision
                data = _ensure_float64_in_dict(data)
                
                task_name = data.get("task", "")
                # Map task name to class label
                label = None
                task_lower = task_name.lower()
                for key, val in task_mapping.items():
                    if key.lower() in task_lower:
                        label = val
                        break
                
                if label is None:
                    continue
                
                # Extract features (strength + betweenness)
                # These are stored as row vectors (1, n_channels) from connectivity_analysis
                # Ensure 64-bit precision (matching MATLAB double)
                strength = np.array(data["strength"], dtype=np.float64).flatten()  # Flatten to 1D
                betweenness = np.array(data["betweenness"], dtype=np.float64).flatten()  # Flatten to 1D
                
                # Combine features: [betweenness, strength] matching MATLAB Features = [betweenness strength_values]
                features = np.concatenate([betweenness, strength])
                # Ensure float64
                features = np.asarray(features, dtype=np.float64)
                
                X_list.append(features)
                y_list.append(label)
                
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")
                continue
    
    if len(X_list) == 0:
        raise ValueError("No valid feature files found")
    
    # Ensure 64-bit precision (matching MATLAB double)
    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list)
    
    # Feature names
    # Features are [betweenness, strength] matching MATLAB Features = [betweenness strength_values]
    # So order is: betweenness_0...betweenness_n, strength_0...strength_n
    if len(X_list) > 0:
        n_channels = len(strength) if hasattr(strength, '__len__') else X.shape[1] // 2
        feature_names = (
            [f"betweenness_{i}" for i in range(n_channels)] +
            [f"strength_{i}" for i in range(n_channels)]
        )
    else:
        feature_names = []
    
    return X, y, feature_names


def design_creativity_entry(
    input_path: str,
    sfreq: float = 500.0,
    out_dir: str = ".",
    channels_file: Optional[str] = None,
    n_channels: Optional[int] = None,
    freq_range: Tuple[float, float] = (8.0, 13.0),
    threshold: float = 0.2,
    max_processors: int = 4,
    run_classification: bool = True,
    log_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main entry point for design creativity analysis pipeline.
    
    This function implements the complete pipeline matching MATLAB Connectivity_Analysis.m:
    1. Compute wPLI connectivity for each task (fs=500, freq_range=[8,13], threshold=0.2)
    2. Extract Strength and Betweenness features
    3. Train and evaluate classifiers (SVM, MLP, KNN)
    
    Supports Creativity_EEG_Dataset directory structure:
    - Input: E:\Creativity_EEG_Dataset\Data_Creativity_Sub_*.mat
    - Variables: Creativity_{subject_id}_{trial}_{state} or Creativity_{subject_id}_{RST}
    
    Parameters:
    -----------
    input_path : str
        Path to folder containing Data_Creativity_Sub_*.mat files or single .mat file
        or folder of subject JSONs or single subject JSON
    sfreq : float
        Sampling frequency (Hz), default: 500.0
    out_dir : str
        Output directory, default: "."
    channels_file : Optional[str]
        Path to channels file (.locs, .txt, .csv)
    n_channels : Optional[int]
        Expected number of channels. If specified and differs from data, conversion will be performed (e.g., 63→64)
    freq_range : Tuple[float, float]
        Frequency range for filtering [low, high] (Hz), default: (8.0, 13.0)
    threshold : float
        Threshold for filtering weak connections, default: 0.2
    max_processors : int
        Maximum number of processors (default: 4)
    run_classification : bool
        Whether to run classification (default: True)
    log_kwargs : Optional[Dict[str, Any]]
        Logging parameters (log_level, log_dir, log_prefix, log_suffix, log_percentage)
    
    Returns:
    --------
    Dict[str, Any]
        Summary dictionary with analysis results
    """
    if log_kwargs is None:
        log_kwargs = {}
    app = BaseApp(**log_kwargs)
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Load subjects
    try:
        subjects = list_subject_jsons(input_path)
        if not subjects:
            raise FileNotFoundError("No JSON files found under input path")
        app.logger.info(f"Found {len(subjects)} subject file(s)")
    except Exception as e:
        app.logger.error(f"Failed to enumerate input: {e}")
        raise
    
    # Schedule tasks
    schedule = []
    for spath in subjects:
        sid = subject_id_from_path(spath)
        try:
            # Step 1: Load data (disable auto-convert here, we'll handle it based on user's n_channels)
            tasks = load_subject_tasks_json(spath, auto_convert_63_to_64=False)
            
            # Step 2: Convert channels if user specified expected dimension
            # This happens BEFORE resolve_channels to ensure channel names match converted data
            if n_channels is not None:
                from eegspec.utils import convert_channels_if_needed
                tasks = convert_channels_if_needed(tasks, expected_n_channels=n_channels, logger=app.logger)
            
            # Step 3: Get channel count AFTER conversion (if any)
            sample_task = next(iter(tasks.values()))
            n_ch = sample_task.shape[1]  # n_channels after potential conversion
            
            # Step 4: Resolve channel names based on FINAL channel count
            # This ensures channels match the converted data
            ch_names = resolve_channels(channels_file, n_channels=n_ch)
            if len(ch_names) != n_ch:
                app.logger.warning(f"Channel count mismatch for {sid}: expected {n_ch}, got {len(ch_names)}. Using placeholders Ch1..Ch{n_ch}")
                ch_names = [f"Ch{i+1}" for i in range(n_ch)]
            
            # Step 4: Schedule all tasks for this subject
            for tname, data in tasks.items():
                schedule.append((sid, tname, data, ch_names))
        except Exception as e:
            app.logger.error(f"Failed to parse subject {sid}: {e}")
    
    summary = {
        "subjects": {},
        "sfreq": sfreq,
        "freq_range": list(freq_range),
        "threshold": threshold,
    }
    
    total = len(schedule)
    app.logger.info(f"Total tasks to run: {total} (max_processors={max_processors})")
    
    if total == 0:
        save_json(summary, os.path.join(out_dir, "summary.json"))
        return summary
    
    # Process tasks
    import concurrent.futures
    idx = 0
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_processors) as ex:
        while idx < total and len(futures) < max_processors:
            sid, tname, data, ch = schedule[idx]
            idx += 1
            task_log_kwargs = dict(log_kwargs)
            task_log_kwargs["log_suffix"] = f"_{sid}_{tname}"
            fut = ex.submit(
                run_design_creativity_analysis,
                sid, tname, data, sfreq, ch,
                freq_range, threshold,
                out_dir, task_log_kwargs
            )
            futures.append(fut)
        
        done_count = 0
        while futures:
            for fut in concurrent.futures.as_completed(futures, timeout=None):
                try:
                    res = fut.result()
                except Exception as e:
                    app.logger.error(f"Worker crashed: {e}")
                    res = {"ok": False, "error": str(e)}
                
                if res.get("ok"):
                    sid = res["subject"]
                    tname = res["task"]
                    summary.setdefault("subjects", {}).setdefault(sid, {})[tname] = {
                        "output_path": res["output_path"]
                    }
                else:
                    app.logger.error(f"Task failed: {res}")
                
                done_count += 1
                futures.remove(fut)
                if idx < total:
                    sid, tname, data, ch = schedule[idx]
                    idx += 1
                    task_log_kwargs = dict(log_kwargs)
                    task_log_kwargs["log_suffix"] = f"_{sid}_{tname}"
                    futures.append(
                        ex.submit(
                            run_design_creativity_analysis,
                            sid, tname, data, sfreq, ch,
                            freq_range, threshold,
                            out_dir, task_log_kwargs
                        )
                    )
                app.logger.info(f"Progress: {done_count}/{total}")
                break
    
    # Run classification if requested
    if run_classification:
        try:
            app.logger.info("Preparing classification data...")
            X, y, feature_names = prepare_classification_data(out_dir)
            app.logger.info(f"Classification data: {X.shape}, labels: {np.unique(y, return_counts=True)}")
            
            app.logger.info("Training classifiers...")
            results = train_classifiers(X, y)
            
            # Save classification results
            classification_path = os.path.join(out_dir, "classification_results.json")
            classification_summary = get_classification_summary(results)
            save_json(classification_summary, classification_path)
            
            # Print report
            target_names = sorted(np.unique(y).tolist())
            print_classification_report(results, target_names=target_names)
            
            summary["classification"] = {
                "results_path": classification_path,
                "summary": classification_summary,
            }
            
            app.logger.info(f"Classification results saved to {classification_path}")
        except Exception as e:
            app.logger.warning(f"Classification failed: {e}")
            import traceback
            app.logger.debug(traceback.format_exc())
    
    # Save summary
    summ_path = os.path.join(out_dir, "summary.json")
    save_json(summary, summ_path)
    app.logger.info(f"Summary written: {summ_path}")
    
    return summary

