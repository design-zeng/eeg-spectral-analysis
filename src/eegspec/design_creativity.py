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
from eegspec.connectivity import compute_wpli, extract_graph_features
from eegspec.classification import train_classifiers, get_classification_summary, print_classification_report


def compute_design_creativity_features(
    data: np.ndarray,
    sfreq: float,
    fmin: float = 1.0,
    fmax: float = 45.0,
    epoch_sec: float = 2.0,
    overlap: float = 0.5,
    threshold: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute design creativity features from EEG data.
    
    This function implements the feature extraction pipeline from the paper:
    1. Compute wPLI connectivity matrix
    2. Extract Strength and Betweenness graph features
    
    Parameters:
    -----------
    data : np.ndarray
        EEG data of shape (n_times, n_channels)
    sfreq : float
        Sampling frequency
    fmin : float
        Minimum frequency for connectivity analysis (default: 1.0 Hz)
    fmax : float
        Maximum frequency for connectivity analysis (default: 45.0 Hz)
    epoch_sec : float
        Epoch length in seconds (default: 2.0)
    overlap : float
        Overlap fraction between epochs (default: 0.5)
    threshold : Optional[float]
        Threshold for binarizing connectivity matrix in betweenness computation
    
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'wpli': wPLI connectivity matrix (n_channels, n_channels)
        - 'strength': Strength features (n_channels,)
        - 'betweenness': Betweenness features (n_channels,)
    """
    # Step 1: Compute wPLI connectivity matrix
    wpli_matrix = compute_wpli(
        data=data,
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        epoch_sec=epoch_sec,
        overlap=overlap,
    )
    
    # Step 2: Extract graph theory features
    graph_features = extract_graph_features(
        connectivity_matrix=wpli_matrix,
        threshold=threshold,
    )
    
    return {
        'wpli': wpli_matrix,
        'strength': graph_features['strength'],
        'betweenness': graph_features['betweenness'],
    }


def run_design_creativity_analysis(
    subject_id: str,
    task_name: str,
    data_txc: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    fmin: float = 1.0,
    fmax: float = 45.0,
    epoch_sec: float = 2.0,
    overlap: float = 0.5,
    threshold: Optional[float] = None,
    out_dir: str = ".",
    log_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run complete design creativity analysis for a single task.
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier
    task_name : str
        Task name (e.g., 'IDG', 'IDE', 'IDR', 'RST')
    data_txc : np.ndarray
        EEG data of shape (n_times, n_channels)
    sfreq : float
        Sampling frequency
    ch_names : List[str]
        Channel names
    fmin : float
        Minimum frequency (default: 1.0 Hz)
    fmax : float
        Maximum frequency (default: 45.0 Hz)
    epoch_sec : float
        Epoch length in seconds (default: 2.0)
    overlap : float
        Overlap fraction (default: 0.5)
    threshold : Optional[float]
        Threshold for betweenness computation
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
        
        # Compute features
        features = compute_design_creativity_features(
            data=data_txc,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            epoch_sec=epoch_sec,
            overlap=overlap,
            threshold=threshold,
        )
        
        # Save results
        subj_dir = os.path.join(out_dir, "subjects", subject_id)
        os.makedirs(subj_dir, exist_ok=True)
        
        output_path = os.path.join(subj_dir, f"design_creativity_{task_name}.json")
        
        output_data = {
            "subject": subject_id,
            "task": task_name,
            "wpli": features['wpli'].tolist(),
            "strength": features['strength'].tolist(),
            "betweenness": features['betweenness'].tolist(),
            "channels": ch_names,
            "fmin": fmin,
            "fmax": fmax,
            "epoch_sec": epoch_sec,
            "overlap": overlap,
        }
        
        save_json(output_data, output_path)
        
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
                strength = np.array(data["strength"])
                betweenness = np.array(data["betweenness"])
                
                # Combine features
                features = np.concatenate([strength, betweenness])
                
                X_list.append(features)
                y_list.append(label)
                
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")
                continue
    
    if len(X_list) == 0:
        raise ValueError("No valid feature files found")
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Feature names
    n_channels = len(strength)
    feature_names = (
        [f"strength_{i}" for i in range(n_channels)] +
        [f"betweenness_{i}" for i in range(n_channels)]
    )
    
    return X, y, feature_names


def design_creativity_entry(
    input_path: str,
    sfreq: float,
    out_dir: str,
    channels_file: Optional[str] = None,
    fmin: float = 1.0,
    fmax: float = 45.0,
    epoch_sec: float = 2.0,
    overlap: float = 0.5,
    threshold: Optional[float] = None,
    max_processors: int = 4,
    run_classification: bool = True,
    log_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main entry point for design creativity analysis pipeline.
    
    This function implements the complete pipeline from the paper:
    1. Compute wPLI connectivity for each task
    2. Extract Strength and Betweenness features
    3. Train and evaluate classifiers (SVM, MLP, KNN)
    
    Parameters:
    -----------
    input_path : str
        Path to folder of subject JSONs or single subject JSON
    sfreq : float
        Sampling frequency
    out_dir : str
        Output directory
    channels_file : Optional[str]
        Path to channels file
    fmin : float
        Minimum frequency (default: 1.0 Hz)
    fmax : float
        Maximum frequency (default: 45.0 Hz)
    epoch_sec : float
        Epoch length in seconds (default: 2.0)
    overlap : float
        Overlap fraction (default: 0.5)
    threshold : Optional[float]
        Threshold for betweenness computation
    max_processors : int
        Maximum number of processors (default: 4)
    run_classification : bool
        Whether to run classification (default: True)
    log_kwargs : Optional[Dict[str, Any]]
        Logging parameters
    
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
            tasks = load_subject_tasks_json(spath)
            sample_task = next(iter(tasks.values()))
            n_ch = sample_task.shape[1]
            ch_names = resolve_channels(channels_file, n_channels=n_ch)
            if len(ch_names) != n_ch:
                app.logger.warning(f"Channel count mismatch for {sid}: using placeholders Ch1..Ch{n_ch}")
            for tname, data in tasks.items():
                schedule.append((sid, tname, data, ch_names))
        except Exception as e:
            app.logger.error(f"Failed to parse subject {sid}: {e}")
    
    summary = {
        "subjects": {},
        "sfreq": sfreq,
        "fmin": fmin,
        "fmax": fmax,
        "epoch_sec": epoch_sec,
        "overlap": overlap,
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
                fmin, fmax, epoch_sec, overlap, threshold,
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
                            fmin, fmax, epoch_sec, overlap, threshold,
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

