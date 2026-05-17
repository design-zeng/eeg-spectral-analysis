"""Microstate-driven dynamic windows for spectral analysis."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from eegspec.bands import DEFAULT_BANDS
from eegspec.faa import faa_from_psd
from eegspec.features import bandpower, median_frequency, spectral_edge, spectral_entropy, spectral_moments
from eegspec.iaf import estimate_iaf
from eegspec.psd import compute_psd_welch
from eegspec.utils import (
    convert_channels_if_needed,
    list_subject_jsons,
    load_subject_tasks_json,
    resolve_channels,
    save_json,
    subject_id_from_path,
)


def _replace_nonfinite(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _replace_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_replace_nonfinite(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _replace_nonfinite(obj.tolist())
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    return obj


def _ensure_microstate_analysis_available() -> None:
    """
    Prefer an installed microstate_analysis package, but support the sibling
    checkout layout used by this workspace without requiring manual PYTHONPATH.
    """
    try:
        import microstate_analysis  # noqa: F401

        return
    except ImportError:
        pass

    here = os.path.abspath(__file__)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(here), "..", ".."))
    sibling_src = os.path.abspath(os.path.join(repo_root, "..", "microstate-analysis", "src"))
    if os.path.isdir(sibling_src) and sibling_src not in sys.path:
        sys.path.insert(0, sibling_src)

    try:
        import microstate_analysis  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "microstate_analysis is required for dynamic-window. Install it with "
            "`python -m pip install -e D:\\Documents\\GitHub\\microstate-analysis`, "
            "or keep the sibling repo at D:\\Documents\\GitHub\\microstate-analysis."
        ) from exc


def _compute_labels_from_eeg(data_txc: np.ndarray, **kwargs: Any) -> Tuple[np.ndarray, Any]:
    _ensure_microstate_analysis_available()
    try:
        from microstate_analysis.protocol_segmentation.labels_from_eeg import compute_labels_from_eeg
    except ModuleNotFoundError as exc:
        missing = exc.name or "a dependency"
        raise ModuleNotFoundError(
            f"microstate_analysis was found, but dependency {missing!r} is missing. "
            "Install the sibling project into the current environment with "
            "`python -m pip install -e D:\\Documents\\GitHub\\microstate-analysis`."
        ) from exc

    return compute_labels_from_eeg(data_txc, **kwargs)


@dataclass
class DynamicWindow:
    start_sample: int
    end_sample: int
    source: str
    label: Optional[int] = None
    cluster_id: Optional[int] = None
    hardness_mean: Optional[float] = None

    def as_dict(self, sfreq: float) -> Dict[str, Any]:
        n_samples = self.end_sample - self.start_sample
        return {
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "start_sec": self.start_sample / sfreq,
            "end_sec": self.end_sample / sfreq,
            "duration_sec": n_samples / sfreq,
            "n_samples": n_samples,
            "source": self.source,
            "label": self.label,
            "cluster_id": self.cluster_id,
            "hardness_mean": self.hardness_mean,
        }


def _find_label_runs(labels: np.ndarray) -> List[DynamicWindow]:
    lab = np.asarray(labels, dtype=np.int64).ravel()
    if lab.size == 0:
        return []
    runs: List[DynamicWindow] = []
    start = 0
    for i in range(1, lab.size + 1):
        if i == lab.size or lab[i] != lab[start]:
            runs.append(DynamicWindow(start, i, source="microstate_run", label=int(lab[start])))
            start = i
    return runs


def _clip_filter_windows(
    windows: List[DynamicWindow],
    *,
    n_times: int,
    min_window_samples: int,
) -> Tuple[List[DynamicWindow], List[Dict[str, Any]]]:
    kept: List[DynamicWindow] = []
    skipped: List[Dict[str, Any]] = []
    for idx, w in enumerate(windows):
        start = max(0, min(n_times, int(w.start_sample)))
        end = max(0, min(n_times, int(w.end_sample)))
        n_samples = end - start
        if n_samples < min_window_samples:
            skipped.append(
                {
                    "index": idx,
                    "start_sample": start,
                    "end_sample": end,
                    "n_samples": n_samples,
                    "reason": "shorter_than_min_window_samples",
                }
            )
            continue
        kept.append(
            DynamicWindow(
                start,
                end,
                source=w.source,
                label=w.label,
                cluster_id=w.cluster_id,
                hardness_mean=w.hardness_mean,
            )
        )
    return kept, skipped


def build_run_windows(
    labels_raw: np.ndarray,
    *,
    sfreq: float,
    smooth_window_w: int,
    min_state_duration_ms: float,
    min_window_sec: float,
) -> Tuple[np.ndarray, List[DynamicWindow], Dict[str, Any]]:
    """Build windows from regularized microstate label runs (scheme A)."""
    _ensure_microstate_analysis_available()
    from microstate_analysis.protocol_segmentation.microstate_regularize import regularize_labels

    min_state_duration_samples = max(1, int(round(min_state_duration_ms * 1e-3 * sfreq)))
    labels_smooth, diag = regularize_labels(
        labels_raw,
        lambda_=0.0,
        smooth_window_w=max(0, smooth_window_w),
        min_state_duration_samples=min_state_duration_samples,
    )
    min_window_samples = max(1, int(round(min_window_sec * sfreq)))
    windows, skipped = _clip_filter_windows(
        _find_label_runs(labels_smooth),
        n_times=int(labels_smooth.size),
        min_window_samples=min_window_samples,
    )
    diag = {
        **diag,
        "mode": "run",
        "min_state_duration_samples": min_state_duration_samples,
        "min_window_samples": min_window_samples,
        "n_windows_before_filter": int(len(_find_label_runs(labels_smooth))),
        "n_windows": int(len(windows)),
        "n_skipped": int(len(skipped)),
        "skipped_windows": skipped,
    }
    return labels_smooth, windows, diag


def build_subtask_windows(
    labels_raw: np.ndarray,
    *,
    sfreq: float,
    group_sec: float,
    macro_window_size: Optional[int],
    macro_window_stride: Optional[int],
    n_clusters: int,
    min_subtask_sec: float,
    smooth_window_w: int,
    min_state_duration_ms: float,
    line: str,
    min_window_sec: float,
) -> Tuple[np.ndarray, List[DynamicWindow], Dict[str, Any]]:
    """Build windows from protocol-segmentation subtasks (scheme B)."""
    _ensure_microstate_analysis_available()
    from microstate_analysis.protocol_segmentation.runner import build_segmentation_config
    from microstate_analysis.protocol_segmentation.pipeline import run_segmentation_pipeline

    cfg = build_segmentation_config(
        sfreq=sfreq,
        line=line,
        group_sec=group_sec,
        macro_window_size=macro_window_size,
        macro_window_stride=macro_window_stride,
        n_clusters=n_clusters,
        min_subtask_sec=min_subtask_sec,
        smooth_window_w=smooth_window_w,
        min_state_duration_ms=min_state_duration_ms,
        lambda_reg=0.25,
        n_microstates=4,
        ms_n_runs=25,
    )
    result = run_segmentation_pipeline(labels_raw, cfg)
    raw_windows = [
        DynamicWindow(
            int(round(s.start_sec * sfreq)),
            int(round(s.end_sec * sfreq)),
            source="protocol_subtask",
            cluster_id=int(s.cluster_id),
            hardness_mean=float(s.hardness_mean),
        )
        for s in result.subtasks
    ]
    min_window_samples = max(1, int(round(min_window_sec * sfreq)))
    windows, skipped = _clip_filter_windows(
        raw_windows,
        n_times=int(np.asarray(labels_raw).size),
        min_window_samples=min_window_samples,
    )
    diag = {
        **result.diagnostics,
        "mode": "subtask",
        "line": line,
        "group_sec": group_sec,
        "macro_window_size": cfg.macro_window_size,
        "macro_window_stride": cfg.macro_window_stride,
        "n_clusters": cfg.n_clusters,
        "min_window_samples": min_window_samples,
        "n_windows_before_filter": int(len(raw_windows)),
        "n_windows": int(len(windows)),
        "n_skipped": int(len(skipped)),
        "skipped_windows": skipped,
    }
    return result.labels_smooth, windows, diag


def _window_metrics(
    data_txc: np.ndarray,
    *,
    sfreq: float,
    nperseg: int,
    noverlap: Optional[int],
    window: str,
    ch_names: List[str],
    alpha_band: Tuple[float, float],
    faa_db: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    freqs, psd = compute_psd_welch(data_txc, sfreq=sfreq, nperseg=nperseg, noverlap=noverlap, window=window)
    bp_abs = bandpower(psd, freqs, DEFAULT_BANDS, relative=False)
    bp_rel = bandpower(psd, freqs, DEFAULT_BANDS, relative=True, total_range=(1.0, 45.0))
    metrics = {
        "bands_abs": {k: v.tolist() for k, v in bp_abs.items()},
        "bands_rel": {k: v.tolist() for k, v in bp_rel.items()},
        "entropy": spectral_entropy(psd, freqs, fmin=1.0, fmax=45.0, log_base=np.e).tolist(),
        "moments": {k: v.tolist() for k, v in spectral_moments(psd, freqs, fmin=1.0, fmax=45.0).items()},
        "SEF95": spectral_edge(psd, freqs, percent=0.95, fmin=1.0, fmax=45.0).tolist(),
        "F50": median_frequency(psd, freqs, fmin=1.0, fmax=45.0).tolist(),
        "IAF": estimate_iaf(psd, freqs, fmin=alpha_band[0], fmax=alpha_band[1], smooth=True).tolist(),
        "FAA": faa_from_psd(psd, freqs, ch_names, left="F3", right="F4", alpha=alpha_band, use_db=faa_db),
    }
    return freqs, psd, metrics


def _load_dynamic_input(
    *,
    eeg_npy: Optional[str],
    input_path: Optional[str],
    task: Optional[str],
    channels_file: Optional[str],
    n_channels: Optional[int],
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    if eeg_npy:
        data = np.asarray(np.load(eeg_npy), dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("--eeg-npy must be shape (n_times, n_channels)")
        ch_names = resolve_channels(channels_file, n_channels=data.shape[1])
        return data, ch_names, {"input_kind": "eeg_npy", "eeg_npy": eeg_npy, "task": None, "subject": None}

    if not input_path:
        raise ValueError("Provide --eeg-npy or --input")
    subjects = list_subject_jsons(input_path)
    if not subjects:
        raise FileNotFoundError(f"No subject JSON/MAT files found under {input_path}")
    spath = subjects[0]
    tasks = load_subject_tasks_json(spath, auto_convert_63_to_64=False)
    if n_channels is not None:
        tasks = convert_channels_if_needed(tasks, expected_n_channels=n_channels, logger=None)
    selected_task = task or next(iter(tasks.keys()))
    if selected_task not in tasks:
        raise ValueError(f"Task {selected_task!r} not found. Available tasks: {sorted(tasks.keys())}")
    data = np.asarray(tasks[selected_task], dtype=np.float64)
    ch_names = resolve_channels(channels_file, n_channels=data.shape[1])
    return data, ch_names, {
        "input_kind": "task_file",
        "input": input_path,
        "subject": subject_id_from_path(spath),
        "task": selected_task,
    }


def dynamic_window_entry(
    *,
    out_dir: str,
    sfreq: float,
    labels_npy: Optional[str],
    mode: str,
    eeg_npy: Optional[str] = None,
    input_path: Optional[str] = None,
    task: Optional[str] = None,
    channels_file: Optional[str] = None,
    n_channels: Optional[int] = None,
    nperseg: int = 1024,
    noverlap: Optional[int] = None,
    window: str = "hann",
    alpha: str = "8,13",
    faa_db: bool = False,
    min_window_sec: float = 1.0,
    smooth_window_w: int = 3,
    min_state_duration_ms: float = 80.0,
    line: str = "B",
    group_sec: float = 5.0,
    macro_window_size: Optional[int] = None,
    macro_window_stride: Optional[int] = None,
    n_clusters: int = 4,
    min_subtask_sec: float = 10.0,
    n_microstates: int = 4,
    ms_n_runs: int = 25,
    ms_n_std: int = 3,
    ms_distance: int = 10,
    ms_peaks_only: bool = True,
    save_psd: bool = False,
) -> Dict[str, Any]:
    """Run scheme A/B dynamic-window spectral analysis and write JSON outputs."""
    if mode not in {"run", "subtask"}:
        raise ValueError("--mode must be 'run' or 'subtask'")
    os.makedirs(out_dir, exist_ok=True)

    data_txc, ch_names, input_meta = _load_dynamic_input(
        eeg_npy=eeg_npy,
        input_path=input_path,
        task=task,
        channels_file=channels_file,
        n_channels=n_channels,
    )
    computed_labels_path = None
    if labels_npy:
        labels_raw = np.asarray(np.load(labels_npy), dtype=np.int64).ravel()
    else:
        labels_raw, _ms = _compute_labels_from_eeg(
            data_txc,
            min_maps=max(2, n_microstates),
            max_maps=max(2, n_microstates),
            n_runs=max(1, ms_n_runs),
            n_std=max(1, ms_n_std),
            distance=max(1, ms_distance),
            peaks_only=ms_peaks_only,
        )
        computed_labels_path = os.path.join(out_dir, "computed_labels.npy")
        np.save(computed_labels_path, labels_raw)
    n = min(data_txc.shape[0], labels_raw.size)
    if n <= 0:
        raise ValueError("EEG and labels must contain at least one aligned sample")
    if data_txc.shape[0] != labels_raw.size:
        data_txc = data_txc[:n]
        labels_raw = labels_raw[:n]

    if mode == "run":
        labels_smooth, windows, diagnostics = build_run_windows(
            labels_raw,
            sfreq=sfreq,
            smooth_window_w=smooth_window_w,
            min_state_duration_ms=min_state_duration_ms,
            min_window_sec=min_window_sec,
        )
    else:
        labels_smooth, windows, diagnostics = build_subtask_windows(
            labels_raw,
            sfreq=sfreq,
            group_sec=group_sec,
            macro_window_size=macro_window_size,
            macro_window_stride=macro_window_stride,
            n_clusters=n_clusters,
            min_subtask_sec=min_subtask_sec,
            smooth_window_w=smooth_window_w,
            min_state_duration_ms=min_state_duration_ms,
            line=line,
            min_window_sec=min_window_sec,
        )

    alpha_band = tuple(map(float, alpha.split(",")))
    metrics_rows: List[Dict[str, Any]] = []
    psd_rows: List[Dict[str, Any]] = []
    for idx, w in enumerate(windows):
        seg = data_txc[w.start_sample : w.end_sample, :]
        freqs, psd, metrics = _window_metrics(
            seg,
            sfreq=sfreq,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
            ch_names=ch_names,
            alpha_band=alpha_band,
            faa_db=faa_db,
        )
        win_meta = {"window_index": idx, **w.as_dict(sfreq)}
        metrics_rows.append({**win_meta, "metrics": metrics})
        if save_psd:
            psd_rows.append({**win_meta, "freqs": freqs.tolist(), "psd": psd.tolist()})

    windows_payload = {
        "mode": mode,
        "sfreq": sfreq,
        "input": input_meta,
        "labels_npy": labels_npy,
        "computed_labels_npy": computed_labels_path,
        "n_times": int(n),
        "n_channels": int(data_txc.shape[1]),
        "channels": ch_names,
        "diagnostics": diagnostics,
        "windows": [{"window_index": idx, **w.as_dict(sfreq)} for idx, w in enumerate(windows)],
    }
    metrics_payload = {
        "mode": mode,
        "sfreq": sfreq,
        "alpha_band": list(alpha_band),
        "nperseg_requested": int(nperseg),
        "noverlap_requested": noverlap,
        "window": window,
        "windows_metrics": metrics_rows,
    }
    windows_path = os.path.join(out_dir, "dynamic_windows.json")
    metrics_path = os.path.join(out_dir, "dynamic_metrics.json")
    save_json(_replace_nonfinite(windows_payload), windows_path)
    save_json(_replace_nonfinite(metrics_payload), metrics_path)
    psd_path = None
    if save_psd:
        psd_path = os.path.join(out_dir, "dynamic_psd.json")
        save_json(_replace_nonfinite({"mode": mode, "windows_psd": psd_rows}), psd_path)

    summary = {
        "mode": mode,
        "out_dir": out_dir,
        "windows": windows_path,
        "metrics": metrics_path,
        "psd": psd_path,
        "computed_labels": computed_labels_path,
        "n_windows": int(len(windows)),
        "n_skipped": int(diagnostics.get("n_skipped", 0)),
    }
    save_json(_replace_nonfinite(summary), os.path.join(out_dir, "dynamic_summary.json"))
    return summary
