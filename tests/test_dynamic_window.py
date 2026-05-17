import json

import numpy as np

from eegspec.dynamic_window import dynamic_window_entry


def _write_fixture(tmp_path):
    sfreq = 100.0
    t = np.arange(0, 20, 1 / sfreq)
    data = np.column_stack(
        [
            np.sin(2 * np.pi * 10 * t),
            2.0 * np.sin(2 * np.pi * 10 * t),
        ]
    )
    labels = np.repeat([0, 1, 0, 2], [500, 500, 500, 500]).astype(np.int64)
    eeg_path = tmp_path / "eeg.npy"
    labels_path = tmp_path / "labels.npy"
    np.save(eeg_path, data)
    np.save(labels_path, labels)
    return sfreq, eeg_path, labels_path


def test_dynamic_window_run_mode(tmp_path):
    sfreq, eeg_path, labels_path = _write_fixture(tmp_path)
    out_dir = tmp_path / "run_out"
    summary = dynamic_window_entry(
        out_dir=str(out_dir),
        sfreq=sfreq,
        labels_npy=str(labels_path),
        eeg_npy=str(eeg_path),
        mode="run",
        nperseg=128,
        min_window_sec=1.0,
        smooth_window_w=0,
        min_state_duration_ms=10.0,
    )
    assert summary["n_windows"] == 4
    with open(out_dir / "dynamic_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert len(metrics["windows_metrics"]) == 4


def test_dynamic_window_subtask_mode(tmp_path):
    sfreq, eeg_path, labels_path = _write_fixture(tmp_path)
    out_dir = tmp_path / "subtask_out"
    summary = dynamic_window_entry(
        out_dir=str(out_dir),
        sfreq=sfreq,
        labels_npy=str(labels_path),
        eeg_npy=str(eeg_path),
        mode="subtask",
        nperseg=128,
        min_window_sec=1.0,
        smooth_window_w=0,
        min_state_duration_ms=10.0,
        group_sec=1.0,
        macro_window_size=2,
        macro_window_stride=2,
        n_clusters=2,
        min_subtask_sec=0.0,
        line="none",
    )
    assert summary["n_windows"] >= 1
    with open(out_dir / "dynamic_windows.json", "r", encoding="utf-8") as f:
        windows = json.load(f)
    assert windows["mode"] == "subtask"


def test_dynamic_window_can_compute_labels(tmp_path, monkeypatch):
    sfreq, eeg_path, _labels_path = _write_fixture(tmp_path)
    out_dir = tmp_path / "computed_labels_out"

    def fake_compute_labels_from_eeg(data_txc, **_kwargs):
        labels = np.repeat([0, 1, 0, 2], [500, 500, 500, 500]).astype(np.int64)
        return labels[: data_txc.shape[0]], object()

    import eegspec.dynamic_window as dynamic_window_mod

    monkeypatch.setattr(dynamic_window_mod, "_compute_labels_from_eeg", fake_compute_labels_from_eeg)
    summary = dynamic_window_entry(
        out_dir=str(out_dir),
        sfreq=sfreq,
        labels_npy=None,
        eeg_npy=str(eeg_path),
        mode="run",
        nperseg=128,
        min_window_sec=1.0,
        smooth_window_w=0,
        min_state_duration_ms=10.0,
        ms_n_runs=1,
    )
    assert summary["computed_labels"] is not None
    assert (out_dir / "computed_labels.npy").exists()
