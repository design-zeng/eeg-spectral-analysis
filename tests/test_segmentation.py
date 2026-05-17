import numpy as np

from eegspec.segmentation import SegmentationConfig, run_segmentation_pipeline


def test_pipeline_shapes_and_merge():
    rng = np.random.default_rng(0)
    T = 5000
    labels = rng.integers(0, 4, size=T, dtype=np.int64)
    cfg = SegmentationConfig(
        sfreq=500.0,
        group_sec=1.0,
        macro_window_size=2,
        macro_window_stride=2,
        n_clusters=3,
        min_state_duration_samples=5,
        smooth_window_w=0,
        merge_adjacent_same_label=True,
        min_subtask_sec=0.0,
        random_state=0,
    )
    res = run_segmentation_pipeline(labels, cfg)
    assert res.labels_raw.shape == (T,)
    assert res.group_features.shape[0] == res.group_edges.shape[0]
    assert res.macro_features.shape[0] == res.cluster_labels.shape[0]
    assert len(res.subtasks) >= 1
