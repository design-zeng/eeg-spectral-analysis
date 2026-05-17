"""
Backward-compatible re-exports for design protocol segmentation.

Implementation lives in ``microstate_analysis.protocol_segmentation`` (package
``microstate_analysis`` ≥ 0.4.0, same repo as the ``microstate-analysis`` CLI).
"""

from microstate_analysis.protocol_segmentation import (
    SegmentationConfig,
    default_config,
    line_a_config,
    line_b_config,
    run_pipeline_with_optional_eeg,
    run_segmentation_pipeline,
)

__all__ = [
    "SegmentationConfig",
    "default_config",
    "line_a_config",
    "line_b_config",
    "run_segmentation_pipeline",
    "run_pipeline_with_optional_eeg",
]
