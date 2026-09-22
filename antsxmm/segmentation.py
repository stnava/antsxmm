"""Segmentation, lesion extraction, tissue masking, and partitioning.

Top-level shim re-exporting from antsxmm.modalities.segmentation.
"""

from __future__ import annotations

from .modalities.segmentation import (
    augment_image,
    boot_wmh,
    crop_mcimage,
    enantiomorphic_filling_without_mask,
    map_scalar_to_labels,
    segment_timeseries_by_bvalue,
    segment_timeseries_by_meanvalue,
    trim_dti_mask,
    warn_if_small_mask,
    wmh,
)

__all__ = [
    "augment_image",
    "boot_wmh",
    "crop_mcimage",
    "enantiomorphic_filling_without_mask",
    "map_scalar_to_labels",
    "segment_timeseries_by_bvalue",
    "segment_timeseries_by_meanvalue",
    "trim_dti_mask",
    "warn_if_small_mask",
    "wmh",
]
