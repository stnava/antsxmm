"""Registration, motion correction, template construction, and spatial transforms.

Top-level shim re-exporting from antsxmm.modalities.registration.
"""

from __future__ import annotations

from .modalities.registration import (
    apply_transforms_mixed_interpolation,
    bvec_reorientation,
    concat_dewarp,
    deformation_gradient_optimized,
    dewarp_imageset,
    distortion_correct_bvecs,
    dti_reg,
    dti_template,
    generate_voxelwise_bvecs,
    get_average_dwi_b0,
    get_average_rsf,
    mc_reg,
    read_ants_transforms_to_numpy,
    timeseries_reg,
    timeseries_transform,
    tra_initializer,
    transform_and_reorient_dti,
)

__all__ = [
    "apply_transforms_mixed_interpolation",
    "bvec_reorientation",
    "concat_dewarp",
    "deformation_gradient_optimized",
    "dewarp_imageset",
    "distortion_correct_bvecs",
    "dti_reg",
    "dti_template",
    "generate_voxelwise_bvecs",
    "get_average_dwi_b0",
    "get_average_rsf",
    "mc_reg",
    "read_ants_transforms_to_numpy",
    "timeseries_reg",
    "timeseries_transform",
    "tra_initializer",
    "transform_and_reorient_dti",
]
