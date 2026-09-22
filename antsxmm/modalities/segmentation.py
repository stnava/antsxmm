"""Segmentation, lesion extraction, tissue masking, and timeseries partitioning algorithms.

This module unifies:
- Tissue and lesion segmentation (wmh, boot_wmh, enantiomorphic_filling_without_mask, augment_image).
- Masking, trimming, and spatial cropping (trim_dti_mask, crop_mcimage, warn_if_small_mask).
- Timeseries and diffusion volume segmentation (segment_timeseries_by_meanvalue, segment_timeseries_by_bvalue).
- Label mapping (map_scalar_to_labels).
"""

from __future__ import annotations

import math
import random
import warnings
from typing import Any

import ants
import numpy as np
import pandas as pd

from .metrics import mask_snr


def augment_image(
    x: ants.ANTsImage,
    max_rot: float = 10.0,
    nzsd: float = 1.0,
) -> tuple[ants.ANTsImage, Any, Any]:
    """Randomly rotate and add noise to an image."""
    r_rot = ants.contrib.RandomRotate3D((max_rot * -1.0, max_rot), reference=x)
    tx = r_rot.transform()
    itx = ants.invert_ants_transform(tx)
    y = ants.apply_ants_transform_to_image(tx, x, x, interpolation="linear")
    y = ants.add_noise_to_image(y, "additivegaussian", [0, nzsd])
    return y, tx, itx


def wmh(
    flair: ants.ANTsImage,
    t1: ants.ANTsImage,
    t1seg: ants.ANTsImage,
    mmfromconvexhull: float = 3.0,
    strict: bool = True,
    probability_mask: ants.ANTsImage | None = None,
    prior_probability: ants.ANTsImage | None = None,
    model: str = "sysu",
    verbose: bool = False,
) -> dict[str, Any]:
    """Outputs WMH probability mask and summary volumetric measurement."""
    import antspynet
    import antspyt1w

    t1_2_flair_reg = ants.registration(flair, t1, type_of_transform="antsRegistrationSyNRepro[r]")

    if probability_mask is None and model == "sysu":
        if verbose:
            print("sysu")
        probability_mask = antspynet.sysu_media_wmh_segmentation(flair)
    elif probability_mask is None and model == "hyper":
        if verbose:
            print("hyper")
        probability_mask = antspynet.hypermapp3r_segmentation(t1_2_flair_reg["warpedmovout"], flair)

    prior_probability_flair = None
    if prior_probability is not None:
        prior_probability_flair = ants.apply_transforms(
            flair, prior_probability, t1_2_flair_reg["fwdtransforms"]
        )

    wmseg_mask = ants.threshold_image(t1seg, low_thresh=3, high_thresh=3).iMath("FillHoles")
    wmseg_mask_use = ants.image_clone(wmseg_mask)
    distmask = None

    if mmfromconvexhull > 0:
        convexhull = ants.threshold_image(t1seg, 1, 4)
        myspc = ants.get_spacing(t1seg)
        voxdist = math.sqrt(sum(s * s for s in myspc[: t1seg.dimension]))
        nmorph = round(2.0 / voxdist)
        convexhull = ants.morphology(convexhull, "close", nmorph).iMath("FillHoles")
        dist = ants.iMath(convexhull, "MaurerDistance") * -1.0
        distmask = ants.threshold_image(dist, mmfromconvexhull, 1.0e80)
        wmseg_mask = wmseg_mask + distmask
        wmseg_mask_use = (
            ants.threshold_image(wmseg_mask, 2, 2)
            if strict
            else ants.threshold_image(wmseg_mask, 1, 2)
        )

    wmseg_2_flair = ants.apply_transforms(
        flair, wmseg_mask_use, transformlist=t1_2_flair_reg["fwdtransforms"], interpolator="nearestNeighbor"
    )
    seg_2_flair = ants.apply_transforms(
        flair, t1seg, transformlist=t1_2_flair_reg["fwdtransforms"], interpolator="nearestNeighbor"
    )
    csfmask = ants.threshold_image(seg_2_flair, 1, 1)
    flairsnr = mask_snr(flair, csfmask, wmseg_2_flair, bias_correct=False)
    probability_mask_wm = wmseg_2_flair * probability_mask
    wmh_sum = float(np.prod(ants.get_spacing(flair)) * probability_mask_wm.sum())

    wmh_sum_prior = math.nan
    probability_mask_posterior = None
    if prior_probability_flair is not None:
        probability_mask_posterior = prior_probability_flair * probability_mask
        wmh_sum_prior = float(np.prod(ants.get_spacing(flair)) * probability_mask_posterior.sum())

    if math.isnan(wmh_sum):
        wmh_sum = 0.0
    if math.isnan(wmh_sum_prior):
        wmh_sum_prior = 0.0

    flair_evr = antspyt1w.patch_eigenvalue_ratio(flair, 512, [16, 16, 16], evdepth=0.9, mask=wmseg_2_flair)

    return {
        "WMH_probability_map_raw": probability_mask,
        "WMH_probability_map": probability_mask_wm,
        "WMH_posterior_probability_map": probability_mask_posterior,
        "wmh_mass": wmh_sum,
        "wmh_mass_prior": wmh_sum_prior,
        "wmh_evr": flair_evr,
        "wmh_SNR": flairsnr,
        "convexhull_mask": distmask,
    }


def boot_wmh(
    flair: ants.ANTsImage,
    t1: ants.ANTsImage,
    t1seg: ants.ANTsImage,
    mmfromconvexhull: float = 0.0,
    strict: bool = True,
    probability_mask: ants.ANTsImage | None = None,
    prior_probability: ants.ANTsImage | None = None,
    n_simulations: int = 8,
    random_seed: int = 42,
    verbose: bool = False,
) -> dict[str, Any]:
    """Bootstrap WMH segmentation with image augmentation."""
    random.seed(random_seed)
    wmh_sum_aug = 0.0
    wmh_sum_prior_aug = 0.0
    augprob = flair * 0.0
    augprob_prior = flair * 0.0 if prior_probability is not None else None
    locwmh: dict[str, Any] = {}

    for n in range(n_simulations):
        augflair, _, itx = augment_image(ants.iMath(flair, "Normalize"), 5, 0.01)
        locwmh = wmh(
            augflair,
            t1,
            t1seg,
            mmfromconvexhull=mmfromconvexhull,
            strict=strict,
            probability_mask=None,
            prior_probability=prior_probability,
        )
        wmh_sum_aug += locwmh["wmh_mass"]
        wmh_sum_prior_aug += locwmh["wmh_mass_prior"]
        augprob = augprob + ants.apply_ants_transform_to_image(itx, locwmh["WMH_probability_map"], flair, interpolation="linear")
        if prior_probability is not None and augprob_prior is not None:
            augprob_prior = augprob_prior + ants.apply_ants_transform_to_image(
                itx, locwmh["WMH_posterior_probability_map"], flair, interpolation="linear"
            )

    augprob = augprob * (1.0 / float(n_simulations))
    if augprob_prior is not None:
        augprob_prior = augprob_prior * (1.0 / float(n_simulations))
    wmh_sum_aug = wmh_sum_aug / float(n_simulations)
    wmh_sum_prior_aug = wmh_sum_prior_aug / float(n_simulations)

    return {
        "flair": ants.iMath(flair, "Normalize"),
        "WMH_probability_map": augprob,
        "WMH_posterior_probability_map": augprob_prior,
        "wmh_mass": wmh_sum_aug,
        "wmh_mass_prior": wmh_sum_prior_aug,
        "wmh_evr": locwmh.get("wmh_evr", 0.0),
        "wmh_SNR": locwmh.get("wmh_SNR", 0.0),
    }


def enantiomorphic_filling_without_mask(
    image: ants.ANTsImage,
    axis: int = 0,
    intensity: str = "low",
) -> tuple[ants.ANTsImage, ants.ANTsImage]:
    """Perform enantiomorphic lesion filling on an image without a prior lesion mask."""
    imagen = ants.iMath(image, "Normalize")
    imagen = ants.iMath(imagen, "TruncateIntensity", 1e-6, 0.98)
    imagen = ants.iMath(imagen, "Normalize")

    mirror_image = ants.reflect_image(imagen, axis=axis, tx="antsRegistrationSyNQuickRepro[s]")["warpedmovout"]
    symmetric_image = imagen * 0.5 + mirror_image * 0.5
    difference_image = image - symmetric_image
    diffseg = ants.threshold_image(difference_image, "Otsu", 3)

    if intensity == "low":
        likely_lesion = ants.threshold_image(diffseg, 1, 1)
    else:
        likely_lesion = ants.threshold_image(diffseg, 3, 3)

    likely_lesion = ants.smooth_image(likely_lesion, 3.0).iMath("Normalize")
    lesionneg = (imagen * 0.0 + 1.0) - likely_lesion
    filled_image = imagen * lesionneg + mirror_image * likely_lesion
    return filled_image, diffseg


def trim_dti_mask(fa: ants.ANTsImage, mask: ants.ANTsImage, param: float = 4.0) -> ants.ANTsImage:
    """Trim DTI mask to remove bright FA rim."""
    spcmin = min(ants.get_spacing(mask))
    param_vox = int(np.round(param / spcmin))
    trim_mask = ants.image_clone(mask)
    trim_mask = ants.iMath(trim_mask, "FillHoles")
    edgemask = trim_mask - ants.iMath(trim_mask, "ME", param_vox)
    maxk = 4
    edgemask = ants.threshold_image(fa * edgemask, "Otsu", maxk)
    edgemask = ants.threshold_image(edgemask, maxk - 1, maxk)
    trim_mask[edgemask >= 1] = 0
    trim_mask = ants.iMath(trim_mask, "ME", param_vox - 1)
    trim_mask = ants.iMath(trim_mask, "GetLargestComponent")
    trim_mask = ants.iMath(trim_mask, "MD", param_vox - 1)
    return trim_mask


def crop_mcimage(x: ants.ANTsImage, mask: ants.ANTsImage, padder: int | None = None) -> ants.ANTsImage:
    """Crop a 4D or multi-component image using a 3D mask bounding box."""
    if padder is None:
        padder = 0
    x_cropped = ants.crop_image(ants.slice_image(x, axis=3, idx=0), mask, label_intensity=1)
    if padder > 0:
        x_cropped = ants.pad_image(x_cropped, pad_width=padder)
    out_img = ants.resample_image_to_target(x, x_cropped, verbose=False)
    return out_img


def warn_if_small_mask(
    mask: ants.ANTsImage,
    threshold_fraction: float = 0.05,
    label: str = " ",
) -> None:
    """Check if the mask foreground fraction is below threshold and issue a warning."""
    mask_np = mask.numpy()
    total_voxels = mask_np.size
    foreground_voxels = np.count_nonzero(mask_np > 0)
    foreground_fraction = foreground_voxels / float(total_voxels)
    if foreground_fraction < threshold_fraction:
        warnings.warn(
            f"Small mask detected for {label}! Fraction of non-zero voxels: "
            f"{foreground_fraction:.4f} (threshold: {threshold_fraction:.4f}).",
            UserWarning,
            stacklevel=2,
        )


def segment_timeseries_by_meanvalue(image: ants.ANTsImage, quantile: float = 0.995) -> dict[str, list[int]]:
    """Partition timeseries into upper and lower signal mean groups."""
    ishape = image.shape
    lastdim = len(ishape) - 1
    meanvalues = [float(ants.slice_image(image, axis=lastdim, idx=x).mean()) for x in range(ishape[lastdim])]
    myhiq = float(np.quantile(meanvalues, quantile))
    myloq = float(np.quantile(meanvalues, 1.0 - quantile))
    lowerindices: list[int] = []
    higherindices: list[int] = []
    for x in range(len(meanvalues)):
        hiabs = abs(meanvalues[x] - myhiq)
        loabs = abs(meanvalues[x] - myloq)
        if hiabs < loabs:
            higherindices.append(x)
        else:
            lowerindices.append(x)

    return {
        "lowermeans": lowerindices,
        "highermeans": higherindices,
        "high": higherindices,
        "low": lowerindices,
    }


def segment_timeseries_by_bvalue(bvals: np.ndarray | list[float]) -> dict[str, list[int]]:
    """Partition diffusion volumes into shells based on nominal b-values."""
    bvals_arr = np.asarray(bvals, dtype=float)
    threshold = 1e-12
    lowbvals = [int(i) for i in np.where(bvals_arr <= threshold)[0]]
    largerbvals = [int(i) for i in np.where(bvals_arr > threshold)[0]]
    if len(lowbvals) == 0:
        minval = float(np.min(bvals_arr))
        lowbvals = [int(i) for i in np.where(bvals_arr <= minval)[0]]
        largerbvals = [int(i) for i in np.where(bvals_arr > minval)[0]]

    shells: dict[str, list[int]] = {
        "largerbvals": largerbvals,
        "lowbvals": lowbvals,
    }
    unique_bvals = np.unique(np.round(bvals_arr, -2))
    for ub in unique_bvals:
        idx = np.where(np.abs(bvals_arr - ub) < 50.0)[0].tolist()
        shells[f"b{int(ub)}"] = idx
    return shells


def map_scalar_to_labels(dataframe: pd.DataFrame, label_image_template: ants.ANTsImage) -> ants.ANTsImage:
    """Map scalar values from a DataFrame to associated integer image labels."""
    mapped_label_image = label_image_template.clone() * 0.0
    for _, row in dataframe.iterrows():
        label = int(row["label"])
        scalar_value = float(row["scalar_value"])
        mapped_label_image[label_image_template == label] = scalar_value
    return mapped_label_image


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
