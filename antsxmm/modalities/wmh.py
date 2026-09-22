from __future__ import annotations

import math
import random
from typing import Any
import numpy as np
import ants

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
