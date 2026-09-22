"""Positron Emission Tomography (PET) modality processing for ANTsXMM."""

from __future__ import annotations

from typing import Any
import ants

from .metrics import convert_np_in_dict
from .registration import register_images


def pet3d_summary(
    pet3d: ants.ANTsImage,
    t1head: ants.ANTsImage,
    t1: ants.ANTsImage,
    t1segmentation: ants.ANTsImage,
    t1dktcit: ants.ANTsImage,
    spa: tuple[float, float, float] = (0.0, 0.0, 0.0),
    type_of_transform: str = "antsRegistrationSyNRepro[r]",
    upsample: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Anatomical registration and DKT parcellation summary of 3D PET data."""
    import antspyt1w

    pet3dr = pet3d
    if upsample:
        spc = ants.get_spacing(pet3d)
        minspc = min(1.0, min(spc))
        pet3dr = ants.resample_image(pet3d, [minspc, minspc, minspc], interp_type=0)

    rig = register_images(pet3dr, t1head, type_of_transform)
    bmask = ants.apply_transforms(
        pet3dr, ants.threshold_image(t1segmentation, 1, 6), rig["fwdtransforms"][0], interpolator="genericLabel"
    )

    und = pet3dr * bmask
    t1reg = rig
    gmseg = (
        ants.threshold_image(t1segmentation, 2, 2) + ants.threshold_image(t1segmentation, 4, 4)
    ).threshold_image(1, 4).iMath("MD", 1)
    gmseg = ants.apply_transforms(und, gmseg, t1reg["fwdtransforms"], interpolator="genericLabel") * bmask

    csfseg = ants.threshold_image(t1segmentation, 1, 1)
    wmseg = ants.threshold_image(t1segmentation, 3, 3)
    csf_and_wm = (csfseg + wmseg).morphology("erode", 1)
    csf_and_wm = ants.apply_transforms(und, csf_and_wm, t1reg["fwdtransforms"], interpolator="nearestNeighbor") * bmask
    csfseg = ants.apply_transforms(und, csfseg, t1reg["fwdtransforms"], interpolator="nearestNeighbor") * bmask
    wmseg = ants.apply_transforms(und, wmseg, t1reg["fwdtransforms"], interpolator="nearestNeighbor") * bmask

    wmsignal = float(pet3dr[ants.iMath(wmseg, "ME", 1) == 1].mean())
    gmsignal = float(pet3dr[gmseg == 1].mean())
    csfsignal = float(pet3dr[csfseg == 1].mean())

    dktseg = ants.apply_transforms(und, t1dktcit, t1reg["fwdtransforms"], interpolator="genericLabel") * bmask
    df_pet3d = antspyt1w.map_intensity_to_dataframe("dkt_cortex_cit_deep_brain", und, dktseg)
    df_pet3d = antspyt1w.merge_hierarchical_csvs_to_wide_format({"pet3d": df_pet3d}, col_names=["Mean"])

    return convert_np_in_dict({
        "pet3d_dataframe": df_pet3d,
        "pet3d": pet3dr,
        "brainmask": bmask,
        "gm_mean": gmsignal,
        "wm_mean": wmsignal,
        "csf_mean": csfsignal,
    })
