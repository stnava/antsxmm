from __future__ import annotations

import math
import os
import random
import tempfile
from typing import Any
import numpy as np
import pandas as pd
import ants

from .metrics import convert_np_in_dict, mask_snr
from .registration import register_images, robust_affine


def tra_initializer(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    n_simulations: int = 32,
    max_rotation: float = 30.0,
    transform: list[str] | None = None,
    compreg: Any = None,
    random_seed: int | None = 42,
    verbose: bool = False,
) -> dict[str, Any]:
    """Multi-start multi-transform registration initialization based on ants.registration."""
    if transform is None:
        transform = ["rigid"]
    if random_seed is not None:
        random.seed(random_seed)

    output_directory = tempfile.mkdtemp()
    output_directory_w = os.path.join(output_directory, "tra_reg")
    os.makedirs(output_directory_w, exist_ok=True)
    bestmi = math.inf
    bestvar = 0.0
    myorig = list(ants.get_origin(fixed))
    mymax = max(abs(x) for x in myorig) if myorig else 1.0
    maxtrans = mymax * 0.05

    if compreg is None:
        bestreg = register_images(fixed, moving, "Translation", outprefix=os.path.join(output_directory_w, "trans"))
        initx = ants.read_transform(bestreg["fwdtransforms"][0])
    else:
        bestreg = compreg
        initx = ants.read_transform(bestreg["fwdtransforms"][0])

    for mytx in transform:
        regtx = "antsRegistrationSyNRepro[r]"
        with tempfile.NamedTemporaryFile(suffix=".h5") as tp:
            if mytx == "translation":
                regtx = "Translation"
                r_rot = ants.contrib.RandomTranslate3D((maxtrans * -1.0, maxtrans), reference=fixed)
            elif mytx == "affine":
                regtx = "Affine"
                r_rot = ants.contrib.RandomRotate3D((maxtrans * -1.0, maxtrans), reference=fixed)
            else:
                r_rot = ants.contrib.RandomRotate3D((max_rotation * -1.0, max_rotation), reference=fixed)

            for k in range(n_simulations):
                simtx = ants.compose_ants_transforms([r_rot.transform(), initx])
                ants.write_transform(simtx, tp.name)
                init_tf = tp.name if k > 0 else None
                reg = register_images(
                    fixed,
                    moving,
                    regtx,
                    initial_transform=init_tf,
                    outprefix=os.path.join(output_directory_w, f"reg{k}"),
                    verbose=False,
                )
                temp = reg["warpedmovout"]
                myvar = float(temp.numpy().var())
                if myvar > 0:
                    mymi = float(ants.image_mutual_information(fixed, temp))
                    if mymi < bestmi:
                        bestmi = mymi
                        bestreg = reg
                        bestvar = myvar

    if bestvar == 0.0 and compreg is not None:
        return compreg
    return bestreg


def neuromelanin(
    list_nm_images: list[ants.ANTsImage],
    t1: ants.ANTsImage,
    t1_head: ants.ANTsImage,
    t1lab: ants.ANTsImage,
    brain_stem_dilation: int = 8,
    bias_correct: bool = True,
    denoise: int | None = None,
    srmodel: Any = None,
    target_range: list[float] | None = None,
    poly_order: str | int | None = "hist",
    normalize_nm: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Averaged and registered neuromelanin quantification."""
    import antspynet
    import antspyt1w

    if target_range is None:
        target_range = [0, 1]

    fnt = os.path.expanduser("~/.antspyt1w/CIT168_T1w_700um_pad_adni.nii.gz")
    fnt_nm = os.path.expanduser("~/.antspymm/CIT168_T1w_700um_pad_adni_NM_norm_avg.nii.gz")
    fnt_bst = os.path.expanduser("~/.antspyt1w/CIT168_T1w_700um_pad_adni_brainstem.nii.gz")
    fn_slab = os.path.expanduser("~/.antspyt1w/CIT168_MT_Slab_adni.nii.gz")

    template = ants.image_read(fnt, reorient=False)
    template_nm = ants.iMath(ants.image_read(fnt_nm, reorient=False), "Normalize")
    template_bstem = ants.image_read(fnt_bst, reorient=False).threshold_image(1, 1000)

    reg = register_images(t1, template, "antsRegistrationSyNQuickRepro[s]")
    nmavg2t1 = ants.apply_transforms(t1, template_nm, reg["fwdtransforms"], interpolator="linear")
    slab2t1 = ants.threshold_image(nmavg2t1, "Otsu", 2).threshold_image(1, 2).iMath("MD", 1).iMath("FillHoles")
    bstem2t1 = ants.apply_transforms(t1, template_bstem, reg["fwdtransforms"], interpolator="nearestNeighbor").iMath("MD", 1)
    bstem2t1 = ants.crop_image(bstem2t1, slab2t1)
    cropper = ants.decrop_image(bstem2t1, slab2t1).iMath("MD", brain_stem_dilation)
    nm_imgs: list[ants.ANTsImage] = []
    for x in list_nm_images:
        xim = ants.image_read(x) if isinstance(x, (str, os.PathLike)) else x
        if hasattr(xim, "dimension") and xim.dimension == 4:
            nm_imgs.extend(ants.ndimage_to_list(xim))
        else:
            nm_imgs.append(xim)

    if not nm_imgs:
        raise ValueError("No valid neuromelanin images provided")

    nm_avg = nm_imgs[0] * 0.0
    for k in range(len(nm_imgs)):
        if denoise is not None:
            nm_imgs[k] = ants.denoise_image(nm_imgs[k], shrink_factor=1, p=denoise, r=denoise + 1, noise_model="Gaussian")
        if bias_correct:
            n4mask = ants.threshold_image(ants.iMath(nm_imgs[k], "Normalize"), 0.05, 1)
            nm_imgs[k] = ants.n4_bias_field_correction(nm_imgs[k], mask=n4mask)
        nm_avg = nm_avg + ants.resample_image_to_target(nm_imgs[k], nm_avg) / len(nm_imgs)

    nm_avg_new = nm_avg * 0.0
    txlist: list[str] = []
    for k in range(len(nm_imgs)):
        current_image = register_images(nm_imgs[k], nm_avg, type_of_transform="antsRegistrationSyNRepro[r]")
        txlist.append(current_image["fwdtransforms"][0])
        nm_avg_new = nm_avg_new + current_image["warpedfixout"] / len(nm_imgs)
    nm_avg = nm_avg_new

    t1c = ants.crop_image(t1_head, slab2t1).iMath("Normalize")
    slabreg = tra_initializer(nm_avg, t1c, verbose=verbose)
    labels2nm = ants.apply_transforms(nm_avg, t1lab, slabreg["fwdtransforms"], interpolator="genericLabel")
    cropper2nm = ants.apply_transforms(nm_avg, cropper, slabreg["fwdtransforms"], interpolator="nearestNeighbor")

    crop_nm_list: list[ants.ANTsImage] = []
    for k in range(len(nm_imgs)):
        concattx = [txlist[k], slabreg["fwdtransforms"][0]]
        cropmask = ants.apply_transforms(nm_imgs[k], cropper, concattx, interpolator="nearestNeighbor")
        crop_nm_list.append(ants.crop_image(nm_imgs[k], cropmask))

    if srmodel is not None:
        for k in range(len(crop_nm_list)):
            temp = antspynet.apply_super_resolution_model_to_image(crop_nm_list[k], srmodel, target_range=target_range)
            if poly_order is not None:
                bilin = ants.resample_image_to_target(crop_nm_list[k], temp)
                temp = (
                    ants.histogram_match_image(temp, bilin)
                    if poly_order == "hist"
                    else antspynet.regression_match_image(temp, bilin, poly_order=poly_order)
                )
            crop_nm_list[k] = temp

    nm_avg_cropped = crop_nm_list[0] * 0.0
    for k in range(len(crop_nm_list)):
        nm_avg_cropped = nm_avg_cropped + ants.apply_transforms(nm_avg_cropped, crop_nm_list[k], txlist[k]) / len(crop_nm_list)

    for _ in range(3):
        nm_avg_cropped_new = nm_avg_cropped * 0.0
        for k in range(len(crop_nm_list)):
            myreg = register_images(
                ants.iMath(nm_avg_cropped, "Normalize"), ants.iMath(crop_nm_list[k], "Normalize"), "antsRegistrationSyNRepro[r]"
            )
            warpednext = ants.apply_transforms(nm_avg_cropped_new, crop_nm_list[k], myreg["fwdtransforms"])
            nm_avg_cropped_new = nm_avg_cropped_new + warpednext
        nm_avg_cropped = nm_avg_cropped_new / len(crop_nm_list)

    slabreg_updated = tra_initializer(nm_avg_cropped, t1c, compreg=slabreg, verbose=verbose)
    temp_orig = ants.apply_transforms(nm_avg_cropped, t1c, slabreg["fwdtransforms"])
    temp_update = ants.apply_transforms(nm_avg_cropped, t1c, slabreg_updated["fwdtransforms"])
    mi_update = float(ants.image_mutual_information(ants.iMath(nm_avg_cropped, "Normalize"), ants.iMath(temp_update, "Normalize")))
    mi_orig = float(ants.image_mutual_information(ants.iMath(nm_avg_cropped, "Normalize"), ants.iMath(temp_orig, "Normalize")))
    if mi_update < mi_orig:
        slabreg = slabreg_updated

    if normalize_nm:
        nm_avg_cropped = ants.iMath(nm_avg_cropped, "Normalize")
        nm_avg_cropped = ants.iMath(nm_avg_cropped, "TruncateIntensity", 0.05, 0.95)
        nm_avg_cropped = ants.iMath(nm_avg_cropped, "Normalize")

    labels2nm = ants.apply_transforms(nm_avg_cropped, t1lab, slabreg["fwdtransforms"], interpolator="nearestNeighbor")

    def get_biggest_part(x: ants.ANTsImage, labeln: int) -> None:
        temp_comp = ants.threshold_image(x, labeln, labeln).iMath("GetLargestComponent")
        x[x == labeln] = 0
        x[temp_comp == 1] = labeln

    get_biggest_part(labels2nm, 33)
    get_biggest_part(labels2nm, 34)

    nmdf = antspyt1w.map_intensity_to_dataframe("CIT168_Reinf_Learn_v1_label_descriptions_pad", nm_avg_cropped, labels2nm)
    nmdf_wide = antspyt1w.merge_hierarchical_csvs_to_wide_format({"NM": nmdf}, col_names=["Mean"])

    rr_mask = ants.mask_image(labels2nm, labels2nm, [33, 34], binarize=True)
    sn_mask = ants.mask_image(labels2nm, labels2nm, [7, 9, 23, 25], binarize=True)
    nmavgsnr = mask_snr(nm_avg_cropped, rr_mask, sn_mask, bias_correct=False)

    snavg = float(nm_avg_cropped[sn_mask == 1].mean())
    rravg = float(nm_avg_cropped[rr_mask == 1].mean())
    snstd = float(nm_avg_cropped[sn_mask == 1].std())
    rrstd = float(nm_avg_cropped[rr_mask == 1].std())
    vol_element = float(np.prod(ants.get_spacing(sn_mask)))
    snvol = float(vol_element * sn_mask.sum())

    if snvol > 0:
        sn_z = float(ants.transform_physical_point_to_index(sn_mask, ants.get_center_of_mass(sn_mask))[2]) / sn_mask.shape[2]
    else:
        sn_z = math.nan

    nm_evr = (
        antspyt1w.patch_eigenvalue_ratio(nm_avg, 512, [6, 6, 6], evdepth=0.9, mask=cropper2nm)
        if cropper2nm.sum() > 0
        else 0.0
    )

    simg = ants.smooth_image(nm_avg_cropped, float(np.min(ants.get_spacing(nm_avg_cropped))))
    nmabovekthresh_mask = sn_mask * ants.threshold_image(simg, rravg + 2.0 * rrstd, math.inf)
    snvolabovethresh = float(vol_element * nmabovekthresh_mask.sum())
    snintmeanabovethresh = float((simg * nmabovekthresh_mask).mean())
    snintsumabovethresh = float((simg * nmabovekthresh_mask).sum())

    nmabovekthresh_mask3 = sn_mask * ants.threshold_image(simg, rravg + 3.0 * rrstd, math.inf)
    snvolabovethresh3 = float(vol_element * nmabovekthresh_mask3.sum())

    nmabovekthresh_mask1 = sn_mask * ants.threshold_image(simg, rravg + 1.0 * rrstd, math.inf)
    snvolabovethresh1 = float(vol_element * nmabovekthresh_mask1.sum())

    return convert_np_in_dict({
        "NM_avg": nm_avg,
        "NM_avg_cropped": nm_avg_cropped,
        "NM_labels": labels2nm,
        "NM_cropped": crop_nm_list,
        "NM_midbrainROI": cropper2nm,
        "NM_dataframe": nmdf,
        "NM_dataframe_wide": nmdf_wide,
        "t1_to_NM": slabreg["warpedmovout"],
        "t1_to_NM_transform": slabreg["fwdtransforms"],
        "NM_avg_signaltonoise": nmavgsnr,
        "NM_avg_substantianigra": snavg,
        "NM_std_substantianigra": snstd,
        "NM_volume_substantianigra": snvol,
        "NM_volume_substantianigra_1std": snvolabovethresh1,
        "NM_volume_substantianigra_2std": snvolabovethresh,
        "NM_intmean_substantianigra_2std": snintmeanabovethresh,
        "NM_intsum_substantianigra_2std": snintsumabovethresh,
        "NM_volume_substantianigra_3std": snvolabovethresh3,
        "NM_avg_refregion": rravg,
        "NM_std_refregion": rrstd,
        "NM_min": float(nm_avg_cropped.min()),
        "NM_max": float(nm_avg_cropped.max()),
        "NM_mean": float(nm_avg_cropped.numpy().mean()),
        "NM_sd": float(np.std(nm_avg_cropped.numpy())),
        "NM_q0pt05": float(np.quantile(nm_avg_cropped.numpy(), 0.05)),
        "NM_q0pt10": float(np.quantile(nm_avg_cropped.numpy(), 0.10)),
        "NM_q0pt90": float(np.quantile(nm_avg_cropped.numpy(), 0.90)),
        "NM_q0pt95": float(np.quantile(nm_avg_cropped.numpy(), 0.95)),
        "NM_substantianigra_z_coordinate": sn_z,
        "NM_evr": nm_evr,
        "NM_count": len(list_nm_images),
    })
