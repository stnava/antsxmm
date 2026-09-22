from __future__ import annotations

import os
import warnings
from typing import Any
import numpy as np
import pandas as pd
import ants
from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor, RANSACRegressor, SGDRegressor, TheilSenRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from .dti import trim_dti_mask
from .fmri import timeseries_reg
from .metrics import (
    convert_np_in_dict,
    dvars,
    loop_timeseries_censoring,
    remove_elements_from_numpy_array,
    remove_volumes_from_timeseries,
    segment_timeseries_by_meanvalue,
    slice_snr,
    tsnr,
)


def calculate_CBF(
    Delta_M: ants.ANTsImage,
    M_0: ants.ANTsImage,
    mask: ants.ANTsImage,
    Lambda: float = 0.9,
    T_1: float = 0.67,
    Alpha: float = 0.68,
    w: float = 1.0,
    Tau: float = 1.5,
) -> ants.ANTsImage:
    """Calculate Cerebral Blood Flow (CBF) from difference and M0 images using the pCASL model."""
    cbf = M_0 * 0.0
    m0_data = M_0.numpy()
    mask_data = mask.numpy()
    m0thresh = float(np.quantile(m0_data[mask_data == 1], 0.1))
    sel = (mask_data == 1) & (m0_data >= m0thresh)

    delta_m_data = Delta_M.numpy()
    cbf_data = cbf.numpy()
    denom = m0_data[sel] * 2.0 * Alpha * (np.exp(-w * T_1) - np.exp(-(Tau + w) * T_1))
    denom = np.where(denom == 0, 1e-10, denom)
    cbf_data[sel] = delta_m_data[sel] * 60.0 * 100.0 * (Lambda * T_1) / denom
    cbf_data[cbf_data < 0.0] = 0.0

    cbf_img = ants.from_numpy(cbf_data)
    return ants.copy_image_info(M_0, cbf_img)


def warn_if_small_mask(mask: ants.ANTsImage, threshold_fraction: float = 0.05, label: str = " ") -> None:
    """Warn if the number of non-zero voxels in mask is below a fraction threshold."""
    image_size = np.prod(mask.shape)
    mask_size = np.count_nonzero(mask.numpy())
    if mask_size / image_size < threshold_fraction:
        percentage = 100.0 * mask_size / image_size
        warnings.warn(
            f"[ants] Warning: {label} contains only {mask_size} voxels ({percentage:.2f}%). "
            f"Below {threshold_fraction * 100:.2f}%.",
            UserWarning,
        )


def _replicate_list(user_list: list[Any], target_size: int) -> list[Any]:
    replication_factor = target_size // len(user_list)
    replicated_list = user_list * replication_factor
    remaining_elements = target_size % len(user_list)
    return replicated_list + user_list[:remaining_elements]


def _one_hot_encode(char_list: list[str]) -> np.ndarray:
    unique_chars = sorted(list(set(char_list)))
    encoding_dict = {char: [1 if char == c else 0 for c in unique_chars] for char in unique_chars}
    return np.array([encoding_dict[char] for char in char_list])


def bold_perfusion_minimal(
    fmri: ants.ANTsImage,
    m0_image: ants.ANTsImage | None = None,
    spa: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    nc: int = 0,
    tc: str = "alternating",
    n_to_trim: int = 0,
    outlier_threshold: float = 0.250,
    plot_brain_mask: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Estimate perfusion minimally from a BOLD time series without T1 image."""
    import antspynet

    fmri_template = ants.get_average_of_timeseries(fmri)
    mytrim = n_to_trim or 0
    perf_total_sigma = 1.5
    corrmo = timeseries_reg(
        fmri,
        fmri_template,
        type_of_transform="antsRegistrationSyNRepro[r]",
        total_sigma=perf_total_sigma,
        fdOffset=2.0,
        trim=mytrim,
        output_directory=None,
        verbose=verbose,
        syn_metric="CC",
        syn_sampling=2,
        reg_iterations=[40, 20, 5],
    )

    ntp = corrmo["motion_corrected"].shape[3]
    fmri_template = ants.get_average_of_timeseries(corrmo["motion_corrected"])
    bmask = antspynet.brain_extraction(fmri_template, "bold").threshold_image(0.5, 1).iMath("GetLargestComponent").morphology("close", 2).iMath("FillHoles")

    tclist = _replicate_list(["C", "T"], ntp) if tc == "alternating" else _replicate_list(["C"], int(ntp / 2)) + _replicate_list(["T"], int(ntp / 2))
    tclist_enc = _one_hot_encode(tclist[:ntp])
    fmrimotcorr = corrmo["motion_corrected"]
    hlinds = None

    if 0.0 < outlier_threshold < 1.0:
        fmrimotcorr, hlinds = loop_timeseries_censoring(fmrimotcorr, outlier_threshold, mask=None, verbose=verbose)
        tclist_enc = remove_elements_from_numpy_array(tclist_enc, hlinds)
        corrmo["FD"] = remove_elements_from_numpy_array(corrmo["FD"], hlinds)

    fmri_template = ants.iMath(ants.get_average_of_timeseries(fmrimotcorr), "Normalize")
    corrmo = timeseries_reg(
        fmri,
        fmri_template,
        type_of_transform="antsRegistrationSyNRepro[r]",
        total_sigma=perf_total_sigma,
        fdOffset=2.0,
        trim=mytrim,
        output_directory=None,
        verbose=verbose,
        syn_metric="CC",
        syn_sampling=2,
        reg_iterations=[40, 20, 5],
    )

    if 0.0 < outlier_threshold < 1.0:
        corrmo["motion_corrected"] = remove_volumes_from_timeseries(corrmo["motion_corrected"], hlinds)
        corrmo["FD"] = remove_elements_from_numpy_array(corrmo["FD"], hlinds)

    bmask = antspynet.brain_extraction(fmri_template, "bold").threshold_image(0.5, 1).iMath("GetLargestComponent").morphology("close", 2).iMath("FillHoles")
    mytsnr = tsnr(corrmo["motion_corrected"], bmask)
    mytsnr_thresh = float(np.quantile(mytsnr.numpy(), 0.995))
    tsnrmask = ants.threshold_image(mytsnr, 0, mytsnr_thresh).morphology("close", 3)
    bmask = bmask * ants.iMath(tsnrmask, "FillHoles")

    fmrimotcorr = corrmo["motion_corrected"]
    und = fmri_template * bmask
    mycompcor = ants.compcor(fmrimotcorr, ncompcor=nc, quantile=0.50, mask=bmask, filter_type="polynomial", degree=2)
    simg = ants.smooth_image(fmrimotcorr, spa, sigma_in_physical_coordinates=True)
    nuisance = np.c_[mycompcor["basis"], mycompcor["components"]]
    nuisance = ants.regress_components(nuisance, tclist_enc)

    regression_mask = bmask.clone()
    gmmat = ants.timeseries_to_matrix(simg, regression_mask)
    regvars = np.hstack((nuisance, tclist_enc))
    regvars = regvars[:, : regvars.shape[1] - 1]

    regression_model = LinearRegression()
    regression_model.fit(regvars, gmmat)
    coefind = regression_model.coef_.shape[1] - 1
    perfimg = ants.make_image(regression_mask, regression_model.coef_[:, coefind])
    gmseg = ants.image_clone(bmask)
    meangmval = float((perfimg[gmseg == 1]).mean())
    if meangmval < 0:
        perfimg = perfimg * -1.0
    negative_voxels = float((perfimg < 0.0).sum() / np.prod(perfimg.shape) * 100.0)
    perfimg[perfimg < 0.0] = 0.0

    if m0_image is None:
        m0 = ants.get_average_of_timeseries(fmrimotcorr)
    else:
        m0reg = ants.registration(fmri_template, m0_image, "antsRegistrationSyNRepro[r]", verbose=False)
        m0 = m0reg["warpedmovout"]

    if ntp == 2:
        img0 = ants.slice_image(corrmo["motion_corrected"], axis=3, idx=0)
        img1 = ants.slice_image(corrmo["motion_corrected"], axis=3, idx=1)
        if m0_image is None:
            perfimg, m0 = (img0, img1) if img0.mean() < img1.mean() else (img1, img0)
        else:
            perfimg = img1 - img0 if img0.mean() < img1.mean() else img0 - img1

    cbf = calculate_CBF(Delta_M=perfimg, M_0=m0, mask=bmask)
    meangmval = float((perfimg[gmseg == 1]).mean())
    meangmvalcbf = float((cbf[gmseg == 1]).mean())

    rsf_nuisance = pd.DataFrame(nuisance)
    rsf_nuisance["FD"] = corrmo["FD"]

    return convert_np_in_dict({
        "meanBold": und,
        "brainmask": bmask,
        "perfusion": perfimg,
        "cbf": cbf,
        "m0": m0,
        "perfusion_gm_mean": meangmval,
        "cbf_gm_mean": meangmvalcbf,
        "motion_corrected": corrmo["motion_corrected"],
        "brain_mask": bmask,
        "nuisance": rsf_nuisance,
        "tsnr": mytsnr,
        "dvars": dvars(corrmo["motion_corrected"], gmseg),
        "FD_max": float(rsf_nuisance["FD"].max()),
        "FD_mean": float(rsf_nuisance["FD"].mean()),
        "FD_sd": float(rsf_nuisance["FD"].std()),
        "outlier_volumes": hlinds,
        "negative_voxels": negative_voxels,
    })


def bold_perfusion(
    fmri: ants.ANTsImage,
    t1head: ants.ANTsImage,
    t1: ants.ANTsImage,
    t1segmentation: ants.ANTsImage,
    t1dktcit: ants.ANTsImage,
    FD_threshold: float = 0.5,
    spa: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    nc: int = 3,
    type_of_transform: str = "antsRegistrationSyNRepro[r]",
    tc: str = "alternating",
    n_to_trim: int = 0,
    m0_image: ants.ANTsImage | None = None,
    m0_indices: list[int] | None = None,
    outlier_threshold: float = 0.250,
    add_FD_to_nuisance: bool = False,
    n3: bool = False,
    segment_timeseries: bool = False,
    trim_the_mask: float = 4.25,
    upsample: bool = True,
    perfusion_regression_model: str = "linear",
    verbose: bool = False,
) -> dict[str, Any]:
    """Estimate perfusion and quantified CBF from a BOLD time series with anatomical guidance."""
    import antspyt1w

    if segment_timeseries:
        lo_vs_high = segment_timeseries_by_meanvalue(fmri)
        fmri = remove_volumes_from_timeseries(fmri, lo_vs_high["lowermeans"])

    if m0_image is not None and m0_image.dimension == 4:
        m0_image = ants.get_average_of_timeseries(m0_image)

    fmri_temp_raw, _ = loop_timeseries_censoring(fmri, 0.10)
    fmri_template = ants.get_average_of_timeseries(fmri_temp_raw)
    rig = ants.registration(fmri_template, t1head, "antsRegistrationSyNRepro[r]")
    bmask = ants.apply_transforms(
        fmri_template, ants.threshold_image(t1segmentation, 1, 6), rig["fwdtransforms"][0], interpolator="genericLabel"
    )

    mytrim = 0 if m0_indices is not None else (n_to_trim or 0)
    perf_total_sigma = 1.5
    corrmo = timeseries_reg(
        fmri,
        fmri_template,
        type_of_transform=type_of_transform,
        total_sigma=perf_total_sigma,
        fdOffset=2.0,
        trim=mytrim,
        output_directory=None,
        verbose=verbose,
        syn_metric="CC",
        syn_sampling=2,
        reg_iterations=[40, 20, 5],
    )

    if m0_image is not None:
        m0 = m0_image
    elif m0_indices is not None:
        not_m0 = [x for x in range(fmri.shape[3]) if x not in m0_indices]
        m0 = ants.get_average_of_timeseries(remove_volumes_from_timeseries(corrmo["motion_corrected"], not_m0))
        corrmo["motion_corrected"] = remove_volumes_from_timeseries(corrmo["motion_corrected"], m0_indices)
        corrmo["FD"] = remove_elements_from_numpy_array(corrmo["FD"], m0_indices)
        fmri = remove_volumes_from_timeseries(fmri, m0_indices)
    else:
        m0 = None

    ntp = corrmo["motion_corrected"].shape[3]
    tclist = _replicate_list(["C", "T"], ntp) if tc == "alternating" else _replicate_list(["C"], int(ntp / 2)) + _replicate_list(["T"], int(ntp / 2))
    tclist_enc = _one_hot_encode(tclist[:ntp])

    fmrimotcorr = corrmo["motion_corrected"]
    hlinds = None
    if 0.0 < outlier_threshold < 1.0:
        fmrimotcorr, hlinds = loop_timeseries_censoring(fmrimotcorr, outlier_threshold, mask=None, verbose=verbose)
        tclist_enc = remove_elements_from_numpy_array(tclist_enc, hlinds)
        corrmo["FD"] = remove_elements_from_numpy_array(corrmo["FD"], hlinds)

    fmri_template = ants.iMath(ants.get_average_of_timeseries(fmrimotcorr), "Normalize")
    if upsample:
        spc = ants.get_spacing(fmri)
        minspc = min(2.0, min(spc[0:3]))
        fmri_template = ants.resample_image(fmri_template, [minspc, minspc, minspc], interp_type=0)

    rig = ants.registration(fmri_template, t1head, "antsRegistrationSyNRepro[r]")
    bmask = ants.apply_transforms(
        fmri_template, ants.threshold_image(t1segmentation, 1, 6), rig["fwdtransforms"][0], interpolator="genericLabel"
    )
    warn_if_small_mask(bmask, label="bold_perfusion:bmask")

    corrmo = timeseries_reg(
        fmri,
        fmri_template,
        type_of_transform=type_of_transform,
        total_sigma=perf_total_sigma,
        fdOffset=2.0,
        trim=mytrim,
        output_directory=None,
        verbose=verbose,
        syn_metric="CC",
        syn_sampling=2,
        reg_iterations=[40, 20, 5],
    )

    if 0.0 < outlier_threshold < 1.0:
        corrmo["motion_corrected"] = remove_volumes_from_timeseries(corrmo["motion_corrected"], hlinds)
        corrmo["FD"] = remove_elements_from_numpy_array(corrmo["FD"], hlinds)

    mytsnr = tsnr(corrmo["motion_corrected"], bmask)
    mytsnr_thresh = float(np.quantile(mytsnr.numpy(), 0.995))
    tsnrmask = ants.threshold_image(mytsnr, 0, mytsnr_thresh).morphology("close", 3)
    bmask = bmask * ants.iMath(tsnrmask, "FillHoles")
    warn_if_small_mask(bmask, label="bold_perfusion:bmask*tsnrmask")

    und = fmri_template * bmask
    t1reg = ants.registration(und, t1, "antsRegistrationSyNRepro[s]")
    gmseg = (
        ants.threshold_image(t1segmentation, 2, 2) + ants.threshold_image(t1segmentation, 4, 4)
    ).threshold_image(1, 4).iMath("MD", 1)
    gmseg = ants.apply_transforms(und, gmseg, t1reg["fwdtransforms"], interpolator="genericLabel") * bmask

    csfseg = ants.threshold_image(t1segmentation, 1, 1)
    wmseg = ants.threshold_image(t1segmentation, 3, 3)
    csf_and_wm = (csfseg + wmseg).morphology("erode", 1)
    csf_and_wm = ants.apply_transforms(und, csf_and_wm, t1reg["fwdtransforms"], interpolator="nearestNeighbor") * bmask
    warn_if_small_mask(csf_and_wm, label="bold_perfusion:csfAndWM")

    mycompcor = ants.compcor(corrmo["motion_corrected"], ncompcor=nc, quantile=0.50, mask=csf_and_wm, filter_type="polynomial", degree=2)
    simg = ants.smooth_image(corrmo["motion_corrected"], spa, sigma_in_physical_coordinates=True)
    nuisance = np.c_[mycompcor["basis"], mycompcor["components"]]
    if add_FD_to_nuisance:
        nuisance = np.c_[nuisance, corrmo["FD"]]
    nuisance = ants.regress_components(nuisance, tclist_enc)

    regression_mask = bmask.clone()
    gmmat = ants.timeseries_to_matrix(simg, regression_mask)
    regvars = np.hstack((nuisance, tclist_enc))
    coefind = regvars.shape[1] - 1
    regvars = regvars[:, range(coefind)]
    predictor_idx = regvars.shape[1] - 1

    if perfusion_regression_model == "linear":
        model = LinearRegression()
        model.fit(regvars, gmmat)
        perfimg = ants.make_image(regression_mask, model.coef_[:, model.coef_.shape[1] - 1])
    elif perfusion_regression_model in ["huber", "quantile", "theilsen", "ransac", "sgd"]:
        models_map = {
            "huber": HuberRegressor,
            "quantile": QuantileRegressor,
            "theilsen": TheilSenRegressor,
            "ransac": RANSACRegressor,
            "sgd": SGDRegressor,
        }
        scaler = StandardScaler()
        gmmat_scaled = scaler.fit_transform(gmmat)
        multi_model = MultiOutputRegressor(models_map[perfusion_regression_model]())
        multi_model.fit(regvars, gmmat_scaled)
        coeffs = np.array([est.coef_[predictor_idx] for est in multi_model.estimators_])
        perfimg = ants.make_image(regression_mask, coeffs)
    else:
        raise ValueError(f"Unknown regression model: {perfusion_regression_model}")

    meangmval = float((perfimg[gmseg == 1]).mean())
    if meangmval < 0:
        perfimg = perfimg * -1.0
    negative_voxels = float((perfimg < 0.0).sum() / np.prod(perfimg.shape) * 100.0)
    perfimg[perfimg < 0.0] = 0.0

    if m0 is None:
        m0 = ants.get_average_of_timeseries(corrmo["motion_corrected"])
    else:
        m0reg = ants.registration(fmri_template, m0, "antsRegistrationSyNRepro[r]", verbose=False)
        m0 = m0reg["warpedmovout"]

    if ntp == 2:
        img0 = ants.slice_image(corrmo["motion_corrected"], axis=3, idx=0)
        img1 = ants.slice_image(corrmo["motion_corrected"], axis=3, idx=1)
        if m0_image is None and m0_indices is None:
            perfimg, m0 = (img0, img1) if img0.mean() < img1.mean() else (img1, img0)
        else:
            perfimg = img1 - img0 if img0.mean() < img1.mean() else img0 - img1

    cbf = calculate_CBF(Delta_M=perfimg, M_0=m0, mask=bmask)
    if trim_the_mask > 0.0:
        bmask = trim_dti_mask(cbf, bmask, trim_the_mask)
        perfimg = perfimg * bmask
        cbf = cbf * bmask

    meangmval = float((perfimg[gmseg == 1]).mean())
    meangmvalcbf = float((cbf[gmseg == 1]).mean())

    rsf_nuisance = pd.DataFrame(nuisance)
    rsf_nuisance["FD"] = corrmo["FD"]

    dktseg = ants.apply_transforms(und, t1dktcit, t1reg["fwdtransforms"], interpolator="genericLabel") * bmask
    df_perf = antspyt1w.map_intensity_to_dataframe("dkt_cortex_cit_deep_brain", perfimg, dktseg)
    df_perf = antspyt1w.merge_hierarchical_csvs_to_wide_format({"perf": df_perf}, col_names=["Mean"])

    df_cbf = antspyt1w.map_intensity_to_dataframe("dkt_cortex_cit_deep_brain", cbf, dktseg)
    df_cbf = antspyt1w.merge_hierarchical_csvs_to_wide_format({"cbf": df_cbf}, col_names=["Mean"]).add_prefix("cbf_")
    df_perf_total = pd.concat([df_perf, df_cbf], axis=1, ignore_index=False)

    return convert_np_in_dict({
        "meanBold": und,
        "brainmask": bmask,
        "perfusion": perfimg,
        "cbf": cbf,
        "m0": m0,
        "perfusion_gm_mean": meangmval,
        "cbf_gm_mean": meangmvalcbf,
        "perf_dataframe": df_perf_total,
        "motion_corrected": corrmo["motion_corrected"],
        "gmseg": gmseg,
        "brain_mask": bmask,
        "nuisance": rsf_nuisance,
        "tsnr": mytsnr,
        "ssnr": slice_snr(corrmo["motion_corrected"], csf_and_wm, gmseg),
        "dvars": dvars(corrmo["motion_corrected"], gmseg),
        "high_motion_count": int((rsf_nuisance["FD"] > FD_threshold).sum()),
        "high_motion_pct": float((rsf_nuisance["FD"] > FD_threshold).sum() / rsf_nuisance.shape[0]),
        "FD_max": float(rsf_nuisance["FD"].max()),
        "FD_mean": float(rsf_nuisance["FD"].mean()),
        "FD_sd": float(rsf_nuisance["FD"].std()),
        "bold_evr": antspyt1w.patch_eigenvalue_ratio(und, 512, [16, 16, 16], evdepth=0.9, mask=bmask),
        "t1reg": t1reg,
        "outlier_volumes": hlinds,
        "n_outliers": len(hlinds) if hlinds else 0,
        "negative_voxels": negative_voxels,
    })


from .pet import pet3d_summary

