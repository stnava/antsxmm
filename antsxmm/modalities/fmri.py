from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
import warnings
from typing import Any
import numpy as np
import pandas as pd
import ants
from scipy import fft, signal, stats
from scipy.stats import pearsonr
from sklearn.decomposition import FastICA, PCA

from .metrics import (
    convert_np_in_dict,
    dvars,
    impute_timeseries,
    loop_timeseries_censoring,
    remove_elements_from_numpy_array,
    remove_volumes_from_timeseries,
    slice_snr,
    tsnr,
)
from .templates import get_data


def spec_taper(x: np.ndarray, p: float | np.ndarray = 0.1) -> np.ndarray:
    """Computes a tapered version of x with tapering p."""
    p_arr = np.r_[p]
    assert np.all((p_arr >= 0) & (p_arr < 0.5)), "'p' must be between 0 and 0.5"
    x_arr = np.r_[x].astype("float64")
    original_shape = x_arr.shape
    while len(x_arr.shape) < 2:
        x_arr = np.expand_dims(x_arr, axis=1)

    nr, nc = x_arr.shape
    p_vec = p_arr * np.ones(nc) if len(p_arr) == 1 else p_arr

    for i in range(nc):
        m = int(np.floor(nr * p_vec[i]))
        if m == 0:
            continue
        w = 0.5 * (1 - np.cos(np.pi * np.arange(1, 2 * m, step=2) / (2 * m)))
        x_arr[:, i] = np.r_[w, np.ones(nr - 2 * m), w[::-1]] * x_arr[:, i]

    return np.reshape(x_arr, original_shape)


def spec_ci(df: float, coverage: float = 0.95) -> np.ndarray:
    """Computes confidence interval for a spectral fit."""
    tail = 1.0 - coverage
    phi = stats.chi2.cdf(x=df, df=df)
    upper_quantile = 1.0 - tail * (1.0 - phi)
    lower_quantile = tail * phi
    return df / stats.chi2.ppf([upper_quantile, lower_quantile], df=df)


def plot_spec(spec_res: dict[str, Any], coverage: float | None = None, ax: Any = None, title: str | None = None) -> None:
    """Plotting method for periodogram results."""
    import matplotlib.pyplot as plt

    f, pxx = spec_res["freq"], spec_res["spec"]
    if ax is None:
        ax = plt.gca()
    ax.plot(f, pxx, color="C0")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Log Spectrum")
    ax.set_yscale("log")
    ax.set_title(spec_res["method"] if title is None else title)


def spec_pgram(
    x: np.ndarray,
    xfreq: float = 1.0,
    spans: Any = None,
    kernel: Any = None,
    taper: float = 0.1,
    pad: int = 0,
    fast: bool = True,
    demean: bool = False,
    detrend: bool = True,
    plot: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Computes the spectral density estimate using a periodogram."""
    def daniell_window_modified(m: int) -> np.ndarray:
        def w(k: np.ndarray) -> np.ndarray:
            return np.where(np.abs(k) < m, 1 / (2 * m), np.where(np.abs(k) == m, 1 / (4 * m), 0))
        return w(np.arange(-m, m + 1))

    def daniell_window_convolve(v: np.ndarray) -> np.ndarray:
        if len(v) == 0:
            return np.r_[1]
        if len(v) == 1:
            return daniell_window_modified(v[0])
        return signal.convolve(daniell_window_modified(v[0]), daniell_window_convolve(v[1:]))

    x_arr = np.r_[x].astype("float64")
    original_shape = x_arr.shape
    while len(x_arr.shape) < 2:
        x_arr = np.expand_dims(x_arr, axis=1)

    n, nser = x_arr.shape
    n0 = n

    if spans is not None:
        kernel = daniell_window_convolve(np.floor_divide(np.r_[spans], 2))

    if detrend:
        t = np.arange(n) - (n - 1) / 2
        sumt2 = n * (n**2 - 1) / 12
        x_arr -= np.repeat(np.expand_dims(np.mean(x_arr, axis=0), 0), n, axis=0) + np.outer(np.sum(x_arr.T * t, axis=1), t / sumt2).T
    elif demean:
        x_arr -= np.mean(x_arr, axis=0)

    x_arr = spec_taper(x_arr, taper)
    u2 = 1 - (5 / 8) * taper * 2
    u4 = 1 - (93 / 128) * taper * 2

    if pad > 0:
        x_arr = np.r_[x_arr, np.zeros((pad * x_arr.shape[0], x_arr.shape[1]))]
        n = x_arr.shape[0]

    if fast:
        new_n = fft.next_fast_len(n, True)
        x_arr = np.r_[x_arr, np.zeros((new_n - n, x_arr.shape[1]))]
        n = new_n

    nspec = int(np.floor(n / 2))
    freq = (np.arange(nspec) + 1) * xfreq / n
    xfft = fft.fft(x_arr.T).T

    pgram = np.empty((n, nser, nser), dtype="complex")
    for i in range(nser):
        for j in range(nser):
            pgram[:, i, j] = xfft[:, i] * np.conj(xfft[:, j]) / (n0 * xfreq)
            pgram[0, i, j] = 0.5 * (pgram[1, i, j] + pgram[-1, i, j])

    if kernel is None:
        df = 2.0
        bandwidth = np.sqrt(1 / 12)
    else:
        def conv_circular(sig: np.ndarray, kern: np.ndarray) -> np.ndarray:
            pad_len = len(sig) - len(kern)
            half_window = int((len(kern) + 1) / 2)
            indices = range(-half_window, len(sig) - half_window)
            orig_conv = np.real(fft.ifft(fft.fft(sig) * fft.fft(np.r_[np.zeros(pad_len), kern])))
            return orig_conv.take(indices, mode="wrap")

        for i in range(nser):
            for j in range(nser):
                pgram[:, i, j] = conv_circular(pgram[:, i, j], kernel)

        df = 2 / np.sum(kernel**2)
        m = (len(kernel) - 1) / 2
        k = np.arange(-m, m + 1)
        bandwidth = np.sqrt(np.sum((1 / 12 + k**2) * kernel))

    df = df / (u4 / u2**2) * (n0 / n)
    bandwidth = bandwidth * xfreq / n
    pgram = pgram[1 : (nspec + 1), :, :]

    spec = np.empty((nspec, nser))
    for i in range(nser):
        spec[:, i] = np.real(pgram[:, i, i])

    spec = (spec / u2).squeeze()
    return {
        "freq": freq,
        "spec": spec,
        "kernel": kernel,
        "df": df,
        "bandwidth": bandwidth,
        "method": "Raw Periodogram" if kernel is None else "Smoothed Periodogram",
    }


def alffmap(x: np.ndarray, flo: float = 0.01, fhi: float = 0.1, tr: float = 1.0, detrend: bool = True) -> dict[str, float]:
    """Amplitude of Low Frequency Fluctuations (ALFF and f/ALFF)."""
    temp = spec_pgram(x, xfreq=1.0 / tr, demean=False, detrend=detrend, taper=0, fast=True, plot=False)
    fselect = np.logical_and(temp["freq"] >= flo, temp["freq"] <= fhi)
    denom = float(temp["spec"].sum())
    numer = float(temp["spec"][fselect].sum())
    falff = numer / denom if denom > 0 else 0.0
    return {"alff": numer, "falff": falff}


def calculate_trimmed_mean(data: np.ndarray, proportion_to_trim: float = 0.01) -> float:
    """Calculate trimmed mean."""
    return float(stats.trim_mean(data, proportion_to_trim))


def alff_image(
    x: ants.ANTsImage,
    mask: ants.ANTsImage,
    flo: float = 0.01,
    fhi: float = 0.1,
    nuisance: np.ndarray | None = None,
) -> dict[str, ants.ANTsImage]:
    """Compute ALFF and fALFF images over mask."""
    xmat = ants.timeseries_to_matrix(x, mask)
    if nuisance is not None:
        xmat = ants.regress_components(xmat, nuisance)
    alffvec = np.zeros(xmat.shape[1])
    falffvec = np.zeros(xmat.shape[1])
    mytr = float(ants.get_spacing(x)[3])
    for n in range(xmat.shape[1]):
        temp = alffmap(xmat[:, n], flo=flo, fhi=fhi, tr=mytr)
        alffvec[n] = temp["alff"]
        falffvec[n] = temp["falff"]
    alffi = ants.make_image(mask, alffvec)
    falffi = ants.make_image(mask, falffvec)
    alff_tm = calculate_trimmed_mean(alffvec, 0.01)
    falff_tm = calculate_trimmed_mean(falffvec, 0.01)
    if alff_tm > 0:
        alffi = alffi / alff_tm
    if falff_tm > 0:
        falffi = falffi / falff_tm
    return {"alff": alffi, "falff": falffi}


def compute_PerAF_voxel(time_series: np.ndarray) -> float:
    """Compute Percentage Amplitude Fluctuation (PerAF) for a 1D time series."""
    n = len(time_series)
    m = np.mean(time_series)
    if m == 0:
        return 0.0
    return float((100.0 / n) * np.sum(np.abs((time_series - m) / m)))


def PerAF(x: ants.ANTsImage, mask: ants.ANTsImage, globalmean: bool = True) -> ants.ANTsImage:
    """Compute Percentage Amplitude Fluctuation (PerAF) image."""
    time_series = ants.timeseries_to_matrix(x, mask)
    n = time_series.shape[1]
    vec = np.zeros(n)
    for i in range(n):
        vec[i] = compute_PerAF_voxel(time_series[:, i])
    outimg = ants.make_image(mask, vec)
    if globalmean:
        tm = calculate_trimmed_mean(vec, 0.01)
        if tm > 0:
            outimg = outimg / tm
    return outimg


def despike_time_series_afni(image: ants.ANTsImage, c1: float = 2.5, c2: float = 4.0) -> tuple[ants.ANTsImage, np.ndarray]:
    """Despike a time series image using polynomial fitting and AFNI-style squashing."""
    data = image.numpy()
    despiked_data = np.copy(data)
    curve = np.zeros_like(despiked_data)

    def l1_fit_polynomial(ts: np.ndarray, degree: int = 2) -> np.ndarray:
        t = np.arange(len(ts))
        coefs = np.polyfit(t, ts, degree)
        return np.polyval(coefs, t)

    for x_idx in range(data.shape[0]):
        for y_idx in range(data.shape[1]):
            for z_idx in range(data.shape[2]):
                curve[x_idx, y_idx, z_idx, :] = l1_fit_polynomial(data[x_idx, y_idx, z_idx, :], degree=2)

    residuals = data - curve
    mad = np.median(np.abs(residuals - np.median(residuals, axis=-1, keepdims=True)), axis=-1, keepdims=True)
    sigma = np.sqrt(np.pi / 2.0) * mad
    sigma_safe = np.where(sigma == 0, 1e-10, sigma)

    spike_counts = np.zeros(image.shape[3])
    for i in range(data.shape[-1]):
        s = (data[..., i] - curve[..., i]) / sigma_safe[..., 0]
        ww = s > c1
        s_prime = np.where(ww, c1 + (c2 - c1) * np.tanh((s - c1) / (c2 - c1)), s)
        spike_counts[i] = ww.sum()
        despiked_data[..., i] = curve[..., i] + s_prime * sigma[..., 0]

    despiked_image = ants.from_numpy(despiked_data)
    return ants.copy_image_info(image, despiked_image), spike_counts


def despike_time_series(image: ants.ANTsImage, threshold: float = 3.0, replacement: str = "threshold") -> tuple[ants.ANTsImage, np.ndarray]:
    """Despike a time series image by z-score thresholding."""
    data = image.numpy()
    mean = np.mean(data, axis=-1)
    std = np.std(data, axis=-1)
    spikes = np.abs(data - mean[..., np.newaxis]) > threshold * std[..., np.newaxis]
    spike_counts = np.zeros(image.shape[3])

    for i in range(data.shape[-1]):
        sl = data[..., i]
        locs = spikes[..., i]
        spike_counts[i] = locs.sum()
        if replacement == "median":
            sl[locs] = np.median(sl)
        else:
            thresh_vals = mean + np.sign(sl - mean) * threshold * std
            sl[locs] = thresh_vals[locs]
        data[..., i] = sl

    despike_image = ants.from_numpy(data)
    return ants.copy_image_info(image, despike_image), spike_counts


def copy_spatial_metadata_from_3d_to_4d(spatial_img: ants.ANTsImage, timeseries_img: ants.ANTsImage) -> ants.ANTsImage:
    """Copy spatial geometry from 3D reference to 4D timeseries."""
    new_origin = list(spatial_img.origin) + [list(timeseries_img.origin)[3]]
    new_spacing = list(spatial_img.spacing) + [list(timeseries_img.spacing)[3]]
    new_direction = timeseries_img.direction.copy()
    new_direction[:3, :3] = spatial_img.direction
    return ants.from_numpy(timeseries_img.numpy(), origin=new_origin, spacing=new_spacing, direction=new_direction)


def timeseries_transform(transform: Any, image: ants.ANTsImage, reference: ants.ANTsImage, interpolation: str = "linear") -> ants.ANTsImage:
    """Apply spatial transform to each volume in 4D timeseries."""
    if image.dimension != 4:
        raise ValueError("Input image must be 4D (X, Y, Z, T).")
    transformed_volumes = []
    for t in range(image.shape[3]):
        vol = ants.slice_image(image, 3, t)
        transformed = ants.apply_ants_transform_to_image(
            transform=transform, image=vol, reference=reference, interpolation=interpolation
        )
        transformed_volumes.append(transformed.numpy())
    transformed_array = np.stack(transformed_volumes, axis=-1)
    out_image = ants.from_numpy(transformed_array)
    out_image = ants.copy_image_info(image, out_image)
    return copy_spatial_metadata_from_3d_to_4d(reference, out_image)


def timeseries_reg(
    image: ants.ANTsImage,
    avg_b0: ants.ANTsImage,
    type_of_transform: str = "antsRegistrationSyNRepro[r]",
    total_sigma: float = 1.0,
    fdOffset: float = 2.0,
    trim: int = 0,
    output_directory: str | None = None,
    return_numpy_motion_parameters: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Correct BOLD time-series data for motion."""
    idim = image.dimension
    ishape = image.shape
    n_time_points = ishape[idim - 1]
    fd = np.zeros(n_time_points)

    if type_of_transform is None:
        return {"motion_corrected": image, "motion_parameters": None, "FD": fd}

    remove_it = False
    if output_directory is None:
        remove_it = True
        output_directory = tempfile.mkdtemp()
    output_directory_w = os.path.join(output_directory, "ts_reg")
    os.makedirs(output_directory_w, exist_ok=True)
    ofn_l = tempfile.NamedTemporaryFile(delete=False, suffix="local_deformation", dir=output_directory_w).name

    motion_parameters: list[Any] = []
    motion_corrected: list[ants.ANTsImage] = []
    mask = ants.get_mask(avg_b0)
    center_of_mass = mask.get_center_of_mass()
    npts = pow(2, idim - 1)
    myrad = np.ones(idim - 1, dtype=int).tolist()
    mask1vals = np.zeros(int(mask.sum()))
    mask1vals[round(len(mask1vals) / 2)] = 1
    mask1 = ants.make_image(mask, mask1vals)
    myoffsets = ants.get_neighborhood_in_mask(mask1, mask1, radius=myrad, spatial_info=True)["offsets"]
    mycols = list("xyz" if idim - 1 == 3 else "xy")
    useinds: list[int] = []
    for k in range(myoffsets.shape[0]):
        if abs(myoffsets[k, :]).sum() == (idim - 2):
            useinds.append(k)
        myoffsets[k, :] = myoffsets[k, :] * fdOffset / 2.0 + center_of_mass
    fdpts = pd.DataFrame(data=myoffsets[useinds, :], columns=mycols)

    counter = round(n_time_points / 10) + 1
    for k in range(n_time_points):
        if verbose and (k % counter == 0 or k == n_time_points - 1):
            print(f"{round(k / n_time_points * 100)}%.", end="", flush=True)
        temp = ants.slice_image(image, axis=idim - 1, idx=k)
        temp = ants.iMath(temp, "Normalize")
        txprefix = f"{ofn_l}{str(k % 2).zfill(4)}_"
        if temp.numpy().var() > 0:
            myrig = ants.registration(avg_b0, temp, type_of_transform="antsRegistrationSyNRepro[r]", outprefix=txprefix)
            if type_of_transform == "SyN":
                myreg = ants.registration(
                    avg_b0, temp, type_of_transform="SyNOnly", total_sigma=total_sigma,
                    initial_transform=myrig["fwdtransforms"][0], outprefix=txprefix, **kwargs
                )
            else:
                myreg = myrig
            fdpts_tx = ants.apply_transforms_to_points(idim - 1, fdpts, myrig["fwdtransforms"])
            fdpts_prev = (
                ants.apply_transforms_to_points(idim - 1, fdpts, motion_parameters[k - 1])
                if k > 0 and motion_parameters[k - 1] != "NA"
                else fdpts_tx
            )
            fd[k] = (fdpts_prev - fdpts_tx).abs().mean().sum()
            motion_parameters.append(myreg["fwdtransforms"])
        else:
            motion_parameters.append("NA")

        temp = ants.slice_image(image, axis=idim - 1, idx=k)
        if temp.numpy().var() > 0:
            img1w = ants.apply_transforms(avg_b0, temp, motion_parameters[k])
            motion_corrected.append(img1w)
        else:
            motion_corrected.append(avg_b0)

    motion_parameters = motion_parameters[trim:]
    if return_numpy_motion_parameters:
        filtered = [[s for s in sublist if s.endswith(".mat")] for sublist in motion_parameters if sublist != "NA"]
        np_params = []
        for fl in filtered:
            tx = ants.read_transform(fl[0])
            np_params.append(np.array(ants.get_ants_transform_parameters(tx)[0:9]))
        motion_parameters = np.array(np_params)

    if remove_it:
        shutil.rmtree(output_directory, ignore_errors=True)

    d4siz = list(avg_b0.shape) + [2]
    spc = list(ants.get_spacing(avg_b0)) + [float(ants.get_spacing(image)[3])]
    mydir4d = ants.get_direction(image)
    mydir4d[0:3, 0:3] = ants.get_direction(avg_b0)
    myorg = list(ants.get_origin(avg_b0)) + [0.0]
    avg_b0_4d = ants.make_image(d4siz, 0, spacing=spc, origin=myorg, direction=mydir4d)

    return {
        "motion_corrected": ants.list_to_ndimage(avg_b0_4d, motion_corrected[trim:]),
        "motion_parameters": motion_parameters,
        "FD": fd[trim:],
    }


def get_average_rsf(x: ants.ANTsImage, min_t: int = 10, max_t: int = 35) -> ants.ANTsImage:
    """Automatically generates average bold image with registration."""
    output_directory = tempfile.mkdtemp()
    ofn = os.path.join(output_directory, "w")
    bavg = ants.slice_image(x, axis=3, idx=0) * 0.0
    oavg = ants.slice_image(x, axis=3, idx=0)
    if x.shape[3] <= min_t:
        min_t = 0
    if x.shape[3] <= max_t:
        max_t = x.shape[3]
    for myidx in range(min_t, max_t):
        b0 = ants.slice_image(x, axis=3, idx=myidx)
        bavg = bavg + ants.registration(oavg, b0, "antsRegistrationSyNRepro[r]", outprefix=ofn)["warpedmovout"]
    bavg = ants.iMath(bavg, "Normalize")
    oavg = ants.image_clone(bavg)
    bavg = oavg * 0.0
    for myidx in range(min_t, max_t):
        b0 = ants.slice_image(x, axis=3, idx=myidx)
        bavg = bavg + ants.registration(oavg, b0, "antsRegistrationSyNRepro[r]", outprefix=ofn)["warpedmovout"]
    shutil.rmtree(output_directory, ignore_errors=True)
    return ants.iMath(bavg, "Normalize")


def estimate_optimal_pca_components(data: np.ndarray, variance_threshold: float = 0.80) -> int:
    """Estimate optimal number of PCA components to represent data."""
    pca = PCA()
    pca.fit(data)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    return int(np.where(cumulative_variance >= variance_threshold)[0][0] + 1)


def resting_state_fmri_networks(
    fmri: ants.ANTsImage,
    fmri_template: ants.ANTsImage,
    t1: ants.ANTsImage,
    t1segmentation: ants.ANTsImage,
    f: list[float] | None = None,
    FD_threshold: float = 5.0,
    spa: float | None = None,
    spt: float | None = None,
    nc: int | float = 5,
    outlier_threshold: float = 0.250,
    ica_components: int = 0,
    impute: bool = True,
    censor: bool = True,
    despike: float = 2.5,
    motion_as_nuisance: bool = True,
    powers: bool = False,
    upsample: float = 3.0,
    clean_tmp: float | None = None,
    paramset: str = "unset",
    verbose: bool = False,
) -> dict[str, Any]:
    """Compute resting state network correlation maps based on Power or Yeo 2023 nodes."""
    import antspynet
    import antspyt1w

    if f is None:
        f = [0.03, 0.08]

    if fmri_template is None:
        fmri_template = get_average_rsf(fmri)

    output_directory = tempfile.mkdtemp()
    output_directory_w = os.path.join(output_directory, "ts_t1_reg")
    os.makedirs(output_directory_w, exist_ok=True)
    ofnt1tx = tempfile.NamedTemporaryFile(delete=False, suffix="t1_deformation", dir=output_directory_w).name

    if upsample > 0.0:
        spc = ants.get_spacing(fmri)
        minspc = min(upsample, min(spc[0:3]))
        fmri_template = ants.resample_image(fmri_template, [minspc, minspc, minspc], interp_type=0)

    fmrispc = list(ants.get_spacing(fmri))
    if spa is None:
        spa = float(np.mean(fmrispc[0:3]))
    if spt is None:
        spt = float(fmrispc[3]) * 0.5

    if powers:
        powers_areal_mni_itk = pd.read_csv(get_data("powers_mni_itk", target_extension=".csv"))
        coords = "powers"
    else:
        powers_areal_mni_itk = pd.read_csv(
            get_data("ppmi_template_500Parcels_Yeo2011_17Networks_2023_homotopic", target_extension=".csv")
        )
        coords = "yeo_17_500_2023"

    fmri = ants.iMath(fmri, "Normalize")
    bmask = antspynet.brain_extraction(fmri_template, "bold").threshold_image(0.5, 1).iMath("FillHoles")

    corrmo = timeseries_reg(
        fmri,
        fmri_template,
        type_of_transform="antsRegistrationSyNQuickRepro[r]",
        total_sigma=0.5,
        fdOffset=2.0,
        trim=8,
        output_directory=None,
        verbose=verbose,
        syn_metric="CC",
        syn_sampling=2,
        reg_iterations=[40, 20, 5],
        return_numpy_motion_parameters=True,
    )

    despiking_count = np.zeros(corrmo["motion_corrected"].shape[3])
    if despike > 0.0:
        corrmo["motion_corrected"], despiking_count = despike_time_series_afni(corrmo["motion_corrected"], c1=despike)

    despiking_count_summary = float(despiking_count.sum() / np.prod(corrmo["motion_corrected"].shape))
    high_motion_count = int((corrmo["FD"] > FD_threshold).sum())
    high_motion_pct = float(high_motion_count / fmri.shape[3])

    mytsnr = tsnr(corrmo["motion_corrected"], bmask)
    mytsnr_thresh = float(np.quantile(mytsnr.numpy(), 0.995))
    tsnrmask = ants.threshold_image(mytsnr, 0, mytsnr_thresh).morphology("close", 2)
    bmask = bmask * tsnrmask

    und = fmri_template * bmask
    t1reg = ants.registration(und, t1, "antsRegistrationSyNQuickRepro[s]", outprefix=ofnt1tx)

    gmseg = (
        ants.threshold_image(t1segmentation, 2, 2) + ants.threshold_image(t1segmentation, 4, 4)
    ).threshold_image(1, 4).iMath("MD", 1)
    gmseg = ants.apply_transforms(und, gmseg, t1reg["fwdtransforms"], interpolator="nearestNeighbor") * bmask

    csf_and_wm = (
        ants.threshold_image(t1segmentation, 1, 1) + ants.threshold_image(t1segmentation, 3, 3)
    ).morphology("erode", 1)
    csf_and_wm = ants.apply_transforms(und, csf_and_wm, t1reg["fwdtransforms"], interpolator="nearestNeighbor") * bmask

    csf = ants.threshold_image(t1segmentation, 1, 1)
    csf = ants.apply_transforms(und, csf, t1reg["fwdtransforms"], interpolator="nearestNeighbor") * bmask
    wm = ants.threshold_image(t1segmentation, 3, 3).morphology("erode", 1)
    wm = ants.apply_transforms(und, wm, t1reg["fwdtransforms"], interpolator="nearestNeighbor") * bmask

    ch2 = ants.image_read(
        ants.get_ants_data("ch2") if powers else get_data("PPMI_template0_brain", target_extension=".nii.gz")
    )
    treg = ants.registration(ants.resample_image(t1, [1.0, 1.0, 1.0], interp_type=0), ch2, "antsRegistrationSyNQuickRepro[s]")

    if powers:
        concatx2 = treg["invtransforms"] + t1reg["invtransforms"]
        pts2bold = ants.apply_transforms_to_points(3, powers_areal_mni_itk, concatx2, whichtoinvert=(True, False, True, False))
        locations = pts2bold.iloc[:, :3].values
        ptImg = ants.make_points_image(locations, bmask, radius=2)
    else:
        concatx2 = t1reg["fwdtransforms"] + treg["fwdtransforms"]
        rsfsegfn = get_data("ppmi_template_500Parcels_Yeo2011_17Networks_2023_homotopic", target_extension=".nii.gz")
        rsfsegimg = ants.image_read(rsfsegfn)
        ptImg = ants.apply_transforms(und, rsfsegimg, concatx2, interpolator="nearestNeighbor") * bmask
        pts2bold = powers_areal_mni_itk

    tr = float(ants.get_spacing(corrmo["motion_corrected"])[3])
    smth = (spa, spa, spa, spt)
    simg = ants.smooth_image(corrmo["motion_corrected"], smth, sigma_in_physical_coordinates=True)

    hlinds = [idx for idx, val in enumerate(corrmo["FD"]) if val > FD_threshold]
    if 0.0 < outlier_threshold < 1.0:
        _, hlinds2 = loop_timeseries_censoring(corrmo["motion_corrected"], threshold=outlier_threshold, verbose=verbose)
        hlinds.extend(list(hlinds2))
    hlinds = list(set(hlinds))

    globalmat = ants.timeseries_to_matrix(corrmo["motion_corrected"], bmask)
    globalsignal = np.nanmean(globalmat, axis=1)

    nc_wm = nc_csf = nc
    if nc < 1:
        compcorquantile = 0.50
        def _get_cc_matrix(img: ants.ANTsImage, msk: ants.ANTsImage) -> np.ndarray:
            im = ants.timeseries_to_matrix(img, msk)
            std_m = np.std(im, axis=0)
            thresh = float(np.percentile(std_m, int(compcorquantile * 100)))
            ts_msk = ants.threshold_image(ants.make_image(msk, std_m), thresh, float(std_m.max()))
            return ants.timeseries_to_matrix(img, ts_msk)

        nc_wm = estimate_optimal_pca_components(_get_cc_matrix(corrmo["motion_corrected"], wm), variance_threshold=float(nc))
        nc_csf = estimate_optimal_pca_components(_get_cc_matrix(corrmo["motion_corrected"], csf), variance_threshold=float(nc))

    mycompcor_csf = ants.compcor(corrmo["motion_corrected"], ncompcor=int(nc_csf), quantile=0.50, mask=csf, filter_type="polynomial", degree=2)
    mycompcor_wm = ants.compcor(corrmo["motion_corrected"], ncompcor=int(nc_wm), quantile=0.50, mask=wm, filter_type="polynomial", degree=2)
    nuisance = np.c_[mycompcor_csf["components"], mycompcor_wm["components"]]

    if motion_as_nuisance:
        deriv = np.vstack((np.zeros((1, corrmo["motion_parameters"].shape[1])), np.diff(corrmo["motion_parameters"], axis=0)))
        nuisance = np.c_[nuisance, corrmo["motion_parameters"], deriv]

    if ica_components > 0:
        ica = FastICA(n_components=ica_components, max_iter=10000, tol=0.001, random_state=42)
        nuisance = np.c_[nuisance, ica.fit_transform(ants.timeseries_to_matrix(corrmo["motion_corrected"], csf_and_wm))]

    nuisance = np.c_[nuisance, globalsignal]
    simgimp = impute_timeseries(simg, hlinds, method="linear") if impute and hlinds else simg
    myfalff = alff_image(simgimp, bmask, flo=f[0], fhi=f[1], nuisance=nuisance)

    if f[0] > 0 and f[1] < 1.0:
        nuisance = ants.bandpass_filter_matrix(nuisance, tr=tr, lowf=f[0], highf=f[1])
        gmat = ants.timeseries_to_matrix(simg, bmask)
        gmat = ants.bandpass_filter_matrix(gmat, tr=tr, lowf=f[0], highf=f[1])
        simg = ants.matrix_to_timeseries(simg, gmat, bmask)

    if hlinds and censor:
        nuisance = remove_elements_from_numpy_array(nuisance, hlinds)
        simg = remove_volumes_from_timeseries(simg, hlinds)

    gmmat = ants.timeseries_to_matrix(simg, bmask)
    gmmat = ants.regress_components(gmmat, nuisance)
    simg = ants.matrix_to_timeseries(simg, gmmat, bmask)

    outdict: dict[str, Any] = {
        "paramset": paramset,
        "upsampling": upsample,
        "coords": coords,
        "dfnname": "DefaultMode",
        "meanBold": und,
    }

    n_points = int(pts2bold["ROI"].max()) if powers else int(ptImg.max())
    mean_roi = np.zeros([simg.shape[3], n_points])
    roi_names: list[str] = []

    for i in range(n_points):
        net_label = re.sub(r"[\s\-/]", "", pts2bold.loc[i, "SystemName"])
        roi_label = f"ROI{pts2bold.loc[i, 'ROI']}_{net_label}"
        roi_names.append(roi_label)
        if powers:
            pt_img = ants.make_points_image(pts2bold.iloc[[i], :3].values, bmask, radius=1).threshold_image(1, 1e9)
        else:
            pt_img = ants.threshold_image(ptImg, pts2bold.loc[i, "ROI"], pts2bold.loc[i, "ROI"])
        if pt_img.sum() > 0:
            mean_roi[:, i] = ants.timeseries_to_matrix(simg, pt_img).mean(axis=1)

    cor_mat = np.corrcoef(mean_roi, rowvar=False)
    output_mat = pd.DataFrame(cor_mat, columns=roi_names)
    output_mat["ROIs"] = roi_names
    outdict["fullCorrMat"] = output_mat

    networks = powers_areal_mni_itk["SystemName"].unique()
    numofnets = [3, 5, 6, 7, 8, 9, 10, 11, 13] if powers else list(range(len(networks)))

    for mynet in numofnets:
        netname = re.sub(r"[\s\-]", "", networks[mynet])
        ww = np.where(powers_areal_mni_itk["SystemName"] == networks[mynet])[0]
        if powers:
            dfn_img = ants.make_points_image(pts2bold.iloc[ww, :3].values, bmask, radius=1).threshold_image(1, 1e9)
        else:
            dfn_img = ants.mask_image(ptImg, ptImg, level=pts2bold["ROI"][pts2bold["SystemName"] == networks[mynet]], binarize=True)
        if dfn_img.max() >= 1:
            dfnmat = ants.timeseries_to_matrix(simg, ants.threshold_image(dfn_img, 1, dfn_img.max()))
            dfnsignal = np.nanmean(dfnmat, axis=1)
            gmmat_dfn_corr = np.zeros(gmmat.shape[1])
            if np.count_nonzero(np.isnan(dfnsignal)) == 0:
                for k in range(gmmat.shape[1]):
                    if np.count_nonzero(np.isnan(gmmat[:, k])) == 0:
                        gmmat_dfn_corr[k] = pearsonr(dfnsignal, gmmat[:, k])[0]
            corr_img = ants.make_image(bmask, gmmat_dfn_corr)
            outdict[netname] = corr_img * gmseg
        else:
            outdict[netname] = None

    a_mat = np.zeros((len(numofnets), len(numofnets)))
    a_wide = np.zeros((1, len(numofnets) * len(numofnets)))
    newnames: list[str] = []
    newnames_wide: list[str] = []
    ct = 0

    for i in range(len(numofnets)):
        netname_i = re.sub(r"[\s\-]", "", networks[numofnets[i]])
        newnames.append(netname_i)
        ww = np.where(powers_areal_mni_itk["SystemName"] == networks[numofnets[i]])[0]
        dfn_img = (
            ants.make_points_image(pts2bold.iloc[ww, :3].values, bmask, radius=1).threshold_image(1, 1e9)
            if powers
            else ants.mask_image(ptImg, ptImg, level=pts2bold["ROI"][pts2bold["SystemName"] == networks[numofnets[i]]], binarize=True)
        )
        for j in range(len(numofnets)):
            netname_j = re.sub(r"[\s\-]", "", networks[numofnets[j]])
            newnames_wide.append(f"{netname_i}_2_{netname_j}")
            a_mat[i, j] = 0.0
            if dfn_img is not None and netname_j in outdict and outdict[netname_j] is not None:
                subbit = dfn_img == 1
                if subbit.sum() > 0:
                    a_mat[i, j] = float(outdict[netname_j][subbit].mean())
            a_wide[0, ct] = a_mat[i, j]
            ct += 1

    df_a = pd.DataFrame(a_mat, columns=newnames)
    df_a["networks"] = newnames
    df_a_wide = pd.DataFrame(a_wide, columns=newnames_wide)

    outdict["corr"] = df_a
    outdict["corr_wide"] = df_a_wide
    outdict["fmri_template"] = fmri_template
    outdict["brainmask"] = bmask
    outdict["gmmask"] = gmseg
    outdict["alff"] = myfalff["alff"]
    outdict["falff"] = myfalff["falff"]
    outdict["alff_mean"] = float((myfalff["alff"][myfalff["alff"] != 0]).mean())
    outdict["alff_sd"] = float((myfalff["alff"][myfalff["alff"] != 0]).std())
    outdict["falff_mean"] = float((myfalff["falff"][myfalff["falff"] != 0]).mean())
    outdict["falff_sd"] = float((myfalff["falff"][myfalff["falff"] != 0]).std())

    perafimg = PerAF(simgimp, bmask)
    for k in range(n_points):
        anatname = pts2bold["AAL"][k]
        anatname_clean = re.sub("_", "", anatname) if isinstance(anatname, str) else "Unk"
        kk = f"{k:0>3}_" if powers else f"{k % int(n_points / 2):0>3}_"
        localsel = ptImg == k
        if localsel.sum() > 0:
            outdict[f"falffPoint{kk}{anatname_clean}"] = float((outdict["falff"][localsel]).mean())
            outdict[f"alffPoint{kk}{anatname_clean}"] = float((outdict["alff"][localsel]).mean())
            outdict[f"perafPoint{kk}{anatname_clean}"] = float((perafimg[localsel]).mean())
        else:
            outdict[f"falffPoint{kk}{anatname_clean}"] = math.nan
            outdict[f"alffPoint{kk}{anatname_clean}"] = math.nan
            outdict[f"perafPoint{kk}{anatname_clean}"] = math.nan

    shutil.rmtree(output_directory, ignore_errors=True)

    if not powers:
        if "DefaultA" in outdict and "DefaultB" in outdict and "DefaultC" in outdict:
            outdict["DefaultMode"] = outdict["DefaultA"] + outdict["DefaultB"] + outdict["DefaultC"]
        if "VisCent" in outdict and "VisPeri" in outdict:
            outdict["Visual"] = outdict["VisCent"] + outdict["VisPeri"]

    outdict["motion_corrected"] = corrmo["motion_corrected"]
    outdict["nuisance"] = pd.DataFrame(nuisance)
    outdict["PerAF"] = perafimg
    outdict["tsnr"] = mytsnr
    outdict["ssnr"] = slice_snr(corrmo["motion_corrected"], csf_and_wm, gmseg)
    outdict["dvars"] = dvars(corrmo["motion_corrected"], gmseg)
    outdict["bandpass_freq_0"] = f[0]
    outdict["bandpass_freq_1"] = f[1]
    outdict["censor"] = int(censor)
    outdict["spatial_smoothing"] = spa
    outdict["outlier_threshold"] = outlier_threshold
    outdict["FD_threshold"] = FD_threshold
    outdict["high_motion_count"] = high_motion_count
    outdict["high_motion_pct"] = high_motion_pct
    outdict["despiking_count_summary"] = despiking_count_summary
    outdict["FD_max"] = float(corrmo["FD"].max())
    outdict["FD_mean"] = float(corrmo["FD"].mean())
    outdict["FD_sd"] = float(corrmo["FD"].std())
    outdict["bold_evr"] = antspyt1w.patch_eigenvalue_ratio(und, 512, [16, 16, 16], evdepth=0.9, mask=bmask)
    outdict["n_outliers"] = len(hlinds)
    outdict["nc_wm"] = int(nc_wm)
    outdict["nc_csf"] = int(nc_csf)
    outdict["minutes_original_data"] = float((tr * fmri.shape[3]) / 60.0)
    outdict["minutes_censored_data"] = float((tr * simg.shape[3]) / 60.0)
    return convert_np_in_dict(outdict)
