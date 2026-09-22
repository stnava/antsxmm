from __future__ import annotations

import math
import os
import re
import warnings
from typing import Any
import numpy as np
import pandas as pd
import ants
import nibabel as nib


def get_antsimage_keys(dictionary: dict[str, Any]) -> list[str]:
    """Return the keys of the dictionary where the values are ANTsImages."""
    return [key for key, value in dictionary.items() if isinstance(value, ants.core.ants_image.ANTsImage)]


def dict_to_dataframe(
    data_dict: dict[str, Any],
    convert_lists: bool = True,
    convert_arrays: bool = True,
    convert_images: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Convert a dictionary to a pandas DataFrame, excluding non-scalar or non-convertible items."""
    processed_data: dict[str, Any] = {}

    def _mean_of_list(lst: list[Any]) -> float | None:
        if not lst:
            return 0.0
        all_numeric = all(isinstance(item, (int, float)) for item in lst)
        if all_numeric:
            return sum(lst) / len(lst)
        return None

    for key, value in data_dict.items():
        if isinstance(value, (int, float, str, bool)):
            processed_data[key] = [value]
        elif isinstance(value, list) and all(isinstance(item, (int, float, str, bool)) for item in value) and convert_lists:
            meanvalue = _mean_of_list(value)
            newkey = f"{key}_mean"
            if verbose:
                print(f" Key {key} is list with mean {meanvalue} -> {newkey}")
            if newkey not in data_dict and convert_lists:
                processed_data[newkey] = meanvalue
        elif isinstance(value, np.ndarray) and all(isinstance(item, (int, float, str, bool)) for item in value) and convert_arrays:
            meanvalue = float(value.mean())
            newkey = f"{key}_mean"
            if verbose:
                print(f" Key {key} is nparray with mean {meanvalue} -> {newkey}")
            if newkey not in data_dict:
                processed_data[newkey] = meanvalue
        elif isinstance(value, ants.core.ants_image.ANTsImage) and convert_images:
            meanvalue = float(value.mean())
            newkey = f"{key}_mean"
            if verbose:
                print(f" Key {key} is antsimage with mean {meanvalue} -> {newkey}")
            if newkey not in data_dict:
                processed_data[newkey] = meanvalue
            elif verbose:
                print(f" Key {key} is antsimage with mean {meanvalue} but {newkey} already exists")

    return pd.DataFrame.from_dict(processed_data)


def image_write_with_thumbnail(
    x: ants.ANTsImage,
    fn: str,
    y: ants.ANTsImage | None = None,
    thumb: bool = True,
) -> None:
    """Write an ANTsImage and optionally an orthopedic thumbnail PNG."""
    ants.image_write(x, fn)
    if not thumb or x.components > 1:
        return

    if x.dimension == 3:
        thumb_fn = re.sub(r"\.nii\.gz$", "_3dthumb.png", fn)
        try:
            if y is None:
                ants.plot_ortho(x, crop=True, filename=thumb_fn, flat=True, xyz_lines=False, orient_labels=False, xyz_pad=0)
            else:
                ants.plot_ortho(y, x, crop=True, filename=thumb_fn, flat=True, xyz_lines=False, orient_labels=False, xyz_pad=0)
        except Exception:
            pass
    elif x.dimension == 4:
        thumb_fn = re.sub(r"\.nii\.gz$", "_4dthumb.png", fn)
        nslices = x.shape[3]
        sl = int(min(round(nslices * 0.5), nslices - 1))
        xview = ants.slice_image(x, axis=3, idx=sl)
        try:
            if y is None:
                ants.plot_ortho(xview, crop=True, filename=thumb_fn, flat=True, xyz_lines=False, orient_labels=False, xyz_pad=0)
            elif y.dimension == 3:
                ants.plot_ortho(y, xview, crop=True, filename=thumb_fn, flat=True, xyz_lines=False, orient_labels=False, xyz_pad=0)
        except Exception:
            pass


def shorten_pymm_names(x: str, verbose: bool = False) -> str:
    """Shortens pymm names by applying a series of regex substitutions."""
    xx = x.lower()
    xx = xx.replace("cit168_description_", "cit_")
    xx = xx.replace("dkt_subcortical_description_", "dkts_")
    xx = xx.replace("dkt_description_", "dkt_")
    xx = xx.replace("volume", "vol")
    xx = xx.replace("vol", "")
    xx = xx.replace("hemisphere", "hemi")
    xx = xx.replace("matter", "mat")
    xx = xx.replace("cortex", "ctx")
    xx = xx.replace("cerebellum", "cer")
    xx = xx.replace("cerebral", "cer")
    xx = xx.replace("superior", "sup")
    xx = xx.replace("middle", "mid")
    xx = xx.replace("inferior", "inf")
    xx = xx.replace("anterior", "ant")
    xx = xx.replace("posterior", "pos")
    xx = xx.replace("medial", "med")
    xx = xx.replace("lateral", "lat")
    xx = xx.replace("ventral", "ven")
    xx = xx.replace("dorsal", "dor")
    xx = xx.replace("central", "cen")
    xx = xx.replace("nucleus", "nuc")
    xx = xx.replace("substantia", "sub")
    xx = xx.replace("nigra", "nig")
    xx = xx.replace("perforated", "perf")
    xx = xx.replace("gyrus", "gyr")
    xx = xx.replace("sulcus", "sul")
    xx = xx.replace("pars", "prs")
    xx = xx.replace("operculum", "opc")
    xx = xx.replace("hippocampus", "hip")
    xx = xx.replace("thalamus", "tha")
    xx = xx.replace("striatum", "str")
    xx = xx.replace("putamen", "put")
    xx = xx.replace("accumbens", "acc")
    xx = xx.replace("pallidum", "pal")
    xx = xx.replace("claustrum", "cla")
    xx = xx.replace("amygdala", "amy")
    xx = xx.replace("septum", "sep")
    xx = xx.replace("olfactory", "olf")
    xx = xx.replace("basal", "bas")
    xx = xx.replace("forebrain", "fb")
    xx = xx.replace("temporal", "tem")
    xx = xx.replace("frontal", "fro")
    xx = xx.replace("parietal", "par")
    xx = xx.replace("occipital", "occ")
    xx = xx.replace("cingulate", "cin")
    xx = xx.replace("callosum", "cal")
    xx = xx.replace("internal", "int")
    xx = xx.replace("external", "ext")
    xx = xx.replace("pole", "pol")
    xx = xx.replace("orbital", "orb")
    xx = xx.replace("insula", "ins")
    xx = xx.replace("operculum", "opc")
    xx = xx.replace("triangularis", "tri")
    xx = xx.replace("orbitalis", "orb")
    xx = xx.replace("opercularis", "opc")
    xx = xx.replace("_description_", "_")
    xx = xx.replace("left", "l")
    xx = xx.replace("right", "r")
    xx = xx.replace("_", "")
    xx = xx.replace("-", "")
    xx = xx.replace(" ", "")
    if len(xx) > 18:
        xx = xx[:18]
    if verbose:
        print(f"{x} -> {xx}")
    return xx


def shorten_pymm_names2(x: str) -> str:
    """Shortens JHU-style tract/region names by regex replacements."""
    substitutions = [
        ("anterior.limb.of.internal.capsule", "alintcap"),
        ("cingulum.cingulate.gyrus", "cinggyrus"),
        ("cingulum.hippocampus", "cinghip"),
        ("corticospinal.tract", "cst"),
        ("inferior.cerebellar.peduncle", "infcerped"),
        ("middle.cerebellar.peduncle", "midcerped"),
        ("superior.cerebellar.peduncle", "supcerped"),
        ("posterior.limb.of.internal.capsule", "plintcap"),
        ("retrolenticular.part.of.internal.capsule", "retlintcap"),
        ("superior.corona.radiata", "supcorrad"),
        ("anterior.corona.radiata", "antcorrad"),
        ("posterior.corona.radiata", "poscorrad"),
        ("posterior.thalamic.radiation", "postharad"),
        ("sagittal.stratum", "sagstratum"),
        ("external.capsule", "extcap"),
        ("superior.longitudinal.fasciculus", "suplongfasc"),
        ("superior.fronto.occipital.fasciculus", "supfroccfasc"),
        ("uncinate.fasciculus", "uncfasc"),
        ("pontine.crossing.tract", "pct"),
        ("of.internal.capsule", ".int.cap"),
        ("fornix.cres.stria.terminalis", "fornix."),
        ("capsule", ""),
        (r"and\.inf\.frnt\.occ\.fasciculus\.", ""),
        (r"crossing\.tract\.a\.part\.of\.mcp\.", ""),
    ]
    cur = x.lower()
    for pattern, replacement in substitutions:
        cur = re.sub(pattern, replacement, cur, flags=re.IGNORECASE)
    return cur[:40]


def hierarchical_modality_summary(
    target_image: ants.ANTsImage,
    hier: dict[str, Any],
    transformlist: list[str],
    modality_name: str,
    return_keys: list[str] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Use output of antspyt1w.hierarchical to summarize a modality in wide format."""
    import antspyt1w

    if return_keys is None:
        return_keys = ["Mean", "Volume"]
    dfout = pd.DataFrame()

    def _helper(target_img: ants.ANTsImage, seg: ants.ANTsImage, mytx: list[str], mapname: str, modname: str, mydf: pd.DataFrame) -> pd.DataFrame:
        target_mask = ants.image_clone(target_img) * 0.0
        target_mask[target_img != 0] = 1.0
        cortmapped = ants.apply_transforms(target_img, seg, mytx, interpolator="nearestNeighbor") * target_mask
        mapped = antspyt1w.map_intensity_to_dataframe(
            mapname,
            target_img,
            cortmapped,
            labels=np.unique(seg.numpy()),
            return_keys=return_keys,
            modality_name=modname,
        )
        if mydf.shape[0] == 0:
            return mapped
        return pd.concat([mydf, mapped], axis=1, ignore_index=False)

    dfout = _helper(target_image, hier["dkt_parc"]["dkt_cortex"], transformlist, "dkt_description", modality_name, dfout)
    dfout = _helper(target_image, hier["dkt_parc"]["dkt_subcortical"], transformlist, "dkt_subcortical_description", modality_name, dfout)
    dfout = _helper(target_image, hier["cit168lab"], transformlist, "cit168_description", modality_name, dfout)
    return dfout


def to_nibabel(img: ants.ANTsImage) -> nib.Nifti1Image:
    """Convert an ANTsPy image to a Nibabel Nifti1Image in-memory, using correct spatial affine."""
    array_data = img.numpy()
    affine = ants_to_nibabel_affine(img)
    return nib.Nifti1Image(array_data, affine)


def ants_to_nibabel_affine(ants_img: ants.ANTsImage) -> np.ndarray:
    """Convert an ANTsPy image (in LPS space) to a Nibabel-compatible affine (in RAS space)."""
    spatial_dim = ants_img.dimension
    spacing = np.array(ants_img.spacing)
    origin = np.array(ants_img.origin)
    direction = np.array(ants_img.direction).reshape((spatial_dim, spatial_dim))
    affine_linear = direction @ np.diag(spacing)
    affine = np.eye(4)
    affine[:spatial_dim, :spatial_dim] = affine_linear
    affine[:spatial_dim, 3] = origin
    affine[3, 3] = 1.0
    lps_to_ras = np.diag([-1, -1, 1, 1])
    affine = lps_to_ras @ affine
    return affine


def apply_transforms_mixed_interpolation(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    transformlist: list[str],
    interpolator: str = "linear",
    imagetype: int = 0,
    whichtoinvert: list[bool] | None = None,
    mask: ants.ANTsImage | None = None,
    **kwargs: Any,
) -> ants.ANTsImage:
    """Apply ANTs transforms with linear interpolation inside mask, nearest neighbor outside."""
    if mask is None:
        raise ValueError("A binary `mask` image must be provided.")

    interp_linear = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=transformlist,
        interpolator=interpolator,
        imagetype=imagetype,
        whichtoinvert=whichtoinvert,
        **kwargs,
    )
    interp_nn = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=transformlist,
        interpolator="nearestNeighbor",
        imagetype=imagetype,
        whichtoinvert=whichtoinvert,
        **kwargs,
    )
    if not ants.image_physical_space_consistency(interp_linear, mask):
        mask = ants.resample_image_to_target(mask, interp_linear, interp_type="nearestNeighbor")
    return (interp_linear * mask) + (interp_nn * (1.0 - mask))


def convert_np_in_dict(data_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy float/int scalars in dictionary to standard Python float/int."""
    converted_dict: dict[str, Any] = {}
    for key, value in data_dict.items():
        if isinstance(value, (np.float32, np.float64)):
            converted_dict[key] = float(value)
        elif isinstance(value, (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64)):
            converted_dict[key] = int(value)
        else:
            converted_dict[key] = value
    return converted_dict


def segment_timeseries_by_meanvalue(image: ants.ANTsImage, quantile: float = 0.995) -> dict[str, list[int]]:
    """Identify indices of a time series with higher and lower mean intensities across volumes."""
    ishape = image.shape
    lastdim = len(ishape) - 1
    meanvalues = [ants.slice_image(image, axis=lastdim, idx=x).mean() for x in range(ishape[lastdim])]
    myhiq = np.quantile(meanvalues, quantile)
    myloq = np.quantile(meanvalues, 1.0 - quantile)
    lowerindices: list[int] = []
    higherindices: list[int] = []
    for x, mv in enumerate(meanvalues):
        hiabs = abs(mv - myhiq)
        loabs = abs(mv - myloq)
        if hiabs < loabs:
            higherindices.append(x)
        else:
            lowerindices.append(x)
    return {"lowermeans": lowerindices, "highermeans": higherindices}


def segment_timeseries_by_bvalue(bvals: np.ndarray | list[float]) -> dict[str, list[int]]:
    """Categorize b-values into non-zero and b0 indices."""
    threshold = 1e-12
    bvals_arr = np.asarray(bvals)
    lowermeans = list(np.where(bvals_arr > threshold)[0])
    highermeans = list(np.where(bvals_arr <= threshold)[0])
    if len(highermeans) == 0:
        minval = float(np.min(bvals_arr))
        lowermeans = list(np.where(bvals_arr > minval)[0])
        highermeans = list(np.where(bvals_arr <= minval)[0])
    return {"largerbvals": lowermeans, "lowbvals": highermeans}


def tsnr(x: ants.ANTsImage, mask: ants.ANTsImage, indices: list[int] | None = None) -> ants.ANTsImage:
    """3D temporal SNR image from a 4D time series image."""
    m_mat = ants.timeseries_to_matrix(x, mask)
    m_mat = m_mat - m_mat.min()
    m_max = m_mat.max()
    if m_max > 0:
        m_mat = m_mat / m_max
    if indices is not None:
        m_mat = m_mat[indices, :]
    std_m = np.std(m_mat, axis=0)
    std_m[np.isnan(std_m)] = 0
    return ants.make_image(mask, std_m)


def dvars(x: ants.ANTsImage, mask: ants.ANTsImage, indices: list[int] | None = None) -> np.ndarray:
    """DVARS on a 4D time series image."""
    m_mat = ants.timeseries_to_matrix(x, mask)
    m_mat = m_mat - m_mat.min()
    m_max = m_mat.max()
    if m_max > 0:
        m_mat = m_mat / m_max
    if indices is not None:
        m_mat = m_mat[indices, :]
    n_pts = m_mat.shape[0]
    dvars_arr = np.zeros(n_pts)
    for i in range(1, n_pts):
        vecdiff = m_mat[i - 1, :] - m_mat[i, :]
        dvars_arr[i] = np.sqrt(np.mean(vecdiff * vecdiff))
    if n_pts > 1:
        dvars_arr[0] = float(np.mean(dvars_arr))
    return dvars_arr


def slice_snr(
    x: ants.ANTsImage,
    background_mask: ants.ANTsImage,
    foreground_mask: ants.ANTsImage,
    indices: list[int] | None = None,
) -> np.ndarray:
    """Slice-wise SNR on a time series image."""
    xuse = ants.iMath(x, "Normalize")
    mb = ants.timeseries_to_matrix(xuse, background_mask)
    mf = ants.timeseries_to_matrix(xuse, foreground_mask)
    if indices is not None:
        mb = mb[indices, :]
        mf = mf[indices, :]
    ssnr = np.zeros(mb.shape[0])
    for i in range(mb.shape[0]):
        b_std = float(mb[i, :].std())
        ssnr[i] = float(mf[i, :].mean()) / b_std if b_std > 0 else 0.0
    ssnr[np.isnan(ssnr)] = 0
    return ssnr


def middle_slice_snr(x: ants.ANTsImage, background_dilation: int = 5) -> float:
    """Estimate SNR in 2D mid-slice from a 3D image."""
    xshp = x.shape
    xmidslice = ants.slice_image(x, 2, int(xshp[2] / 2))
    xmidslice = ants.iMath(xmidslice - xmidslice.min(), "Normalize")
    xmidslice = ants.n3_bias_field_correction(xmidslice)
    xmidslice = ants.n3_bias_field_correction(xmidslice)
    xmidslicemask = ants.threshold_image(xmidslice, "Otsu", 1).morphology("close", 2).iMath("FillHoles")
    xbkgmask = ants.iMath(xmidslicemask, "MD", background_dilation) - xmidslicemask
    signal = float((xmidslice[xmidslicemask == 1]).mean())
    noise = float((xmidslice[xbkgmask == 1]).std())
    return signal / noise if noise > 0 else 0.0


def foreground_background_snr(x: ants.ANTsImage, background_dilation: int = 10, erode_foreground: bool = False) -> float:
    """Estimate SNR in an image using foreground/background segmentation."""
    xbc = ants.iMath(x - x.min(), "Normalize")
    xbc = ants.n3_bias_field_correction(xbc)
    xmask = ants.threshold_image(xbc, "Otsu", 1).morphology("close", 2).iMath("FillHoles")
    xbkgmask = ants.iMath(xmask, "MD", background_dilation) - xmask
    fgmask = xmask
    if erode_foreground:
        fgmask = ants.iMath(xmask, "ME", background_dilation)
        xbkgmask = xmask - fgmask
    signal = float((xbc[fgmask == 1]).mean())
    noise = float((xbc[xbkgmask == 1]).std())
    return signal / noise if noise > 0 else 0.0


def quantile_snr(
    x: ants.ANTsImage,
    lowest_quantile: float = 0.01,
    low_quantile: float = 0.1,
    high_quantile: float = 0.5,
    highest_quantile: float = 0.95,
) -> float:
    """Estimate SNR in an image using intensity quantiles."""
    xbc = ants.iMath(x - x.min(), "Normalize")
    xbc = ants.n3_bias_field_correction(xbc)
    xbc = ants.iMath(xbc - xbc.min(), "Normalize")
    y = xbc.numpy()
    y_pos = y[y > 0]
    if len(y_pos) == 0:
        return 0.0
    ylowest = float(np.quantile(y_pos, lowest_quantile))
    ylo = float(np.quantile(y_pos, low_quantile))
    yhi = float(np.quantile(y_pos, high_quantile))
    yhiest = float(np.quantile(y_pos, highest_quantile))
    xbkgmask = ants.threshold_image(xbc, ylowest, ylo)
    fgmask = ants.threshold_image(xbc, yhi, yhiest)
    signal = float((xbc[fgmask == 1]).mean())
    noise = float((xbc[xbkgmask == 1]).std())
    return signal / noise if noise > 0 else 0.0


def mask_snr(x: ants.ANTsImage, background_mask: ants.ANTsImage, foreground_mask: ants.ANTsImage, bias_correct: bool = True) -> float:
    """Estimate SNR using user-defined foreground and background masks."""
    if foreground_mask.sum() <= 1 or background_mask.sum() <= 1:
        return 0.0
    xbc = ants.iMath(x - x.min(), "Normalize")
    if bias_correct:
        xbc = ants.n3_bias_field_correction(xbc)
    xbc = ants.iMath(xbc - xbc.min(), "Normalize")
    signal = float((xbc[foreground_mask == 1]).mean())
    noise = float((xbc[background_mask == 1]).std())
    return signal / noise if noise > 0 else 0.0


def crop_mcimage(x: ants.ANTsImage, mask: ants.ANTsImage, padder: int | None = None) -> ants.ANTsImage:
    """Crop a time series (4D) image by a 3D mask."""
    cropmask = ants.crop_image(mask, mask)
    myorig = list(ants.get_origin(cropmask))
    if len(x.shape) > 3:
        myorig.append(ants.get_origin(x)[3])
        croplist: list[ants.ANTsImage] = []
        for k in range(x.shape[3]):
            temp = ants.slice_image(x, axis=3, idx=k)
            temp = ants.crop_image(temp, mask)
            if padder is not None:
                temp = ants.pad_image(temp, pad_width=padder)
            croplist.append(temp)
        temp_img = ants.list_to_ndimage(x, croplist)
        temp_img.set_origin(myorig)
        return temp_img
    return ants.crop_image(x, mask)


def flatten_time_series(time_series: np.ndarray) -> np.ndarray:
    """Flatten a 4D time series into a 2D array (n_time_points x n_voxels)."""
    n_volumes = time_series.shape[3]
    return time_series.reshape(-1, n_volumes).T


def calculate_loop_scores_full(flattened_series: np.ndarray, n_neighbors: int = 20, verbose: bool = True) -> np.ndarray:
    """Calculate Local Outlier Probabilities for each volume."""
    from PyNomaly import loop
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    flattened_series = np.nan_to_num(flattened_series, nan=0)
    scaler = StandardScaler()
    data = scaler.fit_transform(flattened_series)
    data = np.nan_to_num(data, nan=0)
    if n_neighbors > int(flattened_series.shape[0] / 2.0):
        n_neighbors = max(1, int(flattened_series.shape[0] / 2.0))
    neigh = NearestNeighbors(n_neighbors=n_neighbors, metric="minkowski")
    neigh.fit(data)
    d, idx = neigh.kneighbors(data, return_distance=True)
    m = loop.LocalOutlierProbability(distance_matrix=d, neighbor_matrix=idx, n_neighbors=n_neighbors).fit()
    return m.local_outlier_probabilities[:]


def calculate_loop_scores(
    flattened_series: np.ndarray,
    n_neighbors: int = 20,
    n_features_sample: int | float = 0.02,
    n_feature_repeats: int = 5,
    seed: int = 42,
    use_approx_knn: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """Memory-efficient LoOP score estimation with feature subsampling."""
    from PyNomaly import loop
    from sklearn.preprocessing import StandardScaler

    try:
        from pynndescent import NNDescent
        has_nn_descent = True
    except ImportError:
        has_nn_descent = False

    rng = np.random.default_rng(seed)
    X = np.nan_to_num(flattened_series, nan=0).astype(np.float32)
    n_samples, n_features = X.shape

    if isinstance(n_features_sample, float):
        if 0 < n_features_sample <= 1.0:
            n_features_sample = max(1, int(n_features_sample * n_features))
        else:
            raise ValueError("If float, n_features_sample must be in (0, 1].")

    n_features_sample = min(n_features, int(n_features_sample))
    if n_neighbors >= n_samples:
        n_neighbors = max(1, n_samples // 2)

    loop_scores = []
    for rep in range(n_feature_repeats):
        feature_idx = rng.choice(n_features, n_features_sample, replace=False)
        X_sub = X[:, feature_idx]
        scaler = StandardScaler(copy=False)
        X_sub = scaler.fit_transform(X_sub)
        X_sub = np.nan_to_num(X_sub, nan=0)

        if use_approx_knn and has_nn_descent and n_samples > 1000:
            ann = NNDescent(X_sub, n_neighbors=n_neighbors, random_state=seed + rep)
            indices, dists = ann.neighbor_graph
        else:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=n_neighbors)
            nn.fit(X_sub)
            dists, indices = nn.kneighbors(X_sub)

        model = loop.LocalOutlierProbability(
            distance_matrix=dists,
            neighbor_matrix=indices,
            n_neighbors=n_neighbors,
        ).fit()
        loop_scores.append(model.local_outlier_probabilities[:])

    return np.mean(np.stack(loop_scores), axis=0)


def remove_elements_from_numpy_array(original_array: np.ndarray | None, indices_to_remove: list[int] | np.ndarray) -> np.ndarray | None:
    """Remove specified elements or rows from a numpy array."""
    if original_array is None:
        return None
    if original_array.ndim == 1:
        return np.delete(original_array, indices_to_remove)
    elif original_array.ndim == 2:
        return np.delete(original_array, indices_to_remove, axis=0)
    raise ValueError("original_array must be either 1D or 2D.")


def remove_volumes_from_timeseries(time_series: ants.ANTsImage, volumes_to_remove: list[int] | np.ndarray) -> ants.ANTsImage:
    """Remove specified volumes from a 4D time series."""
    if not isinstance(time_series, ants.core.ants_image.ANTsImage):
        raise ValueError("time_series must be an ANTsImage.")
    if time_series.dimension != 4:
        raise ValueError("time_series must be a 4D image.")
    remove_set = set(volumes_to_remove)
    volumes_to_keep = [i for i in range(time_series.shape[3]) if i not in remove_set]
    filtered_time_series = ants.from_numpy(time_series.numpy()[..., volumes_to_keep])
    return ants.copy_image_info(time_series, filtered_time_series)


def remove_elements_from_list(original_list: list[Any], elements_to_remove: list[Any]) -> list[Any]:
    """Remove specified elements from a list."""
    remove_set = set(elements_to_remove)
    return [element for element in original_list if element not in remove_set]


def impute_timeseries(
    time_series: ants.ANTsImage,
    volumes_to_impute: list[int],
    method: str = "linear",
    verbose: bool = False,
) -> ants.ANTsImage:
    """Impute specified volumes from a time series with interpolated values."""
    if not isinstance(time_series, ants.core.ants_image.ANTsImage):
        raise ValueError("time_series must be an ANTsImage.")
    if time_series.dimension != 4:
        raise ValueError("time_series must be a 4D image.")

    time_series_np = time_series.numpy()
    total_volumes = time_series_np.shape[3]
    volumes_not_to_impute = [i for i in range(total_volumes) if i not in set(volumes_to_impute)]
    if not volumes_not_to_impute:
        return time_series

    min_valid_index = min(volumes_not_to_impute)
    max_valid_index = max(volumes_not_to_impute)

    for vol_idx in volumes_to_impute:
        if vol_idx < 0 or vol_idx >= total_volumes:
            raise ValueError(f"Volume index {vol_idx} is out of bounds.")
        lower_candidates = [v for v in volumes_not_to_impute if v <= vol_idx]
        lower_idx = max(lower_candidates) if lower_candidates else min_valid_index
        upper_candidates = [v for v in volumes_not_to_impute if v >= vol_idx]
        upper_idx = min(upper_candidates) if upper_candidates else max_valid_index

        if method == "linear":
            lower_volume = time_series_np[..., lower_idx]
            upper_volume = time_series_np[..., upper_idx]
            time_series_np[..., vol_idx] = (lower_volume + upper_volume) / 2.0
        else:
            raise NotImplementedError("Currently, only linear interpolation is implemented.")

    imputed_time_series = ants.from_numpy(time_series_np)
    return ants.copy_image_info(time_series, imputed_time_series)


def loop_timeseries_censoring(
    x: ants.ANTsImage,
    threshold: float = 0.5,
    mask: ants.ANTsImage | None = None,
    n_features_sample: int | float = 0.02,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[ants.ANTsImage, np.ndarray]:
    """Censor high leverage volumes from a time series using LoOP."""
    if x.shape[3] < 20:
        warnings.warn("Warning: the time dimension is < 20 - too few samples for loop.")
        return x, np.array([], dtype=int)
    if mask is None:
        flattened_series = flatten_time_series(x.numpy())
    else:
        flattened_series = ants.timeseries_to_matrix(x, mask)
    loop_scores = calculate_loop_scores(flattened_series, n_features_sample=n_features_sample, seed=seed, verbose=verbose)
    high_leverage_volumes = np.where(loop_scores > threshold)[0]
    new_ts = remove_volumes_from_timeseries(x, high_leverage_volumes)
    return new_ts, high_leverage_volumes


def score_fmri_censoring(
    cbfts: ants.ANTsImage,
    csf_seg: ants.ANTsImage,
    gm_seg: ants.ANTsImage,
    wm_seg: ants.ANTsImage,
) -> tuple[ants.ANTsImage, np.ndarray]:
    """Process CBF time series to remove high-leverage points via SCORE."""
    n_gm_voxels = float(np.sum(gm_seg.numpy())) - 1
    n_wm_voxels = float(np.sum(wm_seg.numpy())) - 1
    n_csf_voxels = float(np.sum(csf_seg.numpy())) - 1
    mask1img = gm_seg + wm_seg + csf_seg

    cbfts_np = cbfts.numpy()
    gmbool = (gm_seg == 1).numpy()
    csfbool = (csf_seg == 1).numpy()
    wmbool = (wm_seg == 1).numpy()
    gm_cbf_ts = ants.timeseries_to_matrix(cbfts, gm_seg)
    gm_cbf_ts = np.squeeze(np.mean(gm_cbf_ts, axis=1))

    median_gm_cbf = float(np.median(gm_cbf_ts))
    mad_gm_cbf = float(np.median(np.abs(gm_cbf_ts - median_gm_cbf))) / 0.675
    indx = np.abs(gm_cbf_ts - median_gm_cbf) > (2.5 * mad_gm_cbf)

    spatmeannp = np.mean(cbfts_np[:, :, :, ~indx], axis=3)
    spatmean = ants.from_numpy(spatmeannp)
    V = (
        n_gm_voxels * np.var(spatmeannp[gmbool])
        + n_wm_voxels * np.var(spatmeannp[wmbool])
        + n_csf_voxels * np.var(spatmeannp[csfbool])
    )
    V1 = math.inf
    while V < V1:
        V1 = V
        CC = np.zeros(cbfts_np.shape[3])
        for s in range(cbfts_np.shape[3]):
            if indx[s]:
                continue
            tmp1 = ants.from_numpy(cbfts_np[:, :, :, s])
            CC[s] = ants.image_similarity(spatmean, tmp1, metric_type="Correlation", fixed_mask=mask1img)
        inx = int(np.argmin(CC))
        indx[inx] = True
        spatmeannp = np.mean(cbfts_np[:, :, :, ~indx], axis=3)
        spatmean = ants.from_numpy(spatmeannp)
        V = (
            n_gm_voxels * np.var(spatmeannp[gmbool])
            + n_wm_voxels * np.var(spatmeannp[wmbool])
            + n_csf_voxels * np.var(spatmeannp[csfbool])
        )
    cbfts_recon = cbfts_np[:, :, :, ~indx]
    cbfts_recon = np.nan_to_num(cbfts_recon)
    cbfts_recon_ants = ants.from_numpy(cbfts_recon)
    cbfts_recon_ants = ants.copy_image_info(cbfts, cbfts_recon_ants)
    return cbfts_recon_ants, indx
