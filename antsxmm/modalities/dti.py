from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
import numpy as np
import pandas as pd
import ants
import nibabel as nib
from scipy.linalg import inv
from dipy.core.gradients import gradient_table, reorient_bvecs
from dipy.io.gradients import read_bvals_bvecs
import dipy.reconst.dti as dti
import dipy.reconst.fwdti as fwdti
from dipy.reconst.dti import fractional_anisotropy, color_fa, mean_diffusivity

from .metrics import (
    ants_to_nibabel_affine,
    apply_transforms_mixed_interpolation,
    convert_np_in_dict,
    dvars,
    impute_timeseries,
    loop_timeseries_censoring,
    mask_snr,
    remove_elements_from_list,
    remove_elements_from_numpy_array,
    remove_volumes_from_timeseries,
    segment_timeseries_by_bvalue,
    segment_timeseries_by_meanvalue,
    slice_snr,
    tsnr,
)
from .super_resolution import super_res_mcimage


def write_bvals_bvecs(bvals: Any, bvecs: Any, prefix: str) -> None:
    """Write FSL FDT bvals and bvecs files."""
    val_fmt = "   %e"
    bvals_tuple = tuple(bvals)
    bvecs_arr = np.asarray(bvecs)
    bvecs_arr[np.isnan(bvecs_arr)] = 0
    n = len(bvals_tuple)

    fname_bval = prefix + ".bval"
    fmt = val_fmt * n + "\n"
    with open(fname_bval, "wt") as f:
        f.write(fmt % bvals_tuple)

    fname_bvec = prefix + ".bvec"
    with open(fname_bvec, "wt") as f:
        for dim_vals in bvecs_arr.T:
            f.write(fmt % tuple(dim_vals))


def repair_bvecs(bvecs: np.ndarray) -> np.ndarray:
    """Normalize bvecs if needed."""
    bvecs_arr = np.asarray(bvecs)
    bvecnorm = np.linalg.norm(bvecs_arr, axis=1).reshape(bvecs_arr.shape[0], 1)
    if abs(np.linalg.norm(bvecs_arr) - 1) > 0.009:
        warnings.warn(
            f"Warning: bvecs are not unit norm - normalizing. Norm: {np.linalg.norm(bvecs_arr)}"
        )
        bvecs_arr = np.where(bvecnorm > 1e-16, bvecs_arr / bvecnorm, 0)
    return bvecs_arr


def triangular_to_tensor(image: ants.ANTsImage, upper_triangular: bool = True) -> np.ndarray:
    """Convert triangular tensor image to a full tensor form (in numpy X, Y, Z, 3, 3)."""
    yyind = 3 if upper_triangular else 2
    xzind = 2 if upper_triangular else 3
    dtinp = np.zeros(image.shape + (3, 3), dtype=float)
    dtix = np.zeros((3, 3), dtype=float)
    dtiut = image.numpy()
    for i in np.ndindex(image.shape):
        dtivec = dtiut[i]
        dtix[0, 0] = dtivec[0]
        dtix[1, 1] = dtivec[yyind]
        dtix[2, 2] = dtivec[5]
        dtix[0, 1] = dtix[1, 0] = dtivec[1]
        dtix[0, 2] = dtix[2, 0] = dtivec[xzind]
        dtix[1, 2] = dtix[2, 1] = dtivec[4]
        dtinp[i] = dtix
    return dtinp


def dti_numpy_to_image(reference_image: ants.ANTsImage, tensorarray: np.ndarray, upper_triangular: bool = True) -> ants.ANTsImage:
    """Convert numpy DTI data (X, Y, Z, 3, 3) to 6-component ANTsImage."""
    dtiut = np.zeros(reference_image.shape + (6,), dtype=float)
    dtivec = np.zeros(6, dtype=float)
    yyind = 3 if upper_triangular else 2
    xzind = 2 if upper_triangular else 3
    for i in np.ndindex(reference_image.shape):
        dtix = tensorarray[i]
        dtivec[0] = dtix[0, 0]
        dtivec[yyind] = dtix[1, 1]
        dtivec[5] = dtix[2, 2]
        dtivec[1] = dtix[0, 1]
        dtivec[xzind] = dtix[2, 0]
        dtivec[4] = dtix[1, 2]
        dtiut[i] = dtivec
    dti_ants = ants.from_numpy(dtiut, has_components=True)
    ants.copy_image_info(reference_image, dti_ants)
    return dti_ants


def get_dti(
    reference_image: ants.ANTsImage,
    tensormodel: Any,
    upper_triangular: bool = True,
    return_image: bool = False,
) -> ants.ANTsImage | np.ndarray:
    """Extract DTI data from a dipy tensormodel."""
    reoind = np.array([0, 1, 3, 2, 4, 5])
    dtiut = dti.lower_triangular(tensormodel.quadratic_form)
    yyind = 3 if upper_triangular else 2
    xzind = 2 if upper_triangular else 3
    if upper_triangular:
        for i in np.ndindex(reference_image.shape):
            dtiut[i] = dtiut[i][reoind]
    if return_image:
        dti_ants = ants.from_numpy(dtiut, has_components=True)
        ants.copy_image_info(reference_image, dti_ants)
        return dti_ants

    dtinp = np.zeros(reference_image.shape + (3, 3), dtype=float)
    dtix = np.zeros((3, 3), dtype=float)
    for i in np.ndindex(reference_image.shape):
        dtivec = dtiut[i]
        dtix[0, 0] = dtivec[0]
        dtix[1, 1] = dtivec[yyind]
        dtix[2, 2] = dtivec[5]
        dtix[0, 1] = dtix[1, 0] = dtivec[1]
        dtix[0, 2] = dtix[2, 0] = dtivec[xzind]
        dtix[1, 2] = dtix[2, 1] = dtivec[4]
        dtinp[i] = dtix
    return dtinp


def distortion_correct_bvecs(
    bvecs: np.ndarray,
    def_grad: np.ndarray,
    a_img: np.ndarray,
    a_ref: np.ndarray,
) -> np.ndarray:
    """Vectorized computation of voxel-wise distortion corrected b-vectors."""
    a = a_ref.T @ a_img
    r_voxel = np.einsum("ij,xyzjk->xyzik", a, def_grad)
    r_voxel_reshaped = r_voxel.reshape(-1, 3, 3)
    rotated = np.einsum("vij,nj->vni", r_voxel_reshaped, bvecs)
    norms = np.linalg.norm(rotated, axis=2, keepdims=True)
    rotated /= np.clip(norms, 1e-8, None)
    return rotated.reshape(def_grad.shape[:3] + (bvecs.shape[0], 3))


def deformation_gradient_optimized(
    warp_image: ants.ANTsImage,
    to_rotation: bool = False,
    to_inverse_rotation: bool = False,
) -> np.ndarray:
    """Compute the deformation gradient tensor from a displacement (warp) field image."""
    if not ants.is_image(warp_image):
        raise RuntimeError("ANTsImage is required")
    dim = warp_image.dimension
    tshp = warp_image.shape
    tdir = warp_image.direction
    spc = warp_image.spacing
    warpnp = warp_image.numpy()
    gradient_list = [np.gradient(warpnp[..., k], *spc, axis=range(dim)) for k in range(dim)]
    dg = np.stack([np.stack(grad_k, axis=-1) for grad_k in gradient_list], axis=-1)
    dg = (tdir @ dg).swapaxes(-1, -2)
    dg += np.eye(dim)
    if to_rotation or to_inverse_rotation:
        u, _, vh = np.linalg.svd(dg)
        z = u @ vh
        dets = np.linalg.det(z)
        reflection_mask = dets < 0
        vh[reflection_mask, -1, :] *= -1
        z[reflection_mask] = u[reflection_mask] @ vh[reflection_mask]
        dg = z
        if to_inverse_rotation:
            dg = np.transpose(dg, axes=(*range(dg.ndim - 2), dg.ndim - 1, dg.ndim - 2))
    new_shape = tshp + (dim, dim)
    return np.reshape(dg, new_shape)


def transform_and_reorient_dti(
    fixed: ants.ANTsImage,
    moving_dti: ants.ANTsImage,
    composite_transform: str,
    verbose: bool = False,
    **kwargs: Any,
) -> ants.ANTsImage:
    """Applies transformation to DTI image using ANTs composite transform and Finite Strain reorientation."""
    if moving_dti.dimension != 3:
        raise ValueError("moving_dti must be 3-dimensional.")
    if moving_dti.components != 6:
        raise ValueError("moving_dti must have 6 components (upper triangular format).")

    dtsplit = moving_dti.split_channels()
    dtiw_channels = [
        ants.apply_transforms(fixed, dtsplit[k], composite_transform, **kwargs)
        for k in range(len(dtsplit))
    ]
    dtiw = ants.merge_channels(dtiw_channels)

    wtx = ants.image_read(composite_transform)
    r_moving_to_fixed_forward = deformation_gradient_optimized(
        wtx, to_rotation=False, to_inverse_rotation=True
    )
    dtiw2tensor_np = triangular_to_tensor(dtiw)

    d_world_moving_orient = np.einsum(
        "ab, ...bc, cd -> ...ad",
        moving_dti.direction,
        dtiw2tensor_np,
        moving_dti.direction.T,
    )
    d_world_fixed_orient = np.einsum(
        "...ab, ...bc, ...cd -> ...ad",
        r_moving_to_fixed_forward,
        d_world_moving_orient,
        np.swapaxes(r_moving_to_fixed_forward, -1, -2),
    )
    final_dti_tensors_numpy = np.einsum(
        "ba, ...bc, cd -> ...ad",
        fixed.direction,
        d_world_fixed_orient,
        fixed.direction,
    )
    return dti_numpy_to_image(fixed, final_dti_tensors_numpy)


def generate_voxelwise_bvecs(global_bvecs: np.ndarray, voxel_rotations: np.ndarray, transpose: bool = False) -> np.ndarray:
    """Generate voxel-wise b-vectors from global bvec and voxel-wise rotation field."""
    x, y, z, _, _ = voxel_rotations.shape
    n = global_bvecs.shape[0]
    bvecs_5d = np.zeros((x, y, z, n, 3), dtype=np.float32)
    for n_idx in range(n):
        bvec = global_bvecs[n_idx]
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    r = voxel_rotations[i, j, k]
                    if transpose:
                        r = r.T
                    bvecs_5d[i, j, k, n_idx, :] = r @ bvec
    return bvecs_5d


def bvec_reorientation(motion_parameters: list[Any] | None, bvecs: np.ndarray, rebase: np.ndarray | None = None) -> np.ndarray:
    """Reorient bvecs based on rigid/affine motion parameters."""
    if motion_parameters is None:
        return bvecs
    n = len(motion_parameters)
    if n < 1:
        return bvecs
    for myidx in range(n):
        if myidx < bvecs.shape[0] and motion_parameters[myidx] != "NA":
            temp = motion_parameters[myidx]
            if len(temp) in [3, 4]:
                tx1 = ants.read_transform(temp[-1])
                p1 = ants.get_ants_transform_parameters(tx1)[0:9].reshape([3, 3])
                tx2 = ants.read_transform(temp[1])
                p2 = ants.get_ants_transform_parameters(tx2)[0:9].reshape([3, 3])
                rinv = inv(p2 @ p1)
            else:
                tx_file = temp[1] if len(temp) == 2 else temp[0]
                tx = ants.read_transform(tx_file)
                p = ants.get_ants_transform_parameters(tx)[0:9].reshape([3, 3])
                rinv = inv(p)
            bvecs[myidx, :] = rinv @ bvecs[myidx, :]
            if rebase is not None:
                bvecs[myidx, :] = rebase @ bvecs[myidx, :]
    return bvecs


def dti_reg(
    image: ants.ANTsImage,
    avg_b0: ants.ANTsImage,
    avg_dwi: ants.ANTsImage,
    bvals: Any = None,
    bvecs: Any = None,
    b0_idx: list[int] | None = None,
    type_of_transform: str = "antsRegistrationSyNRepro[r]",
    total_sigma: float = 3.0,
    fdOffset: float = 2.0,
    mask_csf: bool = False,
    brain_mask_eroded: ants.ANTsImage | None = None,
    output_directory: str | None = None,
    verbose: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Correct DWI time-series data for motion with optional deformation."""
    idim = image.dimension
    ishape = image.shape
    n_time_points = ishape[idim - 1]
    fd = np.zeros(n_time_points)

    if bvals is not None and bvecs is not None:
        if isinstance(bvecs, str):
            bvals, bvecs = read_bvals_bvecs(bvals, bvecs)
        else:
            bvals = bvals.copy()
            bvecs = bvecs.copy()

    if type_of_transform is None:
        return {
            "motion_corrected": image,
            "motion_parameters": None,
            "FD": fd,
            "bvals": bvals,
            "bvecs": bvecs,
        }

    remove_it = False
    if output_directory is None:
        remove_it = True
        output_directory = tempfile.mkdtemp()
    output_directory_w = os.path.join(output_directory, "dti_reg")
    os.makedirs(output_directory_w, exist_ok=True)
    ofn_g = tempfile.NamedTemporaryFile(delete=False, suffix="global_deformation", dir=output_directory_w).name
    ofn_l = tempfile.NamedTemporaryFile(delete=False, suffix="local_deformation", dir=output_directory_w).name

    if b0_idx is None:
        b0_idx = segment_timeseries_by_bvalue(bvals)["lowbvals"]

    ab0, adw = get_average_dwi_b0(image)
    mask = ants.threshold_image(ants.iMath(adw, "Normalize"), 0.1, 1.0)
    if brain_mask_eroded is None:
        brain_mask_eroded = mask * 0 + 1

    motion_parameters: list[Any] = []
    motion_corrected: list[ants.ANTsImage] = []
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

    initrig = ants.registration(avg_b0, ab0, "antsRegistrationSyNRepro[r]", outprefix=ofn_g)
    deftx = ants.registration(
        avg_dwi,
        adw,
        "SyNOnly",
        syn_metric="CC",
        syn_sampling=2,
        reg_iterations=[50, 50, 20],
        multivariate_extras=[["CC", avg_b0, ab0, 1, 2]],
        initial_transform=initrig["fwdtransforms"][0],
        outprefix=ofn_g,
    )["fwdtransforms"]

    counter = round(n_time_points / 10) + 1
    for k in range(n_time_points):
        if verbose and (k % counter == 0 or k == n_time_points - 1):
            print(f"{round(k / n_time_points * 100)}%.", end="", flush=True)
        fixed = ants.image_clone(ab0 if k in b0_idx else adw)
        temp = ants.slice_image(image, axis=idim - 1, idx=k)
        temp = ants.iMath(temp, "Normalize")
        txprefix = f"{ofn_l}{str(k).zfill(4)}rig_"
        txprefix2 = f"{ofn_l}{str(k % 2).zfill(4)}def_"

        if temp.numpy().var() > 0:
            myrig = ants.registration(fixed, temp, type_of_transform="antsRegistrationSyNRepro[r]", outprefix=txprefix, **kwargs)
            if type_of_transform == "SyN":
                myreg = ants.registration(
                    fixed, temp, type_of_transform="SyNOnly", total_sigma=total_sigma,
                    grad_step=0.1, initial_transform=myrig["fwdtransforms"][0], outprefix=txprefix2, **kwargs
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
            motion_parameters[k] = deftx + motion_parameters[k]
            img1w = apply_transforms_mixed_interpolation(
                avg_dwi, ants.slice_image(image, axis=idim - 1, idx=k), motion_parameters[k], mask=brain_mask_eroded
            )
            motion_corrected.append(img1w)
        else:
            motion_corrected.append(fixed)

    if bvecs is not None:
        rebase = np.transpose(avg_b0.direction) @ ab0.direction
        bvecs = bvec_reorientation(motion_parameters, bvecs, rebase)

    if remove_it:
        shutil.rmtree(output_directory, ignore_errors=True)

    d4siz = list(avg_b0.shape) + [2]
    spc = list(ants.get_spacing(avg_b0)) + [1.0]
    mydir4d = ants.get_direction(image)
    mydir4d[0:3, 0:3] = ants.get_direction(avg_b0)
    myorg = list(ants.get_origin(avg_b0)) + [0.0]
    avg_b0_4d = ants.make_image(d4siz, 0, spacing=spc, origin=myorg, direction=mydir4d)

    return {
        "motion_corrected": ants.list_to_ndimage(avg_b0_4d, motion_corrected),
        "motion_parameters": motion_parameters,
        "FD": fd,
        "bvals": bvals,
        "bvecs": bvecs,
    }


def get_average_dwi_b0(
    x: ants.ANTsImage,
    fixed_b0: ants.ANTsImage | None = None,
    fixed_dwi: ants.ANTsImage | None = None,
    fast: bool = False,
) -> tuple[ants.ANTsImage, ants.ANTsImage]:
    """Automatically generates average b0 and dwi; maps dwi to b0 space."""
    output_directory = tempfile.mkdtemp()
    ofn = os.path.join(output_directory, "w")
    temp = segment_timeseries_by_meanvalue(x)
    b0_idx = temp["highermeans"]
    non_b0_idx = temp["lowermeans"]

    if (fixed_b0 is None and fixed_dwi is None) or fast:
        xavg = ants.slice_image(x, axis=3, idx=0) * 0.0
        bavg = ants.slice_image(x, axis=3, idx=0) * 0.0
        fixed_b0_use = ants.slice_image(x, axis=3, idx=b0_idx[0])
        fixed_dwi_use = ants.slice_image(x, axis=3, idx=non_b0_idx[0])
    else:
        temp_b0 = ants.slice_image(x, axis=3, idx=b0_idx[0])
        temp_dwi = ants.slice_image(x, axis=3, idx=non_b0_idx[0])
        xavg = fixed_b0 * 0.0
        bavg = fixed_b0 * 0.0
        tempreg = ants.registration(fixed_b0, temp_b0, "antsRegistrationSyNRepro[r]")
        fixed_b0_use = tempreg["warpedmovout"]
        fixed_dwi_use = ants.apply_transforms(fixed_b0, temp_dwi, tempreg["fwdtransforms"])

    for myidx in range(x.shape[3]):
        b0 = ants.slice_image(x, axis=3, idx=myidx)
        if not fast:
            if myidx not in b0_idx:
                xavg = xavg + ants.registration(fixed_dwi_use, b0, "antsRegistrationSyNRepro[r]", outprefix=ofn)["warpedmovout"]
            else:
                bavg = bavg + ants.registration(fixed_b0_use, b0, "antsRegistrationSyNRepro[r]", outprefix=ofn)["warpedmovout"]
        else:
            if myidx not in b0_idx:
                xavg = xavg + b0
            else:
                bavg = bavg + b0

    bavg = ants.iMath(bavg, "Normalize")
    xavg = ants.iMath(xavg, "Normalize")
    shutil.rmtree(output_directory, ignore_errors=True)
    avgb0 = ants.n4_bias_field_correction(bavg)
    avgdwi = ants.n4_bias_field_correction(xavg)
    avgdwi = ants.registration(avgb0, avgdwi, "antsRegistrationSyNRepro[r]")["warpedmovout"]
    return avgb0, avgdwi


def dti_template(
    b_image_list: list[ants.ANTsImage] | None = None,
    w_image_list: list[ants.ANTsImage] | None = None,
    iterations: int = 5,
    gradient_step: float = 0.5,
    mask_csf: bool = False,
    average_both: bool = True,
    verbose: bool = False,
) -> tuple[ants.ANTsImage, ants.ANTsImage]:
    """Two-channel version of build_template returning (avg_b0, avg_dwi)."""
    if b_image_list is None or w_image_list is None:
        raise ValueError("Both b_image_list and w_image_list must be provided.")

    output_directory = tempfile.mkdtemp()
    mydeftx = tempfile.NamedTemporaryFile(delete=False, dir=output_directory).name
    tmp = tempfile.NamedTemporaryFile(delete=False, dir=output_directory, suffix=".nii.gz")
    wavgfn = tmp.name
    tmp2 = tempfile.NamedTemporaryFile(delete=False, dir=output_directory)
    comptx = tmp2.name
    weights = [1.0 / len(b_image_list)] * len(b_image_list)

    b_initial_template = ants.iMath(b_image_list[0], "Normalize")
    w_initial_template = ants.iMath(w_image_list[0], "Normalize")

    if mask_csf:
        bcsf0 = ants.threshold_image(b_image_list[0], "Otsu", 2).threshold_image(1, 1).morphology("open", 1).iMath("GetLargestComponent")
        bcsf1 = ants.threshold_image(b_image_list[1], "Otsu", 2).threshold_image(1, 1).morphology("open", 1).iMath("GetLargestComponent")
    else:
        bcsf0 = b_image_list[0] * 0 + 1
        bcsf1 = b_image_list[1] * 0 + 1
    bavg = b_initial_template.clone() * bcsf0
    wavg = w_initial_template.clone() * bcsf0
    bcsf = [bcsf0, bcsf1]

    for i in range(iterations):
        for k in range(len(w_image_list)):
            fimg, mimg = wavg, w_image_list[k] * bcsf[k]
            fimg2, mimg2 = bavg, b_image_list[k] * bcsf[k]
            w1 = ants.registration(
                fimg, mimg, type_of_transform="antsRegistrationSyNQuickRepro[s]",
                multivariate_extras=[["CC", fimg2, mimg2, 1, 2]], outprefix=mydeftx, verbose=0
            )
            txname = ants.apply_transforms(wavg, wavg, w1["fwdtransforms"], compose=comptx)
            if k == 0:
                txavg = ants.image_read(txname) * weights[k]
                wavgnew = ants.apply_transforms(wavg, w_image_list[k] * bcsf[k], txname).iMath("Normalize")
                bavgnew = ants.apply_transforms(wavg, b_image_list[k] * bcsf[k], txname).iMath("Normalize")
            else:
                txavg = txavg + ants.image_read(txname) * weights[k]
                if i >= (iterations - 2) and average_both:
                    wavgnew = wavgnew + ants.apply_transforms(wavg, w_image_list[k] * bcsf[k], txname).iMath("Normalize")
                    bavgnew = bavgnew + ants.apply_transforms(wavg, b_image_list[k] * bcsf[k], txname).iMath("Normalize")
        if verbose:
            print(f"iteration: {i} {float(txavg.abs().mean())}")
        txavg = txavg * (-1.0 * gradient_step)
        ants.image_write(txavg, wavgfn)
        wavg = ants.apply_transforms(wavg, wavgnew, wavgfn).iMath("Normalize")
        bavg = ants.apply_transforms(bavg, bavgnew, wavgfn).iMath("Normalize")

    shutil.rmtree(output_directory, ignore_errors=True)
    return bavg, wavg


def read_ants_transforms_to_numpy(transform_files: list[list[str]]) -> np.ndarray:
    """Read a list of ANTs transform files and convert them to a NumPy array."""
    filtered_lists = [[s for s in sublist if s.endswith(".mat")] for sublist in transform_files]
    transforms = []
    for f in filtered_lists:
        tx = ants.read_transform(f[0])
        transforms.append(np.array(ants.get_ants_transform_parameters(tx)[0:9]))
    return np.array(transforms)


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


def impute_fa(fa: ants.ANTsImage, md: ants.ANTsImage) -> ants.ANTsImage:
    """Impute bad values in FA and MD images."""
    def _impute(x: ants.ANTsImage, ref_fa: ants.ANTsImage) -> ants.ANTsImage:
        badfa = ants.threshold_image(ref_fa, 1, 1)
        if badfa.max() == 1:
            temp = ants.image_clone(x)
            temp[badfa == 1] = 0
            temp = ants.iMath(temp, "GD", 2)
            x[badfa == 1] = temp[badfa == 1]
        return x
    return _impute(md, fa)


def impute_dwi(
    dwi: ants.ANTsImage,
    threshold: float = 0.20,
    imputeb0: bool = False,
    mask: ants.ANTsImage | None = None,
    verbose: bool = False,
) -> ants.ANTsImage:
    """Identify bad volumes in DWI and impute them automatically."""
    list1 = segment_timeseries_by_meanvalue(dwi)["highermeans"]
    if imputeb0:
        dwib = impute_timeseries(dwi, list1)
        _, list2 = loop_timeseries_censoring(dwib, threshold, mask)
    else:
        _, list2 = loop_timeseries_censoring(dwi, threshold, mask)
    complement = remove_elements_from_list(list(list2), list1)
    if not complement:
        return dwi
    return impute_timeseries(dwi, complement)


def censor_dwi(
    dwi: ants.ANTsImage,
    bval: np.ndarray,
    bvec: np.ndarray,
    threshold: float = 0.20,
    imputeb0: bool = False,
    mask: ants.ANTsImage | None = None,
    verbose: bool = False,
) -> tuple[ants.ANTsImage, np.ndarray, np.ndarray]:
    """Identify bad volumes in DWI and censor them."""
    list1 = segment_timeseries_by_meanvalue(dwi)["highermeans"]
    if imputeb0:
        dwib = impute_timeseries(dwi, list1)
        _, list2 = loop_timeseries_censoring(dwib, threshold, mask, verbose=verbose)
    else:
        _, list2 = loop_timeseries_censoring(dwi, threshold, mask, verbose=verbose)
    complement = remove_elements_from_list(list(list2), list1)
    if not complement:
        return dwi, bval, bvec
    return (
        remove_volumes_from_timeseries(dwi, complement),
        remove_elements_from_numpy_array(bval, complement),
        remove_elements_from_numpy_array(bvec, complement),
    )


def merge_dwi_data(
    img_lrdwp: ants.ANTsImage,
    bval_lr: np.ndarray,
    bvec_lr: np.ndarray,
    img_rldwp: ants.ANTsImage,
    bval_rl: np.ndarray,
    bvec_rl: np.ndarray,
) -> tuple[ants.ANTsImage, np.ndarray, np.ndarray]:
    """Merge motion and distortion corrected DWI acquisitions."""
    if not ants.image_physical_space_consistency(img_lrdwp, img_rldwp):
        warnings.warn("Corrected image pair must occupy the same physical space; returning 1st only.")
        return img_lrdwp, bval_lr, bvec_lr

    bvals = np.concatenate([bval_lr, bval_rl])
    bvecs = np.concatenate([bvec_lr, bvec_rl])
    mimg = [ants.slice_image(img_lrdwp, axis=3, idx=k) for k in range(img_lrdwp.shape[3])]
    mimg += [ants.slice_image(img_rldwp, axis=3, idx=k) for k in range(img_rldwp.shape[3])]
    return ants.list_to_ndimage(img_lrdwp, mimg), bvals, bvecs


def mc_denoise(x: ants.ANTsImage, ratio: float = 0.5) -> ants.ANTsImage:
    """ANTs denoising for 4D timeseries."""
    dwpimage = []
    for myidx in range(x.shape[3]):
        b0 = ants.slice_image(x, axis=3, idx=myidx)
        dnzb0 = ants.denoise_image(b0, p=1, r=1, noise_model="Gaussian")
        dwpimage.append(dnzb0 * ratio + b0 * (1.0 - ratio))
    return ants.list_to_ndimage(x, dwpimage)


def efficient_tensor_fit(
    gtab: Any,
    fit_method: str,
    imagein: ants.ANTsImage,
    maskin: ants.ANTsImage,
    diffusion_model: str = "DTI",
    chunk_size: int = 10,
    num_threads: int = 1,
    verbose: bool = True,
) -> tuple[Any, ants.ANTsImage, ants.ANTsImage, ants.ANTsImage]:
    """Efficient and optionally parallelized tensor reconstruction using DiPy."""
    assert imagein.dimension == 4, "Input image must be 4D"

    img_data = imagein.numpy()
    mask = maskin.numpy().astype(bool)
    X, Y, Z, _ = img_data.shape

    model = fwdti.FreeWaterTensorModel(gtab) if diffusion_model == "FreeWater" else dti.TensorModel(gtab, fit_method=fit_method)

    def process_chunk(z_start: int) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
        z_end = min(Z, z_start + chunk_size)
        local_data = img_data[:, :, z_start:z_end, :]
        local_mask = mask[:, :, z_start:z_end]
        masked_data = np.nan_to_num(local_data * local_mask[..., None], nan=0)
        fit = model.fit(masked_data)
        fa_chunk = fractional_anisotropy(fit.evals)
        fa_chunk[np.isnan(fa_chunk)] = 1
        fa_chunk = np.clip(fa_chunk, 0, 1)
        md_chunk = mean_diffusivity(fit.evals)
        rgb_chunk = color_fa(fa_chunk, fit.evecs)
        return z_start, z_end, fa_chunk, md_chunk, rgb_chunk

    fa_vol = np.zeros((X, Y, Z), dtype=np.float32)
    md_vol = np.zeros((X, Y, Z), dtype=np.float32)
    rgb_vol = np.zeros((X, Y, Z, 3), dtype=np.float32)

    chunks = range(0, Z, chunk_size)
    if num_threads > 1:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(process_chunk, z): z for z in chunks}
            for f in as_completed(futures):
                z_s, z_e, fa_c, md_c, rgb_c = f.result()
                fa_vol[:, :, z_s:z_e] = fa_c
                md_vol[:, :, z_s:z_e] = md_c
                rgb_vol[:, :, z_s:z_e, :] = rgb_c
    else:
        for z in chunks:
            z_s, z_e, fa_c, md_c, rgb_c = process_chunk(z)
            fa_vol[:, :, z_s:z_e] = fa_c
            md_vol[:, :, z_s:z_e] = md_c
            rgb_vol[:, :, z_s:z_e, :] = rgb_c

    b0 = ants.slice_image(imagein, axis=3, idx=0)
    fa_img = ants.copy_image_info(b0, ants.from_numpy(fa_vol))
    md_img = ants.copy_image_info(b0, ants.from_numpy(md_vol))
    rgb_img = ants.merge_channels([ants.copy_image_info(b0, ants.from_numpy(rgb_vol[..., i])) for i in range(3)])
    return model.fit(img_data * mask[..., None]), fa_img, md_img, rgb_img


def efficient_dwi_fit(
    gtab: Any,
    diffusion_model: str,
    imagein: ants.ANTsImage,
    maskin: ants.ANTsImage,
    model_params: dict[str, Any] | None = None,
    bvals_to_use: list[int] | None = None,
    chunk_size: int = 1024,
    num_threads: int = 1,
    verbose: bool = True,
) -> tuple[Any, ants.ANTsImage | None, ants.ANTsImage | None, ants.ANTsImage | None]:
    """Efficient parallelized diffusion model fitting."""
    import dipy.reconst.dki as dki

    assert imagein.dimension == 4, "Input image must be 4D"
    model_params = model_params or {}

    img_data = imagein.numpy()
    mask = maskin.numpy().astype(bool)
    X, Y, Z, N = img_data.shape
    inplane_size = X * Y
    slices_per_chunk = max(1, chunk_size // inplane_size)

    if bvals_to_use is not None:
        bvals_set = set(bvals_to_use)
        sel = np.isin(gtab.bvals, list(bvals_set))
        img_data = img_data[..., sel]
        gtab = gradient_table(gtab.bvals[sel], bvecs=gtab.bvecs[sel])

    if diffusion_model == "DTI":
        model = dti.TensorModel(gtab, **model_params)
    elif diffusion_model == "FreeWater":
        model = fwdti.FreeWaterTensorModel(gtab)
    elif diffusion_model == "DKI":
        model = dki.DiffusionKurtosisModel(gtab, **model_params)
    else:
        raise ValueError(f"Unsupported model: {diffusion_model}")

    has_tensor_metrics = diffusion_model in ["DTI", "FreeWater"]
    fa_vol = np.zeros((X, Y, Z), dtype=np.float32)
    md_vol = np.zeros((X, Y, Z), dtype=np.float32)
    rgb_vol = np.zeros((X, Y, Z, 3), dtype=np.float32)

    def process_chunk(z_start: int) -> tuple[int, int, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        z_end = min(Z, z_start + slices_per_chunk)
        local_data = img_data[:, :, z_start:z_end, :]
        local_mask = mask[:, :, z_start:z_end]
        masked_data = np.nan_to_num(local_data * local_mask[..., None], nan=0)
        fit = model.fit(masked_data)
        if has_tensor_metrics and hasattr(fit, "evals") and hasattr(fit, "evecs"):
            fa = fractional_anisotropy(fit.evals)
            fa[np.isnan(fa)] = 1
            fa = np.clip(fa, 0, 1)
            md = mean_diffusivity(fit.evals)
            rgb = color_fa(fa, fit.evecs)
            return z_start, z_end, fa, md, rgb
        return z_start, z_end, None, None, None

    chunks = range(0, Z, slices_per_chunk)
    if num_threads > 1:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(process_chunk, z): z for z in chunks}
            for f in as_completed(futures):
                z_s, z_e, fa, md, rgb = f.result()
                if fa is not None:
                    fa_vol[:, :, z_s:z_e] = fa
                    md_vol[:, :, z_s:z_e] = md
                    rgb_vol[:, :, z_s:z_e, :] = rgb
    else:
        for z in chunks:
            z_s, z_e, fa, md, rgb = process_chunk(z)
            if fa is not None:
                fa_vol[:, :, z_s:z_e] = fa
                md_vol[:, :, z_s:z_e] = md
                rgb_vol[:, :, z_s:z_e, :] = rgb

    b0 = ants.slice_image(imagein, axis=3, idx=0)
    fa_img = ants.copy_image_info(b0, ants.from_numpy(fa_vol)) if has_tensor_metrics else None
    md_img = ants.copy_image_info(b0, ants.from_numpy(md_vol)) if has_tensor_metrics else None
    rgb_img = (
        ants.merge_channels([ants.copy_image_info(b0, ants.from_numpy(rgb_vol[..., i])) for i in range(3)])
        if has_tensor_metrics
        else None
    )
    full_fit = model.fit(img_data * mask[..., None])
    return full_fit, fa_img, md_img, rgb_img


def efficient_dwi_fit_voxelwise(
    imagein: ants.ANTsImage,
    maskin: ants.ANTsImage,
    bvals: np.ndarray,
    bvecs_5d: np.ndarray,
    model_params: dict[str, Any] | None = None,
    bvals_to_use: list[int] | None = None,
    num_threads: int = 1,
    verbose: bool = True,
) -> tuple[ants.ANTsImage, ants.ANTsImage, ants.ANTsImage]:
    """Voxel-wise diffusion model fitting with individual b-vectors per voxel."""
    from tqdm import tqdm

    model_params = model_params or {}
    img = imagein.numpy()
    mask = maskin.numpy().astype(bool)
    X, Y, Z, _ = img.shape

    if bvals_to_use is not None:
        sel = np.isin(bvals, bvals_to_use)
        img = img[..., sel]
        bvals = bvals[sel]
        bvecs_5d = bvecs_5d[..., sel, :]

    fa_vol = np.zeros((X, Y, Z), dtype=np.float32)
    md_vol = np.zeros((X, Y, Z), dtype=np.float32)
    rgb_vol = np.zeros((X, Y, Z, 3), dtype=np.float32)

    def fit_voxel(ix: int, iy: int, iz: int) -> None:
        if not mask[ix, iy, iz]:
            return
        sig = img[ix, iy, iz, :]
        if np.all(sig == 0):
            return
        bv = bvecs_5d[ix, iy, iz, :, :]
        gtab = gradient_table(bvals, bvecs=bv)
        try:
            model = dti.TensorModel(gtab, **model_params)
            fit = model.fit(sig)
            fa_vol[ix, iy, iz] = fractional_anisotropy(fit.evals)
            md_vol[ix, iy, iz] = mean_diffusivity(fit.evals)
            rgb_vol[ix, iy, iz, :] = color_fa(fa_vol[ix, iy, iz], fit.evecs)
        except Exception:
            pass

    coords = np.argwhere(mask)
    if num_threads > 1:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(tqdm(executor.map(lambda c: fit_voxel(*c), coords), total=len(coords), disable=not verbose))
    else:
        for c in tqdm(coords, disable=not verbose):
            fit_voxel(*c)

    ref = ants.slice_image(imagein, axis=3, idx=0)
    return (
        ants.copy_image_info(ref, ants.from_numpy(fa_vol)),
        ants.copy_image_info(ref, ants.from_numpy(md_vol)),
        ants.merge_channels([ants.copy_image_info(ref, ants.from_numpy(rgb_vol[..., i])) for i in range(3)]),
    )


def dipy_dti_recon(
    image: ants.ANTsImage,
    bvalsfn: Any,
    bvecsfn: Any,
    mask: ants.ANTsImage | None = None,
    b0_idx: list[int] | None = None,
    mask_dilation: int = 2,
    mask_closing: int = 5,
    fit_method: str = "WLS",
    trim_the_mask: float = 2.0,
    diffusion_model: str = "DTI",
    verbose: bool = False,
) -> dict[str, Any]:
    """DiPy DTI reconstruction."""
    import antspynet

    if isinstance(bvecsfn, str):
        bvals, bvecs = read_bvals_bvecs(bvalsfn, bvecsfn)
    else:
        bvals = bvalsfn.copy()
        bvecs = bvecsfn.copy()

    if bvals.max() < 1.0:
        raise ValueError("DTI recon error: maximum bvalues are too small.")

    b0_idx = segment_timeseries_by_bvalue(bvals)["lowbvals"]
    b0 = ants.slice_image(image, axis=3, idx=b0_idx[0])
    constant_mask = False

    if mask is not None:
        constant_mask = True
        mask = ants.resample_image_to_target(mask, b0, interp_type="nearestNeighbor")
    else:
        mask = antspynet.brain_extraction(b0, "t2").threshold_image(0.5, 1).iMath("GetLargestComponent").morphology("close", 2).iMath("FillHoles")

    if mask_closing > 0 and not constant_mask:
        mask = ants.morphology(mask, "close", mask_closing)
    maskdil = ants.iMath(mask, "MD", mask_dilation)

    bvecs = repair_bvecs(bvecs)
    gtab = gradient_table(bvals, bvecs=bvecs, atol=2.0)
    mynt = int(os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", 1))

    tenfit, fa, md1, rgb = efficient_dwi_fit(gtab, diffusion_model, image, maskdil, num_threads=mynt)
    if trim_the_mask > 0 and fit_method is not None:
        mask = trim_dti_mask(fa, mask, trim_the_mask)
        tenfit, fa, md1, rgb = efficient_dwi_fit(gtab, diffusion_model, image, maskdil, num_threads=mynt)

    return {
        "tensormodel": tenfit,
        "MD": md1,
        "FA": fa,
        "RGB": rgb,
        "dwi_mask": mask,
        "bvals": bvals,
        "bvecs": bvecs,
    }


def concat_dewarp(
    refimg: ants.ANTsImage,
    original_dwi: ants.ANTsImage,
    phys_space_dwi: ants.ANTsImage,
    dwp_tx: list[str],
    motion_parameters: list[Any],
    motion_correct: bool = True,
    verbose: bool = False,
) -> ants.ANTsImage:
    """Apply concatenated motion correction and dewarping transforms to timeseries image."""
    dwpimage = []
    for myidx in range(original_dwi.shape[3]):
        b0 = ants.slice_image(original_dwi, axis=3, idx=myidx)
        concatx = dwp_tx.copy()
        if motion_correct:
            concatx = concatx + motion_parameters[myidx]
        warpedb0 = ants.apply_transforms(refimg, b0, concatx, interpolator="nearestNeighbor")
        dwpimage.append(warpedb0)
    return ants.list_to_ndimage(phys_space_dwi, dwpimage)


def joint_dti_recon(
    img_lr: ants.ANTsImage,
    bval_lr: Any,
    bvec_lr: Any,
    jhu_atlas: ants.ANTsImage | None = None,
    jhu_labels: ants.ANTsImage | None = None,
    reference_b0: ants.ANTsImage | None = None,
    reference_dwi: ants.ANTsImage | None = None,
    srmodel: Any = None,
    img_rl: ants.ANTsImage | None = None,
    bval_rl: Any = None,
    bvec_rl: Any = None,
    t1w: ants.ANTsImage | None = None,
    brain_mask: ants.ANTsImage | None = None,
    motion_correct: str | None = None,
    dewarp_modality: str = "FA",
    denoise: bool = False,
    fit_method: str = "WLS",
    impute: bool = False,
    censor: bool = True,
    diffusion_model: str = "DTI",
    verbose: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Joint reconstruction and atlas-based labeling for DTI data."""
    import antspyt1w

    if "dti_motion_correct" in kwargs and motion_correct is None:
        motion_correct = kwargs["dti_motion_correct"]
    if "dti_denoise" in kwargs:
        denoise = kwargs["dti_denoise"]

    if reference_b0 is None or reference_dwi is None:
        ab0_res = get_average_dwi_b0(img_lr)
        if isinstance(ab0_res, dict):
            if reference_b0 is None:
                reference_b0 = ab0_res.get("b0_avg")
            if reference_dwi is None:
                reference_dwi = ab0_res.get("dwi_avg", reference_b0)
        elif isinstance(ab0_res, (list, tuple)):
            if reference_b0 is None and len(ab0_res) > 0:
                reference_b0 = ab0_res[0]
            if reference_dwi is None:
                reference_dwi = ab0_res[1] if len(ab0_res) > 1 else reference_b0
        else:
            if reference_b0 is None:
                reference_b0 = ab0_res
            if reference_dwi is None:
                reference_dwi = ab0_res

    if jhu_atlas is None:
        try:
            from .templates import get_data
            jhu_fn = get_data("JHU_MNI_SS_FA_1mm", target_extension=".nii.gz")
            if jhu_fn and os.path.exists(jhu_fn):
                jhu_atlas = ants.image_read(jhu_fn)
        except Exception:
            jhu_atlas = None

    if jhu_labels is None:
        try:
            from .templates import get_data
            jhu_lbl_fn = get_data("JHU_MNI_SS_WMPM_Type-I_syn_1mm", target_extension=".nii.gz")
            if jhu_lbl_fn and os.path.exists(jhu_lbl_fn):
                jhu_labels = ants.image_read(jhu_lbl_fn)
        except Exception:
            jhu_labels = None

    def _fix_shape(img: ants.ANTsImage, bval_fn: Any, bvec_fn: Any) -> ants.ANTsImage:
        if isinstance(bvec_fn, str):
            _, bvecs = read_bvals_bvecs(bval_fn, bvec_fn)
        else:
            bvecs = bvec_fn
        if bvecs.shape[0] < img.shape[3]:
            imgout = ants.from_numpy(img[:, :, :, 0 : bvecs.shape[0]])
            return ants.copy_image_info(img, imgout)
        return img

    img_lr = _fix_shape(img_lr, bval_lr, bvec_lr)
    if denoise:
        img_lr = mc_denoise(img_lr)
    if img_rl is not None:
        img_rl = _fix_shape(img_rl, bval_rl, bvec_rl)
        if denoise:
            img_rl = mc_denoise(img_rl)

    brainmaske = None
    if brain_mask is not None:
        if not ants.image_physical_space_consistency(brain_mask, reference_b0):
            raise ValueError("Provided brain mask should be in reference_b0 space")
        brainmaske = ants.iMath(brain_mask, "ME", 2)

    reg_rl = (
        dti_reg(
            img_rl,
            avg_b0=reference_b0,
            avg_dwi=reference_dwi,
            bvals=bval_rl,
            bvecs=bvec_rl,
            type_of_transform=motion_correct,
            brain_mask_eroded=brainmaske,
            verbose=verbose,
        )
        if img_rl is not None
        else None
    )

    reg_lr = dti_reg(
        img_lr,
        avg_b0=reference_b0,
        avg_dwi=reference_dwi,
        bvals=bval_lr,
        bvecs=bvec_lr,
        type_of_transform=motion_correct,
        brain_mask_eroded=brainmaske,
        verbose=verbose,
    )

    reg_its = [100, 50, 10]
    img_lrdwp = ants.image_clone(reg_lr["motion_corrected"])
    if img_rl is not None:
        img_rldwp = ants.image_clone(reg_rl["motion_corrected"])
        if srmodel is not None:
            img_rldwp = super_res_mcimage(img_rldwp, srmodel, isotropic=True, verbose=verbose)
    if srmodel is not None:
        reg_its = [100] + reg_its
        img_lrdwp = super_res_mcimage(img_lrdwp, srmodel, isotropic=True, verbose=verbose)

    if impute:
        img_lrdwp = impute_dwi(img_lrdwp, verbose=verbose)
    elif censor:
        img_lrdwp, reg_lr["bvals"], reg_lr["bvecs"] = censor_dwi(img_lrdwp, reg_lr["bvals"], reg_lr["bvecs"], verbose=verbose)

    if impute and img_rl is not None:
        img_rldwp = impute_dwi(img_rldwp, verbose=verbose)
    elif censor and img_rl is not None:
        img_rldwp, reg_rl["bvals"], reg_rl["bvecs"] = censor_dwi(img_rldwp, reg_rl["bvals"], reg_rl["bvecs"], verbose=verbose)

    if img_rl is not None:
        img_lrdwp, bval_lr_use, bvec_lr_use = merge_dwi_data(
            img_lrdwp, reg_lr["bvals"], reg_lr["bvecs"],
            img_rldwp, reg_rl["bvals"], reg_rl["bvecs"],
        )
    else:
        bval_lr_use = reg_lr["bvals"]
        bvec_lr_use = reg_lr["bvecs"]

    recon_lr_dewarp = dipy_dti_recon(
        img_lrdwp,
        bval_lr_use,
        bvec_lr_use,
        mask=brain_mask,
        fit_method=fit_method,
        mask_dilation=0,
        diffusion_model=diffusion_model,
        verbose=verbose,
    )

    framewise_displacement = (
        np.concatenate([reg_lr["FD"], reg_rl["FD"]]) if img_rl is not None else reg_lr["FD"]
    )
    motion_count = int((framewise_displacement > 1.5).sum())
    recon_fa = recon_lr_dewarp["FA"]
    recon_md = recon_lr_dewarp["MD"]

    if jhu_atlas is not None and jhu_labels is not None:
        or_fa2jhureg = ants.registration(
            recon_fa, jhu_atlas, type_of_transform="antsRegistrationSyNQuickRepro[s]",
            reg_iterations=reg_its, verbose=False
        )
        or_fa_jhulabels = ants.apply_transforms(
            recon_fa, jhu_labels, or_fa2jhureg["fwdtransforms"], interpolator="genericLabel"
        )
        df_fa_jhu = antspyt1w.map_intensity_to_dataframe("FA_JHU_labels_edited", recon_fa, or_fa_jhulabels)
        df_fa_wide = antspyt1w.merge_hierarchical_csvs_to_wide_format({"df_FA_JHU_ORRL": df_fa_jhu}, col_names=["Mean"])
        df_md_jhu = antspyt1w.map_intensity_to_dataframe("MD_JHU_labels_edited", recon_md, or_fa_jhulabels)
        df_md_wide = antspyt1w.merge_hierarchical_csvs_to_wide_format({"df_MD_JHU_ORRL": df_md_jhu}, col_names=["Mean"])
    else:
        or_fa2jhureg = None
        or_fa_jhulabels = None
        df_fa_wide = pd.DataFrame({"FA_mean": [float(recon_fa.mean())]})
        df_md_wide = pd.DataFrame({"MD_mean": [float(recon_md.mean())]})

    temp = segment_timeseries_by_meanvalue(img_lrdwp)
    b0_idx = temp["highermeans"]
    non_b0_idx = temp["lowermeans"]

    fgmask = ants.threshold_image(recon_fa, 0.5, 1.0).iMath("GetLargestComponent")
    bgmask = ants.threshold_image(recon_fa, 1e-4, 0.1)
    fa_snr = mask_snr(recon_fa, bgmask, fgmask, bias_correct=False)
    fa_evr = antspyt1w.patch_eigenvalue_ratio(recon_fa, 512, [16, 16, 16], evdepth=0.9, mask=recon_lr_dewarp["dwi_mask"])
    dti_itself = get_dti(recon_fa, recon_lr_dewarp["tensormodel"], return_image=True)

    return convert_np_in_dict({
        "dti": dti_itself,
        "recon_fa": recon_fa,
        "recon_fa_summary": df_fa_wide,
        "recon_md": recon_md,
        "recon_md_summary": df_md_wide,
        "jhu_labels": or_fa_jhulabels,
        "jhu_registration": or_fa2jhureg,
        "reg_LR": reg_lr,
        "reg_RL": reg_rl,
        "dtrecon_LR_dewarp": recon_lr_dewarp,
        "dwi_LR_dewarped": img_lrdwp,
        "bval_unique_count": len(np.unique(bval_lr_use)),
        "bval_LR": bval_lr_use,
        "bvec_LR": bvec_lr_use,
        "bval_RL": bval_rl,
        "bvec_RL": bvec_rl,
        "b0avg": reference_b0,
        "dwiavg": reference_dwi,
        "framewise_displacement": framewise_displacement,
        "high_motion_count": motion_count,
        "tsnr_b0": tsnr(img_lrdwp, recon_lr_dewarp["dwi_mask"], b0_idx),
        "tsnr_dwi": tsnr(img_lrdwp, recon_lr_dewarp["dwi_mask"], non_b0_idx),
        "dvars_b0": dvars(img_lrdwp, recon_lr_dewarp["dwi_mask"], b0_idx),
        "dvars_dwi": dvars(img_lrdwp, recon_lr_dewarp["dwi_mask"], non_b0_idx),
        "ssnr_b0": slice_snr(img_lrdwp, bgmask, fgmask, b0_idx),
        "ssnr_dwi": slice_snr(img_lrdwp, bgmask, fgmask, non_b0_idx),
        "fa_evr": fa_evr,
        "fa_SNR": fa_snr,
    })


def dwi_deterministic_tracking(
    dwi: ants.ANTsImage,
    fa: ants.ANTsImage,
    bvals: Any,
    bvecs: Any,
    num_processes: int = 1,
    mask: ants.ANTsImage | None = None,
    label_image: ants.ANTsImage | None = None,
    seed_labels: list[int] | None = None,
    fa_thresh: float = 0.05,
    seed_density: int = 1,
    step_size: float = 0.15,
    peak_indices: Any = None,
    fit_method: str = "WLS",
    verbose: bool = False,
) -> dict[str, Any]:
    """Deterministic tractography using DiPy."""
    from dipy.data import get_sphere
    from dipy.direction import peaks_from_model
    from dipy.tracking import utils
    from dipy.tracking.local_tracking import LocalTracking
    from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
    from dipy.tracking.streamline import Streamlines

    affine = ants_to_nibabel_affine(dwi)
    if isinstance(bvals, str) or isinstance(bvecs, str):
        bvals, bvecs = read_bvals_bvecs(bvals, bvecs)
    bvecs = repair_bvecs(bvecs)
    gtab = gradient_table(bvals, bvecs=bvecs, atol=2.0)

    if mask is None:
        mask = ants.threshold_image(fa, fa_thresh, 2.0).iMath("GetLargestComponent")
    dwi_data = dwi.numpy()
    dwi_mask = mask.numpy() == 1
    dti_model = dti.TensorModel(gtab, fit_method=fit_method)

    if peak_indices is None:
        sphere = get_sphere(name="symmetric362")
        peak_indices = peaks_from_model(
            model=dti_model,
            data=dwi_data,
            sphere=sphere,
            relative_peak_threshold=0.5,
            min_separation_angle=25,
            mask=dwi_mask,
            npeaks=3,
            return_odf=False,
            return_sh=False,
            parallel=num_processes > 1,
            num_processes=num_processes,
        )

    stopping_criterion = ThresholdStoppingCriterion(fa.numpy(), fa_thresh)
    if label_image is None or seed_labels is None:
        seed_mask = (fa.numpy() >= fa_thresh).astype(float)
    else:
        labels = label_image.numpy()
        seed_mask = np.isin(labels, seed_labels).astype(float)

    seeds = utils.seeds_from_mask(seed_mask, affine=affine, density=seed_density)
    generator = LocalTracking(peak_indices, stopping_criterion, seeds, affine=affine, step_size=step_size)
    streamlines = Streamlines(generator)

    return {
        "tractogram": None,
        "streamlines": streamlines,
        "peak_indices": peak_indices,
    }


def dwi_closest_peak_tracking(
    dwi: ants.ANTsImage,
    fa: ants.ANTsImage,
    bvals: Any,
    bvecs: Any,
    num_processes: int = 1,
    mask: ants.ANTsImage | None = None,
    label_image: ants.ANTsImage | None = None,
    seed_labels: list[int] | None = None,
    fa_thresh: float = 0.05,
    seed_density: int = 1,
    step_size: float = 0.15,
    verbose: bool = False,
) -> dict[str, Any]:
    """Closest peak tractography using CSD."""
    from dipy.data import small_sphere
    from dipy.direction import ClosestPeakDirectionGetter
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
    from dipy.reconst.shm import CsaOdfModel
    from dipy.tracking import utils
    from dipy.tracking.local_tracking import LocalTracking
    from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
    from dipy.tracking.streamline import Streamlines

    affine = ants_to_nibabel_affine(dwi)
    if isinstance(bvals, str) or isinstance(bvecs, str):
        bvals, bvecs = read_bvals_bvecs(bvals, bvecs)
    bvecs = repair_bvecs(bvecs)
    gtab = gradient_table(bvals, bvecs=bvecs, atol=2.0)

    if mask is None:
        mask = ants.threshold_image(fa, fa_thresh, 2.0).iMath("GetLargestComponent")
    dwi_data = dwi.numpy()
    dwi_mask = mask.numpy() == 1

    response, _ = auto_response_ssst(gtab, dwi_data, roi_radii=10, fa_thr=0.7)
    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
    csd_fit = csd_model.fit(dwi_data, mask=dwi_mask)
    csa_model = CsaOdfModel(gtab, sh_order=6)
    gfa = csa_model.fit(dwi_data, mask=dwi_mask).gfa
    stopping_criterion = ThresholdStoppingCriterion(gfa, 0.25)

    if label_image is None or seed_labels is None:
        seed_mask = (fa.numpy() >= fa_thresh).astype(float)
    else:
        labels = label_image.numpy()
        seed_mask = np.isin(labels, seed_labels).astype(float)

    seeds = utils.seeds_from_mask(seed_mask, affine=affine, density=seed_density)
    pmf = csd_fit.odf(small_sphere).clip(min=0)
    peak_dg = ClosestPeakDirectionGetter.from_pmf(pmf, max_angle=30.0, sphere=small_sphere)
    generator = LocalTracking(peak_dg, stopping_criterion, seeds, affine, step_size=step_size)
    streamlines = Streamlines(generator)

    return {
        "tractogram": None,
        "streamlines": streamlines,
    }


def dwi_streamline_pairwise_connectivity(
    streamlines: Any,
    label_image: ants.ANTsImage,
    labels_to_connect: list[int | None] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Return streamlines connecting all regions in the label set (ideal for 2 regions)."""
    from dipy.tracking import utils
    from dipy.tracking.streamline import Streamlines

    if labels_to_connect is None:
        labels_to_connect = [1, None]
    keep_streamlines = Streamlines()
    affine = ants_to_nibabel_affine(label_image)
    lin_t, offset = utils._mapping_to_voxel(affine)
    label_image_np = label_image.numpy()

    def check_it(sl: np.ndarray) -> tuple[bool, int]:
        for idx in range(sl.shape[0]):
            pt = utils._to_voxel_coordinates(sl[idx, :], lin_t, offset)
            mylab = int(label_image_np[pt[0], pt[1], pt[2]])
            if mylab == labels_to_connect[0] or mylab == labels_to_connect[1]:
                return True, mylab
        return False, 0

    ct = 0
    for k in range(len(streamlines)):
        sl = streamlines[k]
        ok, mylab = check_it(sl)
        if ok:
            otherind = 0 if mylab == labels_to_connect[1] else 1
            lsl = len(sl) - 1
            pt = utils._to_voxel_coordinates(sl[lsl, :], lin_t, offset)
            mylab_end = int(label_image_np[pt[0], pt[1], pt[2]])
            accept_point = mylab_end != 0 if labels_to_connect[1] is None else mylab_end == labels_to_connect[otherind]
            if accept_point:
                keep_streamlines.append(sl)
                ct += 1

    return {"streamlines": keep_streamlines, "count": ct}


def dwi_streamline_connectivity(
    streamlines: Any,
    label_image: ants.ANTsImage,
    label_dataframe: pd.DataFrame,
    verbose: bool = False,
) -> dict[str, Any]:
    """Summarize network connectivity of streamlines between all regions."""
    import antspyt1w
    from dipy.tracking import utils

    affine = ants_to_nibabel_affine(label_image)
    ulabs = label_dataframe["Label"]
    labels_to_connect = set(ulabs[ulabs > 0])
    lin_t, offset = utils._mapping_to_voxel(affine)
    label_image_np = label_image.numpy()

    def check_it(sl: np.ndarray, index: int, not_label: int | None = None) -> tuple[bool, int]:
        pt = utils._to_voxel_coordinates(sl[index, :], lin_t, offset)
        mylab = int(label_image_np[pt[0], pt[1], pt[2]])
        if not_label is None:
            if mylab in labels_to_connect:
                return True, mylab
        else:
            if mylab in labels_to_connect and mylab != not_label:
                return True, mylab
        return False, 0

    my_count = np.zeros([len(ulabs), len(ulabs)])
    for k in range(len(streamlines)):
        sl = streamlines[k]
        ok1, lab1 = check_it(sl, 0)
        if ok1:
            ok2, lab2 = check_it(sl, len(sl) - 1, not_label=lab1)
            if ok2:
                my_count[ulabs == lab1, ulabs == lab2] += 1

    ctdf = label_dataframe.copy()
    for k in range(len(ulabs)):
        nn3 = f"CnxCount{str(k).zfill(3)}"
        ctdf.insert(ctdf.shape[1], nn3, my_count[k, :])
    ctdfw = antspyt1w.merge_hierarchical_csvs_to_wide_format(
        {"networkc": ctdf}, ctdf.keys()[2 : ctdf.shape[1]]
    )
    return {"connectivity_matrix": my_count, "connectivity_wide": ctdfw}
