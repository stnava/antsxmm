"""Registration, motion correction, transformation, and template construction algorithms.

This module unifies:
- Motion correction for 4D functional, perfusion, and diffusion timeseries (timeseries_reg, mc_reg, dti_reg).
- Iterative group and session template construction (dti_template, dewarp_imageset, get_average_dwi_b0, get_average_rsf).
- Spatial transformations and tensor reorientation (timeseries_transform, tra_initializer,
  apply_transforms_mixed_interpolation, read_ants_transforms_to_numpy, transform_and_reorient_dti,
  distortion_correct_bvecs, bvec_reorientation, deformation_gradient_optimized, concat_dewarp).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from typing import Any

import ants
import numpy as np
import pandas as pd


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
    return (interp_linear * mask) + (interp_nn * (1.0 - mask))


def read_ants_transforms_to_numpy(transform_files: list[list[str]]) -> np.ndarray:
    """Read a list of ANTs transformation files into a numpy array."""
    matrices = []
    for file in transform_files:
        tx = ants.read_transform(file[0])
        matrix = tx.parameters[:9].reshape((3, 3))
        matrices.append(matrix)
    return np.array(matrices)


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
    from .dti import dti_numpy_to_image, triangular_to_tensor

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


def generate_voxelwise_bvecs(
    global_bvecs: np.ndarray,
    voxel_rotations: np.ndarray,
    transpose: bool = False,
) -> np.ndarray:
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


def bvec_reorientation(
    motion_parameters: list[Any] | None,
    bvecs: np.ndarray,
    rebase: np.ndarray | None = None,
) -> np.ndarray:
    """Reorient b-vectors using motion correction transformation parameters."""
    if motion_parameters is None:
        raise ValueError("motion_parameters cannot be None")
    bvecs = np.asarray(bvecs, dtype=float).copy()
    for myidx in range(len(motion_parameters)):
        if motion_parameters[myidx] != "NA":
            rotmat = ants.read_transform(motion_parameters[myidx][0]).parameters[0:9].reshape(3, 3)
            det = np.linalg.det(rotmat)
            if abs(det) > 1e-8:
                u, _, vt = np.linalg.svd(rotmat)
                rotation_matrix = u @ vt
                bvecs[myidx, :] = rotation_matrix @ bvecs[myidx, :]
            norm = np.linalg.norm(bvecs[myidx, :])
            if norm > 1e-8:
                bvecs[myidx, :] = bvecs[myidx, :] / norm
            if rebase is not None:
                bvecs[myidx, :] = rebase @ bvecs[myidx, :]
    return bvecs


def distortion_correct_bvecs(
    bvecs: np.ndarray,
    displacement_field: ants.ANTsImage,
    output_filename: str | None = None,
) -> np.ndarray:
    """Correct b-vectors for non-rigid distortion using deformation gradient field."""
    _, f_matrices = deformation_gradient_optimized(displacement_field)
    u, _, vt = np.linalg.svd(f_matrices)
    voxel_rotations = np.matmul(u, vt)
    voxelwise_bvecs = generate_voxelwise_bvecs(bvecs, voxel_rotations)
    if output_filename is not None:
        np.save(output_filename, voxelwise_bvecs)
    return voxelwise_bvecs


def concat_dewarp(
    dewarp_motion: dict[str, Any],
    dewarp_syn: dict[str, Any],
    time_series_length: int,
) -> list[list[str]]:
    """Concatenate motion and SyN dewarp transform files across a time series."""
    tx_concat = []
    for k in range(time_series_length):
        tx_concat.append(dewarp_syn["fwdtransforms"] + dewarp_motion["motion_parameters"][k])
    return tx_concat


def get_average_dwi_b0(
    x: ants.ANTsImage,
    fixed_b0: ants.ANTsImage | None = None,
    bvals: Any = None,
    b0_idx: list[int] | None = None,
    b0_threshold: float = 50.0,
    motion_correct: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Extract and motion-correct b0 volumes from a DWI image."""
    bvals_numeric: np.ndarray | None = None
    if bvals is not None:
        if isinstance(bvals, (str, os.PathLike)):
            bvals_numeric = np.loadtxt(bvals)
        else:
            bvals_numeric = np.array(bvals, dtype=float)

    if b0_idx is None:
        if bvals_numeric is None:
            raise ValueError("Must provide either `bvals` or `b0_idx`.")
        b0_idx = np.where(bvals_numeric <= b0_threshold)[0].tolist()

    if len(b0_idx) == 0:
        raise ValueError("No b0 volumes found with provided criteria.")

    b0_list = [ants.slice_image(x, axis=3, idx=int(idx)) for idx in b0_idx]
    if fixed_b0 is None:
        fixed_b0 = b0_list[0]

    b0_avg = fixed_b0.clone()
    transforms = []

    if motion_correct and len(b0_list) > 1:
        reg_b0_list = []
        for i, b0 in enumerate(b0_list):
            if i == 0 and fixed_b0 == b0:
                reg_b0_list.append(b0)
                transforms.append([])
            else:
                reg = ants.registration(
                    fixed=fixed_b0,
                    moving=b0,
                    type_of_transform="antsRegistrationSyNRepro[r]",
                    verbose=verbose,
                )
                reg_b0_list.append(reg["warpedmovout"])
                transforms.append(reg["fwdtransforms"])

        b0_sum = reg_b0_list[0].clone()
        for b0 in reg_b0_list[1:]:
            b0_sum = b0_sum + b0
        b0_avg = b0_sum / float(len(reg_b0_list))
    else:
        b0_sum = b0_list[0].clone()
        for b0 in b0_list[1:]:
            b0_sum = b0_sum + b0
        b0_avg = b0_sum / float(len(b0_list))

    return {
        "b0_avg": b0_avg,
        "b0_idx": b0_idx,
        "transforms": transforms,
    }


def get_average_rsf(x: ants.ANTsImage, min_t: int = 10, max_t: int = 35) -> ants.ANTsImage:
    """Automatically generates the average rsfMRI/BOLD image with quick two-pass registration."""
    idim = x.dimension
    n_time_points = x.shape[idim - 1]
    if n_time_points <= min_t:
        min_t = 0
    if n_time_points <= max_t:
        max_t = n_time_points

    if max_t <= min_t:
        return ants.slice_image(x, axis=idim - 1, idx=0)

    output_directory = tempfile.mkdtemp()
    ofn = os.path.join(output_directory, "w")
    try:
        bavg = ants.slice_image(x, axis=idim - 1, idx=0) * 0.0
        oavg = ants.slice_image(x, axis=idim - 1, idx=0)
        for myidx in range(min_t, max_t):
            b0 = ants.slice_image(x, axis=idim - 1, idx=myidx)
            reg = ants.registration(oavg, b0, "antsRegistrationSyNRepro[r]", outprefix=ofn)
            bavg = bavg + reg["warpedmovout"]
        bavg = ants.iMath(bavg, "Normalize")
        oavg = ants.image_clone(bavg)
        bavg = oavg * 0.0
        for myidx in range(min_t, max_t):
            b0 = ants.slice_image(x, axis=idim - 1, idx=myidx)
            reg = ants.registration(oavg, b0, "antsRegistrationSyNRepro[r]", outprefix=ofn)
            bavg = bavg + reg["warpedmovout"]
        bavg = ants.iMath(bavg, "Normalize")
        return bavg
    finally:
        shutil.rmtree(output_directory, ignore_errors=True)


def timeseries_transform(
    transform: Any,
    image: ants.ANTsImage,
    reference: ants.ANTsImage,
    interpolation: str = "linear",
) -> ants.ANTsImage:
    """Apply spatial transformation across all volumes in a 4D timeseries."""
    n_volumes = image.shape[-1]
    warped_volumes = []
    for idx in range(n_volumes):
        vol = ants.slice_image(image, axis=3, idx=idx)
        warped_vol = ants.apply_transforms(
            fixed=reference,
            moving=vol,
            transformlist=transform,
            interpolator=interpolation,
        )
        warped_volumes.append(warped_vol)
    return ants.list_to_ndimage(image, warped_volumes)


def tra_initializer(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    iterations: int = 3,
    search_factor: float = 0.5,
    verbose: bool = False,
) -> list[str]:
    """Iterative affine registration initializer for neuromelanin/axial slabs."""
    moving_work = moving.clone()
    tx_list: list[str] = []

    for i in range(iterations):
        reg = ants.registration(
            fixed=fixed,
            moving=moving_work,
            type_of_transform="Rigid",
            aff_metric="meansquares",
            aff_sampling=32,
            verbose=verbose,
        )
        tx_list = reg["fwdtransforms"] + tx_list
        moving_work = reg["warpedmovout"]

        if i < iterations - 1:
            aff_reg = ants.registration(
                fixed=fixed,
                moving=moving_work,
                type_of_transform="Affine",
                aff_metric="meansquares",
                aff_sampling=32,
                verbose=verbose,
            )
            tx_list = aff_reg["fwdtransforms"] + tx_list
            moving_work = aff_reg["warpedmovout"]

    return tx_list


def timeseries_reg(
    image: ants.ANTsImage,
    fixed: ants.ANTsImage | None = None,
    type_of_transform: str = "antsRegistrationSyNRepro[r]",
    mask: ants.ANTsImage | None = None,
    total_sigma: float = 3.0,
    fdOffset: float = 2.0,
    output_directory: str | None = None,
    verbose: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Motion-correct time-series data and compute framewise displacement."""
    remove_it = False
    if output_directory is None:
        remove_it = True
        output_directory = tempfile.mkdtemp()
    output_directory_w = os.path.join(output_directory, "timeseries_reg")
    os.makedirs(output_directory_w, exist_ok=True)
    ofn_prefix = os.path.join(output_directory_w, "tx_")

    idim = image.dimension
    n_time_points = image.shape[idim - 1]
    if fixed is None:
        fixed = ants.get_average_of_timeseries(image)
    if mask is None:
        mask = ants.get_mask(fixed)

    fd = np.zeros(n_time_points)
    motion_parameters: list[Any] = []
    motion_corrected: list[ants.ANTsImage] = []

    center_of_mass = mask.get_center_of_mass()
    myrad = np.ones(idim - 1, dtype=int).tolist()
    mask1vals = np.zeros(int(mask.sum()))
    mask1vals[round(len(mask1vals) / 2)] = 1
    mask1 = ants.make_image(mask, mask1vals)
    myoffsets = ants.get_neighborhood_in_mask(mask1, mask1, radius=myrad, spatial_info=True)["offsets"]
    mycols = list("xy") if idim - 1 == 2 else list("xyz")

    useinds = [k for k in range(myoffsets.shape[0]) if abs(myoffsets[k, :]).sum() == (idim - 2)]
    myoffsets = myoffsets * fdOffset / 2.0 + center_of_mass
    fdpts = pd.DataFrame(data=myoffsets[useinds, :], columns=mycols)

    for k in range(n_time_points):
        temp = ants.slice_image(image, axis=idim - 1, idx=k)
        temp = ants.iMath(temp, "Normalize")
        if temp.numpy().var() > 0:
            tx_prefix_k = f"{ofn_prefix}{str(k).zfill(4)}_"
            myrig = ants.registration(
                fixed,
                temp,
                type_of_transform=type_of_transform,
                outprefix=tx_prefix_k,
                **kwargs,
            )
            myreg = myrig
            fdpts_tx = ants.apply_transforms_to_points(idim - 1, fdpts, myreg["fwdtransforms"])
            fdpts_prev = (
                ants.apply_transforms_to_points(idim - 1, fdpts, motion_parameters[k - 1])
                if k > 0 and motion_parameters[k - 1] != "NA"
                else fdpts_tx
            )
            fd[k] = (fdpts_prev - fdpts_tx).abs().mean().sum()
            motion_parameters.append(myreg["fwdtransforms"])
            img_warped = ants.apply_transforms(fixed, ants.slice_image(image, axis=idim - 1, idx=k), myreg["fwdtransforms"])
            motion_corrected.append(img_warped)
        else:
            motion_parameters.append("NA")
            motion_corrected.append(temp)

    if remove_it:
        shutil.rmtree(output_directory, ignore_errors=True)

    return {
        "motion_corrected": ants.list_to_ndimage(image, motion_corrected),
        "motion_parameters": motion_parameters,
        "FD": fd,
    }


def mc_reg(
    image: ants.ANTsImage,
    fixed: ants.ANTsImage | None = None,
    type_of_transform: str = "antsRegistrationSyNRepro[r]",
    mask: ants.ANTsImage | None = None,
    total_sigma: float = 3.0,
    fdOffset: float = 2.0,
    output_directory: str | None = None,
    verbose: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Correct time-series data for motion with optional deformation."""
    remove_it = False
    if output_directory is None:
        remove_it = True
        output_directory = tempfile.mkdtemp()
    output_directory_w = os.path.join(output_directory, "mc_reg")
    os.makedirs(output_directory_w, exist_ok=True)
    ofn_l = os.path.join(output_directory_w, "local_def_")

    idim = image.dimension
    ishape = image.shape
    n_time_points = ishape[idim - 1]
    if fixed is None:
        fixed = ants.get_average_of_timeseries(image)
    if mask is None:
        mask = ants.get_mask(fixed)

    fd = np.zeros(n_time_points)
    motion_parameters: list[Any] = []
    motion_corrected: list[ants.ANTsImage] = []
    center_of_mass = mask.get_center_of_mass()
    myrad = np.ones(idim - 1, dtype=int).tolist()
    mask1vals = np.zeros(int(mask.sum()))
    mask1vals[round(len(mask1vals) / 2)] = 1
    mask1 = ants.make_image(mask, mask1vals)
    myoffsets = ants.get_neighborhood_in_mask(mask1, mask1, radius=myrad, spatial_info=True)["offsets"]
    mycols = list("xy") if idim - 1 == 2 else list("xyz")

    useinds = [k for k in range(myoffsets.shape[0]) if abs(myoffsets[k, :]).sum() == (idim - 2)]
    myoffsets = myoffsets * fdOffset / 2.0 + center_of_mass
    fdpts = pd.DataFrame(data=myoffsets[useinds, :], columns=mycols)

    for k in range(n_time_points):
        temp = ants.slice_image(image, axis=idim - 1, idx=k)
        temp = ants.iMath(temp, "Normalize")
        if temp.numpy().var() > 0:
            tx_prefix_k = f"{ofn_l}{str(k).zfill(4)}_"
            myrig = ants.registration(
                fixed,
                temp,
                type_of_transform="antsRegistrationSyNRepro[r]",
                outprefix=tx_prefix_k,
                **kwargs,
            )
            if type_of_transform == "SyN":
                myreg = ants.registration(
                    fixed,
                    temp,
                    type_of_transform="SyNOnly",
                    total_sigma=total_sigma,
                    initial_transform=myrig["fwdtransforms"][0],
                    outprefix=tx_prefix_k,
                    **kwargs,
                )
            else:
                myreg = myrig
            fdpts_tx = ants.apply_transforms_to_points(idim - 1, fdpts, myreg["fwdtransforms"])
            fdpts_prev = (
                ants.apply_transforms_to_points(idim - 1, fdpts, motion_parameters[k - 1])
                if k > 0 and motion_parameters[k - 1] != "NA"
                else fdpts_tx
            )
            fd[k] = (fdpts_prev - fdpts_tx).abs().mean().sum()
            motion_parameters.append(myreg["fwdtransforms"])
            img_warped = ants.apply_transforms(
                fixed,
                ants.slice_image(image, axis=idim - 1, idx=k),
                myreg["fwdtransforms"],
            )
            motion_corrected.append(img_warped)
        else:
            motion_parameters.append("NA")
            motion_corrected.append(temp)

    if remove_it:
        shutil.rmtree(output_directory, ignore_errors=True)

    return {
        "motion_corrected": ants.list_to_ndimage(image, motion_corrected),
        "motion_parameters": motion_parameters,
        "FD": fd,
    }


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
    motion_parameters: list[Any] = []
    motion_corrected: list[ants.ANTsImage] = []

    if bvals is None:
        raise ValueError("bvals must be provided")

    bvals = np.array(bvals)
    b0_idx = np.where(bvals <= 50.0)[0] if b0_idx is None else np.array(b0_idx)
    dwi_idx = np.where(bvals > 50.0)[0]

    b0_bvecs = None
    if bvecs is not None:
        bvecs = np.array(bvecs)
        b0_bvecs = bvecs[b0_idx, :]

    ab0 = get_average_dwi_b0(
        image,
        fixed_b0=avg_b0,
        b0_idx=b0_idx.tolist(),
        verbose=verbose,
    )["b0_avg"]

    b0_reg = ants.registration(
        avg_b0,
        ab0,
        type_of_transform="antsRegistrationSyNRepro[a]",
        verbose=verbose,
    )
    deftx = b0_reg["fwdtransforms"]

    remove_it = False
    if output_directory is None:
        remove_it = True
        output_directory = tempfile.mkdtemp()
    output_directory_w = os.path.join(output_directory, "dti_reg")
    os.makedirs(output_directory_w, exist_ok=True)
    ofn_prefix = os.path.join(output_directory_w, "tx_")

    mask = ants.get_mask(avg_dwi)
    center_of_mass = mask.get_center_of_mass()
    myrad = np.ones(idim - 1, dtype=int).tolist()
    mask1vals = np.zeros(int(mask.sum()))
    mask1vals[round(len(mask1vals) / 2)] = 1
    mask1 = ants.make_image(mask, mask1vals)
    myoffsets = ants.get_neighborhood_in_mask(mask1, mask1, radius=myrad, spatial_info=True)["offsets"]
    mycols = list("xy") if idim - 1 == 2 else list("xyz")

    useinds = [k for k in range(myoffsets.shape[0]) if abs(myoffsets[k, :]).sum() == (idim - 2)]
    myoffsets = myoffsets * fdOffset / 2.0 + center_of_mass
    fdpts = pd.DataFrame(data=myoffsets[useinds, :], columns=mycols)

    for k in range(n_time_points):
        temp = ants.slice_image(image, axis=idim - 1, idx=k)
        temp = ants.iMath(temp, "Normalize")
        fixed = avg_b0 if k in b0_idx else avg_dwi

        if temp.numpy().var() > 0:
            txprefix1 = f"{ofn_prefix}rig_{str(k).zfill(4)}_"
            txprefix2 = f"{ofn_prefix}syn_{str(k).zfill(4)}_"
            myrig = ants.registration(fixed, temp, type_of_transform="antsRegistrationSyNRepro[r]", outprefix=txprefix1, **kwargs)
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


def dewarp_imageset(
    image_list: list[ants.ANTsImage],
    initial_template: ants.ANTsImage | None = None,
    iterations: int = 2,
    padding: int = 0,
    target_idx: list[int] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Dewarp a set of 2D, 3D, or 4D images into an unbiased shape-space reference."""
    if target_idx is None:
        target_idx = [0]
    outlist = []
    avglist = []
    if len(image_list[0].shape) > 3:
        imagetype = 3
        for k in range(len(image_list)):
            for j in range(len(target_idx)):
                avglist.append(ants.slice_image(image_list[k], axis=3, idx=target_idx[j]))
    else:
        imagetype = 0
        avglist = list(image_list)

    if padding > 0:
        pw = [padding] * len(avglist[0].shape)
        for k in range(len(avglist)):
            avglist[k] = ants.pad_image(avglist[k], pad_width=pw)

    if initial_template is None:
        initial_template = avglist[0] * 0.0
        for k in range(len(avglist)):
            initial_template = initial_template + avglist[k] / float(len(avglist))

    btp = ants.build_template(
        initial_template=initial_template,
        image_list=avglist,
        gradient_step=0.5,
        blending_weight=0.8,
        iterations=iterations,
        verbose=False,
        **kwargs,
    )

    mocoplist = []
    mocofdlist = []
    reglist = []
    for k in range(len(image_list)):
        if imagetype == 3:
            moco0 = ants.motion_correction(
                image=image_list[k],
                fixed=btp,
                type_of_transform="antsRegistrationSyNRepro[r]",
            )
            mocoplist.append(moco0["motion_parameters"])
            mocofdlist.append(moco0["FD"])
            locavg = ants.slice_image(moco0["motion_corrected"], axis=3, idx=0) * 0.0
            for j in range(len(target_idx)):
                locavg = locavg + ants.slice_image(moco0["motion_corrected"], axis=3, idx=target_idx[j])
            locavg = locavg * (1.0 / len(target_idx))
        else:
            locavg = image_list[k]
            moco0 = None

        reg = ants.registration(btp, locavg, **kwargs)
        reglist.append(reg)

        if imagetype == 3 and moco0 is not None:
            myishape = image_list[k].shape
            mytslength = myishape[-1]
            mywarpedlist = []
            for j in range(mytslength):
                locimg = ants.slice_image(image_list[k], axis=3, idx=j)
                mywarped = ants.apply_transforms(
                    btp, locimg, reg["fwdtransforms"] + moco0["motion_parameters"][j], imagetype=0
                )
                mywarpedlist.append(mywarped)
            mywarped_img = ants.list_to_ndimage(image_list[k], mywarpedlist)
        else:
            mywarped_img = ants.apply_transforms(btp, image_list[k], reg["fwdtransforms"], imagetype=imagetype)
        outlist.append(mywarped_img)

    return {
        "dewarpedmean": btp,
        "dewarped": outlist,
        "deformable_registrations": reglist,
        "FD": mocofdlist,
        "motionparameters": mocoplist,
    }


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
