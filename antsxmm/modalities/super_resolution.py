from __future__ import annotations

from typing import Any
import numpy as np
import ants


def down2iso(
    x: ants.ANTsImage,
    interpolation: str = "linear",
    takemin: bool = False,
) -> ants.ANTsImage:
    """Downsample an anisotropic image to an isotropic resolution."""
    spc = ants.get_spacing(x)
    if takemin:
        newspc = np.asarray(spc).min()
    else:
        newspc = np.asarray(spc).max()
    newspc_arr = np.repeat(newspc, x.dimension)
    if interpolation == "linear":
        return ants.resample_image(x, newspc_arr, interp_type=0)
    return ants.resample_image(x, newspc_arr, interp_type=1)


def super_res_mcimage(
    image: ants.ANTsImage,
    srmodel: Any,
    truncation: list[float] | None = None,
    poly_order: str | int | None = "hist",
    target_range: list[float] | tuple[float, float] | None = None,
    isotropic: bool = False,
    verbose: bool = False,
) -> ants.ANTsImage:
    """Super resolution on a timeseries or multi-channel image."""
    import antspynet

    if truncation is None:
        truncation = [0.0001, 0.995]
    if target_range is None:
        target_range = [0, 1]

    idim = image.dimension
    ishape = image.shape
    n_time_points = ishape[idim - 1]
    mcsr: list[ants.ANTsImage] = []
    mysr: ants.ANTsImage | None = None

    for k in range(n_time_points):
        if verbose and ((k % 5) == 0):
            mycount = round(k / n_time_points * 100)
            print(f"{mycount}%.", end="", flush=True)
        temp = ants.slice_image(image, axis=idim - 1, idx=k)
        temp = ants.iMath(temp, "TruncateIntensity", truncation[0], truncation[1])
        mysr = antspynet.apply_super_resolution_model_to_image(
            temp, srmodel, target_range=target_range
        )
        if poly_order is not None:
            bilin = ants.resample_image_to_target(temp, mysr)
            if poly_order == "hist":
                mysr = ants.histogram_match_image(mysr, bilin)
            else:
                mysr = ants.regression_match_image(mysr, bilin, poly_order=poly_order)
        if isotropic:
            mysr = down2iso(mysr)
        mcsr.append(mysr)

    if mysr is None:
        raise ValueError("Cannot super-resolve empty time series.")

    upshape: list[int] = [mysr.shape[j] for j in range(len(ishape) - 1)]
    upshape.append(ishape[idim - 1])
    if verbose:
        print(f"SR will be of voxel size: {upshape}")

    imageup = ants.resample_image(image, upshape, use_voxels=True)
    if verbose:
        print("Done")

    return ants.list_to_ndimage(imageup, mcsr)


def t1w_super_resolution_with_hemispheres(
    t1img: ants.ANTsImage,
    model: Any,
    dilation_amount: int = 8,
    truncation: list[float] | None = None,
    target_range: list[float] | None = None,
    poly_order: str | int = "hist",
    min_spacing: float = 0.8,
    verbose: bool = True,
) -> ants.ANTsImage:
    """Perform hemisphere-aware super-resolution on a T1-weighted image."""
    import antspynet
    import antspyt1w
    import siq

    if truncation is None:
        truncation = [0.001, 0.999]
    if target_range is None:
        target_range = [0, 1]
    if float(np.min(ants.get_spacing(t1img))) < min_spacing:
        if verbose:
            print("Image resolution too high — skipping SR.")
        return t1img

    if verbose:
        print("Performing brain extraction...")
    brain_mask = antspyt1w.brain_extraction(t1img)
    brain = t1img * brain_mask

    if verbose:
        print("Begin template loading")
    tlrfn = antspyt1w.get_data("T_template0_LR", target_extension=".nii.gz")
    tfn = antspyt1w.get_data("T_template0", target_extension=".nii.gz")
    template = ants.image_read(tfn)
    template = (template * antspynet.brain_extraction(template, "t1")).iMath("Normalize")
    template_lr = ants.image_read(tlrfn)
    if verbose:
        print("Done template loading")

    if verbose:
        print("Labeling hemispheres...")
    hemi_seg = antspyt1w.label_hemispheres(brain, template, template_lr)
    hemisphere_mask = hemi_seg + 2.0 * brain_mask

    if verbose:
        print("Starting segmentation-aware super-resolution...")
    sr_result = siq.inference(
        t1img,
        model,
        segmentation=hemisphere_mask,
        truncation=truncation,
        target_range=target_range,
        dilation_amount=dilation_amount,
        poly_order=poly_order,
        verbose=verbose,
    )

    sr_image = sr_result["super_resolution"] if isinstance(sr_result, dict) else sr_result
    if verbose:
        print("Done super-resolution.")
    return sr_image
