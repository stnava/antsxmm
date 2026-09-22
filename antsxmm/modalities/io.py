"""Serialization and artifact writing for multimodal neuroimaging outputs.

This module provides clean, typed serialization of modality outputs directly to
canonical directory structures and creates the `<prefix>+mmwide.csv` summary tables
without relying on legacy monolithic write_mm routines.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import ants
import numpy as np
import pandas as pd

from .metrics import dict_to_dataframe, get_antsimage_keys, image_write_with_thumbnail


def ensure_parent_dir(file_or_prefix: str | Path) -> Path:
    """Ensure the directory containing the given path or prefix exists."""
    p = Path(file_or_prefix)
    parent = p.parent if not p.is_dir() else p
    parent.mkdir(parents=True, exist_ok=True)
    return parent


def write_modality_mmwide(output_prefix: str, df: pd.DataFrame, separator: str = "+") -> str:
    """Write a wide metrics DataFrame to canonical `<output_prefix>+mmwide.csv`."""
    ensure_parent_dir(output_prefix)
    out_csv = f"{output_prefix}{separator}mmwide.csv"
    df.to_csv(out_csv, index=False)
    return out_csv


def write_flair_outputs(
    output_prefix: str,
    flair_result: dict[str, Any],
    separator: str = "+",
    visualize: bool = True,
) -> pd.DataFrame:
    """Serialize FLAIR/WMH segmentation results and write mmwide.csv."""
    ensure_parent_dir(output_prefix)
    op = f"{output_prefix}{separator}"

    # Write probability masks
    if flair_result.get("WMH_probability_map") is not None:
        wmh_img = flair_result["WMH_probability_map"]
        if ants.is_image(wmh_img):
            image_write_with_thumbnail(wmh_img, f"{op}wmh.nii.gz", thumb=False)

    if flair_result.get("WMH_posterior_probability_map") is not None:
        post_img = flair_result["WMH_posterior_probability_map"]
        if ants.is_image(post_img):
            image_write_with_thumbnail(post_img, f"{op}wmh_posterior.nii.gz", thumb=False)

    # Optional visualization
    if visualize and flair_result.get("flair") is not None and flair_result.get("WMH_posterior_probability_map") is not None:
        try:
            ants.plot(
                flair_result["flair"],
                flair_result["WMH_posterior_probability_map"],
                axis=2,
                nslices=21,
                ncol=7,
                filename=f"{op}wmh_seg.png",
                crop=True,
            )
        except Exception:
            pass

    # Produce tabular summary
    flwide = dict_to_dataframe(flair_result)
    write_modality_mmwide(output_prefix, flwide, separator=separator)
    return flwide


def write_dti_outputs(
    output_prefix: str,
    dti_result: dict[str, Any],
    separator: str = "+",
) -> pd.DataFrame:
    """Serialize DTI reconstruction outputs, maps, and write mmwide.csv."""
    from .dti import write_bvals_bvecs

    ensure_parent_dir(output_prefix)
    op = f"{output_prefix}{separator}"

    if ants.is_image(dti_result.get("dti")):
        ants.image_write(dti_result["dti"], f"{op}dti.nii.gz")

    if dti_result.get("bval_LR") is not None and dti_result.get("bvec_LR") is not None:
        try:
            write_bvals_bvecs(dti_result["bval_LR"], dti_result["bvec_LR"], f"{op}reoriented")
        except Exception:
            pass

    for key, suffix in [
        ("dwi_LR_dewarped", "dwi.nii.gz"),
        ("recon_fa", "dtifa.nii.gz"),
        ("recon_md", "dtimd.nii.gz"),
        ("b0avg", "b0avg.nii.gz"),
        ("dwiavg", "dwiavg.nii.gz"),
    ]:
        val = dti_result.get(key)
        if ants.is_image(val):
            image_write_with_thumbnail(val, f"{op}{suffix}", thumb=False)

    # JHU labels overlaid on FA
    if ants.is_image(dti_result.get("jhu_labels")):
        image_write_with_thumbnail(
            dti_result["jhu_labels"],
            f"{op}dtijhulabels.nii.gz",
            reference=dti_result.get("recon_fa"),
            thumb=False,
        )

    # Summary tabular data
    dfs_to_concat = []
    if isinstance(dti_result.get("recon_fa_summary"), pd.DataFrame):
        fa_sum = dti_result["recon_fa_summary"]
        dfs_to_concat.append(fa_sum.iloc[:, 1:] if fa_sum.shape[1] > 1 else fa_sum)

    if isinstance(dti_result.get("recon_md_summary"), pd.DataFrame):
        md_sum = dti_result["recon_md_summary"]
        dfs_to_concat.append(md_sum.iloc[:, 1:] if md_sum.shape[1] > 1 else md_sum)

    if dfs_to_concat:
        dti_wide = pd.concat(dfs_to_concat, axis=1)
    else:
        dti_wide = pd.DataFrame({"u_dti_id": [output_prefix]})

    # Add scalar QC summary metrics if present
    for metric_key in ["tsnr_b0", "tsnr_dwi", "dvars_b0", "dvars_dwi", "ssnr_b0", "ssnr_dwi"]:
        val = dti_result.get(metric_key)
        if val is not None:
            try:
                dti_wide[f"dti_{metric_key}_mean"] = float(np.mean(val))
            except Exception:
                pass

    for scalar_key in ["fa_evr", "fa_SNR", "high_motion_count"]:
        if scalar_key in dti_result and dti_result[scalar_key] is not None:
            dti_wide[f"dti_{scalar_key}"] = dti_result[scalar_key]

    write_modality_mmwide(output_prefix, dti_wide, separator=separator)
    return dti_wide


def write_rsf_outputs(
    output_prefix: str,
    rsf_results: dict[str, Any] | list[dict[str, Any]],
    separator: str = "+",
) -> pd.DataFrame:
    """Serialize resting-state fMRI networks outputs and write mmwide.csv."""
    ensure_parent_dir(output_prefix)
    op = f"{output_prefix}{separator}"

    results_list = rsf_results if isinstance(rsf_results, list) else [rsf_results]
    wide_dfs = []

    for rsfpro in results_list:
        if not isinstance(rsfpro, dict):
            continue
        paramset = str(rsfpro.get("paramset", 99))
        pronum = f"fcnxpro{paramset}_"

        rsf_wide = dict_to_dataframe(rsfpro)
        if isinstance(rsfpro.get("corr_wide"), pd.DataFrame):
            rsf_wide = pd.concat([rsf_wide, rsfpro["corr_wide"]], axis=1)
        rsf_wide = rsf_wide.add_prefix(pronum)
        wide_dfs.append(rsf_wide)

        # Write correlation matrix if available
        if isinstance(rsfpro.get("corr"), pd.DataFrame):
            rsfpro["corr"].to_csv(f"{op}{pronum}rsfcorr.csv")

        # Write derivative images (alff, falff, networks, etc.)
        for key in get_antsimage_keys(rsfpro):
            val = rsfpro[key]
            if ants.is_image(val):
                image_write_with_thumbnail(val, f"{op}{pronum}{key}.nii.gz", thumb=False)

    if wide_dfs:
        combined_wide = pd.concat(wide_dfs, axis=1)
    else:
        combined_wide = pd.DataFrame({"u_rsf_id": [output_prefix]})

    write_modality_mmwide(output_prefix, combined_wide, separator=separator)
    return combined_wide


def write_nm_outputs(
    output_prefix: str,
    nm_result: dict[str, Any],
    separator: str = "+",
) -> pd.DataFrame:
    """Serialize Neuromelanin outputs and write mmwide.csv."""
    ensure_parent_dir(output_prefix)
    op = f"{output_prefix}{separator}"

    # Write images
    for key in get_antsimage_keys(nm_result):
        val = nm_result[key]
        if ants.is_image(val):
            image_write_with_thumbnail(val, f"{op}{key}.nii.gz", thumb=False)

    # Wide dataframe
    if isinstance(nm_result.get("NM_dataframe_wide"), pd.DataFrame):
        nm_wide = nm_result["NM_dataframe_wide"].copy()
    else:
        nm_wide = dict_to_dataframe(nm_result)

    write_modality_mmwide(output_prefix, nm_wide, separator=separator)
    return nm_wide


def write_perf_outputs(
    output_prefix: str,
    perf_result: dict[str, Any],
    separator: str = "+",
) -> pd.DataFrame:
    """Serialize Perfusion/ASL outputs and write mmwide.csv."""
    ensure_parent_dir(output_prefix)
    op = f"{output_prefix}{separator}"

    # Write images
    for key in get_antsimage_keys(perf_result):
        val = perf_result[key]
        if ants.is_image(val):
            image_write_with_thumbnail(val, f"{op}{key}.nii.gz", thumb=False)

    perf_wide = dict_to_dataframe(perf_result)
    if isinstance(perf_result.get("perf_dataframe"), pd.DataFrame):
        pderk = perf_result["perf_dataframe"]
        pderk_use = pderk.iloc[:, 1:] if pderk.shape[1] > 1 else pderk
        perf_wide = pd.concat([perf_wide, pderk_use], axis=1)

    write_modality_mmwide(output_prefix, perf_wide, separator=separator)
    return perf_wide


def write_pet_outputs(
    output_prefix: str,
    pet_result: dict[str, Any],
    separator: str = "+",
) -> pd.DataFrame:
    """Serialize PET summary outputs and write mmwide.csv."""
    ensure_parent_dir(output_prefix)

    pet_wide = dict_to_dataframe(pet_result)
    if isinstance(pet_result.get("pet3d_dataframe"), pd.DataFrame):
        pderk = pet_result["pet3d_dataframe"]
        pderk_use = pderk.iloc[:, 1:] if pderk.shape[1] > 1 else pderk
        pet_wide = pd.concat([pet_wide, pderk_use], axis=1)

    write_modality_mmwide(output_prefix, pet_wide, separator=separator)
    return pet_wide


__all__ = [
    "ensure_parent_dir",
    "write_dti_outputs",
    "write_flair_outputs",
    "write_modality_mmwide",
    "write_nm_outputs",
    "write_perf_outputs",
    "write_pet_outputs",
    "write_rsf_outputs",
]
