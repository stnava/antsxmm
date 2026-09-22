"""Native modality dispatcher and execution engine for antsxmm.

Directly orchestrates execution of each modality using modular algorithmic engines
in antsxmm.modalities without legacy DataFrame packing or monkeypatched globals.
"""

from __future__ import annotations

import os
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ants
import numpy as np
import pandas as pd

from ..execution_plan import ExecutionUnit
from .dti import joint_dti_recon
from .fmri import resting_state_fmri_networks
from .io import (
    ensure_parent_dir,
    write_dti_outputs,
    write_flair_outputs,
    write_nm_outputs,
    write_perf_outputs,
    write_pet_outputs,
    write_rsf_outputs,
)
from .neuromelanin import neuromelanin
from .perfusion import bold_perfusion
from .pet import pet3d_summary
from .templates import get_data
from .wmh import boot_wmh, wmh


@dataclass
class SessionContext:
    """Anatomical reference context derived from the primary T1w image."""
    project_id: str
    subject_id: str
    session_id: str
    t1_image: ants.ANTsImage
    t1_path: str
    hier: dict[str, Any]
    hier_dir: str
    template_tx: dict[str, Any] | None = None
    group_template: ants.ANTsImage | None = None
    group_tx: Any | None = None
    separator: str = "+"


def resolve_bids_sidecars(nifti_path: str) -> tuple[str | None, str | None]:
    """Resolve associated .bval and .bvec files for a DWI NIfTI image."""
    base_no_ext = nifti_path
    for ext in [".nii.gz", ".nii"]:
        if base_no_ext.endswith(ext):
            base_no_ext = base_no_ext[: -len(ext)]
            break

    bval = f"{base_no_ext}.bval"
    bvec = f"{base_no_ext}.bvec"

    bval_out = bval if os.path.exists(bval) else None
    bvec_out = bvec if os.path.exists(bvec) else None
    return bval_out, bvec_out


def initialize_session_context(
    t1_path: str,
    output_prefix_t1_hier: str,
    project_id: str,
    subject_id: str,
    session_id: str,
    separator: str = "+",
    normalization_template: ants.ANTsImage | None = None,
    verbose: bool = False,
) -> SessionContext:
    """Run T1w hierarchical processing and build session context."""
    import antspyt1w

    ensure_parent_dir(output_prefix_t1_hier)
    hier_dir = str(Path(output_prefix_t1_hier).parent)
    hier_prefix = f"{output_prefix_t1_hier}{separator}"

    t1 = ants.image_read(t1_path)

    # Hierarchical outputs
    hier_exists = os.path.exists(f"{hier_prefix}cerebellum.csv")
    if not hier_exists:
        if verbose:
            print(f"[ANTsXMM] Running antspyt1w.hierarchical on {t1_path}...")
        ants.image_write(t1, f"{hier_prefix}head.nii.gz")
        hier = antspyt1w.hierarchical(t1, hier_prefix, labels_to_register=None)
        antspyt1w.write_hierarchical(hier, hier_prefix)
        t1wide = antspyt1w.merge_hierarchical_csvs_to_wide_format(hier["dataframes"], identifier=None)
        t1wide.to_csv(f"{output_prefix_t1_hier}{separator}mmwide.csv", index=False)
    else:
        if verbose:
            print(f"[ANTsXMM] Reading existing hierarchical results from {hier_prefix}...")
        hier = antspyt1w.read_hierarchical(hier_prefix)

    # Template registration
    regout = f"{hier_prefix}syn"
    template_tx = None
    warp_file = f"{regout}1Warp.nii.gz"
    affine_file = f"{regout}0GenericAffine.mat"

    if os.path.exists(warp_file) and os.path.exists(affine_file):
        template_tx = {
            "fwdtransforms": [warp_file, affine_file],
            "invtransforms": [affine_file, f"{regout}1InverseWarp.nii.gz"],
        }
    else:
        try:
            cit_fn = get_data("CIT168_T1w_700um_pad_adni", target_extension=".nii.gz")
            if cit_fn and os.path.exists(cit_fn):
                template = ants.image_read(cit_fn)
                template = ants.resample_image(template, [1, 1, 1], use_voxels=False)
                t1reg = ants.registration(
                    template,
                    hier["brain_n4_dnz"],
                    "antsRegistrationSyNQuickRepro[s]",
                    outprefix=regout,
                    verbose=False,
                )
                template_tx = {
                    "fwdtransforms": t1reg["fwdtransforms"],
                    "invtransforms": t1reg["invtransforms"],
                }
        except Exception as e:
            if verbose:
                print(f"[ANTsXMM] Template registration warning: {e}")

    return SessionContext(
        project_id=project_id,
        subject_id=subject_id,
        session_id=session_id,
        t1_image=t1,
        t1_path=t1_path,
        hier=hier,
        hier_dir=hier_dir,
        template_tx=template_tx,
        group_template=normalization_template,
        separator=separator,
    )


def execute_unit(
    unit: ExecutionUnit,
    context: SessionContext,
    verbose: bool = True,
    **kwargs: Any,
) -> pd.DataFrame | None:
    """Execute a single execution unit (modality) and persist its outputs."""
    mod = unit.modality
    prefix = unit.output_prefix
    sep = context.separator
    mmwide_path = f"{prefix}{sep}mmwide.csv"

    # If outputs already exist, skip execution
    if os.path.exists(mmwide_path):
        if verbose:
            print(f"[ANTsXMM] Skipping {mod} - outputs already complete at {mmwide_path}")
        return pd.read_csv(mmwide_path)

    if not unit.input_paths:
        if verbose:
            print(f"[ANTsXMM] No inputs found for {mod}, skipping.")
        return None

    if verbose:
        print(f"[ANTsXMM] Processing {mod} (run: {unit.run}) -> {prefix}")

    try:
        if mod in ("T1w", "T1wHierarchical"):
            # T1w/T1wHierarchical is processed as part of hierarchical context
            if os.path.exists(mmwide_path):
                return pd.read_csv(mmwide_path)
            # Check if companion (T1wHierarchical or T1w) already wrote an mmwide.csv
            companion_mod = "T1wHierarchical" if mod == "T1w" else "T1w"
            companion_prefix = prefix.replace(f"{sep}{mod}{sep}", f"{sep}{companion_mod}{sep}")
            companion_mmwide = f"{companion_prefix}{sep}mmwide.csv"
            if os.path.exists(companion_mmwide):
                df = pd.read_csv(companion_mmwide)
                ensure_parent_dir(prefix)
                df.to_csv(mmwide_path, index=False)
                return df
            if context.hier and isinstance(context.hier, dict) and "dataframes" in context.hier:
                try:
                    import antspyt1w
                    t1wide = antspyt1w.merge_hierarchical_csvs_to_wide_format(context.hier["dataframes"], identifier=None)
                    ensure_parent_dir(prefix)
                    t1wide.to_csv(mmwide_path, index=False)
                    return t1wide
                except Exception:
                    pass
            return None

        elif mod == "T2Flair":
            flair_img = ants.image_read(unit.input_paths[0])
            t1seg = context.hier["dkt_parc"]["tissue_segmentation"]
            res = wmh(
                flair=flair_img,
                t1=context.t1_image,
                t1seg=t1seg,
                verbose=verbose,
            )
            return write_flair_outputs(prefix, res, separator=sep)

        elif mod == "DTI":
            dwi_path = unit.input_paths[0]
            dwi_img = ants.image_read(dwi_path)
            bval_fn, bvec_fn = resolve_bids_sidecars(dwi_path)
            if not bval_fn or not bvec_fn:
                warnings.warn(f"Missing .bval/.bvec sidecars for {dwi_path}, cannot perform DTI reconstruction.")
                return None

            b0_template = context.group_template
            t1brn = context.hier.get("brain_n4_dnz", context.t1_image)
            dti_res = joint_dti_recon(
                dwi_img,
                bval_fn,
                bvec_fn,
                t1brn,
                verbose=verbose,
                **kwargs,
            )
            return write_dti_outputs(prefix, dti_res, separator=sep)

        elif mod == "rsfMRI":
            rsf_img = ants.image_read(unit.input_paths[0])
            t1seg = context.hier["dkt_parc"]["tissue_segmentation"]
            t1head = context.t1_image
            t1brn = context.hier.get("brain_n4_dnz", t1head)
            rsf_res = resting_state_fmri_networks(
                fmri=rsf_img,
                fmri_template=None,
                t1=t1brn,
                t1segmentation=t1seg,
                verbose=verbose,
                **kwargs,
            )
            return write_rsf_outputs(prefix, rsf_res, separator=sep)

        elif mod == "NM2DMT":
            nm_imgs = [ants.image_read(p) for p in unit.input_paths]
            nm_res = neuromelanin(
                t1=context.t1_image,
                hier=context.hier,
                nm_image_list=nm_imgs,
                verbose=verbose,
                **kwargs,
            )
            return write_nm_outputs(prefix, nm_res, separator=sep)

        elif mod == "perf":
            perf_img = ants.image_read(unit.input_paths[0])
            perf_res = bold_perfusion(
                perf_img,
                verbose=verbose,
                **kwargs,
            )
            return write_perf_outputs(prefix, perf_res, separator=sep)

        elif mod == "pet3d":
            pet_img = ants.image_read(unit.input_paths[0])
            t1seg = context.hier["dkt_parc"]["tissue_segmentation"]
            t1brn = context.hier.get("brain_n4_dnz", context.t1_image)
            pet_res = pet3d_summary(
                pet3d=pet_img,
                t1head=context.t1_image,
                t1=t1brn,
                t1segmentation=t1seg,
                t1dktcit=context.hier.get("dkt_parc", {}).get("dkt_cortical_cit168"),
                verbose=verbose,
            )
            return write_pet_outputs(prefix, pet_res, separator=sep)

        else:
            if verbose:
                print(f"[ANTsXMM] Unhandled modality '{mod}', skipping.")
            return None

    except Exception as e:
        warnings.warn(f"[ANTsXMM] Error executing modality {mod} for unit {unit.run}: {e}")
        if verbose:
            traceback.print_exc()
        return None


def run_session_plan_natively(
    execution_plan: list[ExecutionUnit],
    context: SessionContext,
    verbose: bool = True,
    **kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """Execute all units in an execution plan directly using antsxmm.modalities."""
    results: dict[str, pd.DataFrame] = {}
    for unit in execution_plan:
        res_df = execute_unit(unit, context, verbose=verbose, **kwargs)
        if res_df is not None:
            results[unit.modality] = res_df
    return results


__all__ = [
    "SessionContext",
    "execute_unit",
    "initialize_session_context",
    "resolve_bids_sidecars",
    "run_session_plan_natively",
]
