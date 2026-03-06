import os
import math
import types
import pandas as pd

from . import core as _core_api

# Compatibility aliases: keep these names importable from antsxmm.session.
# During core<->session import bootstrap, core may not have populated these
# attributes yet, so fall back safely and bind the live objects inside
# process_session().
antspymm = getattr(_core_api, 'antspymm', types.SimpleNamespace())
ants = getattr(_core_api, 'ants', types.SimpleNamespace())
sanitize_and_stage_file = getattr(_core_api, 'sanitize_and_stage_file', None)
build_wide_table_from_mmwide = getattr(_core_api, 'build_wide_table_from_mmwide', None)

import tempfile
import shutil
import re
import traceback
import json
from pathlib import Path
from datetime import datetime, timezone

from .inputs import plan_session_inputs, _extract_run_id_from_filename, _collect_discovered_inputs
from .pymm_execution import generate_xmm_dataframe, run_xmm_mm_csv
from .execution_plan import build_execution_plan, validate_execution_plan
from .status import _ensure_dir, _write_json
from .fingerprint import compute_input_fingerprint
from .status import write_session_status
from .staging import extract_image_id, get_modality_variant, sanitize_and_stage_file
from .wide_table import build_wide_table_from_mmwide

def print_expected_tree(output_root, project_id, sub_id, date_id, image_uid, 
                        flair_info, rsf_infos, dti_infos, nm_infos, perf_info, pet_info, sep="_"):
    """
    Prints the expected directory structure based on staged files.
    """
    base = Path(output_root) / project_id / sub_id / date_id
    
    print("\n[PRE-CHECK] Processing Plan for {} : {}".format(sub_id, date_id))
    print("ROOT OUTPUT: {}".format(base))
    
    # T1 Hierarchy
    print("├── T1wHierarchical/ (ID: {}) [FOUND]".format(image_uid))
    # FIXED: Corrected format string
    print("│   └── .../T1wHierarchical{}{}".format(sep, image_uid))

    # FLAIR
    if flair_info[0]:
        print("├── T2Flair/ (ID: {}) [FOUND]".format(flair_info[2]))
    else:
        print("├── T2Flair/ [MISSING] (Skipping)")

    # rsfMRI
    if rsf_infos:
        print("├── rsfMRI/ [FOUND: {} scan(s)]".format(len(rsf_infos)))
        for p, m, fid in rsf_infos:
            print("│   └── Variant: {} (ID: {}) -> {}".format(m, fid, os.path.basename(p)))
    else:
        print("├── rsfMRI/ [MISSING] (Skipping)")

    # DTI
    if dti_infos:
        print("├── DTI/ [FOUND: {} scan(s)]".format(len(dti_infos)))
        for p, m, fid in dti_infos:
            print("│   └── Variant: {} (ID: {}) -> {}".format(m, fid, os.path.basename(p)))
    else:
        print("├── DTI/ [MISSING] (Skipping)")

    # Neuromelanin
    if nm_infos:
        print("└── NM2DMT/ (ID: ...) [FOUND: {} scan(s)]".format(len(nm_infos)))
    else:
        print("└── NM2DMT/ [MISSING] (Skipping)")

    # Perfusion
    if perf_info[0]:
        print("├── perf/ (ID: {}) [FOUND]".format(perf_info[2]))
    else:
        print("├── perf/ [MISSING] (Skipping)")

    # PET
    if pet_info[0]:
        print("├── pet3d/ (ID: {}) [FOUND]".format(pet_info[2]))
    else:
        print("├── pet3d/ [MISSING] (Skipping)")

    print("\n")


def process_session(
    session_data,
    output_root,
    project_id: str = "ANTsX",
    *,
    denoise: bool | None = None,
    denoise_dti: bool = True,
    dti_moco='SyN',
    separator: str = '_',
    verbose: bool = True,
    build_wide_table: bool = True,
    t1_run_match=None,
    write_input_manifest: bool = True,
    tool_version: str | None = None,
    resume_mode: str | None = None,
):
    """
    Runs the full ANTsPyMM pipeline on one session.
    """
    result = {
        'success': False,
        'wide_df': None,
        'session_dir': None
    }

    # Resolve shared runtime seams late so tests can patch either antsxmm.session
    # or antsxmm.core. Prefer explicit session-level overrides, then fall back to
    # the core-owned defaults.
    import antsxmm.core as _core_api  # local import avoids import-time cycles

    session_antspymm = globals().get('antspymm', types.SimpleNamespace())
    session_ants = globals().get('ants', types.SimpleNamespace())

    antspymm = session_antspymm if (
        hasattr(session_antspymm, 'mm_csv') or hasattr(session_antspymm, 'generate_mm_dataframe')
    ) else _core_api.antspymm
    ants = session_ants if type(session_ants) is not types.SimpleNamespace else _core_api.ants
    sanitize_and_stage_file = _core_api.sanitize_and_stage_file
    build_wide_table_from_mmwide = _core_api.build_wide_table_from_mmwide


    # 1. Plan inputs (selection/truncation) + compute fingerprint for resumability
    plan = plan_session_inputs(session_data, t1_run_match=t1_run_match)
    sub_id = plan['subjectID']
    date_id = plan['sessionID']

    session_out_dir = os.path.join(output_root, project_id, sub_id, date_id)
    input_fingerprint = compute_input_fingerprint(session_data, t1_run_match=t1_run_match)

    if not plan.get('processable', False):
        if verbose:
            print("Error: No T1w found for {} {}".format(sub_id, date_id))
        # Persist failure status for transparency
        try:
            write_session_status(
                session_out_dir,
                project_id=project_id,
                subject_id=sub_id,
                session_id=date_id,
                success=False,
                input_fingerprint=input_fingerprint,
                args={'t1_run_match': t1_run_match, 'tool_version': tool_version, 'resume_mode': resume_mode},
                error='no_T1w',
            )
        except Exception:
            pass
        return result

    t1_fn = plan['t1_fn']
    if verbose and t1_run_match:
        # Only emit a message when the user requested a match token.
        print("Selected T1 (requested match='{}'): {}".format(t1_run_match, os.path.basename(t1_fn)))

    image_uid = _extract_run_id_from_filename(t1_fn)

    # 2. Setup Staging Area
    # Use a randomized directory under the system temp dir to avoid collisions and
    # to reduce the risk of symlink tricks in shared environments.
    staging_root = tempfile.mkdtemp(prefix=f"antsxmm_staging_{sub_id}_{date_id}_")

    # 3. Stage Files
    # T1w
    t1_path, _, _ = sanitize_and_stage_file(t1_fn, project_id, sub_id, date_id, "T1w", image_uid, separator, staging_root, verbose)

    # FLAIR (fallback: T2w if FLAIR absent)
    # Support both scalar and list-valued columns.
    flair_raw = plan.get('flair_raw')

    flair_path, flair_mod, flair_id = sanitize_and_stage_file(
        flair_raw, project_id, sub_id, date_id, "T2Flair", image_uid, separator, staging_root, verbose
    )
    flair_info = (flair_path, flair_mod, flair_id)

    # rsfMRI
    rsf_raw = list(plan.get('rsf_selected_raw', []))
    rsf_infos = []
    rsf_paths = []
    rsf_selected_raw = []
    for f in rsf_raw:
        this_id = extract_image_id(f)
        
        path, mod, unique_id = sanitize_and_stage_file(f, project_id, sub_id, date_id, "rsfMRI", this_id, separator, staging_root, verbose)
        if path:
            rsf_infos.append((path, mod, unique_id))
            rsf_paths.append(path)
            rsf_selected_raw.append(f)

    # Selection/truncation is already handled in plan_session_inputs


    # DTI
    dti_raw = list(plan.get('dti_selected_raw', []))
    dti_infos = []
    dti_paths = []
    dti_selected_raw = []
    for f in dti_raw:
        this_id = extract_image_id(f)
        path, mod, unique_id = sanitize_and_stage_file(f, project_id, sub_id, date_id, "DTI", this_id, separator, staging_root, verbose)
        if path:
            dti_infos.append((path, mod, unique_id))
            dti_paths.append(path)
            dti_selected_raw.append(f)

    # Selection/truncation is already handled in plan_session_inputs

    # NM
    nm_raw = list(plan.get('nm_selected_raw', []))
    nm_infos = []
    nm_paths = []
    nm_selected_raw = []
    for f in nm_raw:
        rid = extract_image_id(f)
        path, mod, unique_id = sanitize_and_stage_file(f, project_id, sub_id, date_id, "NM2DMT", rid, separator, staging_root, verbose=verbose)
        if path:
            nm_infos.append((path, mod, unique_id))
            nm_paths.append(path)
            nm_selected_raw.append(f)

    # Perf
    perf_raw = plan.get('perf_raw')
    perf_path, perf_mod, perf_id = sanitize_and_stage_file(
        perf_raw, project_id, sub_id, date_id, "perf", image_uid, separator, staging_root, verbose=verbose
    )
    perf_info = (perf_path, perf_mod, perf_id)

    # PET
    pet_raw = plan.get('pet_raw')
    pet_path, pet_mod, pet_id = sanitize_and_stage_file(
        pet_raw, project_id, sub_id, date_id, "pet3d", image_uid, separator, staging_root, verbose=verbose
    )
    pet_info = (pet_path, pet_mod, pet_id)

    mock_source_dir = staging_root

    # Persist an input manifest that enumerates exactly which NIfTIs will be processed.
    if write_input_manifest:
        _ensure_dir(session_out_dir)

        # Discoveries (as seen from the BIDS parser row)
        discovered = _collect_discovered_inputs(session_data)

        # IMPORTANT: used_inputs are expressed in the same coordinate system as
        # discovered (absolute realpaths). Staged paths live under a temp directory
        # and must not leak into the manifest.
        used = plan.get('used', {})

        if verbose:
            print("[VERBOSE] Discovered inputs:")
            for k, v in discovered.items():
                print(f"  - {k}: {v}")
            print("[VERBOSE] Selected inputs (will be staged/processed):")
            for k, v in used.items():
                print(f"  - {k}: {v}")

        # Exclusions due to truncation / selection.
        excluded = {
            # Truncation: items discovered but not selected for processing.
            'rsf_truncated': [p for p in discovered.get('rsf_filenames', []) if p not in used['rsf_filenames']],
            'dti_truncated': [p for p in discovered.get('dti_filenames', []) if p not in used['dti_filenames']],
            # Single-selection modalities
            't1_not_selected': [p for p in discovered.get('t1_filenames', []) if used['t1_filename'] and p != used['t1_filename']],
            'flair_candidates_not_selected': [
                p
                for p in (discovered.get('flair_filenames', []) + discovered.get('t2w_filenames', []))
                if used['flair_or_t2_as_flair_filename'] and p != used['flair_or_t2_as_flair_filename']
            ],
        }

        manifest = {
            'schema_version': 1,
            'project_id': project_id,
            'subjectID': sub_id,
            'sessionID': date_id,
            'session_path': session_data.get('session_path'),
            'input_fingerprint': input_fingerprint,
            'selection_rules': {
                'requires_T1w': True,
                't1_run_match': t1_run_match,
                'flair_fallback_to_T2w': True,
                'truncate_rsfMRI_to_first_n': 2,
                'truncate_DTI_to_first_n': 2,
                'perf_select_single': True,
                'pet_select_single': True,
                # For compatibility with older CLIs; not currently wired into antspymm.
                'denoise_requested': bool(denoise) if denoise is not None else None,
            },
            'discovered': discovered,
            'used_inputs': used,
            'nifti_inputs_that_will_be_processed': sorted(
                [p for p in [
                    used['t1_filename'],
                    used['flair_or_t2_as_flair_filename'],
                    used['perf_filename'],
                    used['pet3d_filename'],
                ] if p] +
                list(used['rsf_filenames']) +
                list(used['dti_filenames']) +
                list(used['nm_filenames'])
            ),
            'excluded': excluded,
        }

        manifest_path = os.path.join(
            session_out_dir,
            f"{project_id}{separator}{sub_id}{separator}{date_id}{separator}mm_inputs.json",
        )
        _write_json(manifest_path, manifest)
        if verbose:
            print(f"[INFO] Wrote input manifest: {manifest_path}")

    try:
        if not hasattr(antspymm, 'mm_csv'):
            raise ModuleNotFoundError(
                "antspymm.mm_csv is required to run antsxmm processing. Install antspymm (and ants) to execute the pipeline."
            )

        # Pre-execution check
        if verbose:
            print_expected_tree(output_root, project_id, sub_id, date_id, image_uid, 
                        flair_info, rsf_infos, dti_infos, nm_infos, perf_info, pet_info, separator)

        if verbose:
            print("\n{}".format('='*80))
            print("Processing: {} | {}".format(sub_id, date_id))
            print("Image UID: {}".format(image_uid))

        # Compatibility path: preserve antspymm.generate_mm_dataframe outputs if available,
        # but let antsxmm own deterministic layout/prefix metadata.
        compat_df = None
        if hasattr(antspymm, 'generate_mm_dataframe'):
            compat_df = antspymm.generate_mm_dataframe(
                projectID=project_id,
                subjectID=sub_id,
                date=date_id,
                imageUniqueID=image_uid,
                modality='T1w',
                source_image_directory=mock_source_dir,
                output_image_directory=output_root,
                t1_filename=t1_path,
                flair_filename=flair_path,
                rsf_filenames=rsf_paths,
                dti_filenames=dti_paths,
                nm_filenames=nm_paths,
                perf_filename=perf_path,
                pet3d_filename=pet_path,
            )

        execution_session_data = {
            'subjectID': sub_id,
            'sessionID': date_id,
            'session_path': session_data.get('session_path'),
            't1_filenames': [t1_path],
            'flair_filenames': [flair_path] if flair_path else [],
            'dti_filenames': list(dti_paths),
            'rsf_filenames': list(rsf_paths),
            'nm_filenames': list(nm_paths),
            'perf_filenames': [perf_path] if perf_path else [],
            'pet3d_filenames': [pet_path] if pet_path else [],
        }

        execution_plan = build_execution_plan(
            execution_session_data,
            output_root=output_root,
            project_id=project_id,
        )
        validate_execution_plan(execution_plan)
        xmm_df = generate_xmm_dataframe(
            execution_session_data,
            output_root=output_root,
            project_id=project_id,
        )

        if compat_df is None or getattr(compat_df, 'empty', False):
            study_csv = xmm_df
        else:
            study_csv = compat_df.copy()
            for col in xmm_df.columns:
                if col.startswith('xmm_'):
                    study_csv[col] = xmm_df[col]
            for col in ('filename', 'flairid', 'perfid', 'pet3did', 'rsfid1', 'rsfid2', 'rsfid3', 'dtid1', 'dtid2', 'dtid3'):
                if col not in study_csv.columns and col in xmm_df.columns:
                    study_csv[col] = xmm_df[col]
            for i in range(1, 12):
                col = f'nmid{i}'
                if col not in study_csv.columns and col in xmm_df.columns:
                    study_csv[col] = xmm_df[col]
            for col in ('projectID', 'subjectID', 'date', 'imageID', 'modality', 'sourcedir', 'outputdir'):
                if col not in study_csv.columns and col in xmm_df.columns:
                    study_csv[col] = xmm_df[col]

        study_csv_clean = study_csv.dropna(axis=1, how='all')

        try:
            template_path = antspymm.get_data("PPMI_template0", target_extension=".nii.gz")
            mask_path = antspymm.get_data("PPMI_template0_brainmask", target_extension=".nii.gz")
            if not template_path or not mask_path:
                template = None
            else:
                template = ants.image_read(template_path)
                template_mask = ants.image_read(mask_path)
                template = template * template_mask
                template = ants.crop_image(template, ants.iMath(template_mask, "MD", 12))
        except:
            template = None
            if verbose:
                print("Warning: Using default template (None)")

        if verbose:
            print("Running antspymm.mm_csv()...")

        run_xmm_mm_csv(
            study_csv_clean,
            antspymm,
            mysep=separator,
            dti_motion_correct=dti_moco,
            dti_denoise=denoise_dti,
            normalization_template=template,
            normalization_template_output='ppmi',
            normalization_template_transform_type='antsRegistrationSyNQuickRepro[s]',
            normalization_template_spacing=[1,1,1],
            srmodel_T1=None, srmodel_NM=None, srmodel_DTI=None,
        )


        result['success'] = True
        result['session_dir'] = os.path.join(output_root, project_id, sub_id, date_id)

        # Persist resumability status
        try:
            write_session_status(
                result['session_dir'],
                project_id=project_id,
                subject_id=sub_id,
                session_id=date_id,
                success=True,
                input_fingerprint=input_fingerprint,
                args={
                    't1_run_match': t1_run_match,
                    'separator': separator,
                    'denoise_dti': denoise_dti,
                    'dti_moco': dti_moco,
                    'tool_version': tool_version,
                    'resume_mode': resume_mode,
                },
                error=None,
            )
        except Exception:
            pass

        # Persist cleaned study CSV for provenance/debugging
        try:
            os.makedirs(result['session_dir'], exist_ok=True)
            study_csv_path = os.path.join(
                result['session_dir'],
                f"{project_id}+{sub_id}+{date_id}+study.csv",
            )
            study_csv_clean.to_csv(study_csv_path, index=False)
        except Exception:
            # Never fail the pipeline due to artifact persistence
            pass

        if build_wide_table:
            session_output_dir = result['session_dir']
            if os.path.exists(session_output_dir):
                try:
                    wide_df = build_wide_table_from_mmwide(
                        root_dir=session_output_dir,
                        sep=separator,
                        verbose=verbose
                    )
                    result['wide_df'] = wide_df
                    
                    t1_hier_dir = os.path.join(session_output_dir, "T1wHierarchical", image_uid)
                    if os.path.exists(t1_hier_dir):
                        prefix = project_id + separator + sub_id + separator + date_id + separator
                        prefix = prefix + "T1wHierarchical" + separator + image_uid
                        filename = prefix+separator+"mmwidemerged.csv".format(separator)
                        out_path = os.path.join(t1_hier_dir, filename)
                        wide_df.to_csv(out_path, index=False)
                        if verbose:
                            print("[SUCCESS] Session merged wide table written to:\n {}".format(out_path))
                    else:
                        if verbose:
                            print("[WARNING] T1wHierarchical directory not found: {}".format(t1_hier_dir))

                except Exception as e:
                    if verbose:
                        print("Warning: Failed to build wide table: {}".format(e))
                    result['wide_df'] = None

        return result

    except Exception as e:
        print("Error processing {} {}: {}".format(sub_id, date_id, str(e)))
        traceback.print_exc()
        # Persist failure status
        try:
            write_session_status(
                session_out_dir,
                project_id=project_id,
                subject_id=sub_id,
                session_id=date_id,
                success=False,
                input_fingerprint=input_fingerprint,
                args={
                    't1_run_match': t1_run_match,
                    'separator': separator,
                    'denoise_dti': denoise_dti,
                    'dti_moco': dti_moco,
                    'tool_version': tool_version,
                    'resume_mode': resume_mode,
                },
                error=str(e),
            )
        except Exception:
            pass
        return result
    finally:
        if os.path.exists(staging_root):
            shutil.rmtree(staging_root)
