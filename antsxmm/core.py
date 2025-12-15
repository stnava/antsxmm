import os
import antspymm
import pandas as pd
import ants
import tempfile
import shutil
import re
import traceback
from pathlib import Path

def extract_image_id(filename):
    """
    Extracts a run/image ID from a BIDS-like filename.
    Prioritizes 'rXXXX' or 'run-XXXX'. Defaults to '000'.
    """
    fname = os.path.basename(filename)
    match = re.search(r"_(r\d+|run-\d+)[_.]", fname)
    if match:
        return match.group(1)
    return "000"

def get_modality_variant(filename, base_modality, sep):
    """
    Returns the specific antspymm modality string with direction appended.
    """
    fname = os.path.basename(filename)
    suffix = ""
    
    if "dir-RL" in fname or "dir-PA" in fname or "_RL" in fname or "_PA" in fname:
        suffix = "RL"
    elif "dir-LR" in fname or "dir-AP" in fname or "_LR" in fname or "_AP" in fname:
        suffix = "LR"
        
    if suffix:
        if sep == "_":
            return f"{base_modality}{suffix}"
        else:
            return f"{base_modality}{sep}{suffix}"
    
    # Default mappings
    if base_modality == "dwi": return "DTI"
    if base_modality == "func": return "rsfMRI"
    
    return base_modality

def sanitize_and_stage_file(filepath, project, subject, date, base_modality, image_id, sep, staging_root, verbose=False):
    """
    Stages a file into a strict NRG directory structure in tmp.
    ALSO stages accompanying .bval, .bvec, and .json files if they exist.
    """
    if not filepath:
        return None, None

    modality = get_modality_variant(filepath, base_modality, sep)
    
    # Construct base name without extension for sidecar search
    name = os.path.basename(filepath)
    if name.endswith(".nii.gz"):
        base_name = name[:-7]
        ext = ".nii.gz"
    elif name.endswith(".nii"):
        base_name = name[:-4]
        ext = ".nii"
    else:
        base_name, ext = os.path.splitext(name)

    # FORCE NRG FILENAME base
    new_filename_base = f"{project}{sep}{subject}{sep}{date}{sep}{modality}{sep}{image_id}"
    
    # Construct NRG path
    dest_dir = os.path.join(staging_root, project, subject, date, modality, image_id)
    os.makedirs(dest_dir, exist_ok=True)
    
    # Helper to symlink a file
    def stage_one(src, dst_name):
        dst = os.path.join(dest_dir, dst_name)
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
        return dst

    # Stage the main image
    symlink_path = stage_one(filepath, new_filename_base + ext)
    
    if verbose:
        print(f"  Staged: {name} -> .../{modality}/{image_id}/{new_filename_base}{ext}")

    # Stage sidecars (bval, bvec, json)
    src_dir = os.path.dirname(filepath)
    sidecars = [".bval", ".bvec", ".json"]
    
    for side_ext in sidecars:
        src_side = os.path.join(src_dir, base_name + side_ext)
        if os.path.exists(src_side):
            stage_one(src_side, new_filename_base + side_ext)
            if verbose:
                print(f"    + Sidecar: {base_name}{side_ext} -> {new_filename_base}{side_ext}")
        
    return symlink_path, modality

def print_expected_tree(output_root, project_id, sub_id, date_id, image_uid, 
                        flair_info, rsf_infos, dti_infos, nm_infos, perf_info, pet_info, sep="_"):
    """
    Prints the expected directory structure based on staged files.
    """
    base = Path(output_root) / project_id / sub_id / date_id
    
    print(f"\n[PRE-CHECK] Processing Plan for {sub_id} {date_id}:")
    print(f"ROOT OUTPUT: {base}")
    
    # T1 Hierarchy
    print(f"├── T1wHierarchical/ (ID: {image_uid}) [FOUND]")
    print(f"│   └── .../{project_id}{sep}{sub_id}{sep}{date_id}{sep}T1wHierarchical{sep}{image_uid}{sep}...")

    # FLAIR
    if flair_info[0]:
        print(f"├── T2Flair/ (ID: {image_uid}) [FOUND]")
    else:
        print(f"├── T2Flair/ [MISSING] (Skipping)")

    # rsfMRI
    if rsf_infos:
        print(f"├── rsfMRI/ [FOUND: {len(rsf_infos)} scan(s)]")
        for p, m in rsf_infos:
            print(f"│   └── Variant: {m} (ID: {image_uid}) -> {os.path.basename(p)}")
    else:
        print(f"├── rsfMRI/ [MISSING] (Skipping)")

    # DTI
    if dti_infos:
        print(f"├── DTI/ [FOUND: {len(dti_infos)} scan(s)]")
        for p, m in dti_infos:
            print(f"│   └── Variant: {m} (ID: {image_uid}) -> {os.path.basename(p)}")
    else:
        print(f"├── DTI/ [MISSING] (Skipping)")

    # Neuromelanin
    if nm_infos:
        nm_id = extract_image_id(nm_infos[0][0])
        print(f"└── NM2DMT/ (ID: {nm_id}...) [FOUND: {len(nm_infos)} scan(s)]")
    else:
        print(f"└── NM2DMT/ [MISSING] (Skipping)")

    # Perfusion
    if perf_info[0]:
        print(f"├── perf/ (ID: {image_uid}) [FOUND]")
    else:
        print(f"├── perf/ [MISSING] (Skipping)")

    # PET
    if pet_info[0]:
        print(f"├── pet3d/ (ID: {image_uid}) [FOUND]")
    else:
        print(f"├── pet3d/ [MISSING] (Skipping)")

    print("\n")

def bind_mm_rows(named_dataframes, sep="_"):
    if not named_dataframes:
        return pd.DataFrame()

    processed = []

    for mod_name, df in named_dataframes:
        df = df.copy()
        df = df.replace(["", "NA"], pd.NA)

        id_col = df.columns[0]
        df = df.set_index(id_col)

        if len(df) > 1:
            df = df.groupby(level=0).last()

        new_cols = {c: mod_name + sep + c for c in df.columns}
        df = df.rename(columns=new_cols)

        processed.append(df)

    combined = pd.concat(processed, axis=1, join="outer")
    combined = combined.loc[:, ~combined.columns.duplicated(keep="first")]
    combined = combined.reindex(sorted(combined.columns), axis=1)

    return combined.reset_index().rename(columns={"index": "subject_id"})


def check_modality_order(ordered_data, expected_order):
    actual_mods = [mod for mod, _ in ordered_data]
    filtered_expected = [mod for mod in expected_order if mod in actual_mods]

    if actual_mods != filtered_expected:
        # Warn but don't fail, data might still be usable
        print(f"Warning: Modality order mismatch. Expected: {filtered_expected}, Got: {actual_mods}")
    return True


def build_wide_table_from_mmwide(root_dir, pattern="**/*_mmwide.csv", sep="_", verbose=True):
    root = Path(root_dir).expanduser().resolve()
    csv_files = sorted(root.rglob(pattern))

    if verbose:
        print(f"\nFound {len(csv_files)} *_mmwide.csv files")
        for f in csv_files:
            print(" ->", f.relative_to(root))

    MODALITY_MAP = {
        "T1wHierarchical": "T1Hier",
        "T1Hier":     "T1Hier",
        "T1w":        "T1w",
        "NM2DMT":     "NM2DMT",
        "NM":         "NM2DMT",
        "DTI":        "DTI",
        "rsfMRI":     "rsfMRI",
        "T2Flair":    "T2Flair",
        "FLAIR":      "T2Flair",
        "perf":       "perf",
        "pet3d":      "pet3d",
        "PET":        "pet3d",
    }

    MODALITY_ORDER = ["T1Hier", "T1w", "DTI", "rsfMRI", "T2Flair", "NM2DMT", "perf", "pet3d"]

    raw_data = []

    for csv_path in csv_files:
        sub = next((p for p in csv_path.parts if p.startswith("sub-")), None)
        ses = next((p for p in csv_path.parts if p.startswith("ses-")), None)
        subject_key = f"{sub}_{ses}" if sub and ses else "UNKNOWN"

        matched_prefix = None
        best_len = 0
        for clue, prefix in MODALITY_MAP.items():
            if any(clue in part for part in csv_path.parts) or clue in csv_path.name:
                if len(clue) > best_len:
                    matched_prefix = prefix
                    best_len = len(clue)

        if not matched_prefix:
            if verbose:
                print(f" SKIP: unknown modality for file {csv_path.name}")
            continue

        if verbose:
            print(f"\nProcessing: {matched_prefix.ljust(10)} | {csv_path.name}")

        df = pd.read_csv(csv_path)

        drop_cols = [c for c in df.columns if "hier_id" in c.lower()]
        if drop_cols:
            df = df.drop(columns=drop_cols)
            if verbose:
                print(f" Dropped columns: {drop_cols}")

        if len(df) > 1:
            if verbose:
                print(f" Collapsing {len(df)} rows to last")
            df = df.iloc[[-1]].copy()

        df.insert(0, "bids_subject", subject_key)

        raw_data.append((matched_prefix, df))

    if not raw_data:
        if verbose: print("No valid *_mmwide.csv files were loaded!")
        return pd.DataFrame()

    ordered_data = []
    seen_mods = set()

    for mod in MODALITY_ORDER:
        for m, d in raw_data:
            if m == mod:
                ordered_data.append((m, d))
                seen_mods.add(m)
                break

    for m, d in raw_data:
        if m not in seen_mods:
            ordered_data.append((m, d))

    check_modality_order(ordered_data, MODALITY_ORDER)

    t1hier_df = next((df for mod, df in ordered_data if mod == "T1Hier"), None)
    t1hier_raw_cols = set()
    if t1hier_df is not None:
        t1hier_raw_cols = set(t1hier_df.columns) - {"bids_subject"}

    processed_data = []
    for mod, df in ordered_data:
        if mod != "T1Hier" and t1hier_raw_cols:
            overlap = t1hier_raw_cols & set(df.columns)
            if overlap and verbose:
                print(f" Excluding {len(overlap)} overlapping columns from {mod}")
            df = df.drop(columns=overlap, errors='ignore')
        processed_data.append((mod, df))

    wide = bind_mm_rows(processed_data, sep=sep)

    if "bids_subject" in wide.columns:
        wide = wide.drop_duplicates(subset="bids_subject", keep="last")
        wide = wide.rename(columns={"bids_subject": "subject_id"})

    ordered_cols = ["subject_id"]
    for mod_prefix in MODALITY_ORDER:
        mod_cols = [c for c in wide.columns if c.startswith(mod_prefix + sep)]
        mod_cols.sort()
        ordered_cols.extend(mod_cols)

    remaining = [c for c in wide.columns if c not in ordered_cols]
    if remaining:
        remaining.sort()
        ordered_cols.extend(remaining)

    wide = wide[ordered_cols]
    return wide


def process_session(session_data, output_root, project_id="ANTsX",
                    denoise_dti=True, dti_moco='SyN', separator='_', verbose=True,
                    build_wide_table=True):
    """
    Runs the full ANTsPyMM pipeline on one session.
    """
    result = {
        'success': False,
        'wide_df': None,
        'session_dir': None
    }

    # 1. Setup paths
    sub_id = session_data['subjectID']
    date_id = session_data['date']

    # 2. Extract run ID (image_uid)
    t1_fn = session_data['t1_filename']
    image_uid = extract_image_id(t1_fn)

    # 3. Setup Staging Area (Root of our temp NRG structure)
    staging_root = os.path.join(tempfile.gettempdir(), f"antsxmm_staging_{sub_id}_{date_id}")
    if os.path.exists(staging_root):
        shutil.rmtree(staging_root)
    os.makedirs(staging_root, exist_ok=True)

    # 4. Stage Files
    # T1w
    t1_path, _ = sanitize_and_stage_file(t1_fn, project_id, sub_id, date_id, "T1w", image_uid, separator, staging_root, verbose)

    # FLAIR
    flair_raw = session_data.get('flair_filename', None)
    flair_path, _ = sanitize_and_stage_file(flair_raw, project_id, sub_id, date_id, "T2Flair", image_uid, separator, staging_root, verbose)
    flair_info = (flair_path, _)

    # rsfMRI (Handle List & Variants)
    rsf_raw = session_data.get('rsf_filenames', [])
    rsf_infos = []
    rsf_paths = []
    for f in rsf_raw:
        path, mod = sanitize_and_stage_file(f, project_id, sub_id, date_id, "rsfMRI", image_uid, separator, staging_root, verbose)
        if path:
            rsf_infos.append((path, mod))
            rsf_paths.append(path)

    # DTI (Handle List & Variants)
    dti_raw = session_data.get('dti_filenames', [])
    dti_infos = []
    dti_paths = []
    for f in dti_raw:
        path, mod = sanitize_and_stage_file(f, project_id, sub_id, date_id, "DTI", image_uid, separator, staging_root, verbose)
        if path:
            dti_infos.append((path, mod))
            dti_paths.append(path)

    # NM (Handle List - Unique Run IDs)
    nm_raw = session_data.get('nm_filenames', [])
    nm_infos = []
    nm_paths = []
    for f in nm_raw:
        rid = extract_image_id(f)
        if rid == "000": rid = image_uid
        path, mod = sanitize_and_stage_file(f, project_id, sub_id, date_id, "NM2DMT", rid, separator, staging_root, verbose)
        if path:
            nm_infos.append((path, mod))
            nm_paths.append(path)

    # Perf
    perf_raw = session_data.get('perf_filename', None)
    perf_path, _ = sanitize_and_stage_file(perf_raw, project_id, sub_id, date_id, "perf", image_uid, separator, staging_root, verbose)
    perf_info = (perf_path, _)

    # PET
    pet_raw = session_data.get('pet3d_filename', None)
    pet_path, _ = sanitize_and_stage_file(pet_raw, project_id, sub_id, date_id, "pet3d", image_uid, separator, staging_root, verbose)
    pet_info = (pet_path, _)

    mock_source_dir = staging_root

    try:
        # Pre-execution check
        if verbose:
            print_expected_tree(output_root, project_id, sub_id, date_id, image_uid, 
                                flair_info, rsf_infos, dti_infos, nm_infos, perf_info, pet_info, separator)

        if verbose:
            print(f"\n{'='*80}")
            print(f"Processing: {sub_id} | {date_id}")
            print(f"Image UID: {image_uid}")

        # Run antspymm preprocessing
        study_csv = antspymm.generate_mm_dataframe(
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
            pet3d_filename=pet_path
        )
        
        # Explicitly set IDs in dataframe to match the ones we used for staging
        if 'flairid' in study_csv.columns and flair_path: 
            study_csv['flairid'] = image_uid
        
        # Align IDs for DTI/RSF
        if rsf_infos:
            if 'rsfid1' in study_csv.columns: study_csv['rsfid1'] = image_uid
            if 'rsfid2' in study_csv.columns and len(rsf_infos) > 1: study_csv['rsfid2'] = image_uid
            
        if dti_infos:
            if 'dtid1' in study_csv.columns: study_csv['dtid1'] = image_uid
            if 'dtid2' in study_csv.columns and len(dti_infos) > 1: study_csv['dtid2'] = image_uid

        if perf_path and 'perfid' in study_csv.columns: 
            study_csv['perfid'] = image_uid
        if pet_path and 'pet3did' in study_csv.columns: 
            study_csv['pet3did'] = image_uid

        study_csv_clean = study_csv.dropna(axis=1)

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

        antspymm.mm_csv(
            study_csv_clean,
            mysep=separator,
            dti_motion_correct=dti_moco,
            dti_denoise=denoise_dti,
            normalization_template=template,
            normalization_template_output='ppmi',
            normalization_template_transform_type='antsRegistrationSyNQuickRepro[s]',
            normalization_template_spacing=[1,1,1],
            srmodel_T1=None, srmodel_NM=None, srmodel_DTI=None
        )

        result['success'] = True
        result['session_dir'] = os.path.join(output_root, project_id, sub_id, date_id)

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
                        filename = f"{project_id}{separator}{sub_id}{separator}{date_id}{separator}T1wHierarchical{separator}{image_uid}{separator}mmwide_merged.csv"
                        out_path = os.path.join(t1_hier_dir, filename)
                        wide_df.to_csv(out_path, index=False)
                        if verbose:
                            print(f"[SUCCESS] Session merged wide table written to:\n  {out_path}")
                    else:
                        if verbose:
                            print(f"[WARNING] T1wHierarchical directory not found: {t1_hier_dir}")

                except Exception as e:
                    if verbose:
                        print("Warning: Failed to build wide table:", e)
                    result['wide_df'] = None

        return result

    except Exception as e:
        print("Error processing {} {}: {}".format(sub_id, date_id, str(e)))
        traceback.print_exc()
        return result
    finally:
        if os.path.exists(staging_root):
            shutil.rmtree(staging_root)