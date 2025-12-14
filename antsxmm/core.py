import os
import antspymm
import pandas as pd
import ants
import tempfile
import shutil
import re
import traceback
from pathlib import Path

def sanitize_filename(filepath, requirements, verbose=False):
    """
    Checks if filepath meets antspymm naming requirements.
    If not, creates a symlink in a temporary directory that does.
    """
    if not filepath:
        return None
        
    filename = os.path.basename(filepath)
    
    meets_req = False
    for req in requirements:
        if req in filename:
            meets_req = True
            break
            
    if meets_req:
        return filepath
        
    if verbose:
        print(f"Sanitizing filename: {filename} does not contain {requirements}")
        
    staging_dir = os.path.join(tempfile.gettempdir(), "antsxmm_staging")
    os.makedirs(staging_dir, exist_ok=True)
    
    injector = requirements[0]
    new_name = filename
    replaced = False
    for req in requirements:
        pattern = re.compile(re.escape(req), re.IGNORECASE)
        if pattern.search(filename):
            new_name = pattern.sub(req, filename)
            replaced = True
            break
            
    if not replaced:
        name, ext = os.path.splitext(filename)
        if name.endswith(".nii"): 
            name, ext2 = os.path.splitext(name)
            ext = ext2 + ext
        new_name = f"{name}_{injector}{ext}"
        
    symlink_path = os.path.join(staging_dir, new_name)
    
    if os.path.exists(symlink_path) or os.path.islink(symlink_path):
        os.remove(symlink_path)
        
    os.symlink(os.path.abspath(filepath), symlink_path)
    return symlink_path

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

def print_expected_tree(output_root, project_id, sub_id, date_id, image_uid, 
                        flair_fn, rsf_fns, dti_fns, nm_fns, perf_fn, pet_fn, sep="_"):
    """
    Prints the expected directory structure for validation before processing.
    Also lists MISSING modalities.
    """
    base = Path(output_root) / project_id / sub_id / date_id
    
    print(f"\n[PRE-CHECK] Processing Plan for {sub_id} {date_id}:")
    print(f"ROOT OUTPUT: {base}")
    
    # T1 Hierarchy (Always present if running)
    print(f"├── T1wHierarchical/ (ID: {image_uid}) [FOUND]")
    print(f"│   └── .../{project_id}{sep}{sub_id}{sep}{date_id}{sep}T1wHierarchical{sep}{image_uid}{sep}...")

    # FLAIR
    if flair_fn:
        print(f"├── T2Flair/ (ID: {image_uid}) [FOUND]")
        print(f"│   └── .../{project_id}{sep}{sub_id}{sep}{date_id}{sep}T2Flair{sep}{image_uid}{sep}...")
    else:
        print(f"├── T2Flair/ [MISSING] (Skipping)")

    # rsfMRI
    if rsf_fns:
        print(f"├── rsfMRI/ (ID: {image_uid}) [FOUND: {len(rsf_fns)} scan(s)]")
        print(f"│   └── .../{project_id}{sep}{sub_id}{sep}{date_id}{sep}rsfMRI{sep}{image_uid}{sep}...")
    else:
        print(f"├── rsfMRI/ [MISSING] (Skipping)")

    # DTI
    if dti_fns:
        print(f"├── DTI/ (ID: {image_uid}) [FOUND: {len(dti_fns)} scan(s)]")
        print(f"│   └── .../{project_id}{sep}{sub_id}{sep}{date_id}{sep}DTI{sep}{image_uid}{sep}...")
    else:
        print(f"├── DTI/ [MISSING] (Skipping)")

    # Neuromelanin
    if nm_fns:
        nm_id = extract_image_id(nm_fns[0])
        print(f"└── NM2DMT/ (ID: {nm_id}) [FOUND: {len(nm_fns)} scan(s)]")
        print(f"    └── .../{project_id}{sep}{sub_id}{sep}{date_id}{sep}NM2DMT{sep}{nm_id}{sep}...")
    else:
        print(f"└── NM2DMT/ [MISSING] (Skipping)")

    # Perfusion
    if perf_fn:
        print(f"├── perf/ (ID: {image_uid}) [FOUND]")
        print(f"│   └── .../{project_id}{sep}{sub_id}{sep}{date_id}{sep}perf{sep}{image_uid}{sep}...")
    else:
        print(f"├── perf/ [MISSING] (Skipping)")

    # PET
    if pet_fn:
        print(f"├── pet3d/ (ID: {image_uid}) [FOUND]")
        print(f"│   └── .../{project_id}{sep}{sub_id}{sep}{date_id}{sep}pet3d{sep}{image_uid}{sep}...")
    else:
        print(f"├── pet3d/ [MISSING] (Skipping)")

    print("\n")

def bind_mm_rows(named_dataframes, sep="_"):
    """
    Combine multiple modality-wide CSVs into one super-wide DataFrame.
    """
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
    """
    Check if modalities in ordered_data appear in the expected order.
    """
    actual_mods = [mod for mod, _ in ordered_data]
    filtered_expected = [mod for mod in expected_order if mod in actual_mods]

    if actual_mods != filtered_expected:
        raise AssertionError(
            f"Modality order incorrect!\n"
            f"Expected (filtered): {filtered_expected}\n"
            f"Actual:      {actual_mods}"
        )
    return True


def build_wide_table_from_mmwide(root_dir, pattern="**/*_mmwide.csv", sep="_", verbose=True):
    """
    Build a single-row wide feature table from all *_mmwide.csv files in a session directory.
    """

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

    # 2. Extract and sanitize filenames
    t1_fn = session_data['t1_filename']
    image_uid = extract_image_id(t1_fn)

    flair_raw = session_data.get('flair_filename', None)
    flair_fn = sanitize_filename(flair_raw, ["lair"], verbose)

    rsf_raw = session_data.get('rsf_filenames', [])
    rsf_fns = [sanitize_filename(f, ["fMRI", "func"], verbose) for f in rsf_raw]

    dti_fns = session_data.get('dti_filenames', [])
    nm_fns = session_data.get('nm_filenames', [])

    perf_raw = session_data.get('perf_filename', None)
    perf_fn = sanitize_filename(perf_raw, ["perf"], verbose)

    pet_raw = session_data.get('pet3d_filename', None)
    pet_fn = pet_raw

    mock_source_dir = os.path.dirname(os.path.dirname(t1_fn))

    try:
        # Pre-execution check
        if verbose:
            print_expected_tree(output_root, project_id, sub_id, date_id, image_uid, 
                                flair_fn, rsf_fns, dti_fns, nm_fns, perf_fn, pet_fn, separator)

        if verbose:
            print(f"\n{'='*80}")
            print(f"Processing: {sub_id} | {date_id}")
            print(f"Image UID derived from T1: {image_uid}")

        # Run antspymm preprocessing
        study_csv = antspymm.generate_mm_dataframe(
            projectID=project_id,
            subjectID=sub_id,
            date=date_id,
            imageUniqueID=image_uid, # Used for T1
            modality='T1w',
            source_image_directory=mock_source_dir,
            output_image_directory=output_root,
            t1_filename=t1_fn,
            flair_filename=flair_fn,
            rsf_filenames=rsf_fns,
            dti_filenames=dti_fns,
            nm_filenames=nm_fns,
            perf_filename=perf_fn,
            pet3d_filename=pet_fn
        )
        
        # Override IDs to ensure alignment across all modalities
        if 'flairid' in study_csv.columns: study_csv['flairid'] = image_uid
        if 'rsfid1' in study_csv.columns: study_csv['rsfid1'] = image_uid
        if 'rsfid2' in study_csv.columns: study_csv['rsfid2'] = image_uid
        if 'dtid1' in study_csv.columns: study_csv['dtid1'] = image_uid
        if 'dtid2' in study_csv.columns: study_csv['dtid2'] = image_uid
        if 'perfid' in study_csv.columns: study_csv['perfid'] = image_uid
        if 'pet3did' in study_csv.columns: study_csv['pet3did'] = image_uid

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