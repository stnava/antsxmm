import os
from unittest import result
import antspymm
import pandas as pd
import ants
import tempfile
import shutil
import re

def sanitize_filename(filepath, requirements, verbose=False):
    """
    Checks if filepath meets antspymm naming requirements.
    If not, creates a symlink in a temporary directory that does.
    """
    if not filepath:
        return None
        
    filename = os.path.basename(filepath)
    
    # Check if any requirement is met exactly
    meets_req = False
    for req in requirements:
        if req in filename:
            meets_req = True
            break
            
    if meets_req:
        return filepath
        
    # Validation failed, creating symlink
    if verbose:
        print(f"Sanitizing filename: {filename} does not contain {requirements}")
        
    # Create temp directory
    staging_dir = os.path.join(tempfile.gettempdir(), "antsxmm_staging")
    os.makedirs(staging_dir, exist_ok=True)
    
    # Construct new name
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

import re
from pathlib import Path
from typing import Dict, Tuple, List

def parse_bids_filename(filename: str, separator: str = "_") -> Dict[str, str]:
    """
    Rock-solid parser for your exact naming scheme:
        sub-XXXX_ses-YYYY_r0001_<anything>-<modality>[_ph][_t12345].nii.gz|.json|.nrrd|...

    Returns:
        subjectID, sessionID, repeat, modality, extension, extras, original
    """
    path = Path(filename)
    
    # === Handle double extensions like .nii.gz ===
    if path.suffixes[-2:] == ['.nii', '.gz']:
        stem = path.stem.rstrip('.nii')  # removes .nii
        extension = '.nii.gz'
    else:
        stem = path.stem
        extension = ''.join(path.suffixes)  # .json, .nrrd, etc.

    parts: List[str] = stem.split(separator)
    if len(parts) < 4:
        raise ValueError(f"Too few parts: {filename}")

    # === POSITION 0,1,2 — always subject, session, repeat ===
    sub, ses, repeat = parts[0], parts[1], parts[2]
    if not (sub.startswith("sub-") or 
            ses.startswith("ses-") ):
        raise ValueError(f"Invalid sub/ses in: {filename}")

    # === Payload: everything after repeat ===
    payload = parts[3:]

    # === Pop trailing known extras (ph, t12345, etc.) ===
    extras: List[str] = []
    while payload and (payload[-1] == "ph" or re.match(r"^t\d+$", payload[-1])):
        extras.insert(0, payload.pop())

    if not payload:
        raise ValueError(f"No modality found: {filename}")

    # === Modality = last segment of the final chunk after '-' ===
    final_chunk = payload[-1]
    modality = final_chunk.split("-")[-1]

    return {
        "subjectID": sub,
        "sessionID": ses,
        "repeat": repeat,
        "modality": modality,
        "extension": extension,
        "extras": extras,
        "original": str(path),
    }


def extract_key_parts(filename: str, separator: str = "_") -> Tuple[str, str, str, str, str]:
    """
    Returns the 5-tuple you probably want most:
    (subjectID, sessionID, repeat, modality, extension)
    """
    p = parse_bids_filename(filename, separator)
    return (p["subjectID"], p["sessionID"], p["repeat"], p["modality"], p["extension"])


import pandas as pd
import numpy as np
from pathlib import Path

def bind_mm_rows(named_dataframes, sep="_"):
    """
    Combine multiple modality-wide CSVs into one super-wide DataFrame.

    Parameters
    ----------
    named_dataframes : list of tuples (modality_name, pandas.DataFrame)
        Example:
            [
                ("T1Hier",   t1_df),
                ("NM",       nm_df),
                ("DTI",      dti_df),
                ("FLAIR",    flair_df)
            ]
    
    sep : str, default "_"
        Separator between modality name and original column name.

    Behaviour
    ---------
    • Within one modality: if multiple rows → keep the LAST non-missing value
    • Across modalities: if final column name repeats → keep the LEFTMOST one
    • Result: one row per subject, thousands of columns, perfectly clean names

    Returns
    -------
    pandas.DataFrame
        Wide table with subject_id as first column
    """
    if not named_dataframes:
        return pd.DataFrame()

    processed = []

    for mod_name, df in named_dataframes:
        df = df.copy()
        df = df.replace(["", "NA"], np.nan)

        # Use first column as subject identifier (u_hier_id, etc.)
        id_col = df.columns[0]
        df = df.set_index(id_col)

        # If multiple rows → keep last non-NA value per column
        if len(df) > 1:
            df = df.groupby(level=0).last()

        # Prepend modality name with exactly one separator
        # Note: do NOT prefix subject ID column (which is index now)
        new_cols = {c: mod_name + sep + c for c in df.columns}
        df = df.rename(columns=new_cols)

        processed.append(df)

    # Combine all modalities side-by-side
    combined = pd.concat(processed, axis=1, join="outer")

    # Resolve duplicate column names: keep first (leftmost) occurrence
    combined = combined.loc[:, ~combined.columns.duplicated(keep="first")]

    # Sort columns alphabetically for consistent output (excluding index)
    combined = combined.reindex(sorted(combined.columns), axis=1)

    # Bring subject ID back as regular column
    return combined.reset_index().rename(columns={"index": "subject_id"})


def check_modality_order(ordered_data, expected_order):
    """
    Check if modalities in ordered_data appear in the expected order.
    Returns True if order is correct (modality sequence matches expected_order, allowing skips),
    else raises an AssertionError describing the mismatch.
    """
    actual_mods = [mod for mod, _ in ordered_data]
    filtered_expected = [mod for mod in expected_order if mod in actual_mods]

    if actual_mods != filtered_expected:
        raise AssertionError(
            f"Modality order incorrect!\n"
            f"Expected (filtered): {filtered_expected}\n"
            f"Actual:             {actual_mods}"
        )
    return True


def build_wide_table_from_mmwide(root_dir, pattern="**/*_mmwide.csv", sep="_", verbose=True):
    """
    Build a single-row wide feature table from all *_mmwide.csv files in a session directory.

    Modality order:
        T1Hier → T1w → DTI → rsfMRI → T2Flair → NM2DMT
    """

    root = Path(root_dir).expanduser().resolve()
    csv_files = sorted(root.rglob(pattern))

    if verbose:
        print(f"\nFound {len(csv_files)} *_mmwide.csv files")
        for f in csv_files:
            print("  →", f.relative_to(root))

    MODALITY_MAP = {
        "T1wHierarchical": "T1Hier",
        "T1Hier":          "T1Hier",
        "T1w":             "T1w",
        "NM2DMT":          "NM2DMT",
        "NM":              "NM2DMT",
        "DTI":             "DTI",
        "rsfMRI":          "rsfMRI",
        "T2Flair":         "T2Flair",
        "FLAIR":           "T2Flair",
        "perf":            "perf",
        "pet3d":           "pet3d",
    }

    MODALITY_ORDER = ["T1Hier", "T1w", "DTI", "rsfMRI", "T2Flair", "NM2DMT"]

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
                print(f"  SKIP: unknown modality for file {csv_path.name}")
            continue

        if verbose:
            print(f"\nProcessing: {matched_prefix.ljust(10)} | {csv_path.name}")

        df = pd.read_csv(csv_path)

        drop_cols = [c for c in df.columns if "hier_id" in c.lower()]
        if drop_cols:
            df = df.drop(columns=drop_cols)
            if verbose:
                print(f"  Dropped columns: {drop_cols}")

        if len(df) > 1:
            if verbose:
                print(f"  Collapsing {len(df)} rows to last")
            df = df.iloc[[-1]].copy()

        df.insert(0, "bids_subject", subject_key)

        raw_data.append((matched_prefix, df))

    if not raw_data:
        raise RuntimeError("No valid *_mmwide.csv files were loaded!")

    if verbose:
        print("\nRaw modalities found:")
        for m, d in raw_data:
            print(f"  • {m.ljust(10)} → {d.shape}  subject = {d['bids_subject'].iloc[0]}")

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

    # Check modality order correctness
    check_modality_order(ordered_data, MODALITY_ORDER)

    if verbose:
        print("\n====================================================================================================")
        print("CALLING bind_mm_rows()")
        for mod, df in ordered_data:
            print(f"  • {mod.ljust(10)} → {df.shape}  subject = {df['bids_subject'].iloc[0]}")
        print("====================================================================================================")

    t1hier_df = next((df for mod, df in ordered_data if mod == "T1Hier"), None)
    t1hier_raw_cols = set()
    if t1hier_df is not None:
        t1hier_raw_cols = set(t1hier_df.columns) - {"bids_subject"}

    processed_data = []
    for mod, df in ordered_data:
        if mod != "T1Hier" and t1hier_raw_cols:
            overlap = t1hier_raw_cols & set(df.columns)
            if overlap and verbose:
                print(f"  Excluding {len(overlap)} overlapping columns from {mod} (e.g. {list(overlap)[:5]})")
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

    if verbose:
        print("\n====================================================================================================")
        print("FINAL WIDE TABLE+ASS")
        print(f"   Shape       : {wide.shape}")
        print(f"   Subject     : {wide['subject_id'].iloc[0]}")
        print(f"   First 20 columns : {wide.columns[:20].tolist()}")
        print(f"   vol_wmtissues from : {wide.filter(like='vol_wmtissues').columns.tolist()}")

    return wide


def process_session(session_data, output_root, project_id="ANTsX",
                    denoise_dti=True, dti_moco='SyN', separator='_', verbose=True,
                    build_wide_table=True):
    """
    Runs the full ANTsPyMM pipeline on one session AND optionally builds
    the super-wide feature table from all *_mmwide.csv files in the session folder.
    
    Returns
    -------
    dict with keys:
        'success': bool
        'wide_df': pd.DataFrame or None
        'session_dir': str
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
    t1_fn_parsed = parse_bids_filename(t1_fn)
    image_uid = t1_fn_parsed['repeat']

    flair_raw = session_data.get('flair_filename', None)
    flair_fn = sanitize_filename(flair_raw, ["lair"], verbose)

    rsf_raw = session_data.get('rsf_filenames', [])
    rsf_fns = [sanitize_filename(f, ["fMRI", "func"], verbose) for f in rsf_raw]

    dti_fns = session_data.get('dti_filenames', [])
    nm_fns = session_data.get('nm_filenames', [])

    mock_source_dir = os.path.dirname(os.path.dirname(t1_fn))

    try:
        if verbose:
            print(f"\n{'='*80}")
            print(f"Processing: {sub_id} | {date_id}")
            print(f"Generating MM DataFrame...")

        # Run antspymm preprocessing
        study_csv = antspymm.generate_mm_dataframe(
            projectID=project_id,
            subjectID=sub_id,
            date=date_id,
            imageUniqueID=image_uid,
            modality='T1w',
            source_image_directory=mock_source_dir,
            output_image_directory=output_root,
            t1_filename=t1_fn,
            flair_filename=flair_fn,
            rsf_filenames=rsf_fns,
            dti_filenames=dti_fns,
            nm_filenames=nm_fns
        )

        study_csv_clean = study_csv.dropna(axis=1)

        # Optional: load template (same as before)
        try:
            template_path = antspymm.get_data("PPMI_template0", target_extension=".nii.gz")
            mask_path = antspymm.get_data("PPMI_template0_brainmask", target_extension=".nii.gz")
            template = ants.image_read(template_path)
            template_mask = ants.image_read(mask_path)
            template = template * template_mask
            template = ants.crop_image(template, ants.iMath(template_mask, "MD", 12))
        except:
            template = None
            if verbose:
                print("Warning: Using default template (None)")

        if verbose:
            print("Running antspymm.mm_csv() — this may take 5–15 minutes...")

        antspymm.mm_csv(
            study_csv_clean,
            mysep=separator,
            dti_motion_correct=dti_moco,
            dti_denoise=denoise_dti,
            normalization_template=template,
            normalization_template_output='ppmi',
            normalization_template_transform_type='antsRegistrationSyNQuickRepro[s]',
            normalization_template_spacing=[1,1,1]
        )

        result['success'] = True
        result['session_dir'] = os.path.join(output_root, sub_id, f"ses-{date_id}")

        # ——————————————————————————————————————————————————————————————
        # NEW: Automatically build wide table from this session's output
        # ——————————————————————————————————————————————————————————————
        if build_wide_table:
            session_output_dir = result['session_dir']
            if os.path.exists(session_output_dir):
                if verbose:
                    print(f"\nBuilding wide feature table from session output...")
                    print(f"   Directory: {session_output_dir}")

                try:
                    wide_df = build_wide_table_from_mmwide(
                        root_dir=session_output_dir,
                        sep=separator,
                        verbose=verbose
                    )
                    result['wide_df'] = wide_df

                    if verbose:
                        print(f"Success: Wide table built → {wide_df.shape[0]} × {wide_df.shape[1]}")
                        print(f"   Subject ID: {wide_df['subject_id'].iloc[0]}")

                except Exception as e:
                    if verbose:
                        print("Warning: Failed to build wide table:", e)
                    result['wide_df'] = None
            else:
                if verbose:
                    print("Warning: Session output directory not found yet — skipping wide table.")

        return result

    except Exception as e:
        print("Error processing {} {}: {}".format(sub_id, date_id, str(e)))
        traceback.print_exc()
        return result
