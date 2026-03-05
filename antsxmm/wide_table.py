import os
import re
import math
import json
from pathlib import Path
import pandas as pd

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

    combined = combined.reset_index()
    cols = list(combined.columns)
    cols[0] = "subject_id"
    combined.columns = cols
    
    return combined


def check_modality_order(ordered_data, expected_order):
    actual_mods = [mod for mod, _ in ordered_data]
    filtered_expected = [mod for mod in expected_order if mod in actual_mods]

    if actual_mods != filtered_expected:
        print("Warning: Modality order mismatch. Expected: {}, Got: {}".format(filtered_expected, actual_mods))
    return True


def build_wide_table_from_mmwide(root_dir, sep="_", verbose=True):
    root = Path(root_dir).expanduser().resolve()
    pattern="*/*" + sep + "mmwide.csv"
    csv_files = sorted(root.rglob(pattern))

    if verbose:
        print("\nFound {} *_mmwide.csv files".format(len(csv_files)))
        for f in csv_files:
            try:
                print(" -> {}".format(f.relative_to(root)))
            except ValueError:
                print(" -> {}".format(f))

    MODALITY_MAP = {
        "T1wHierarchical": "T1Hier",
        "T1Hier":  "T1Hier",
        "T1w":    "T1w",
        "NM2DMT":  "NM2DMT",
        "NM":    "NM2DMT",
        "DTI":    "DTI",
        "rsfMRI":  "rsfMRI",
        "T2Flair":  "T2Flair",
        "FLAIR":   "T2Flair",
        "perf":   "perf",
        "pet3d":   "pet3d",
        "PET":    "pet3d",
    }

    MODALITY_ORDER = ["T1Hier", "T1w", "DTI", "rsfMRI", "T2Flair", "NM2DMT", "perf", "pet3d"]

    raw_data = []

    for csv_path in csv_files:
        sub = next((p for p in csv_path.parts if p.startswith("sub-")), None)
        ses = next((p for p in csv_path.parts if p.startswith("ses-")), None)
        if sub and ses:
            subject_key = "{}_{}".format(sub, ses)
        else:
            subject_key = "UNKNOWN"

        matched_prefix = None
        best_len = 0
        for clue, prefix in MODALITY_MAP.items():
            if any(clue in part for part in csv_path.parts) or clue in csv_path.name:
                if len(clue) > best_len:
                    matched_prefix = prefix
                    best_len = len(clue)

        if not matched_prefix:
            if verbose:
                print(" SKIP: unknown modality for file {}".format(csv_path.name))
            continue

        if verbose:
            print("\nProcessing: {} | {}".format(matched_prefix.ljust(10), csv_path.name))

        df = pd.read_csv(csv_path)

        drop_cols = [c for c in df.columns if "hier_id" in c.lower()]
        if drop_cols:
            df = df.drop(columns=drop_cols)
            if verbose:
                print(" Dropped columns: {}".format(drop_cols))

        if len(df) > 1:
            if verbose:
                print(" Collapsing {} rows to last".format(len(df)))
            df = df.iloc[[-1]].copy()
        
        # FIXED: Handle duplicate columns if CSV already has bids_subject
        if "bids_subject" in df.columns:
            df = df.drop(columns=["bids_subject"])
            
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
                print(" Excluding {} overlapping columns from {}".format(len(overlap), mod))
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
