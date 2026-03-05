import os
import math
import types
import pandas as pd

try:  # optional dependency for lightweight installs / unit tests
    import antspymm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    antspymm = types.SimpleNamespace()  # tests can monkeypatch attributes

try:  # optional dependency for lightweight installs / unit tests
    import ants  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    ants = types.SimpleNamespace()
import tempfile
import shutil
import re
import traceback
import json
from pathlib import Path



def _extract_run_id(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"(run-\d+)", text)
    return m.group(1) if m else None


def _rename_prefix_run_segment(
    path: Path, *, project_id: str, subject_id: str, date_id: str, modality: str, run_id: str
) -> None:
    """Rename files whose 5th '+'-segment is a verbose run label to the canonical run_id."""
    prefix = f"{project_id}+{subject_id}+{date_id}+{modality}+"
    for p in path.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if not name.startswith(prefix):
            continue
        parts = name.split("+")
        # Expect at least: project, sub, date, modality, runseg, rest...
        if len(parts) < 6:
            continue
        if parts[4] == run_id:
            continue
        parts[4] = run_id
        new_name = "+".join(parts)
        p.rename(p.with_name(new_name))


def _merge_tree(src: Path, dst: Path) -> None:
    """Move all contents of src into dst, merging directories."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            _merge_tree(item, target)
            try:
                item.rmdir()
            except OSError:
                pass
        else:
            if target.exists():
                # If collision, keep existing and skip to avoid data loss.
                continue
            item.rename(target)
    # best-effort cleanup
    try:
        src.rmdir()
    except OSError:
        pass


def _normalize_session_output_tree(
    session_out: Path, *, project_id: str, subject_id: str, date_id: str
) -> None:
    """Normalize output layout to <modality>/<run-id>/ and canonical filename prefixes.

    Handles two common antspymm output quirks:
      1) modality/<full-stem-containing-run>/ instead of modality/run-XXX/
      2) modality/run-XXX plus an extra modality/<full-stem-containing-run>/ (merge)
    Also rewrites the '+'-delimited filename prefix run segment to run-XXX.
    """
    if not session_out.exists():
        return

    for modality_dir in [p for p in session_out.iterdir() if p.is_dir()]:
        modality = modality_dir.name
        # Determine candidate run id from directory names or files
        run_id = _extract_run_id(modality_dir.name)
        child_dirs = [d for d in modality_dir.iterdir() if d.is_dir()]
        # If there's a canonical run dir, prefer it
        canonical_run_dir = None
        for d in child_dirs:
            rid = _extract_run_id(d.name)
            if rid and d.name == rid:
                canonical_run_dir = d
                run_id = rid
                break

        # If no canonical run dir and exactly one child dir, rename it
        if canonical_run_dir is None and len(child_dirs) == 1:
            only = child_dirs[0]
            rid = _extract_run_id(only.name)
            if rid:
                run_id = rid
                canonical_run_dir = modality_dir / rid
                if only != canonical_run_dir:
                    only.rename(canonical_run_dir)
            else:
                canonical_run_dir = only

        # If we have a run_id but no canonical run dir yet, create one if needed
        if run_id and canonical_run_dir is None:
            canonical_run_dir = modality_dir / run_id
            canonical_run_dir.mkdir(exist_ok=True)

        # Refresh child directory listing after any renames/creates
        child_dirs = [d for d in modality_dir.iterdir() if d.is_dir()]

        # Merge any extra dirs that contain the same run id into canonical
        if run_id and canonical_run_dir is not None:
            for d in child_dirs:
                if d == canonical_run_dir:
                    continue
                if _extract_run_id(d.name) == run_id:
                    _merge_tree(d, canonical_run_dir)

            # Rewrite filename prefix run segment in canonical
            _rename_prefix_run_segment(
                canonical_run_dir,
                project_id=project_id,
                subject_id=subject_id,
                date_id=date_id,
                modality=modality,
                run_id=run_id,
            )
def _is_nifti(path: str) -> bool:
    if not path:
        return False
    p = str(path).lower()
    return p.endswith('.nii') or p.endswith('.nii.gz')



def _as_path_list(value) -> list[str]:
    """Normalize a possibly-missing BIDS field into a list of path strings.

    The parser may yield NaN (float) for missing list-valued columns when using pandas.
    """
    if value is None:
        return []
    # pandas missing values often appear as float('nan')
    if isinstance(value, float):
        try:
            if math.isnan(value):
                return []
        except Exception:
            return []
    if isinstance(value, (str, os.PathLike)):
        s = str(value)
        return [s] if s else []
    # Avoid iterating over scalars like numpy.float64 / numpy.int64
    if isinstance(value, (int, float, bool)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v is not None and not (isinstance(v, float) and math.isnan(v))]
    # Best-effort support for other iterables
    try:
        out = []
        for v in value:
            if v is None:
                continue
            if isinstance(v, float):
                try:
                    if math.isnan(v):
                        continue
                except Exception:
                    continue
            if isinstance(v, (str, os.PathLike)):
                out.append(str(v))
        return out
    except TypeError:
        return []


def _collect_discovered_inputs(session_data):
    """Collect discovered inputs from a BIDS session row, robust to NaN fields.

    IMPORTANT: For manifests, preserve the parser-provided path representation
    (often relative paths like 'BIDS/...') to avoid mixing relative and absolute
    paths across discovered/used/excluded sections.

    Some real-world datasets place ASL under an 'asl/' directory; if perf fields are
    empty, we opportunistically discover ASL NIfTIs there and report them as perf.
    """

    def _filter_existing(paths: list[str]) -> list[str]:
        return [os.path.realpath(p) for p in paths if _is_nifti(p) and os.path.exists(p)]

    discovered = {
        't1_filenames': _filter_existing(_as_path_list(session_data.get('t1_filenames'))),
        'flair_filenames': _filter_existing(_as_path_list(session_data.get('flair_filenames'))),
        't2w_filenames': _filter_existing(_as_path_list(session_data.get('t2w_filenames'))),
        'dti_filenames': _filter_existing(_as_path_list(session_data.get('dti_filenames'))),
        'rsf_filenames': _filter_existing(_as_path_list(session_data.get('rsf_filenames'))),
        'nm_filenames': _filter_existing(_as_path_list(session_data.get('nm_filenames'))),
        'perf_filenames': _filter_existing(_as_path_list(session_data.get('perf_filenames'))),
        'pet3d_filenames': _filter_existing(_as_path_list(session_data.get('pet3d_filenames'))),
    }

    # If perfusion wasn't populated, try to discover ASL under session_path/asl.
    if not discovered['perf_filenames']:
        session_path = session_data.get('session_path')
        if isinstance(session_path, (str, os.PathLike)) and str(session_path):
            ses = Path(session_path)
            candidates: list[Path] = []
            for perf_dir in (ses / 'perf', ses / 'asl'):
                if perf_dir.exists() and perf_dir.is_dir():
                    candidates.extend(perf_dir.glob('*.nii'))
                    candidates.extend(perf_dir.glob('*.nii.gz'))

            perf_found: list[str] = []
            for p in candidates:
                sp = str(p)
                if not os.path.exists(sp) or not _is_nifti(sp):
                    continue
                perf_found.append(os.path.realpath(sp))
            discovered['perf_filenames'] = sorted(set(perf_found))

    return discovered


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

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
            return "{}{}".format(base_modality, suffix)
        else:
            return "{}{}{}".format(base_modality, sep, suffix)
    
    # Default mappings
    if base_modality == "dwi": return "DTI"
    if base_modality == "func": return "rsfMRI"
    
    return base_modality

def sanitize_and_stage_file(filepath, project, subject, date, base_modality, image_id, sep, staging_root, verbose=False):
    """
    Stages a file into a strict NRG directory structure in tmp.
    """
    # Pandas rows may contain NaN for missing entries; those come through as
    # floats and will break os.path operations. Treat all missing/non-path
    # values as absent.
    if filepath is None:
        return None, None, None
    if isinstance(filepath, float) and pd.isna(filepath):
        return None, None, None
    if not isinstance(filepath, (str, os.PathLike)):
        return None, None, None

    filepath = os.fspath(filepath)
    if not filepath:
        return None, None, None

    modality = get_modality_variant(filepath, base_modality, sep)
    
    # Clean modality string for filename construction if separator matches
    filename_modality = modality
    if sep in modality:
        filename_modality = modality.replace(sep, "")

    # Ensure extension is handled
    name = os.path.basename(filepath)
    if name.endswith(".nii.gz"):
        ext = ".nii.gz"
    elif name.endswith(".nii"):
        ext = ".nii"
    else:
        ext = os.path.splitext(name)[1]

    # FORCE NRG FILENAME
    safe_sub = subject if subject else "sub"
    safe_date = date if date else "ses"
    
    new_filename = "{}_{}_{}_{}{}".format(safe_sub, safe_date, filename_modality, image_id, ext)
    
    # Construct NRG path
    dest_dir = os.path.join(staging_root, project, subject, date, modality, image_id)
    os.makedirs(dest_dir, exist_ok=True)
    
    symlink_path = os.path.join(dest_dir, new_filename)
    
    if os.path.exists(symlink_path) or os.path.islink(symlink_path):
        os.remove(symlink_path)
        
    os.symlink(os.path.abspath(filepath), symlink_path)
    
    if verbose:
        print(" Staged: {} -> {}/{}/{}".format(name, modality, image_id, new_filename))
        
    # Stage sidecars (bval, bvec, json)
    new_filename_base = new_filename.replace(ext, "")
    
    if name.endswith(".nii.gz"):
        orig_base = name[:-7]
    elif name.endswith(".nii"):
        orig_base = name[:-4]
    else:
        orig_base = os.path.splitext(name)[0]
        
    src_dir = os.path.dirname(filepath)
    sidecars = [".bval", ".bvec", ".json"]
    
    for side_ext in sidecars:
        src_side = os.path.join(src_dir, orig_base + side_ext)
        if os.path.exists(src_side):
            dst_side = os.path.join(dest_dir, new_filename_base + side_ext)
            if os.path.exists(dst_side) or os.path.islink(dst_side):
                os.remove(dst_side)
            os.symlink(os.path.abspath(src_side), dst_side)
            if verbose:
                print("   + Sidecar: {} -> {}".format(side_ext, os.path.basename(dst_side)))

    return symlink_path, modality, image_id

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
):
    """
    Runs the full ANTsPyMM pipeline on one session.
    """
    result = {
        'success': False,
        'wide_df': None,
        'session_dir': None
    }

    # 1. Setup paths
    sub_id = session_data.get('subjectID')
    if not sub_id:
        raise KeyError("session_data missing required key: 'subjectID'")

    # Historically, callers used a variety of keys for session/date.
    # Prefer 'date' when present but accept 'sessionID' as an alias.
    date_id = session_data.get('date')
    if not date_id:
        date_id = session_data.get('sessionID')
    if not date_id:
        raise KeyError("session_data missing required key: 'date' (or alias 'sessionID')")

    # 2. Select T1 based on run_match if provided
    all_t1s = _as_path_list(session_data.get('t1_filenames'))
    if not all_t1s:
        if 't1_filename' in session_data:
            t1_fn = session_data['t1_filename']
        else:
            print("Error: No T1w found for {} {}".format(sub_id, date_id))
            return result
    else:
        t1_fn = all_t1s[0] # Default
        if t1_run_match:
            matches = [f for f in all_t1s if t1_run_match in os.path.basename(f)]
            if matches:
                t1_fn = matches[0]
                if verbose: print("Selected T1 match: {}".format(os.path.basename(t1_fn)))
            else:
                if verbose: print("Warning: No T1 matched '{}'. Using default: {}".format(t1_run_match, os.path.basename(t1_fn)))

    image_uid = extract_image_id(t1_fn)

    # 3. Setup Staging Area
    staging_root = os.path.join(tempfile.gettempdir(), "antsxmm_staging_{}_{}".format(sub_id, date_id))
    if os.path.exists(staging_root):
        shutil.rmtree(staging_root)
    os.makedirs(staging_root, exist_ok=True)

    # 4. Stage Files
    # T1w
    t1_path, _, _ = sanitize_and_stage_file(t1_fn, project_id, sub_id, date_id, "T1w", image_uid, separator, staging_root, verbose)

    # FLAIR (fallback: T2w if FLAIR absent)
    # Support both scalar and list-valued columns.
    flair_raw = session_data.get('flair_filename', None)
    # guard NaN / non-paths
    if isinstance(flair_raw, float) and pd.isna(flair_raw):
        flair_raw = None
    if not isinstance(flair_raw, (str, os.PathLike)):
        flair_raw = None
    if not flair_raw:
        flair_list = _as_path_list(session_data.get('flair_filenames'))
        flair_raw = flair_list[0] if flair_list else None

    if not flair_raw:
        t2_raw = session_data.get('t2w_filename', None)
        if isinstance(t2_raw, float) and pd.isna(t2_raw):
            t2_raw = None
        if isinstance(t2_raw, (str, os.PathLike)):
            flair_raw = os.fspath(t2_raw)
        else:
            t2_list = _as_path_list(session_data.get('t2w_filenames'))
            flair_raw = t2_list[0] if t2_list else None

    flair_path, flair_mod, flair_id = sanitize_and_stage_file(
        flair_raw, project_id, sub_id, date_id, "T2Flair", image_uid, separator, staging_root, verbose
    )
    flair_info = (flair_path, flair_mod, flair_id)

    # rsfMRI
    rsf_raw = _as_path_list(session_data.get('rsf_filenames'))
    rsf_infos = []
    rsf_paths = []
    rsf_selected_raw = []
    for f in rsf_raw:
        this_id = extract_image_id(f)
        if this_id == "000": this_id = image_uid
        
        path, mod, unique_id = sanitize_and_stage_file(f, project_id, sub_id, date_id, "rsfMRI", this_id, separator, staging_root, verbose)
        if path:
            rsf_infos.append((path, mod, unique_id))
            rsf_paths.append(path)
            rsf_selected_raw.append(f)

    # NOTE: Truncate to first 2 rsf images if > 2 found.
    # This prevents antspymm ValueError: len( ... ) > 3
    if len(rsf_paths) > 2:
        if verbose:
            print("NOTE: Found more then 2 sets of rsfMRI images. Selecting the first 2 only to satisfy antspymm requirements.")
        rsf_paths = rsf_paths[:2]
        rsf_infos = rsf_infos[:2]
        rsf_selected_raw = rsf_selected_raw[:2]


    # DTI
    dti_raw = _as_path_list(session_data.get('dti_filenames'))
    dti_infos = []
    dti_paths = []
    dti_selected_raw = []
    for f in dti_raw:
        this_id = extract_image_id(f)
        if this_id == "000": this_id = image_uid
        path, mod, unique_id = sanitize_and_stage_file(f, project_id, sub_id, date_id, "DTI", this_id, separator, staging_root, verbose)
        if path:
            dti_infos.append((path, mod, unique_id))
            dti_paths.append(path)
            dti_selected_raw.append(f)

    # NOTE: Truncate to first 2 DTI images if > 2 found.
    # This prevents antspymm ValueError: len( dti_filenames ) > 3
    if len(dti_paths) > 2:
        if verbose:
            print("NOTE: Found more then 2 sets of DTI images. Selecting the first 2 only to satisfy antspymm requirements.")
        dti_paths = dti_paths[:2]
        dti_infos = dti_infos[:2]
        dti_selected_raw = dti_selected_raw[:2]

    # NM
    nm_raw = _as_path_list(session_data.get('nm_filenames'))
    nm_infos = []
    nm_paths = []
    nm_selected_raw = []
    for f in nm_raw:
        rid = extract_image_id(f)
        if rid == "000": rid = image_uid
        path, mod, unique_id = sanitize_and_stage_file(f, project_id, sub_id, date_id, "NM2DMT", rid, separator, staging_root, verbose=verbose)
        if path:
            nm_infos.append((path, mod, unique_id))
            nm_paths.append(path)
            nm_selected_raw.append(f)

    # Perf
    perf_raw = session_data.get('perf_filename', None)
    if isinstance(perf_raw, float) and pd.isna(perf_raw):
        perf_raw = None
    if not isinstance(perf_raw, (str, os.PathLike)):
        perf_raw = None
    if not perf_raw:
        perf_list = _as_path_list(session_data.get('perf_filenames'))
        perf_raw = perf_list[0] if perf_list else None

    # Some BIDS variants store ASL under an 'asl/' directory and parsers may not
    # populate perf fields. If perf is still missing, attempt discovery under the
    # session_path (preferring perf/, then asl/).
    if not perf_raw:
        ses_path = session_data.get('session_path')
        if isinstance(ses_path, (str, os.PathLike)) and str(ses_path):
            ses = Path(ses_path)
            candidates: list[Path] = []
            for perf_dir in (ses / 'perf', ses / 'asl'):
                if perf_dir.exists() and perf_dir.is_dir():
                    candidates.extend(sorted(perf_dir.glob('*.nii')))
                    candidates.extend(sorted(perf_dir.glob('*.nii.gz')))
            for c in candidates:
                sp = str(c)
                if _is_nifti(sp) and os.path.exists(sp):
                    perf_raw = sp
                    break
    perf_path, perf_mod, perf_id = sanitize_and_stage_file(
        perf_raw, project_id, sub_id, date_id, "perf", image_uid, separator, staging_root, verbose=verbose
    )
    perf_info = (perf_path, perf_mod, perf_id)

    # PET
    pet_raw = session_data.get('pet3d_filename', None)
    if isinstance(pet_raw, float) and pd.isna(pet_raw):
        pet_raw = None
    if not isinstance(pet_raw, (str, os.PathLike)):
        pet_raw = None
    if not pet_raw:
        pet_list = _as_path_list(session_data.get('pet3d_filenames'))
        pet_raw = pet_list[0] if pet_list else None
    pet_path, pet_mod, pet_id = sanitize_and_stage_file(
        pet_raw, project_id, sub_id, date_id, "pet3d", image_uid, separator, staging_root, verbose=verbose
    )
    pet_info = (pet_path, pet_mod, pet_id)

    mock_source_dir = staging_root

    # Persist an input manifest that enumerates exactly which NIfTIs will be processed.
    session_out_dir = os.path.join(output_root, project_id, sub_id, date_id)
    if write_input_manifest:
        _ensure_dir(session_out_dir)

        # Discoveries (as seen from the BIDS parser row)
        discovered = _collect_discovered_inputs(session_data)

        # IMPORTANT: used_inputs are expressed in the same coordinate system as
        # discovered (absolute realpaths). Staged paths live under a temp directory
        # and must not leak into the manifest.
        rp = os.path.realpath
        used = {
            't1_filename': rp(t1_fn) if t1_fn else None,
            'flair_or_t2_as_flair_filename': rp(flair_raw) if flair_raw else None,
            'rsf_filenames': [rp(p) for p in rsf_selected_raw],
            'dti_filenames': [rp(p) for p in dti_selected_raw],
            'nm_filenames': [rp(p) for p in nm_selected_raw],
            'perf_filename': rp(perf_raw) if perf_raw else None,
            'pet3d_filename': rp(pet_raw) if pet_raw else None,
        }

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
        if not hasattr(antspymm, 'generate_mm_dataframe') or not hasattr(antspymm, 'mm_csv'):
            raise ModuleNotFoundError(
                "antspymm is required to run antsxmm processing. Install antspymm (and ants) to execute the pipeline."
            )

        # Pre-execution check
        if verbose:
            print_expected_tree(output_root, project_id, sub_id, date_id, image_uid, 
                        flair_info, rsf_infos, dti_infos, nm_infos, perf_info, pet_info, separator)

        if verbose:
            print("\n{}".format('='*80))
            print("Processing: {} | {}".format(sub_id, date_id))
            print("Image UID: {}".format(image_uid))

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


        # Normalize antspymm output layout and filename prefixes to stable run-id form
        session_out = Path(output_root) / project_id / sub_id / date_id
        _normalize_session_output_tree(
            session_out,
            project_id=project_id,
            subject_id=sub_id,
            date_id=date_id,
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
        return result
    finally:
        if os.path.exists(staging_root):
            shutil.rmtree(staging_root)