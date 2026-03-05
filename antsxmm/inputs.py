
import os
import math
import re
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

def _extract_run_id_from_filename(path: str) -> str:
    """
    Extract BIDS run identifier from filename.
    Default to run-01 if missing.
    """
    name = Path(path).name

    m = re.search(r"run-(\d+)", name)
    if m:
        return f"run-{int(m.group(1)):02d}"

    # Legacy: r0002-style tokens (not strictly BIDS) are common in older exports.
    m = re.search(r"(?:^|_)(?:r)(\d+)(?:[_.]|$)", name)
    if m:
        return f"run-{int(m.group(1)):02d}"

    return "run-01"


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


def _sidecar_paths_for_nifti(nifti_path: str) -> list[str]:
    """Return existing sidecar file paths for a NIfTI.

    Includes common BIDS sidecars (.json) and DWI sidecars (.bval/.bvec).
    """
    p = Path(nifti_path)
    # Handle .nii.gz specially
    if p.name.endswith('.nii.gz'):
        stem = p.name[:-7]
        base = p.with_name(stem)
    elif p.name.endswith('.nii'):
        stem = p.name[:-4]
        base = p.with_name(stem)
    else:
        base = p.with_suffix('')

    out: list[str] = []
    for ext in ('.json', '.bval', '.bvec'):
        candidate = str(base) + ext
        if os.path.exists(candidate):
            out.append(os.path.realpath(candidate))
    return out


def plan_session_inputs(session_data, *, t1_run_match: str | None = None) -> dict:
    """Plan which inputs will be processed for a session.

    This centralizes selection/truncation logic so the pipeline can make
    idempotent/resume decisions without executing antspymm.
    """
    sub_id = session_data.get('subjectID')
    if not sub_id:
        raise KeyError("session_data missing required key: 'subjectID'")

    date_id = session_data.get('date') or session_data.get('sessionID')
    if not date_id:
        raise KeyError("session_data missing required key: 'date' (or alias 'sessionID')")

    # T1 selection
    all_t1s = _as_path_list(session_data.get('t1_filenames'))
    if not all_t1s:
        t1_fn = session_data.get('t1_filename')
    else:
        t1_fn = all_t1s[0]
        if t1_run_match:
            matches = [f for f in all_t1s if t1_run_match in os.path.basename(f)]
            if matches:
                t1_fn = matches[0]

    if not t1_fn:
        # No T1 means we cannot process this session.
        return {
            'subjectID': sub_id,
            'sessionID': date_id,
            'processable': False,
            'reason': 'no_T1w',
            'used': {},
            'nifti_inputs': [],
        }

    # FLAIR (fallback to T2w)
    flair_raw = session_data.get('flair_filename', None)
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

    # rsfMRI (truncate to 2)
    rsf_raw = _as_path_list(session_data.get('rsf_filenames'))
    rsf_selected_raw = rsf_raw[:2]

    # DTI (truncate to 2)
    dti_raw = _as_path_list(session_data.get('dti_filenames'))
    dti_selected_raw = dti_raw[:2]

    # NM (no truncation)
    nm_selected_raw = _as_path_list(session_data.get('nm_filenames'))

    # Perf: scalar/list with asl/perf discovery
    perf_raw = session_data.get('perf_filename', None)
    if isinstance(perf_raw, float) and pd.isna(perf_raw):
        perf_raw = None
    if not isinstance(perf_raw, (str, os.PathLike)):
        perf_raw = None
    if not perf_raw:
        perf_list = _as_path_list(session_data.get('perf_filenames'))
        perf_raw = perf_list[0] if perf_list else None
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

    # PET
    pet_raw = session_data.get('pet3d_filename', None)
    if isinstance(pet_raw, float) and pd.isna(pet_raw):
        pet_raw = None
    if not isinstance(pet_raw, (str, os.PathLike)):
        pet_raw = None
    if not pet_raw:
        pet_list = _as_path_list(session_data.get('pet3d_filenames'))
        pet_raw = pet_list[0] if pet_list else None

    rp = os.path.realpath
    used = {
        't1_filename': rp(t1_fn) if t1_fn else None,
        'flair_or_t2_as_flair_filename': rp(flair_raw) if flair_raw else None,
        'rsf_filenames': [rp(p) for p in rsf_selected_raw if p],
        'dti_filenames': [rp(p) for p in dti_selected_raw if p],
        'nm_filenames': [rp(p) for p in nm_selected_raw if p],
        'perf_filename': rp(perf_raw) if perf_raw else None,
        'pet3d_filename': rp(pet_raw) if pet_raw else None,
    }

    nifti_inputs: list[str] = []
    for p in [used['t1_filename'], used['flair_or_t2_as_flair_filename'], used['perf_filename'], used['pet3d_filename']]:
        if p:
            nifti_inputs.append(p)
    nifti_inputs.extend(list(used['rsf_filenames']))
    nifti_inputs.extend(list(used['dti_filenames']))
    nifti_inputs.extend(list(used['nm_filenames']))
    nifti_inputs = sorted([p for p in nifti_inputs if p and os.path.exists(p) and _is_nifti(p)])

    return {
        'subjectID': sub_id,
        'sessionID': date_id,
        'processable': True,
        't1_fn': t1_fn,
        'flair_raw': flair_raw,
        'rsf_selected_raw': rsf_selected_raw,
        'dti_selected_raw': dti_selected_raw,
        'nm_selected_raw': nm_selected_raw,
        'perf_raw': perf_raw,
        'pet_raw': pet_raw,
        'used': used,
        'nifti_inputs': nifti_inputs,
    }
