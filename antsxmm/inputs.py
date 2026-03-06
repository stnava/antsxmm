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


def _load_json_sidecar(nifti_path: str) -> dict:
    for p in _sidecar_paths_for_nifti(nifti_path):
        if p.endswith('.json'):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}
    return {}


def _filename_without_nifti_ext(path: str) -> str:
    name = Path(path).name
    if name.endswith('.nii.gz'):
        return name[:-7]
    if name.endswith('.nii'):
        return name[:-4]
    return name


def _bids_tokens(path: str) -> list[str]:
    return _filename_without_nifti_ext(path).split('_')


def _get_bids_entity(path: str, key: str) -> str | None:
    prefix = f"{key}-"
    for tok in _bids_tokens(path):
        if tok.startswith(prefix) and len(tok) > len(prefix):
            return tok[len(prefix):]
    return None


def _suffix_token(path: str) -> str:
    toks = _bids_tokens(path)
    return toks[-1] if toks else ''


def _read_phase_direction(path: str) -> str | None:
    direction = _get_bids_entity(path, 'dir')
    if direction:
        return direction.upper()

    meta = _load_json_sidecar(path)
    phase = meta.get('PhaseEncodingDirection')
    if not isinstance(phase, str) or not phase:
        return None
    phase = phase.strip().lower().rstrip('-')
    mapping = {
        'i': 'LR',
        'i+': 'LR',
        'i-': 'RL',
        'j': 'AP',
        'j+': 'AP',
        'j-': 'PA',
        'k': 'SI',
        'k+': 'SI',
        'k-': 'IS',
        'lr': 'LR',
        'rl': 'RL',
        'ap': 'AP',
        'pa': 'PA',
        'si': 'SI',
        'is': 'IS',
    }
    return mapping.get(phase)


def _exact_suffix_kind(path: str, expected_suffix: str) -> bool:
    return _suffix_token(path).lower() == expected_suffix.lower()


def _variant_suffix_kind(path: str, expected_suffix: str) -> bool:
    suffix = _suffix_token(path).lower()
    expected = expected_suffix.lower()
    return suffix.startswith(expected) and suffix != expected


def _run_number(path: str) -> int:
    run_id = _extract_run_id_from_filename(path)
    m = re.search(r"run-(\d+)", run_id)
    return int(m.group(1)) if m else 1


def _modality_rank(path: str, modality: str) -> tuple[int, ...]:
    suffix = _suffix_token(path).lower()
    task = (_get_bids_entity(path, 'task') or '').lower()
    dir_code = _read_phase_direction(path)

    if modality == 't1':
        return (
            0 if _exact_suffix_kind(path, 'T1w') else 1,
            0 if task == '' else 1,
            0 if 't1w' in suffix else 1,
            _run_number(path),
            len(Path(path).name),
            Path(path).name.lower(),
        )
    if modality == 'flair_or_t2':
        return (
            0 if _exact_suffix_kind(path, 'FLAIR') else 1,
            0 if _exact_suffix_kind(path, 'T2w') else 1,
            0 if suffix in ('flair', 't2w') else 1,
            _run_number(path),
            len(Path(path).name),
            Path(path).name.lower(),
        )
    if modality == 'rsf':
        return (
            0 if task == 'rest' else 1,
            0 if _exact_suffix_kind(path, 'bold') else 1,
            0 if dir_code in {'LR', 'RL', 'AP', 'PA', 'SI', 'IS'} else 1,
            _run_number(path),
            len(Path(path).name),
            Path(path).name.lower(),
        )
    if modality == 'dti':
        return (
            0 if _exact_suffix_kind(path, 'dwi') else 1,
            0 if dir_code in {'LR', 'RL', 'AP', 'PA', 'SI', 'IS'} else 1,
            _run_number(path),
            len(Path(path).name),
            Path(path).name.lower(),
        )
    if modality == 'perf':
        return (
            0 if _exact_suffix_kind(path, 'asl') else 1,
            0 if _exact_suffix_kind(path, 'm0scan') else 1,
            0 if 'm0' not in Path(path).name.lower() else 1,
            _run_number(path),
            len(Path(path).name),
            Path(path).name.lower(),
        )
    if modality == 'pet3d':
        tracer = str(_load_json_sidecar(path).get('TracerRadionuclide', '') or '').strip()
        return (
            0 if _exact_suffix_kind(path, 'pet') else 1,
            0 if tracer else 1,
            _run_number(path),
            len(Path(path).name),
            Path(path).name.lower(),
        )
    if modality == 'nm':
        return (
            _run_number(path),
            0 if _exact_suffix_kind(path, 'NM') else 1,
            len(Path(path).name),
            Path(path).name.lower(),
        )
    return (_run_number(path), len(Path(path).name), Path(path).name.lower())


def _direction_pair_candidates(paths: list[str], modality: str) -> list[str]:
    if not paths:
        return []
    ranked = sorted(paths, key=lambda p: _modality_rank(p, modality))
    pairs = [('LR', 'RL'), ('AP', 'PA'), ('SI', 'IS')]
    for a, b in pairs:
        first = next((p for p in ranked if _read_phase_direction(p) == a), None)
        second = next((p for p in ranked if _read_phase_direction(p) == b and p != first), None)
        if first and second:
            return [first, second]
    return ranked[:2]


def _selection_reason_lines(path: str, modality: str) -> list[str]:
    reasons: list[str] = []
    suffix = _suffix_token(path)
    task = _get_bids_entity(path, 'task')
    direction = _read_phase_direction(path)

    if modality == 't1' and _exact_suffix_kind(path, 'T1w'):
        reasons.append('exact_suffix:T1w')
    if modality == 'flair_or_t2':
        if _exact_suffix_kind(path, 'FLAIR'):
            reasons.append('exact_suffix:FLAIR')
        elif _exact_suffix_kind(path, 'T2w'):
            reasons.append('fallback_suffix:T2w')
    if modality == 'rsf':
        if (task or '').lower() == 'rest':
            reasons.append('task:rest')
        if _exact_suffix_kind(path, 'bold'):
            reasons.append('exact_suffix:bold')
    if modality == 'dti' and _exact_suffix_kind(path, 'dwi'):
        reasons.append('exact_suffix:dwi')
    if modality == 'perf':
        if _exact_suffix_kind(path, 'asl'):
            reasons.append('exact_suffix:asl')
        if _exact_suffix_kind(path, 'm0scan'):
            reasons.append('supporting_scan:m0scan')
    if modality == 'pet3d' and _exact_suffix_kind(path, 'pet'):
        reasons.append('exact_suffix:pet')
    if modality == 'nm' and suffix.lower() == 'nm':
        reasons.append('exact_suffix:NM')
    if direction:
        reasons.append(f'phase_direction:{direction}')
    reasons.append(f'run:{_extract_run_id_from_filename(path)}')
    return reasons


def _ranked_selection(paths: list[str], modality: str, *, limit: int | None, preferred_pair: bool = False) -> tuple[list[str], dict]:
    existing = [os.path.realpath(p) for p in paths if _is_nifti(p) and os.path.exists(p)]
    unique_existing = sorted(set(existing))
    ranked = sorted(unique_existing, key=lambda p: _modality_rank(p, modality))

    if preferred_pair and limit == 2:
        selected = _direction_pair_candidates(ranked, modality)
    elif limit is None:
        selected = ranked
    else:
        selected = ranked[:limit]

    selected_set = set(selected)
    excluded = [p for p in ranked if p not in selected_set]
    tracking = {
        'strategy': modality,
        'limit': limit,
        'preferred_pair': bool(preferred_pair),
        'ranked_candidates': [
            {
                'path': p,
                'selected': p in selected_set,
                'rank': idx + 1,
                'reasons': _selection_reason_lines(p, modality),
            }
            for idx, p in enumerate(ranked)
        ],
        'selected': selected,
        'excluded': excluded,
    }
    return selected, tracking


def _discover_perf_candidates(session_data) -> list[str]:
    perf_list = _as_path_list(session_data.get('perf_filenames'))
    if perf_list:
        return perf_list
    perf_single = session_data.get('perf_filename')
    if isinstance(perf_single, (str, os.PathLike)) and str(perf_single):
        return [os.fspath(perf_single)]

    ses_path = session_data.get('session_path')
    if isinstance(ses_path, (str, os.PathLike)) and str(ses_path):
        ses = Path(ses_path)
        candidates: list[str] = []
        for perf_dir in (ses / 'perf', ses / 'asl'):
            if perf_dir.exists() and perf_dir.is_dir():
                candidates.extend([str(p) for p in sorted(perf_dir.glob('*.nii'))])
                candidates.extend([str(p) for p in sorted(perf_dir.glob('*.nii.gz'))])
        return candidates
    return []



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

    selection_tracking: dict[str, dict] = {}

    # T1 selection
    t1_candidates = [p for p in _as_path_list(session_data.get('t1_filenames')) if p]
    if not t1_candidates:
        t1_fn = session_data.get('t1_filename')
        t1_candidates = [t1_fn] if t1_fn else []

    t1_selected, t1_tracking = _ranked_selection(t1_candidates, 't1', limit=1)
    if t1_run_match and len(t1_candidates) > 1:
        matched = [os.path.realpath(p) for p in t1_candidates if t1_run_match in os.path.basename(p) and os.path.exists(p) and _is_nifti(p)]
        if matched:
            ranked_match = sorted(set(matched), key=lambda p: _modality_rank(p, 't1'))
            t1_selected = ranked_match[:1]
            sel = set(t1_selected)
            ranked_all = [c['path'] for c in t1_tracking['ranked_candidates']]
            t1_tracking['ranked_candidates'] = [
                {
                    **c,
                    'selected': c['path'] in sel,
                    'reasons': list(c['reasons']) + (['matched_t1_run'] if t1_run_match in os.path.basename(c['path']) else []),
                }
                for c in t1_tracking['ranked_candidates']
            ]
            t1_tracking['selected'] = t1_selected
            t1_tracking['excluded'] = [p for p in ranked_all if p not in sel]
            t1_tracking['t1_run_match'] = t1_run_match
    selection_tracking['t1'] = t1_tracking
    t1_fn = t1_selected[0] if t1_selected else None

    if not t1_fn:
        return {
            'subjectID': sub_id,
            'sessionID': date_id,
            'processable': False,
            'reason': 'no_T1w',
            'used': {},
            'nifti_inputs': [],
            'selection_tracking': selection_tracking,
        }

    # FLAIR (fallback to T2w)
    flair_candidates = _as_path_list(session_data.get('flair_filenames')) + _as_path_list(session_data.get('t2w_filenames'))
    flair_selected, flair_tracking = _ranked_selection(flair_candidates, 'flair_or_t2', limit=1)
    selection_tracking['flair_or_t2'] = flair_tracking
    flair_raw = flair_selected[0] if flair_selected else None

    # rsfMRI (pair-aware, max 2)
    rsf_candidates = _as_path_list(session_data.get('rsf_filenames'))
    rsf_selected_raw, rsf_tracking = _ranked_selection(rsf_candidates, 'rsf', limit=2, preferred_pair=True)
    selection_tracking['rsf'] = rsf_tracking

    # DTI (pair-aware, max 2)
    dti_candidates = _as_path_list(session_data.get('dti_filenames'))
    dti_selected_raw, dti_tracking = _ranked_selection(dti_candidates, 'dti', limit=2, preferred_pair=True)
    selection_tracking['dti'] = dti_tracking

    # NM (select all, but deterministically rank/order)
    nm_candidates = _as_path_list(session_data.get('nm_filenames'))
    nm_selected_raw, nm_tracking = _ranked_selection(nm_candidates, 'nm', limit=None)
    selection_tracking['nm'] = nm_tracking

    # Perf (select one)
    perf_candidates = _discover_perf_candidates(session_data)
    perf_selected, perf_tracking = _ranked_selection(perf_candidates, 'perf', limit=1)
    selection_tracking['perf'] = perf_tracking
    perf_raw = perf_selected[0] if perf_selected else None

    # PET (select one)
    pet_candidates = _as_path_list(session_data.get('pet3d_filenames'))
    pet_single = session_data.get('pet3d_filename', None)
    if isinstance(pet_single, (str, os.PathLike)) and str(pet_single):
        pet_candidates = [os.fspath(pet_single)] + pet_candidates
    pet_selected, pet_tracking = _ranked_selection(pet_candidates, 'pet3d', limit=1)
    selection_tracking['pet3d'] = pet_tracking
    pet_raw = pet_selected[0] if pet_selected else None

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
        'selection_tracking': selection_tracking,
    }
