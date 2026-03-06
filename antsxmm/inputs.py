
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




_PHASE_FORWARD_LABELS = ("LR", "AP", "SI")
_PHASE_REVERSE_LABELS = ("RL", "PA", "IS")


def _safe_read_json(path: str) -> dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _json_sidecar_for_nifti(nifti_path: str) -> str | None:
    p = Path(nifti_path)
    if p.name.endswith('.nii.gz'):
        base = p.with_name(p.name[:-7])
    elif p.name.endswith('.nii'):
        base = p.with_name(p.name[:-4])
    else:
        return None
    sidecar = str(base) + '.json'
    return sidecar if os.path.exists(sidecar) else None


def _phase_bucket_from_json_metadata(meta: dict) -> str | None:
    ped = str(meta.get('PhaseEncodingDirection', '')).strip().lower()
    mapping = {
        'i': 'LR',
        'i-': 'RL',
        'j': 'AP',
        'j-': 'PA',
        'k': 'SI',
        'k-': 'IS',
    }
    if ped in mapping:
        return mapping[ped]

    direction = str(meta.get('PhaseEncodingAxis', '')).strip().upper()
    polarity = str(meta.get('PhaseEncodingPolarityGE', '')).strip().lower()
    if direction in ('ROW', 'COL') and polarity in ('flipped', 'unflipped'):
        if direction == 'COL':
            return 'PA' if polarity == 'flipped' else 'AP'
        return 'RL' if polarity == 'flipped' else 'LR'
    return None


def _extract_terminal_suffix(path: str) -> str:
    name = Path(path).name
    if name.endswith('.nii.gz'):
        stem = name[:-7]
    else:
        stem = Path(name).stem
    return stem.split('_')[-1].lower() if stem else ''


def _extract_phase_direction_label(path: str) -> str | None:
    name = Path(path).name
    for pat in (
        r'(?:^|[_-])dir-(LR|RL|AP|PA|SI|IS)(?:[_-]|$)',
        r'(?:^|[_-])(LR|RL|AP|PA|SI|IS)(?:[_-]|(?:dwi|bold)(?:a)?(?:[_-]|$))',
    ):
        m = re.search(pat, name, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    sidecar = _json_sidecar_for_nifti(path)
    if sidecar:
        meta = _safe_read_json(sidecar)
        bucket = _phase_bucket_from_json_metadata(meta)
        if bucket:
            return bucket
    return None


def _extract_task_label(path: str) -> str | None:
    name = Path(path).name
    m = re.search(r'(?:^|[_-])task-([A-Za-z0-9]+)(?:[_-]|$)', name, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    sidecar = _json_sidecar_for_nifti(path)
    if sidecar:
        meta = _safe_read_json(sidecar)
        task_name = str(meta.get('TaskName', '')).strip().lower()
        return task_name or None
    return None


def _run_number(path: str) -> int:
    run_id = _extract_run_id_from_filename(path)
    m = re.search(r'run-(\d+)', run_id)
    return int(m.group(1)) if m else 1


def _selection_score(
    path: str,
    *,
    exact_suffixes: tuple[str, ...],
    preferred_task: str | None = None,
) -> tuple:
    direction = _extract_phase_direction_label(path) or 'ZZ'
    exact_suffix_rank = 0 if _extract_terminal_suffix(path) in exact_suffixes else 1
    known_direction_rank = 0 if direction != 'ZZ' else 1
    preferred_task_rank = 1
    if preferred_task is not None:
        task_label = _extract_task_label(path)
        preferred_task_rank = 0 if task_label == preferred_task else 1
    reverse_rank = 0 if direction in _PHASE_REVERSE_LABELS else 1
    run_rank = _run_number(path)
    basename = Path(path).name.lower()
    return (preferred_task_rank, known_direction_rank, exact_suffix_rank, run_rank, reverse_rank, basename)


def _select_phase_encoded_filenames(
    paths,
    *,
    exact_suffixes: tuple[str, ...],
    preferred_task: str | None = None,
) -> list[str]:
    candidates = [p for p in _as_path_list(paths) if p]
    if not candidates:
        return []

    ranked = sorted(
        candidates,
        key=lambda p: _selection_score(p, exact_suffixes=exact_suffixes, preferred_task=preferred_task),
    )

    by_direction: dict[str, list[str]] = {}
    for path in ranked:
        direction = _extract_phase_direction_label(path)
        if direction is not None:
            by_direction.setdefault(direction, []).append(path)

    def _best(direction: str) -> str | None:
        items = by_direction.get(direction, [])
        return items[0] if items else None

    preferred_pairs = [
        ('LR', 'RL'),
        ('AP', 'PA'),
        ('SI', 'IS'),
        ('LR', 'PA'),
        ('AP', 'RL'),
    ]

    selected: list[str] = []
    used: set[str] = set()

    for a, b in preferred_pairs:
        pa = _best(a)
        pb = _best(b)
        if pa and pb:
            selected.extend([pa, pb])
            used.update([pa, pb])
            break

    if not selected:
        for path in ranked:
            direction = _extract_phase_direction_label(path)
            if direction is None or path in used:
                continue
            selected.append(path)
            used.add(path)
            first_direction = direction
            complementary = {
                'LR': ('RL', 'PA'),
                'RL': ('LR', 'AP'),
                'AP': ('PA', 'RL'),
                'PA': ('AP', 'LR'),
                'SI': ('IS',),
                'IS': ('SI',),
            }.get(first_direction, ())
            for comp_direction in complementary:
                pb = _best(comp_direction)
                if pb and pb not in used:
                    selected.append(pb)
                    used.add(pb)
                    break
            break

    for path in ranked:
        if len(selected) >= 2:
            break
        if path not in used:
            selected.append(path)
            used.add(path)

    return selected[:2]


def _select_dti_filenames(paths) -> list[str]:
    return _select_phase_encoded_filenames(paths, exact_suffixes=('dwi',))


def _select_rsf_filenames(paths) -> list[str]:
    return _select_phase_encoded_filenames(paths, exact_suffixes=('bold',), preferred_task='rest')


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

    # rsfMRI (select the best complementary pair, capped at 2)
    rsf_raw = _as_path_list(session_data.get('rsf_filenames'))
    rsf_selected_raw = _select_rsf_filenames(rsf_raw)

    # DTI (select the best complementary pair, capped at 2)
    dti_raw = _as_path_list(session_data.get('dti_filenames'))
    dti_selected_raw = _select_dti_filenames(dti_raw)

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
