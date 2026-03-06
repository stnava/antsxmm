import os
from pathlib import Path

from .selectors import (
    as_path_list,
    extract_run_id_from_filename,
    is_nifti,
    selector_for_modality,
    sidecar_paths_for_nifti,
)


def _extract_run_id_from_filename(path: str) -> str:
    return extract_run_id_from_filename(path)


def _is_nifti(path: str) -> bool:
    return is_nifti(path)


def _as_path_list(value) -> list[str]:
    """Normalize a possibly-missing BIDS field into a list of path strings."""
    return as_path_list(value)


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
    return sidecar_paths_for_nifti(nifti_path)


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


def _apply_t1_run_match_override(selection_result, t1_candidates: list[str], t1_run_match: str | None):
    tracking = selection_result.as_tracking()
    selected = list(selection_result.selected)
    if not t1_run_match or len(t1_candidates) <= 1:
        return selected, tracking

    matched = [
        os.path.realpath(p)
        for p in t1_candidates
        if t1_run_match in os.path.basename(p) and _is_nifti(p)
    ]
    if not matched:
        return selected, tracking

    selector = selector_for_modality('t1')
    matched_result = selector.select(matched)
    selected = list(matched_result.selected[:1])
    selected_set = set(selected)
    ranked_all = [c['path'] for c in tracking['ranked_candidates']]
    tracking['ranked_candidates'] = [
        {
            **c,
            'selected': c['path'] in selected_set,
            'reasons': list(c['reasons']) + (['matched_t1_run'] if t1_run_match in os.path.basename(c['path']) else []),
        }
        for c in tracking['ranked_candidates']
    ]
    tracking['selected'] = selected
    tracking['excluded'] = [p for p in ranked_all if p not in selected_set]
    tracking['t1_run_match'] = t1_run_match
    return selected, tracking




def _select_rsf_filenames(paths) -> list[str]:
    """Backward-compatible rsf selector wrapper.

    Preserves the legacy helper name expected by older tests/callers while
    delegating to the explicit RestingStateSelector-based policy layer.
    """
    return list(selector_for_modality('rsf').select(_as_path_list(paths)).selected)


def _select_dti_filenames(paths) -> list[str]:
    """Backward-compatible dti selector wrapper."""
    return list(selector_for_modality('dti').select(_as_path_list(paths)).selected)


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

    t1_candidates = [p for p in _as_path_list(session_data.get('t1_filenames')) if p]
    if not t1_candidates:
        t1_fn = session_data.get('t1_filename')
        t1_candidates = [t1_fn] if t1_fn else []

    t1_result = selector_for_modality('t1').select(t1_candidates)
    t1_selected, t1_tracking = _apply_t1_run_match_override(t1_result, t1_candidates, t1_run_match)
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

    flair_candidates = _as_path_list(session_data.get('flair_filenames')) + _as_path_list(session_data.get('t2w_filenames'))
    flair_result = selector_for_modality('flair_or_t2').select(flair_candidates)
    selection_tracking['flair_or_t2'] = flair_result.as_tracking()
    flair_raw = flair_result.selected[0] if flair_result.selected else None

    rsf_result = selector_for_modality('rsf').select(_as_path_list(session_data.get('rsf_filenames')))
    selection_tracking['rsf'] = rsf_result.as_tracking()
    rsf_selected_raw = list(rsf_result.selected)

    dti_result = selector_for_modality('dti').select(_as_path_list(session_data.get('dti_filenames')))
    selection_tracking['dti'] = dti_result.as_tracking()
    dti_selected_raw = list(dti_result.selected)

    nm_result = selector_for_modality('nm').select(_as_path_list(session_data.get('nm_filenames')))
    selection_tracking['nm'] = nm_result.as_tracking()
    nm_selected_raw = list(nm_result.selected)

    perf_result = selector_for_modality('perf').select(_discover_perf_candidates(session_data))
    selection_tracking['perf'] = perf_result.as_tracking()
    perf_raw = perf_result.selected[0] if perf_result.selected else None

    pet_candidates = _as_path_list(session_data.get('pet3d_filenames'))
    pet_single = session_data.get('pet3d_filename', None)
    if isinstance(pet_single, (str, os.PathLike)) and str(pet_single):
        pet_candidates = [os.fspath(pet_single)] + pet_candidates
    pet_result = selector_for_modality('pet3d').select(pet_candidates)
    selection_tracking['pet3d'] = pet_result.as_tracking()
    pet_raw = pet_result.selected[0] if pet_result.selected else None

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
