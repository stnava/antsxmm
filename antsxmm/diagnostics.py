import json
import os
from pathlib import Path
from typing import Any, Iterable

from .selectors import as_path_list, is_nifti, selector_for_modality, sidecar_paths_for_nifti


_MODALITY_FIELDS = {
    't1': ('t1_filenames',),
    'flair': ('flair_filenames',),
    't2w': ('t2w_filenames',),
    'rsf': ('rsf_filenames',),
    'dti': ('dti_filenames',),
    'nm': ('nm_filenames',),
    'perf': ('perf_filenames', 'perf_filename'),
    'pet3d': ('pet3d_filenames', 'pet3d_filename'),
}


__all__ = [
    'inspect_input_path',
    'diagnose_session_inputs',
    'summarize_input_diagnostics',
    'diagnose_bids_tree',
    'write_study_diagnostics_json',
    'format_study_diagnostics_summary',
]


# ---------- session-level diagnostics ----------

def _unique_paths(paths: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for p in paths:
        if not p:
            continue
        s = os.fspath(p)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _collect_declared_inputs(session_data) -> dict[str, list[str]]:
    declared: dict[str, list[str]] = {}
    for modality, fields in _MODALITY_FIELDS.items():
        paths: list[str] = []
        for field in fields:
            value = session_data.get(field)
            if field.endswith('_filename'):
                if isinstance(value, (str, os.PathLike)) and str(value):
                    paths.append(os.fspath(value))
            else:
                paths.extend(as_path_list(value))

        if modality == 'perf' and not paths:
            session_path = session_data.get('session_path')
            if isinstance(session_path, (str, os.PathLike)) and str(session_path):
                ses = Path(session_path)
                for perf_dir in (ses / 'perf', ses / 'asl'):
                    if perf_dir.exists() and perf_dir.is_dir():
                        paths.extend(str(p) for p in sorted(perf_dir.glob('*.nii')))
                        paths.extend(str(p) for p in sorted(perf_dir.glob('*.nii.gz')))

        declared[modality] = _unique_paths(paths)
    return declared


def inspect_input_path(path: str | Path) -> dict[str, Any]:
    s = os.fspath(path)
    lexists = os.path.lexists(s)
    exists = os.path.exists(s)
    is_link = os.path.islink(s)
    real = os.path.realpath(s)
    entry: dict[str, Any] = {
        'path': s,
        'path_exists': exists,
        'path_lexists': lexists,
        'exists': exists,
        'is_symlink': is_link,
        'symlink_target': os.readlink(s) if is_link else None,
        'symlink_target_exists': exists if is_link else None,
        'realpath': real,
        'is_nifti': is_nifti(s),
        'looks_like_nifti': is_nifti(s),
        'is_file': os.path.isfile(s),
        'size_bytes': os.path.getsize(s) if exists and os.path.isfile(s) else None,
        'sidecars_present': [],
        'reasons': [],
        'usable': False,
    }

    if not entry['is_nifti']:
        entry['reasons'].append('not_nifti')
        return entry

    if not lexists:
        entry['reasons'].append('declared_path_missing')
        return entry

    if is_link and not exists:
        entry['reasons'].extend(['broken_symlink', 'target_missing'])
        return entry

    if not exists:
        entry['reasons'].append('target_missing')
        return entry

    if not os.path.isfile(s):
        entry['reasons'].append('not_regular_file')
        return entry

    entry['sidecars_present'] = sidecar_paths_for_nifti(s)
    entry['usable'] = True
    entry['reasons'].append('usable_candidate')
    return entry


def _selected_paths_for_modality(plan: dict, modality: str) -> list[str]:
    used = plan.get('used', {}) if isinstance(plan, dict) else {}
    if modality == 't1':
        return [used.get('t1_filename')] if used.get('t1_filename') else []
    if modality in ('flair', 't2w'):
        return [used.get('flair_or_t2_as_flair_filename')] if used.get('flair_or_t2_as_flair_filename') else []
    if modality == 'rsf':
        return list(used.get('rsf_filenames', []))
    if modality == 'dti':
        return list(used.get('dti_filenames', []))
    if modality == 'nm':
        return list(used.get('nm_filenames', []))
    if modality == 'perf':
        return [used.get('perf_filename')] if used.get('perf_filename') else []
    if modality == 'pet3d':
        return [used.get('pet3d_filename')] if used.get('pet3d_filename') else []
    return []


def diagnose_session_inputs(session_data, plan: dict | None = None) -> dict[str, Any]:
    declared = _collect_declared_inputs(session_data)
    selection_tracking = plan.get('selection_tracking', {}) if isinstance(plan, dict) else {}
    modalities: dict[str, dict[str, Any]] = {}
    overall_failures: list[str] = []

    for modality, paths in declared.items():
        inspected = [inspect_input_path(p) for p in paths]
        usable_paths = [item['realpath'] for item in inspected if item['usable']]
        selected = _selected_paths_for_modality(plan or {}, modality)
        selected_set = {os.path.realpath(p) for p in selected if p}
        rejected = [item for item in inspected if not item['usable']]
        usable_not_selected = [p for p in usable_paths if p not in selected_set]

        if modality in ('flair', 't2w'):
            selector_name = 'FlairOrT2Selector'
        else:
            try:
                selector_name = selector_for_modality(modality).selector_name
            except Exception:
                selector_name = None

        modality_diag = {
            'selector': selector_name,
            'declared_paths': paths,
            'declared_count': len(paths),
            'inspected': inspected,
            'usable_candidates': usable_paths,
            'usable_count': len(usable_paths),
            'selected': selected,
            'selected_count': len(selected),
            'usable_but_not_selected': usable_not_selected,
            'selection_tracking': selection_tracking.get('flair_or_t2' if modality in ('flair', 't2w') else modality),
            'failure_reasons': [],
        }

        if paths and not usable_paths:
            reason_codes = sorted({reason for item in rejected for reason in item['reasons'] if reason != 'not_nifti'})
            modality_diag['failure_reasons'] = reason_codes or ['declared_but_not_usable']
        elif not paths:
            modality_diag['failure_reasons'] = ['no_declared_candidates']
        elif usable_paths and not selected:
            modality_diag['failure_reasons'] = ['usable_candidates_filtered_out']

        modalities[modality] = modality_diag

    if not modalities['t1']['selected']:
        if modalities['t1']['declared_count'] == 0:
            overall_failures.append('no_declared_T1w_candidates')
        elif modalities['t1']['usable_count'] == 0:
            overall_failures.append('declared_T1w_candidates_not_usable')
        else:
            overall_failures.append('usable_T1w_candidates_not_selected')

    return {
        'schema_version': 1,
        'subjectID': session_data.get('subjectID'),
        'sessionID': session_data.get('date') or session_data.get('sessionID'),
        'session_path': session_data.get('session_path'),
        'modalities': modalities,
        'overall_failures': overall_failures,
    }


def summarize_input_diagnostics(diag: dict[str, Any]) -> str:
    lines: list[str] = []
    for modality, info in diag.get('modalities', {}).items():
        failures = info.get('failure_reasons') or []
        if failures and failures != ['no_declared_candidates']:
            lines.append(
                f"{modality}: declared={info.get('declared_count', 0)} usable={info.get('usable_count', 0)} selected={info.get('selected_count', 0)} reasons={','.join(failures)}"
            )
    if not lines:
        overall = diag.get('overall_failures') or []
        return ', '.join(overall) if overall else 'no specific input diagnostics available'
    return ' | '.join(lines)


# ---------- study-level diagnostics ----------

def _is_nifti_name(name: str) -> bool:
    lower = name.lower()
    return lower.endswith('.nii') or lower.endswith('.nii.gz')


def _base_without_nifti_suffix(path: Path) -> str:
    name = path.name
    if name.endswith('.nii.gz'):
        return str(path.with_name(name[:-7]))
    if name.endswith('.nii'):
        return str(path.with_name(name[:-4]))
    return str(path)


def diagnose_bids_tree(bids_root: str | Path) -> dict[str, Any]:
    root = Path(bids_root)
    subjects = sorted([p.name for p in root.glob('sub*') if p.is_dir()]) if root.exists() else []
    sessions = sorted([str(p.relative_to(root)) for p in root.glob('sub*/ses*') if p.is_dir()]) if root.exists() else []

    broken_symlinks: list[dict[str, Any]] = []
    images_without_json: list[str] = []
    json_without_image: list[str] = []
    modality_counts = {
        'T1w': 0,
        'FLAIR': 0,
        'T2w': 0,
        'DTI': 0,
        'rsfMRI': 0,
        'NM': 0,
        'perf': 0,
        'pet3d': 0,
    }
    existing_images: set[str] = set()
    existing_json_bases: set[str] = set()
    image_paths: list[Path] = []
    json_paths: list[Path] = []
    all_symlinks = 0

    if root.exists():
        for p in root.rglob('*'):
            if p.is_symlink():
                all_symlinks += 1
                if not p.exists():
                    broken_symlinks.append(inspect_input_path(p))
            if p.is_file():
                name = p.name
                lower = name.lower()
                if _is_nifti_name(name):
                    image_paths.append(p)
                    existing_images.add(_base_without_nifti_suffix(p))
                    if 't1w' in lower:
                        modality_counts['T1w'] += 1
                    if 'flair' in lower:
                        modality_counts['FLAIR'] += 1
                    if 't2w' in lower:
                        modality_counts['T2w'] += 1
                    if 'dwi' in lower:
                        modality_counts['DTI'] += 1
                    if 'bold' in lower:
                        modality_counts['rsfMRI'] += 1
                    if 'nm' in lower:
                        modality_counts['NM'] += 1
                    if any(tok in lower for tok in ('asl', 'cbf', 'm0', 'perfusion')):
                        modality_counts['perf'] += 1
                    if any(tok in lower for tok in ('pet', 'suv', 'suvr', 'dyn', 'static')):
                        modality_counts['pet3d'] += 1
                elif lower.endswith('.json'):
                    json_paths.append(p)
                    existing_json_bases.add(str(p.with_suffix('')))

    for img in image_paths:
        if _base_without_nifti_suffix(img) not in existing_json_bases:
            images_without_json.append(str(img))
    for js in json_paths:
        if str(js.with_suffix('')) not in existing_images:
            json_without_image.append(str(js))

    suspicious_reasons: list[str] = []
    if broken_symlinks:
        suspicious_reasons.append('broken_symlink')
    if json_without_image:
        suspicious_reasons.append('json_without_image')
    if not image_paths and json_paths:
        suspicious_reasons.append('json_only_tree')
    if not subjects:
        suspicious_reasons.append('no_subject_directories')

    return {
        'bids_root': str(root),
        'root_exists': root.exists(),
        'subjects': subjects,
        'sessions': sessions,
        'counts': {
            'subjects': len(subjects),
            'sessions': len(sessions),
            'image_files': len(image_paths),
            'json_files': len(json_paths),
            'symlinks': all_symlinks,
            'broken_symlinks': len(broken_symlinks),
            'images_without_json': len(images_without_json),
            'json_without_image': len(json_without_image),
        },
        'modality_image_counts': modality_counts,
        'broken_symlinks': broken_symlinks[:25],
        'images_without_json': images_without_json[:25],
        'json_without_image': json_without_image[:25],
        'suspicious_reasons': suspicious_reasons,
    }


def write_study_diagnostics_json(output_dir: str | Path, project: str, diagnostics: dict[str, Any]) -> str:
    out_dir = Path(output_dir) / project
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / '.antsxmm_study_input_diagnostics.json'
    out_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding='utf-8')
    return str(out_path)


def format_study_diagnostics_summary(
    diagnostics: dict[str, Any], *, participant_label: str | None = None, session_label: str | None = None
) -> list[str]:
    counts = diagnostics.get('counts', {}) or {}
    reasons = diagnostics.get('suspicious_reasons', []) or []
    lines = [
        f"No usable BIDS sessions were discovered under {diagnostics.get('bids_root')}",
        f"subjects={counts.get('subjects', 0)} sessions={counts.get('sessions', 0)} image_files={counts.get('image_files', 0)} json_files={counts.get('json_files', 0)} broken_symlinks={counts.get('broken_symlinks', 0)}",
    ]
    if participant_label:
        lines.append(f"requested participant={participant_label}")
    if session_label:
        lines.append(f"requested session={session_label}")
    if reasons:
        lines.append("possible causes: " + ', '.join(reasons))
    broken = diagnostics.get('broken_symlinks', []) or []
    if broken:
        first = broken[0]
        lines.append(f"example broken symlink: {first.get('path')} -> {first.get('symlink_target')}")
    jwo = diagnostics.get('json_without_image', []) or []
    if jwo:
        lines.append(f"example json without image: {jwo[0]}")
    iwo = diagnostics.get('images_without_json', []) or []
    if iwo:
        lines.append(f"example image without json: {iwo[0]}")
    return lines
