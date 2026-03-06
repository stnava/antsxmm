from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .bids_entities import parse_entities
from .run_id import normalize_run_id
from .inputs import _as_path_list, plan_session_inputs

_SUFFIX_TO_MODALITY = {
    'T1w': 'T1w',
    'FLAIR': 'T2Flair',
    'T2w': 'T2Flair',
    'dwi': 'DTI',
    'bold': 'rsfMRI',
    'asl': 'perf',
    'pet': 'pet3d',
    'PET': 'pet3d',
    'NM': 'NM2DMT',
    'nm': 'NM2DMT',
}


@dataclass(frozen=True)
class ExecutionUnit:
    project_id: str
    subject: str
    session: str
    modality: str
    run: str
    input_paths: tuple[str, ...]
    output_prefix: str
    role: str = 'primary'

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def modality_from_path(path: str) -> str | None:
    entities = parse_entities(path)
    return _SUFFIX_TO_MODALITY.get(entities.get('suffix', ''))


def canonical_output_prefix(output_root: str, project_id: str, subject: str, session: str, modality: str, run: str) -> str:
    out_dir = Path(output_root) / project_id / subject / session / modality / run
    prefix = f"{project_id}+{subject}+{session}+{modality}+{run}"
    return str(out_dir / prefix)


def _pick_primary_run(session_data: dict[str, Any]) -> str:
    t1_paths = _as_path_list(session_data.get('t1_filenames'))
    if not t1_paths:
        t1_single = session_data.get('t1_filename')
        if t1_single:
            t1_paths = [str(t1_single)]
    if not t1_paths:
        return 'run-01'
    entities = parse_entities(sorted(t1_paths)[0])
    return normalize_run_id(entities.get('run'))


def _coerce_paths(session_data: dict[str, Any], list_key: str, single_key: str | None = None) -> list[str]:
    paths = list(_as_path_list(session_data.get(list_key)))
    if single_key:
        single = session_data.get(single_key)
        if single:
            s = str(single)
            if s not in paths:
                paths.insert(0, s)
    return sorted([str(p) for p in paths if p])


def build_execution_plan(session_data: dict[str, Any], output_root: str, project_id: str) -> list[ExecutionUnit]:
    subject = str(session_data.get('subjectID'))
    session = str(session_data.get('date') or session_data.get('sessionID'))
    if not subject:
        raise KeyError("session_data missing required key: 'subjectID'")
    if not session:
        raise KeyError("session_data missing required key: 'date' (or alias 'sessionID')")

    canonical_run = _pick_primary_run(session_data)
    plan_inputs = plan_session_inputs(session_data)
    used = plan_inputs.get('used', {})
    t1_paths = tuple([used['t1_filename']]) if used.get('t1_filename') else tuple()
    if not t1_paths:
        raise ValueError('build_execution_plan requires at least one T1w input')

    flair_paths = tuple([used['flair_or_t2_as_flair_filename']]) if used.get('flair_or_t2_as_flair_filename') else tuple()

    modality_inputs = {
        'T1w': t1_paths,
        'T2Flair': flair_paths,
        'perf': tuple([used['perf_filename']]) if used.get('perf_filename') else tuple(),
        'pet3d': tuple([used['pet3d_filename']]) if used.get('pet3d_filename') else tuple(),
        'DTI': tuple(used.get('dti_filenames', [])),
        'rsfMRI': tuple(used.get('rsf_filenames', [])),
        'NM2DMT': tuple(used.get('nm_filenames', [])),
        'T1wHierarchical': t1_paths,
    }

    plan: list[ExecutionUnit] = []
    for modality, input_paths in modality_inputs.items():
        if not input_paths:
            continue
        plan.append(
            ExecutionUnit(
                project_id=project_id,
                subject=subject,
                session=session,
                modality=modality,
                run=canonical_run,
                input_paths=input_paths,
                output_prefix=canonical_output_prefix(output_root, project_id, subject, session, modality, canonical_run),
                role='derived' if modality == 'T1wHierarchical' else 'primary',
            )
        )
    validate_execution_plan(plan)
    return plan


def validate_execution_plan(plan: list[ExecutionUnit]) -> None:
    if not plan:
        raise ValueError('execution plan is empty')

    seen_outputs: set[str] = set()
    seen_keys: set[tuple[str, str, str, str]] = set()
    anchor = (plan[0].project_id, plan[0].subject, plan[0].session, plan[0].run)

    for unit in plan:
        current = (unit.project_id, unit.subject, unit.session, unit.run)
        if current != anchor:
            raise ValueError('execution plan mixes multiple project/subject/session/run anchors')
        if not unit.input_paths:
            raise ValueError(f'execution unit {unit.modality} has no input paths')
        key = (unit.subject, unit.session, unit.modality, unit.run)
        if key in seen_keys:
            raise ValueError(f'duplicate execution unit for {key}')
        seen_keys.add(key)
        if unit.output_prefix in seen_outputs:
            raise ValueError(f'duplicate output prefix: {unit.output_prefix}')
        seen_outputs.add(unit.output_prefix)


def plan_to_row(plan: list[ExecutionUnit], output_root: str) -> dict[str, Any]:
    validate_execution_plan(plan)
    by_modality = {unit.modality: unit for unit in plan}
    anchor = plan[0]
    nm = list(by_modality.get('NM2DMT', ExecutionUnit(anchor.project_id, anchor.subject, anchor.session, 'NM2DMT', anchor.run, tuple(), '')).input_paths)
    rsf = list(by_modality.get('rsfMRI', ExecutionUnit(anchor.project_id, anchor.subject, anchor.session, 'rsfMRI', anchor.run, tuple(), '')).input_paths)
    dti = list(by_modality.get('DTI', ExecutionUnit(anchor.project_id, anchor.subject, anchor.session, 'DTI', anchor.run, tuple(), '')).input_paths)

    row = {
        'projectID': anchor.project_id,
        'subjectID': anchor.subject,
        'date': anchor.session,
        'imageID': anchor.run,
        'modality': 'T1w',
        'sourcedir': str(Path(by_modality['T1w'].input_paths[0]).parent),
        'outputdir': output_root,
        'filename': by_modality['T1w'].input_paths[0],
        'flairid': by_modality['T2Flair'].input_paths[0] if 'T2Flair' in by_modality else None,
        'perfid': by_modality['perf'].input_paths[0] if 'perf' in by_modality else None,
        'pet3did': by_modality['pet3d'].input_paths[0] if 'pet3d' in by_modality else None,
        'rsfid1': rsf[0] if len(rsf) > 0 else None,
        'rsfid2': rsf[1] if len(rsf) > 1 else None,
        'rsfid3': None,
        'dtid1': dti[0] if len(dti) > 0 else None,
        'dtid2': dti[1] if len(dti) > 1 else None,
        'dtid3': None,
        'xmm_run': anchor.run,
    }
    for i in range(1, 12):
        row[f'nmid{i}'] = nm[i - 1] if len(nm) >= i else None
    for modality in ('T1w', 'T2Flair', 'perf', 'pet3d', 'DTI', 'rsfMRI', 'NM2DMT', 'T1wHierarchical'):
        unit = by_modality.get(modality)
        row[f'xmm_prefix_{modality}'] = unit.output_prefix if unit else None
    return row
