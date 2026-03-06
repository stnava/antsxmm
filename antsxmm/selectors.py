import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def extract_run_id_from_filename(path: str) -> str:
    """Extract BIDS run identifier from filename.
    Default to run-01 if missing.
    """
    name = Path(path).name

    m = re.search(r"run-(\d+)", name)
    if m:
        return f"run-{int(m.group(1)):02d}"

    m = re.search(r"(?:^|_)(?:r)(\d+)(?:[_.]|$)", name)
    if m:
        return f"run-{int(m.group(1)):02d}"

    return "run-01"


def is_nifti(path: str) -> bool:
    if not path:
        return False
    p = str(path).lower()
    return p.endswith('.nii') or p.endswith('.nii.gz')


def as_path_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float):
        try:
            if math.isnan(value):
                return []
        except Exception:
            return []
    if isinstance(value, (str, os.PathLike)):
        s = str(value)
        return [s] if s else []
    if isinstance(value, (int, float, bool)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v is not None and not (isinstance(v, float) and math.isnan(v))]
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


def sidecar_paths_for_nifti(nifti_path: str) -> list[str]:
    p = Path(nifti_path)
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


def load_json_sidecar(nifti_path: str) -> dict:
    for p in sidecar_paths_for_nifti(nifti_path):
        if p.endswith('.json'):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}
    return {}


def filename_without_nifti_ext(path: str) -> str:
    name = Path(path).name
    if name.endswith('.nii.gz'):
        return name[:-7]
    if name.endswith('.nii'):
        return name[:-4]
    return name


def bids_tokens(path: str) -> list[str]:
    return filename_without_nifti_ext(path).split('_')


def get_bids_entity(path: str, key: str) -> str | None:
    prefix = f"{key}-"
    for tok in bids_tokens(path):
        if tok.startswith(prefix) and len(tok) > len(prefix):
            return tok[len(prefix):]
    return None


def suffix_token(path: str) -> str:
    toks = bids_tokens(path)
    return toks[-1] if toks else ''


def read_phase_direction(path: str) -> str | None:
    direction = get_bids_entity(path, 'dir')
    if direction:
        return direction.upper()

    meta = load_json_sidecar(path)
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


def exact_suffix_kind(path: str, expected_suffix: str) -> bool:
    return suffix_token(path).lower() == expected_suffix.lower()


def variant_suffix_kind(path: str, expected_suffix: str) -> bool:
    suffix = suffix_token(path).lower()
    expected = expected_suffix.lower()
    return suffix.startswith(expected) and suffix != expected


def run_number(path: str) -> int:
    run_id = extract_run_id_from_filename(path)
    m = re.search(r"run-(\d+)", run_id)
    return int(m.group(1)) if m else 1


@dataclass(frozen=True)
class RankedCandidate:
    path: str
    rank: tuple[int, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SelectionResult:
    modality: str
    selector: str
    limit: int | None
    preferred_pair: bool
    ranked: tuple[RankedCandidate, ...]
    selected: tuple[str, ...]
    excluded: tuple[str, ...]

    def as_tracking(self) -> dict:
        selected_set = set(self.selected)
        return {
            'strategy': self.modality,
            'selector': self.selector,
            'limit': self.limit,
            'preferred_pair': self.preferred_pair,
            'ranked_candidates': [
                {
                    'path': c.path,
                    'selected': c.path in selected_set,
                    'rank': idx + 1,
                    'reasons': list(c.reasons),
                }
                for idx, c in enumerate(self.ranked)
            ],
            'selected': list(self.selected),
            'excluded': list(self.excluded),
        }


class ModalitySelector:
    modality = 'generic'
    selector_name = 'GenericSelector'
    limit: int | None = 1
    preferred_pair = False

    def normalize_existing(self, paths: Iterable[str]) -> list[str]:
        normalized = [os.path.realpath(p) for p in paths if is_nifti(p)]
        return sorted(set(normalized))

    def rank_key(self, path: str) -> tuple[int, ...]:
        return (run_number(path), len(Path(path).name), Path(path).name.lower())

    def reason_lines(self, path: str) -> list[str]:
        direction = read_phase_direction(path)
        reasons = [f'run:{extract_run_id_from_filename(path)}']
        if direction:
            reasons.insert(0, f'phase_direction:{direction}')
        return reasons

    def rank_candidates(self, paths: Iterable[str]) -> list[RankedCandidate]:
        ranked_paths = sorted(self.normalize_existing(paths), key=self.rank_key)
        return [
            RankedCandidate(path=p, rank=self.rank_key(p), reasons=tuple(self.reason_lines(p)))
            for p in ranked_paths
        ]

    def select_from_ranked(self, ranked: list[RankedCandidate]) -> list[str]:
        if self.limit is None:
            return [c.path for c in ranked]
        return [c.path for c in ranked[: self.limit]]

    def select(self, paths: Iterable[str]) -> SelectionResult:
        ranked = self.rank_candidates(paths)
        selected = self.select_from_ranked(ranked)
        selected_set = set(selected)
        excluded = [c.path for c in ranked if c.path not in selected_set]
        return SelectionResult(
            modality=self.modality,
            selector=self.selector_name,
            limit=self.limit,
            preferred_pair=self.preferred_pair,
            ranked=tuple(ranked),
            selected=tuple(selected),
            excluded=tuple(excluded),
        )


class T1Selector(ModalitySelector):
    modality = 't1'
    selector_name = 'T1Selector'
    limit = 1

    def rank_key(self, path: str) -> tuple[int, ...]:
        suffix = suffix_token(path).lower()
        task = (get_bids_entity(path, 'task') or '').lower()
        return (
            0 if exact_suffix_kind(path, 'T1w') else 1,
            0 if task == '' else 1,
            0 if 't1w' in suffix else 1,
            run_number(path),
            len(Path(path).name),
            Path(path).name.lower(),
        )

    def reason_lines(self, path: str) -> list[str]:
        reasons = []
        if exact_suffix_kind(path, 'T1w'):
            reasons.append('exact_suffix:T1w')
        reasons.append(f'run:{extract_run_id_from_filename(path)}')
        return reasons


class FlairOrT2Selector(ModalitySelector):
    modality = 'flair_or_t2'
    selector_name = 'FlairOrT2Selector'
    limit = 1

    def rank_key(self, path: str) -> tuple[int, ...]:
        suffix = suffix_token(path).lower()
        return (
            0 if exact_suffix_kind(path, 'FLAIR') else 1,
            0 if exact_suffix_kind(path, 'T2w') else 1,
            0 if suffix in ('flair', 't2w') else 1,
            run_number(path),
            len(Path(path).name),
            Path(path).name.lower(),
        )

    def reason_lines(self, path: str) -> list[str]:
        reasons = []
        if exact_suffix_kind(path, 'FLAIR'):
            reasons.append('exact_suffix:FLAIR')
        elif exact_suffix_kind(path, 'T2w'):
            reasons.append('fallback_suffix:T2w')
        reasons.append(f'run:{extract_run_id_from_filename(path)}')
        return reasons


class PairAwareSelector(ModalitySelector):
    preferred_pair = True
    limit = 2
    pair_order = (('LR', 'RL'), ('AP', 'PA'), ('SI', 'IS'))

    def select_from_ranked(self, ranked: list[RankedCandidate]) -> list[str]:
        for a, b in self.pair_order:
            first = next((c.path for c in ranked if read_phase_direction(c.path) == a), None)
            second = next((c.path for c in ranked if read_phase_direction(c.path) == b and c.path != first), None)
            if first and second:
                return [first, second]
        return super().select_from_ranked(ranked)


class RestingStateSelector(PairAwareSelector):
    modality = 'rsf'
    selector_name = 'RestingStateSelector'

    def rank_key(self, path: str) -> tuple[int, ...]:
        task = (get_bids_entity(path, 'task') or '').lower()
        direction = read_phase_direction(path)
        return (
            0 if task == 'rest' else 1,
            0 if exact_suffix_kind(path, 'bold') else 1,
            0 if direction in {'LR', 'RL', 'AP', 'PA', 'SI', 'IS'} else 1,
            run_number(path),
            len(Path(path).name),
            Path(path).name.lower(),
        )

    def reason_lines(self, path: str) -> list[str]:
        reasons = []
        task = (get_bids_entity(path, 'task') or '').lower()
        if task == 'rest':
            reasons.append('task:rest')
        if exact_suffix_kind(path, 'bold'):
            reasons.append('exact_suffix:bold')
        direction = read_phase_direction(path)
        if direction:
            reasons.append(f'phase_direction:{direction}')
        reasons.append(f'run:{extract_run_id_from_filename(path)}')
        return reasons


class DwiSelector(PairAwareSelector):
    modality = 'dti'
    selector_name = 'DwiSelector'

    def rank_key(self, path: str) -> tuple[int, ...]:
        direction = read_phase_direction(path)
        return (
            0 if exact_suffix_kind(path, 'dwi') else 1,
            0 if direction in {'LR', 'RL', 'AP', 'PA', 'SI', 'IS'} else 1,
            run_number(path),
            len(Path(path).name),
            Path(path).name.lower(),
        )

    def reason_lines(self, path: str) -> list[str]:
        reasons = []
        if exact_suffix_kind(path, 'dwi'):
            reasons.append('exact_suffix:dwi')
        direction = read_phase_direction(path)
        if direction:
            reasons.append(f'phase_direction:{direction}')
        reasons.append(f'run:{extract_run_id_from_filename(path)}')
        return reasons


class PerfusionSelector(ModalitySelector):
    modality = 'perf'
    selector_name = 'PerfusionSelector'
    limit = 1

    def rank_key(self, path: str) -> tuple[int, ...]:
        return (
            0 if exact_suffix_kind(path, 'asl') else 1,
            0 if exact_suffix_kind(path, 'm0scan') else 1,
            0 if 'm0' not in Path(path).name.lower() else 1,
            run_number(path),
            len(Path(path).name),
            Path(path).name.lower(),
        )

    def reason_lines(self, path: str) -> list[str]:
        reasons = []
        if exact_suffix_kind(path, 'asl'):
            reasons.append('exact_suffix:asl')
        if exact_suffix_kind(path, 'm0scan'):
            reasons.append('supporting_scan:m0scan')
        reasons.append(f'run:{extract_run_id_from_filename(path)}')
        return reasons


class Pet3DSelector(ModalitySelector):
    modality = 'pet3d'
    selector_name = 'Pet3DSelector'
    limit = 1

    def rank_key(self, path: str) -> tuple[int, ...]:
        tracer = str(load_json_sidecar(path).get('TracerRadionuclide', '') or '').strip()
        return (
            0 if exact_suffix_kind(path, 'pet') else 1,
            0 if tracer else 1,
            run_number(path),
            len(Path(path).name),
            Path(path).name.lower(),
        )

    def reason_lines(self, path: str) -> list[str]:
        reasons = []
        if exact_suffix_kind(path, 'pet'):
            reasons.append('exact_suffix:pet')
        tracer = str(load_json_sidecar(path).get('TracerRadionuclide', '') or '').strip()
        if tracer:
            reasons.append(f'tracer:{tracer}')
        reasons.append(f'run:{extract_run_id_from_filename(path)}')
        return reasons


class NeuromelaninSelector(ModalitySelector):
    modality = 'nm'
    selector_name = 'NeuromelaninSelector'
    limit = None

    def rank_key(self, path: str) -> tuple[int, ...]:
        return (
            run_number(path),
            0 if exact_suffix_kind(path, 'NM') else 1,
            len(Path(path).name),
            Path(path).name.lower(),
        )

    def reason_lines(self, path: str) -> list[str]:
        reasons = []
        if suffix_token(path).lower() == 'nm':
            reasons.append('exact_suffix:NM')
        reasons.append(f'run:{extract_run_id_from_filename(path)}')
        return reasons


def selector_for_modality(modality: str) -> ModalitySelector:
    registry = {
        't1': T1Selector,
        'flair_or_t2': FlairOrT2Selector,
        'rsf': RestingStateSelector,
        'dti': DwiSelector,
        'perf': PerfusionSelector,
        'pet3d': Pet3DSelector,
        'nm': NeuromelaninSelector,
    }
    selector_cls = registry.get(modality, ModalitySelector)
    return selector_cls()
