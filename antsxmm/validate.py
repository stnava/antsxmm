from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set

from .tree import predict_tree


@dataclass(frozen=True)
class ValidationResult:
    missing: List[str]
    unexpected: List[str]
    ok: List[str]


def _expected_paths(project: str, subject: str, tree: Dict[str, List[Tuple[str, str]]]) -> Set[Path]:
    expected: Set[Path] = set()
    for ses, runs in tree.items():
        for modality, run in runs:
            expected.add(Path("pymm") / project / subject / ses / modality / run)
    return expected


def validate_project(bids_project_dir: str | Path, *, pymm_dir: str | Path = "pymm") -> Dict[str, ValidationResult]:
    """Validate BIDS project against existing antsxmm outputs under pymm_dir.

    Parameters
    ----------
    bids_project_dir:
        Path like: <bids>/<project>
    pymm_dir:
        Output root containing <project>/<subject>/<session>/...

    Returns
    -------
    Mapping: "<subject>/<session>" -> ValidationResult
    """
    bids_project_dir = Path(bids_project_dir)
    project = bids_project_dir.name
    pymm_dir = Path(pymm_dir)

    results: Dict[str, ValidationResult] = {}

    for subject_dir in sorted(bids_project_dir.glob("sub-*")):
        _, subject, tree = predict_tree(subject_dir)

        for ses_name, runs in tree.items():
            key = f"{subject}/{ses_name}"

            expected = set()
            for modality, run in runs:
                expected.add(pymm_dir / project / subject / ses_name / modality / run)

            existing = set()
            root = pymm_dir / project / subject / ses_name
            if root.exists():
                # consider any direct run-like or legacy leaf dirs under each modality
                for modality_dir in root.iterdir():
                    if not modality_dir.is_dir():
                        continue
                    for child in modality_dir.iterdir():
                        if child.is_dir():
                            existing.add(child)

            missing = sorted(str(p.relative_to(pymm_dir)) for p in expected - existing)
            unexpected = sorted(str(p.relative_to(pymm_dir)) for p in existing - expected)
            ok = sorted(str(p.relative_to(pymm_dir)) for p in expected & existing)

            results[key] = ValidationResult(missing=missing, unexpected=unexpected, ok=ok)

    return results
