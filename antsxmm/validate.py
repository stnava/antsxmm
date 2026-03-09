from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from .tree import predict_tree


@dataclass(frozen=True)
class ValidationResult:
    missing: List[str]
    unexpected: List[str]
    ok: List[str]
    missing_mmwide_files: List[str]
    ok_mmwide_files: List[str]


def _normalize_participant_labels(participant_labels: str | Iterable[str] | None) -> set[str] | None:
    if participant_labels is None:
        return None
    raw_values = [participant_labels] if isinstance(participant_labels, str) else list(participant_labels)
    normalized: set[str] = set()
    for value in raw_values:
        for token in str(value).split(","):
            token = token.strip().rstrip("/\\")
            if token:
                normalized.add(token)
    return normalized or None


def _expected_mmwide_file(project: str, subject: str, session: str, modality: str, run: str) -> Path:
    return Path(project) / subject / session / modality / run / f"{project}+{subject}+{session}+{modality}+{run}+mmwide.csv"


def validate_project(
    bids_project_dir: str | Path,
    *,
    output_dir: str | Path = "pymm",
    pymm_dir: str | Path | None = None,
    participant_labels: str | Iterable[str] | None = None,
) -> Dict[str, ValidationResult]:
    """Validate a processed antsxmm output tree against one BIDS project."""
    bids_project_dir = Path(bids_project_dir)
    project = bids_project_dir.name
    resolved_output_dir = Path(pymm_dir) if pymm_dir is not None else Path(output_dir)
    requested_participants = _normalize_participant_labels(participant_labels)

    results: Dict[str, ValidationResult] = {}
    for subject_dir in sorted(bids_project_dir.glob("sub-*")):
        _, subject, tree = predict_tree(subject_dir)
        if requested_participants is not None and subject not in requested_participants:
            continue

        for ses_name, runs in tree.items():
            key = f"{subject}/{ses_name}"
            expected_dirs: set[Path] = set()
            expected_mmwide_files: set[Path] = set()
            for modality, run in runs:
                expected_dirs.add(resolved_output_dir / project / subject / ses_name / modality / run)
                expected_mmwide_files.add(resolved_output_dir / _expected_mmwide_file(project, subject, ses_name, modality, run))

            existing_dirs: set[Path] = set()
            existing_mmwide_files: set[Path] = set()
            root = resolved_output_dir / project / subject / ses_name
            if root.exists():
                for modality_dir in root.iterdir():
                    if not modality_dir.is_dir():
                        continue
                    for run_dir in modality_dir.iterdir():
                        if not run_dir.is_dir():
                            continue
                        existing_dirs.add(run_dir)
                        for mmwide_file in run_dir.glob("*+mmwide.csv"):
                            if mmwide_file.is_file():
                                existing_mmwide_files.add(mmwide_file)

            missing = sorted(str(p.relative_to(resolved_output_dir)) for p in expected_dirs - existing_dirs)
            unexpected = sorted(str(p.relative_to(resolved_output_dir)) for p in existing_dirs - expected_dirs)
            ok = sorted(str(p.relative_to(resolved_output_dir)) for p in expected_dirs & existing_dirs)
            missing_mmwide_files = sorted(str(p.relative_to(resolved_output_dir)) for p in expected_mmwide_files - existing_mmwide_files)
            ok_mmwide_files = sorted(str(p.relative_to(resolved_output_dir)) for p in expected_mmwide_files & existing_mmwide_files)

            results[key] = ValidationResult(
                missing=missing,
                unexpected=unexpected,
                ok=ok,
                missing_mmwide_files=missing_mmwide_files,
                ok_mmwide_files=ok_mmwide_files,
            )

    return results
