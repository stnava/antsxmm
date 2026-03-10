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
    missing_modalities: List[str]


@dataclass(frozen=True)
class ValidationSummary:
    session_count: int
    ok_count: int
    missing_count: int
    unexpected_count: int
    missing_mmwide_count: int
    clean_session_count: int
    affected_session_count: int


@dataclass(frozen=True)
class ValidationTableRow:
    session_key: str
    ok_count: int
    missing_count: int
    unexpected_count: int
    missing_mmwide_count: int
    missing_modalities: List[str]
    status: str


@dataclass(frozen=True)
class SessionModalityRow:
    subject_id: str
    session_id: str
    modality: str
    run_id: str
    status: str
    expected_mmwide_csv: str


@dataclass(frozen=True)
class MissingPercentageRow:
    modality: str
    expected_count: int
    present_dir_count: int
    missing_dir_count: int
    missing_dir_pct: float
    present_mmwide_count: int
    missing_mmwide_count: int
    missing_mmwide_pct: float


def _normalize_participant_labels(participant_labels: Iterable[str] | None) -> set[str] | None:
    if participant_labels is None:
        return None
    normalized = {str(label).strip() for label in participant_labels if str(label).strip()}
    return normalized or None


def _expected_mmwide_path(
    output_dir: Path,
    project: str,
    subject: str,
    session: str,
    modality: str,
    run: str,
) -> Path:
    filename = f"{project}+{subject}+{session}+{modality}+{run}+mmwide.csv"
    return output_dir / project / subject / session / modality / run / filename


def _extract_modality_from_relpath(relpath: str) -> str:
    parts = Path(relpath).parts
    return parts[3] if len(parts) >= 5 else "unknown"


def _extract_run_from_relpath(relpath: str) -> str:
    parts = Path(relpath).parts
    return parts[4] if len(parts) >= 5 else "unknown"


def validate_project(
    bids_project_dir: str | Path,
    output_dir: str | Path = "pymm",
    *,
    participant_labels: Iterable[str] | None = None,
) -> Dict[str, ValidationResult]:
    """Validate a BIDS project against antsxmm outputs under output_dir.

    Parameters
    ----------
    bids_project_dir:
        Path like: <bids>/<project>
    output_dir:
        Processed output root containing <project>/<subject>/<session>/...
    participant_labels:
        Optional subset of subject IDs, e.g. ["sub-01", "sub-02"]

    Returns
    -------
    Mapping: "<subject>/<session>" -> ValidationResult
    """
    bids_project_dir = Path(bids_project_dir)
    project = bids_project_dir.name
    output_dir = Path(output_dir)
    wanted_subjects = _normalize_participant_labels(participant_labels)

    results: Dict[str, ValidationResult] = {}

    for subject_dir in sorted(bids_project_dir.glob("sub-*")):
        subject_name = subject_dir.name
        if wanted_subjects and subject_name not in wanted_subjects:
            continue

        _, subject, tree = predict_tree(subject_dir)

        for ses_name, runs in tree.items():
            key = f"{subject}/{ses_name}"

            expected_dirs: set[Path] = set()
            expected_mmwide_files: set[Path] = set()
            for modality, run in runs:
                expected_dirs.add(output_dir / project / subject / ses_name / modality / run)
                expected_mmwide_files.add(
                    _expected_mmwide_path(output_dir, project, subject, ses_name, modality, run)
                )

            existing_dirs: set[Path] = set()
            root = output_dir / project / subject / ses_name
            if root.exists():
                for modality_dir in root.iterdir():
                    if not modality_dir.is_dir():
                        continue
                    for child in modality_dir.iterdir():
                        if child.is_dir():
                            existing_dirs.add(child)

            missing = sorted(str(p.relative_to(output_dir)) for p in expected_dirs - existing_dirs)
            unexpected = sorted(str(p.relative_to(output_dir)) for p in existing_dirs - expected_dirs)
            ok = sorted(str(p.relative_to(output_dir)) for p in expected_dirs & existing_dirs)
            missing_mmwide_files = sorted(
                str(p.relative_to(output_dir))
                for p in expected_mmwide_files
                if not p.exists()
            )
            missing_modalities = sorted(
                {
                    _extract_modality_from_relpath(path)
                    for path in [*missing, *missing_mmwide_files]
                }
            )

            results[key] = ValidationResult(
                missing=missing,
                unexpected=unexpected,
                ok=ok,
                missing_mmwide_files=missing_mmwide_files,
                missing_modalities=missing_modalities,
            )

    return results


def summarize_results(results: Dict[str, ValidationResult]) -> ValidationSummary:
    session_count = len(results)
    ok_count = sum(len(res.ok) for res in results.values())
    missing_count = sum(len(res.missing) for res in results.values())
    unexpected_count = sum(len(res.unexpected) for res in results.values())
    missing_mmwide_count = sum(len(res.missing_mmwide_files) for res in results.values())
    clean_session_count = sum(
        1
        for res in results.values()
        if not res.missing and not res.unexpected and not res.missing_mmwide_files
    )
    affected_session_count = session_count - clean_session_count
    return ValidationSummary(
        session_count=session_count,
        ok_count=ok_count,
        missing_count=missing_count,
        unexpected_count=unexpected_count,
        missing_mmwide_count=missing_mmwide_count,
        clean_session_count=clean_session_count,
        affected_session_count=affected_session_count,
    )


def build_summary_table(results: Dict[str, ValidationResult]) -> List[ValidationTableRow]:
    rows: List[ValidationTableRow] = []
    for session_key in sorted(results):
        res = results[session_key]
        if res.missing or res.unexpected or res.missing_mmwide_files:
            status = "issues"
        else:
            status = "clean"
        rows.append(
            ValidationTableRow(
                session_key=session_key,
                ok_count=len(res.ok),
                missing_count=len(res.missing),
                unexpected_count=len(res.unexpected),
                missing_mmwide_count=len(res.missing_mmwide_files),
                missing_modalities=res.missing_modalities,
                status=status,
            )
        )
    return rows


def build_session_modality_table(results: Dict[str, ValidationResult]) -> List[SessionModalityRow]:
    rows: List[SessionModalityRow] = []
    for session_key in sorted(results):
        subject_id, session_id = session_key.split("/", 1)
        res = results[session_key]

        ok_by_key = {
            (_extract_modality_from_relpath(relpath), _extract_run_from_relpath(relpath))
            for relpath in res.ok
        }
        missing_dir_by_key = {
            (_extract_modality_from_relpath(relpath), _extract_run_from_relpath(relpath))
            for relpath in res.missing
        }

        expected_mmwide_by_key: dict[tuple[str, str], str] = {}
        missing_mmwide_by_key: set[tuple[str, str]] = set()

        for relpath in res.missing_mmwide_files:
            key = (_extract_modality_from_relpath(relpath), _extract_run_from_relpath(relpath))
            expected_mmwide_by_key[key] = relpath
            missing_mmwide_by_key.add(key)

        for modality, run_id in sorted(ok_by_key | missing_dir_by_key | set(expected_mmwide_by_key.keys())):
            expected_mmwide_csv = expected_mmwide_by_key.get(key := (modality, run_id), "")
            if key in missing_dir_by_key:
                status = "MISSING"
            elif key in missing_mmwide_by_key:
                status = "MISSING_CSV"
            else:
                status = "OK"
            rows.append(
                SessionModalityRow(
                    subject_id=subject_id,
                    session_id=session_id,
                    modality=modality,
                    run_id=run_id,
                    status=status,
                    expected_mmwide_csv=expected_mmwide_csv,
                )
            )
    return rows


def build_missing_percentage_table(results: Dict[str, ValidationResult]) -> List[MissingPercentageRow]:
    expected_by_modality: dict[str, int] = {}
    missing_dirs_by_modality: dict[str, int] = {}
    missing_mmwide_by_modality: dict[str, int] = {}

    for res in results.values():
        for relpath in res.ok:
            modality = _extract_modality_from_relpath(relpath)
            expected_by_modality[modality] = expected_by_modality.get(modality, 0) + 1
        for relpath in res.missing:
            modality = _extract_modality_from_relpath(relpath)
            expected_by_modality[modality] = expected_by_modality.get(modality, 0) + 1
            missing_dirs_by_modality[modality] = missing_dirs_by_modality.get(modality, 0) + 1
        for relpath in res.missing_mmwide_files:
            modality = _extract_modality_from_relpath(relpath)
            missing_mmwide_by_modality[modality] = missing_mmwide_by_modality.get(modality, 0) + 1

    rows: List[MissingPercentageRow] = []
    for modality in sorted(expected_by_modality):
        expected_count = expected_by_modality[modality]
        missing_dir_count = missing_dirs_by_modality.get(modality, 0)
        present_dir_count = expected_count - missing_dir_count
        missing_mmwide_count = missing_mmwide_by_modality.get(modality, 0)
        present_mmwide_count = expected_count - missing_mmwide_count
        missing_dir_pct = (100.0 * missing_dir_count / expected_count) if expected_count else 0.0
        missing_mmwide_pct = (100.0 * missing_mmwide_count / expected_count) if expected_count else 0.0
        rows.append(
            MissingPercentageRow(
                modality=modality,
                expected_count=expected_count,
                present_dir_count=present_dir_count,
                missing_dir_count=missing_dir_count,
                missing_dir_pct=missing_dir_pct,
                present_mmwide_count=present_mmwide_count,
                missing_mmwide_count=missing_mmwide_count,
                missing_mmwide_pct=missing_mmwide_pct,
            )
        )
    return rows
