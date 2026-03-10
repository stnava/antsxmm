from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from .validation import build_validation_report as build_validation_report_v1
from .validation.models import FindingCode, StudyValidationReport
from .validation.reporting import build_issue_code_table, serialize_validation_report, write_validation_json_report


@dataclass(frozen=True)
class ValidationResult:
    missing: List[str]
    unexpected: List[str]
    ok: List[str]
    missing_mmwide_files: List[str]
    missing_modalities: List[str]
    invalid_mmwide_files: List[str]
    missing_status_files: List[str]


@dataclass(frozen=True)
class ValidationSummary:
    session_count: int
    ok_count: int
    missing_count: int
    unexpected_count: int
    missing_mmwide_count: int
    invalid_mmwide_count: int
    missing_status_count: int
    clean_session_count: int
    affected_session_count: int


@dataclass(frozen=True)
class ValidationTableRow:
    session_key: str
    ok_count: int
    missing_count: int
    unexpected_count: int
    missing_mmwide_count: int
    invalid_mmwide_count: int
    missing_status_count: int
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
    invalid_mmwide_count: int
    invalid_mmwide_pct: float


@dataclass(frozen=True)
class IssueCodeRow:
    code: str
    count: int


@dataclass(frozen=True)
class ValidationReport:
    study_report: StudyValidationReport
    legacy_results: Dict[str, ValidationResult]


def _normalize_participant_labels(participant_labels: Iterable[str] | None) -> set[str] | None:
    if participant_labels is None:
        return None
    normalized = {str(label).strip() for label in participant_labels if str(label).strip()}
    return normalized or None


def _extract_modality_from_relpath(relpath: str) -> str:
    parts = Path(relpath).parts
    return parts[3] if len(parts) >= 5 else "unknown"


def _extract_run_from_relpath(relpath: str) -> str:
    parts = Path(relpath).parts
    return parts[4] if len(parts) >= 5 else "unknown"


def _extract_expected_artifact_key(relpath: str) -> tuple[str, str, str, str] | None:
    parts = Path(relpath).parts
    if len(parts) < 5:
        return None
    return (parts[1], parts[2], parts[3], parts[4])


def build_validation_report(
    bids_project_dir: str | Path,
    output_dir: str | Path = "pymm",
    *,
    participant_labels: Iterable[str] | None = None,
    check_mmwide_content: bool = True,
    check_status_files: bool = True,
    strict_schema: bool = False,
) -> ValidationReport:
    study_report = build_validation_report_v1(
        bids_project_dir,
        output_dir,
        participant_labels=participant_labels,
        check_mmwide_content=check_mmwide_content,
        check_status_files=check_status_files,
        strict_schema=strict_schema,
    )
    return ValidationReport(
        study_report=study_report,
        legacy_results=_build_legacy_results(study_report, Path(output_dir)),
    )



def validate_project(
    bids_project_dir: str | Path,
    output_dir: str | Path = "pymm",
    *,
    participant_labels: Iterable[str] | None = None,
) -> Dict[str, ValidationResult]:
    report = build_validation_report(
        bids_project_dir,
        output_dir,
        participant_labels=participant_labels,
    )
    return report.legacy_results


def _build_legacy_results(study_report: StudyValidationReport, output_dir: Path) -> Dict[str, ValidationResult]:
    expected_paths_by_session: dict[str, dict[tuple[str, str], str]] = defaultdict(dict)
    ok_paths_by_session: dict[str, list[str]] = defaultdict(list)
    missing_paths_by_session: dict[str, list[str]] = defaultdict(list)
    missing_csv_by_session: dict[str, list[str]] = defaultdict(list)
    invalid_csv_by_session: dict[str, list[str]] = defaultdict(list)
    missing_status_by_session: dict[str, list[str]] = defaultdict(list)
    unexpected_by_session: dict[str, list[str]] = defaultdict(list)

    for record in study_report.records:
        session_key = record.session.label
        rel_dir = str(record.expected.output_dir.relative_to(output_dir))
        rel_csv = str(record.expected.mmwide_csv.relative_to(output_dir))
        expected_paths_by_session[session_key][(record.run.modality, record.run.run_id)] = rel_dir
        if record.dir_exists and record.mmwide_exists and not any(f.code in {FindingCode.INVALID_MMWIDE_CSV, FindingCode.EMPTY_MMWIDE_CSV} for f in record.findings):
            ok_paths_by_session[session_key].append(rel_dir)
        if any(f.code == FindingCode.MISSING_RUN_DIR for f in record.findings):
            missing_paths_by_session[session_key].append(rel_dir)
        if any(f.code == FindingCode.MISSING_MMWIDE_CSV for f in record.findings):
            missing_csv_by_session[session_key].append(rel_csv)
        if any(f.code in {FindingCode.INVALID_MMWIDE_CSV, FindingCode.EMPTY_MMWIDE_CSV} for f in record.findings):
            invalid_csv_by_session[session_key].append(rel_csv)
        if any(f.code == FindingCode.MISSING_STATUS_FILE for f in record.findings):
            missing_status_by_session[session_key].append(str(record.expected.status_file.relative_to(output_dir)))

    for finding in study_report.findings:
        if finding.code not in {FindingCode.UNEXPECTED_RUN_DIR, FindingCode.ORPHAN_OUTPUT} or finding.path is None:
            continue
        unexpected_by_session[finding.session.label].append(str(finding.path.relative_to(output_dir)))

    session_keys = sorted(
        set(expected_paths_by_session)
        | set(unexpected_by_session)
        | set(missing_status_by_session)
    )
    results: Dict[str, ValidationResult] = {}
    for session_key in session_keys:
        missing_modalities = sorted(
            {
                _extract_modality_from_relpath(path)
                for path in [
                    *missing_paths_by_session[session_key],
                    *missing_csv_by_session[session_key],
                    *invalid_csv_by_session[session_key],
                ]
            }
        )
        results[session_key] = ValidationResult(
            missing=sorted(missing_paths_by_session[session_key]),
            unexpected=sorted(unexpected_by_session[session_key]),
            ok=sorted(ok_paths_by_session[session_key]),
            missing_mmwide_files=sorted(missing_csv_by_session[session_key]),
            missing_modalities=missing_modalities,
            invalid_mmwide_files=sorted(invalid_csv_by_session[session_key]),
            missing_status_files=sorted(set(missing_status_by_session[session_key])),
        )
    return results


def summarize_results(results: Dict[str, ValidationResult]) -> ValidationSummary:
    session_count = len(results)
    ok_count = sum(len(res.ok) for res in results.values())
    missing_count = sum(len(res.missing) for res in results.values())
    unexpected_count = sum(len(res.unexpected) for res in results.values())
    missing_mmwide_count = sum(len(res.missing_mmwide_files) for res in results.values())
    invalid_mmwide_count = sum(len(res.invalid_mmwide_files) for res in results.values())
    missing_status_count = sum(len(res.missing_status_files) for res in results.values())
    clean_session_count = sum(
        1
        for res in results.values()
        if not res.missing and not res.unexpected and not res.missing_mmwide_files and not res.invalid_mmwide_files
    )
    affected_session_count = session_count - clean_session_count
    return ValidationSummary(
        session_count=session_count,
        ok_count=ok_count,
        missing_count=missing_count,
        unexpected_count=unexpected_count,
        missing_mmwide_count=missing_mmwide_count,
        invalid_mmwide_count=invalid_mmwide_count,
        missing_status_count=missing_status_count,
        clean_session_count=clean_session_count,
        affected_session_count=affected_session_count,
    )


def build_summary_table(results: Dict[str, ValidationResult]) -> List[ValidationTableRow]:
    rows: List[ValidationTableRow] = []
    for session_key in sorted(results):
        res = results[session_key]
        has_issues = bool(
            res.missing or res.unexpected or res.missing_mmwide_files or res.invalid_mmwide_files or res.missing_status_files
        )
        rows.append(
            ValidationTableRow(
                session_key=session_key,
                ok_count=len(res.ok),
                missing_count=len(res.missing),
                unexpected_count=len(res.unexpected),
                missing_mmwide_count=len(res.missing_mmwide_files),
                invalid_mmwide_count=len(res.invalid_mmwide_files),
                missing_status_count=len(res.missing_status_files),
                missing_modalities=res.missing_modalities,
                status="issues" if has_issues else "clean",
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
        invalid_mmwide_by_key: set[tuple[str, str]] = set()

        for relpath in res.missing_mmwide_files:
            key = (_extract_modality_from_relpath(relpath), _extract_run_from_relpath(relpath))
            expected_mmwide_by_key[key] = relpath
            missing_mmwide_by_key.add(key)
        for relpath in res.invalid_mmwide_files:
            key = (_extract_modality_from_relpath(relpath), _extract_run_from_relpath(relpath))
            expected_mmwide_by_key[key] = relpath
            invalid_mmwide_by_key.add(key)

        all_keys = ok_by_key | missing_dir_by_key | set(expected_mmwide_by_key.keys())
        for modality, run_id in sorted(all_keys):
            key = (modality, run_id)
            expected_mmwide_csv = expected_mmwide_by_key.get(key, "")
            if key in missing_dir_by_key:
                status = "MISSING"
            elif key in missing_mmwide_by_key:
                status = "MISSING_CSV"
            elif key in invalid_mmwide_by_key:
                status = "INVALID_CSV"
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
    expected_keys_by_modality: dict[str, set[tuple[str, str, str, str]]] = defaultdict(set)
    missing_dirs_by_modality: dict[str, int] = {}
    missing_mmwide_by_modality: dict[str, int] = {}
    invalid_mmwide_by_modality: dict[str, int] = {}

    for res in results.values():
        for relpath in [*res.ok, *res.missing, *res.missing_mmwide_files, *res.invalid_mmwide_files]:
            modality = _extract_modality_from_relpath(relpath)
            key = _extract_expected_artifact_key(relpath)
            if key is not None:
                expected_keys_by_modality[modality].add(key)
        for relpath in res.missing:
            modality = _extract_modality_from_relpath(relpath)
            missing_dirs_by_modality[modality] = missing_dirs_by_modality.get(modality, 0) + 1
        for relpath in res.missing_mmwide_files:
            modality = _extract_modality_from_relpath(relpath)
            missing_mmwide_by_modality[modality] = missing_mmwide_by_modality.get(modality, 0) + 1
        for relpath in res.invalid_mmwide_files:
            modality = _extract_modality_from_relpath(relpath)
            invalid_mmwide_by_modality[modality] = invalid_mmwide_by_modality.get(modality, 0) + 1

    rows: List[MissingPercentageRow] = []
    for modality in sorted(expected_keys_by_modality):
        expected_count = len(expected_keys_by_modality[modality])
        missing_dir_count = missing_dirs_by_modality.get(modality, 0)
        present_dir_count = expected_count - missing_dir_count
        missing_mmwide_count = missing_mmwide_by_modality.get(modality, 0)
        invalid_mmwide_count = invalid_mmwide_by_modality.get(modality, 0)
        present_mmwide_count = max(expected_count - missing_mmwide_count - invalid_mmwide_count, 0)
        missing_dir_pct = (100.0 * missing_dir_count / expected_count) if expected_count else 0.0
        missing_mmwide_pct = (100.0 * missing_mmwide_count / expected_count) if expected_count else 0.0
        invalid_mmwide_pct = (100.0 * invalid_mmwide_count / expected_count) if expected_count else 0.0
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
                invalid_mmwide_count=invalid_mmwide_count,
                invalid_mmwide_pct=invalid_mmwide_pct,
            )
        )
    return rows


def build_issue_code_summary(results: Dict[str, ValidationResult]) -> List[IssueCodeRow]:
    counts: dict[str, int] = defaultdict(int)
    for res in results.values():
        counts[str(FindingCode.MISSING_RUN_DIR)] += len(res.missing)
        counts[str(FindingCode.UNEXPECTED_RUN_DIR)] += len(res.unexpected)
        counts[str(FindingCode.MISSING_MMWIDE_CSV)] += len(res.missing_mmwide_files)
        counts[str(FindingCode.INVALID_MMWIDE_CSV)] += len(res.invalid_mmwide_files)
        counts[str(FindingCode.MISSING_STATUS_FILE)] += len(res.missing_status_files)
    return [IssueCodeRow(code=code, count=counts[code]) for code in sorted(code for code, count in counts.items() if count)]



def serialize_report_to_json(report: ValidationReport) -> dict:
    return serialize_validation_report(report.study_report)


def write_report_json(report: ValidationReport, output_path: str | Path) -> Path:
    return write_validation_json_report(report.study_report, output_path)
