from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .contracts import iter_expected_artifacts
from .models import (
    ExpectedArtifact,
    FindingCode,
    RunKey,
    RunValidationRecord,
    SessionKey,
    Severity,
    StudyValidationReport,
    ValidationFinding,
)
from .scanner import scan_output_inventory, validate_mmwide_csv


def build_validation_report(
    bids_project_dir: str | Path,
    output_dir: str | Path,
    *,
    participant_labels: Iterable[str] | None = None,
    check_mmwide_content: bool = True,
    check_status_files: bool = True,
    strict_schema: bool = False,
) -> StudyValidationReport:
    bids_project_dir = Path(bids_project_dir)
    project_id = bids_project_dir.name
    expected = iter_expected_artifacts(
        bids_project_dir=bids_project_dir,
        output_dir=output_dir,
        participant_labels=participant_labels,
    )
    expected_sessions = tuple(sorted({artifact.session for artifact in expected}))
    inventory = scan_output_inventory(output_dir, project_id)

    discovered_by_key = {(run.session, run.run): run for run in inventory.discovered_runs}
    expected_keys = {(artifact.session, artifact.run) for artifact in expected}
    findings: list[ValidationFinding] = []
    records: list[RunValidationRecord] = []

    for artifact in expected:
        record, record_findings = _validate_expected_artifact(
            artifact=artifact,
            discovered=discovered_by_key.get((artifact.session, artifact.run)),
            status=inventory.status_by_session.get(artifact.session),
            check_mmwide_content=check_mmwide_content,
            check_status_files=check_status_files,
            strict_schema=strict_schema,
        )
        records.append(record)
        findings.extend(record_findings)

    for discovered in inventory.discovered_runs:
        if (discovered.session, discovered.run) in expected_keys:
            continue
        if discovered.session in expected_sessions:
            code = FindingCode.UNEXPECTED_RUN_DIR
            severity = Severity.WARNING
            message = "Unexpected modality/run output exists for an expected session."
        else:
            code = FindingCode.ORPHAN_OUTPUT
            severity = Severity.WARNING
            message = "Output exists for a subject/session that is not present in the BIDS input."
        findings.append(
            ValidationFinding(
                code=code,
                severity=severity,
                session=discovered.session,
                run=discovered.run,
                path=discovered.output_dir,
                message=message,
            )
        )

    return StudyValidationReport(
        records=tuple(sorted(records, key=lambda r: (r.session.subject_id, r.session.session_id, r.run.modality, r.run.run_id))),
        findings=tuple(sorted(findings, key=lambda f: (f.session.subject_id, f.session.session_id, f.run.modality if f.run else "", f.run.run_id if f.run else "", f.code))),
        expected_sessions=expected_sessions,
        discovered_sessions=tuple(sorted(inventory.status_by_session.keys())),
        status_by_session=inventory.status_by_session,
        strict_schema=strict_schema,
    )


def _validate_expected_artifact(
    *,
    artifact: ExpectedArtifact,
    discovered,
    status: dict | None,
    check_mmwide_content: bool,
    check_status_files: bool,
    strict_schema: bool,
) -> tuple[RunValidationRecord, list[ValidationFinding]]:
    findings: list[ValidationFinding] = []
    dir_exists = discovered is not None and artifact.output_dir.exists()
    mmwide_exists = artifact.mmwide_csv.exists()
    mmwide_valid: bool | None = None
    csv_columns: tuple[str, ...] = ()
    csv_row_count = 0
    csv_issue: str | None = None
    csv_profile: str | None = None
    csv_metric_matches: tuple[str, ...] = ()

    if not dir_exists:
        findings.append(
            ValidationFinding(
                code=FindingCode.MISSING_RUN_DIR,
                severity=Severity.ERROR,
                session=artifact.session,
                run=artifact.run,
                path=artifact.output_dir,
                message="Expected output directory is missing.",
            )
        )
    if not mmwide_exists:
        findings.append(
            ValidationFinding(
                code=FindingCode.MISSING_MMWIDE_CSV,
                severity=Severity.ERROR,
                session=artifact.session,
                run=artifact.run,
                path=artifact.mmwide_csv,
                message="Expected mmwide.csv is missing.",
            )
        )
    elif check_mmwide_content:
        csv_validation = validate_mmwide_csv(artifact.mmwide_csv, modality=artifact.run.modality, strict_schema=strict_schema)
        mmwide_valid = csv_validation.is_valid
        csv_columns = csv_validation.columns
        csv_row_count = csv_validation.row_count
        csv_issue = csv_validation.issue
        csv_profile = csv_validation.profile_name
        csv_metric_matches = csv_validation.metric_matches
        if csv_validation.is_valid is False:
            code = FindingCode.EMPTY_MMWIDE_CSV if csv_validation.issue in {"empty_header", "no_rows", "blank_header"} else FindingCode.INVALID_MMWIDE_CSV
            findings.append(
                ValidationFinding(
                    code=code,
                    severity=Severity.ERROR,
                    session=artifact.session,
                    run=artifact.run,
                    path=artifact.mmwide_csv,
                    message=f"mmwide.csv failed validation: {csv_validation.issue}",
                )
            )
    if check_status_files:
        findings.extend(_status_findings(artifact.session, artifact.run, artifact.status_file, status))

    return (
        RunValidationRecord(
            expected=artifact,
            dir_exists=dir_exists,
            mmwide_exists=mmwide_exists,
            mmwide_valid=mmwide_valid,
            findings=tuple(findings),
            csv_columns=csv_columns,
            csv_row_count=csv_row_count,
            csv_issue=csv_issue,
            csv_profile=csv_profile,
            strict_schema_applied=strict_schema,
            csv_metric_matches=csv_metric_matches,
        ),
        findings,
    )


def _status_findings(session: SessionKey, run: RunKey, status_path: Path, status: dict | None) -> list[ValidationFinding]:
    if status is None:
        return [
            ValidationFinding(
                code=FindingCode.MISSING_STATUS_FILE,
                severity=Severity.WARNING,
                session=session,
                run=run,
                path=status_path,
                message="Per-session status file is missing.",
            )
        ]
    if status.get("_invalid"):
        return [
            ValidationFinding(
                code=FindingCode.INVALID_STATUS_FILE,
                severity=Severity.WARNING,
                session=session,
                run=run,
                path=status_path,
                message="Per-session status file could not be parsed as JSON.",
            )
        ]
    mismatches: list[ValidationFinding] = []
    if str(status.get("project_id", "")) not in {"", session.project_id} or str(status.get("subjectID", "")) not in {"", session.subject_id} or str(status.get("sessionID", "")) not in {"", session.session_id}:
        mismatches.append(
            ValidationFinding(
                code=FindingCode.STATUS_SESSION_MISMATCH,
                severity=Severity.WARNING,
                session=session,
                run=run,
                path=status_path,
                message="Status file metadata does not match the expected project/subject/session.",
            )
        )
    return mismatches
