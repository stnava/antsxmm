from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import StudyValidationReport


@dataclass(frozen=True)
class IssueCodeRow:
    code: str
    count: int


def build_issue_code_table(report: StudyValidationReport) -> list[IssueCodeRow]:
    counts = Counter(str(f.code) for f in report.findings)
    return [IssueCodeRow(code=code, count=counts[code]) for code in sorted(counts)]


def build_session_issue_counts(report: StudyValidationReport) -> list[tuple[str, int]]:
    counts: dict[str, int] = defaultdict(int)
    for finding in report.findings:
        counts[finding.session.label] += 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def build_summary_payload(report: StudyValidationReport) -> dict[str, Any]:
    session_issue_counts = {session.label: 0 for session in report.expected_sessions}
    for finding in report.findings:
        session_issue_counts[finding.session.label] = session_issue_counts.get(finding.session.label, 0) + 1
    return {
        "session_count": len(report.expected_sessions),
        "clean_session_count": sum(1 for count in session_issue_counts.values() if count == 0),
        "affected_session_count": sum(1 for count in session_issue_counts.values() if count > 0),
        "finding_counts": {row.code: row.count for row in build_issue_code_table(report)},
    }


def serialize_validation_report(report: StudyValidationReport) -> dict[str, Any]:
    return {
        "summary": build_summary_payload(report),
        "config": {
            "strict_schema": report.strict_schema,
        },
        "expected_sessions": [
            {
                "project_id": session.project_id,
                "subject_id": session.subject_id,
                "session_id": session.session_id,
            }
            for session in report.expected_sessions
        ],
        "records": [
            {
                "project_id": record.session.project_id,
                "subject_id": record.session.subject_id,
                "session_id": record.session.session_id,
                "modality": record.run.modality,
                "run_id": record.run.run_id,
                "output_dir": str(record.expected.output_dir),
                "mmwide_csv": str(record.expected.mmwide_csv),
                "status_file": str(record.expected.status_file),
                "dir_exists": record.dir_exists,
                "mmwide_exists": record.mmwide_exists,
                "mmwide_valid": record.mmwide_valid,
                "csv_columns": list(record.csv_columns),
                "csv_row_count": record.csv_row_count,
                "csv_issue": record.csv_issue,
                "csv_profile": record.csv_profile,
                "strict_schema_applied": record.strict_schema_applied,
                "csv_metric_matches": list(record.csv_metric_matches),
                "finding_codes": [str(finding.code) for finding in record.findings],
            }
            for record in report.records
        ],
        "findings": [
            {
                "code": str(finding.code),
                "severity": str(finding.severity),
                "project_id": finding.session.project_id,
                "subject_id": finding.session.subject_id,
                "session_id": finding.session.session_id,
                "modality": finding.run.modality if finding.run else None,
                "run_id": finding.run.run_id if finding.run else None,
                "path": str(finding.path) if finding.path is not None else None,
                "message": finding.message,
            }
            for finding in report.findings
        ],
    }


def write_validation_json_report(report: StudyValidationReport, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = serialize_validation_report(report)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path
