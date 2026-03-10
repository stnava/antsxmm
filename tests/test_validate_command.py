from pathlib import Path

from click.testing import CliRunner

from antsxmm.pipeline import main
from antsxmm.validate import (
    build_missing_percentage_table,
    build_session_modality_table,
    build_validation_report,
    summarize_results,
    validate_project,
)


def _mk_expected_outputs(pymm_root: Path, project: str, subject: str, session: str, paths: list[tuple[str, str]]):
    for modality, run in paths:
        d = pymm_root / project / subject / session / modality / run
        d.mkdir(parents=True, exist_ok=True)


def _write_mmwide(path: Path, *, rows: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows is None:
        rows = ["metric,value", "foo,1"]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

def test_validate_perfect_tree(tmp_path):
    bids_proj = tmp_path / "BIDS" / "breacher"
    subj = bids_proj / "sub-9162" / "ses-followup-day2"
    (subj / "anat").mkdir(parents=True)
    (subj / "anat" / "sub-9162_ses-followup-day2_T1w.nii.gz").touch()

    pymm = tmp_path / "pymm"
    _mk_expected_outputs(pymm, "breacher", "sub-9162", "ses-followup-day2", [("T1w", "run-01"), ("T1wHierarchical", "run-01")])
    _write_mmwide(pymm / "breacher" / "sub-9162" / "ses-followup-day2" / "T1w" / "run-01" / "breacher+sub-9162+ses-followup-day2+T1w+run-01+mmwide.csv")
    _write_mmwide(pymm / "breacher" / "sub-9162" / "ses-followup-day2" / "T1wHierarchical" / "run-01" / "breacher+sub-9162+ses-followup-day2+T1wHierarchical+run-01+mmwide.csv")

    results = validate_project(bids_proj, pymm)
    res = results["sub-9162/ses-followup-day2"]
    assert res.missing == []
    assert res.unexpected == []
    assert res.missing_mmwide_files == []
    assert res.invalid_mmwide_files == []
    assert "breacher/sub-9162/ses-followup-day2/T1w/run-01" in res.ok


def test_validate_missing_modality_and_mmwide(tmp_path):
    bids_proj = tmp_path / "BIDS" / "breacher"
    subj = bids_proj / "sub-9162" / "ses-followup-day2"
    (subj / "perf").mkdir(parents=True)
    (subj / "perf" / "sub-9162_ses-followup-day2_asl.nii.gz").touch()

    pymm = tmp_path / "pymm"
    results = validate_project(bids_proj, pymm)
    res = results["sub-9162/ses-followup-day2"]
    assert any("perf/run-01" in m for m in res.missing)
    assert any(path.endswith("+perf+run-01+mmwide.csv") for path in res.missing_mmwide_files)
    assert res.missing_modalities == ["perf"]


def test_validate_unexpected_directory(tmp_path):
    bids_proj = tmp_path / "BIDS" / "breacher"
    subj = bids_proj / "sub-9162" / "ses-followup-day2"
    (subj / "anat").mkdir(parents=True)
    (subj / "anat" / "sub-9162_ses-followup-day2_T1w.nii.gz").touch()

    pymm = tmp_path / "pymm"
    _mk_expected_outputs(pymm, "breacher", "sub-9162", "ses-followup-day2", [("T1w", "run-01"), ("T1wHierarchical", "run-01")])
    (pymm / "breacher" / "sub-9162" / "ses-followup-day2" / "perf" / "sub-9162_perf_000").mkdir(parents=True, exist_ok=True)

    results = validate_project(bids_proj, pymm)
    res = results["sub-9162/ses-followup-day2"]
    assert any("perf/sub-9162_perf_000" in u for u in res.unexpected)


def test_validate_participant_filter_and_summary(tmp_path):
    bids_proj = tmp_path / "BIDS" / "breacher"
    for subject in ["sub-0102", "sub-0202"]:
        subj = bids_proj / subject / "ses-initial-day1"
        (subj / "anat").mkdir(parents=True)
        (subj / "anat" / f"{subject}_ses-initial-day1_T1w.nii.gz").touch()

    pymm = tmp_path / "pymm"
    _mk_expected_outputs(pymm, "breacher", "sub-0102", "ses-initial-day1", [("T1w", "run-01"), ("T1wHierarchical", "run-01")])
    for modality in ["T1w", "T1wHierarchical"]:
        _write_mmwide(pymm / "breacher" / "sub-0102" / "ses-initial-day1" / modality / "run-01" / f"breacher+sub-0102+ses-initial-day1+{modality}+run-01+mmwide.csv")

    results = validate_project(bids_proj, pymm, participant_labels=["sub-0102"])
    assert list(results.keys()) == ["sub-0102/ses-initial-day1"]
    summary = summarize_results(results)
    assert summary.session_count == 1
    assert summary.clean_session_count == 1
    rows = build_session_modality_table(results)
    assert rows[0].status == "OK"


def test_missing_percentage_table(tmp_path):
    bids_proj = tmp_path / "BIDS" / "breacher"

    subj1 = bids_proj / "sub-0102" / "ses-initial-day1"
    (subj1 / "anat").mkdir(parents=True)
    (subj1 / "anat" / "sub-0102_ses-initial-day1_T1w.nii.gz").touch()

    subj2 = bids_proj / "sub-0103" / "ses-initial-day1"
    (subj2 / "anat").mkdir(parents=True)
    (subj2 / "anat" / "sub-0103_ses-initial-day1_T1w.nii.gz").touch()

    pymm = tmp_path / "pymm"
    _mk_expected_outputs(pymm, "breacher", "sub-0102", "ses-initial-day1", [("T1w", "run-01")])
    _write_mmwide(pymm / "breacher" / "sub-0102" / "ses-initial-day1" / "T1w" / "run-01" / "breacher+sub-0102+ses-initial-day1+T1w+run-01+mmwide.csv")

    results = validate_project(bids_proj, pymm)
    pct_rows = build_missing_percentage_table(results)
    t1w_row = next(row for row in pct_rows if row.modality == "T1w")
    assert t1w_row.expected_count == 2
    assert t1w_row.missing_dir_count == 1
    assert t1w_row.missing_dir_pct == 50.0
    assert t1w_row.missing_mmwide_count == 1
    assert t1w_row.missing_mmwide_pct == 50.0


def test_validate_detects_invalid_mmwide_csv(tmp_path):
    bids_proj = tmp_path / "BIDS" / "breacher"
    subj = bids_proj / "sub-9162" / "ses-followup-day2"
    (subj / "anat").mkdir(parents=True)
    (subj / "anat" / "sub-9162_ses-followup-day2_T1w.nii.gz").touch()

    output_dir = tmp_path / "pymm"
    _mk_expected_outputs(output_dir, "breacher", "sub-9162", "ses-followup-day2", [("T1w", "run-01")])
    _write_mmwide(
        output_dir / "breacher" / "sub-9162" / "ses-followup-day2" / "T1w" / "run-01" / "breacher+sub-9162+ses-followup-day2+T1w+run-01+mmwide.csv",
        rows=["metric,metric", "foo,1"],
    )

    report = build_validation_report(bids_proj, output_dir)
    result = report.legacy_results["sub-9162/ses-followup-day2"]
    assert len(result.invalid_mmwide_files) == 1
    assert result.ok == []


def test_validate_detects_orphan_output(tmp_path):
    bids_proj = tmp_path / "BIDS" / "breacher"
    subj = bids_proj / "sub-9162" / "ses-followup-day2"
    (subj / "anat").mkdir(parents=True)
    (subj / "anat" / "sub-9162_ses-followup-day2_T1w.nii.gz").touch()

    output_dir = tmp_path / "pymm"
    orphan_dir = output_dir / "breacher" / "sub-9999" / "ses-ghost" / "T1w" / "run-01"
    orphan_dir.mkdir(parents=True)
    _write_mmwide(orphan_dir / "breacher+sub-9999+ses-ghost+T1w+run-01+mmwide.csv")

    report = build_validation_report(bids_proj, output_dir)
    orphan_findings = [f for f in report.study_report.findings if f.code.value == "orphan_output"]
    assert len(orphan_findings) == 1
    assert orphan_findings[0].session.subject_id == "sub-9999"


def test_validate_cli_summary_first_and_issue_only_default(tmp_path):
    bids_proj = tmp_path / "BIDS" / "breacher"
    for subject in ["sub-0102", "sub-0103"]:
        subj = bids_proj / subject / "ses-initial-day1"
        (subj / "anat").mkdir(parents=True)
        (subj / "anat" / f"{subject}_ses-initial-day1_T1w.nii.gz").touch()

    output_dir = tmp_path / "pymm"
    _mk_expected_outputs(output_dir, "breacher", "sub-0102", "ses-initial-day1", [("T1w", "run-01"), ("T1wHierarchical", "run-01")])
    _write_mmwide(output_dir / "breacher" / "sub-0102" / "ses-initial-day1" / "T1w" / "run-01" / "breacher+sub-0102+ses-initial-day1+T1w+run-01+mmwide.csv")

    runner = CliRunner()
    result = runner.invoke(main, ["validate", str(bids_proj), str(output_dir)])
    assert result.exit_code == 0
    assert "Validation summary" in result.output
    assert "Missing percentage table" in result.output
    assert result.output.index("Missing percentage table") < result.output.index("Per-run validation table")
    per_run_section = result.output.split("Per-run validation table", 1)[1]
    assert " run-01     OK" not in per_run_section


def test_validate_cli_participant_shows_all_rows_by_default(tmp_path):
    bids_proj = tmp_path / "BIDS" / "breacher"
    subj = bids_proj / "sub-9162" / "ses-followup-day2"
    (subj / "anat").mkdir(parents=True)
    (subj / "anat" / "sub-9162_ses-followup-day2_T1w.nii.gz").touch()

    output_dir = tmp_path / "pymm"
    _mk_expected_outputs(output_dir, "breacher", "sub-9162", "ses-followup-day2", [("T1w", "run-01"), ("T1wHierarchical", "run-01")])
    _write_mmwide(output_dir / "breacher" / "sub-9162" / "ses-followup-day2" / "T1w" / "run-01" / "breacher+sub-9162+ses-followup-day2+T1w+run-01+mmwide.csv")
    _write_mmwide(output_dir / "breacher" / "sub-9162" / "ses-followup-day2" / "T1wHierarchical" / "run-01" / "breacher+sub-9162+ses-followup-day2+T1wHierarchical+run-01+mmwide.csv")

    runner = CliRunner()
    result = runner.invoke(main, ["validate", str(bids_proj), str(output_dir), "--participant-label", "sub-9162"])
    assert result.exit_code == 0
    assert "Per-run validation table" in result.output
    assert "sub-9162" in result.output
    assert "OK" in result.output


def test_validate_detects_schema_mismatch_mmwide_csv(tmp_path):
    bids_proj = tmp_path / "BIDS" / "breacher"
    subj = bids_proj / "sub-9162" / "ses-followup-day2"
    (subj / "anat").mkdir(parents=True)
    (subj / "anat" / "sub-9162_ses-followup-day2_T1w.nii.gz").touch()

    output_dir = tmp_path / "pymm"
    _mk_expected_outputs(output_dir, "breacher", "sub-9162", "ses-followup-day2", [("T1w", "run-01")])
    _write_mmwide(
        output_dir / "breacher" / "sub-9162" / "ses-followup-day2" / "T1w" / "run-01" / "breacher+sub-9162+ses-followup-day2+T1w+run-01+mmwide.csv",
        rows=["foo,bar", "1,2"],
    )

    report = build_validation_report(bids_proj, output_dir)
    finding_messages = [f.message for f in report.study_report.findings if f.code.value == "invalid_mmwide_csv"]
    assert any("schema_missing_identifier:T1w" in message for message in finding_messages)


def test_validate_cli_writes_json_report(tmp_path):
    bids_proj = tmp_path / "BIDS" / "breacher"
    subj = bids_proj / "sub-9162" / "ses-followup-day2"
    (subj / "anat").mkdir(parents=True)
    (subj / "anat" / "sub-9162_ses-followup-day2_T1w.nii.gz").touch()

    output_dir = tmp_path / "pymm"
    report_json = tmp_path / "reports" / "validate.json"

    runner = CliRunner()
    result = runner.invoke(main, ["validate", str(bids_proj), str(output_dir), "--summary-only", "--report-json", str(report_json)])
    assert result.exit_code == 0
    assert report_json.exists()
    payload = __import__("json").loads(report_json.read_text(encoding="utf-8"))
    assert payload["summary"]["session_count"] == 1
    assert payload["records"][0]["modality"] == "T1w"
    assert "missing_mmwide_csv" in payload["summary"]["finding_counts"]
    assert payload["config"]["strict_schema"] is False
    assert "JSON report:" in result.output


def test_validate_strict_schema_detects_missing_modality_metrics(tmp_path):
    bids_proj = tmp_path / "BIDS" / "breacher"
    subj = bids_proj / "sub-9162" / "ses-followup-day2"
    (subj / "dwi").mkdir(parents=True)
    (subj / "dwi" / "sub-9162_ses-followup-day2_dwi.nii.gz").touch()

    output_dir = tmp_path / "pymm"
    _mk_expected_outputs(output_dir, "breacher", "sub-9162", "ses-followup-day2", [("DTI", "run-01")])
    _write_mmwide(
        output_dir / "breacher" / "sub-9162" / "ses-followup-day2" / "DTI" / "run-01" / "breacher+sub-9162+ses-followup-day2+DTI+run-01+mmwide.csv",
        rows=["bids_subject,value", "sub-9162,1"],
    )

    report = build_validation_report(bids_proj, output_dir, strict_schema=True)
    finding_messages = [f.message for f in report.study_report.findings if f.code.value == "invalid_mmwide_csv"]
    assert any("strict_schema_missing_metrics:DTI" in message for message in finding_messages)
    record = report.study_report.records[0]
    assert record.strict_schema_applied is True
    assert record.csv_profile == "DTI"


def test_validate_strict_schema_accepts_modality_metrics(tmp_path):
    bids_proj = tmp_path / "BIDS" / "breacher"
    subj = bids_proj / "sub-9162" / "ses-followup-day2"
    (subj / "dwi").mkdir(parents=True)
    (subj / "dwi" / "sub-9162_ses-followup-day2_dwi.nii.gz").touch()

    output_dir = tmp_path / "pymm"
    _mk_expected_outputs(output_dir, "breacher", "sub-9162", "ses-followup-day2", [("DTI", "run-01")])
    _write_mmwide(
        output_dir / "breacher" / "sub-9162" / "ses-followup-day2" / "DTI" / "run-01" / "breacher+sub-9162+ses-followup-day2+DTI+run-01+mmwide.csv",
        rows=["bids_subject,fa,md", "sub-9162,0.5,0.8"],
    )

    report = build_validation_report(bids_proj, output_dir, strict_schema=True)
    invalid_findings = [f for f in report.study_report.findings if f.code.value == "invalid_mmwide_csv"]
    assert invalid_findings == []
    record = report.study_report.records[0]
    assert record.csv_metric_matches == ("fa", "md")


def test_validate_cli_strict_schema_writes_json_flag(tmp_path):
    bids_proj = tmp_path / "BIDS" / "breacher"
    subj = bids_proj / "sub-9162" / "ses-followup-day2"
    (subj / "dwi").mkdir(parents=True)
    (subj / "dwi" / "sub-9162_ses-followup-day2_dwi.nii.gz").touch()

    output_dir = tmp_path / "pymm"
    _mk_expected_outputs(output_dir, "breacher", "sub-9162", "ses-followup-day2", [("DTI", "run-01")])
    _write_mmwide(
        output_dir / "breacher" / "sub-9162" / "ses-followup-day2" / "DTI" / "run-01" / "breacher+sub-9162+ses-followup-day2+DTI+run-01+mmwide.csv",
        rows=["bids_subject,value", "sub-9162,1"],
    )
    report_json = tmp_path / "reports" / "validate_strict.json"

    runner = CliRunner()
    result = runner.invoke(main, ["validate", str(bids_proj), str(output_dir), "--summary-only", "--strict-schema", "--report-json", str(report_json)])
    assert result.exit_code == 0
    payload = __import__("json").loads(report_json.read_text(encoding="utf-8"))
    assert payload["config"]["strict_schema"] is True
    assert payload["records"][0]["csv_issue"] == "strict_schema_missing_metrics:DTI"
    assert "Strict schema mode: on" in result.output
