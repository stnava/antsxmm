from pathlib import Path

from click.testing import CliRunner

from antsxmm.pipeline import main
from antsxmm.validate import (
    build_missing_percentage_table,
    build_summary_table,
    summarize_results,
    validate_project,
)


def _mk_expected_outputs(pymm_root: Path, project: str, subject: str, session: str, paths: list[tuple[str, str]]):
    for modality, run in paths:
        d = pymm_root / project / subject / session / modality / run
        d.mkdir(parents=True, exist_ok=True)


def test_validate_perfect_tree(tmp_path):
    bids_proj = tmp_path / "BIDS" / "breacher"
    subj = bids_proj / "sub-9162" / "ses-followup-day2"
    (subj / "anat").mkdir(parents=True)
    (subj / "anat" / "sub-9162_ses-followup-day2_T1w.nii.gz").touch()

    pymm = tmp_path / "pymm"
    _mk_expected_outputs(pymm, "breacher", "sub-9162", "ses-followup-day2", [("T1w", "run-01"), ("T1wHierarchical", "run-01")])
    (pymm / "breacher" / "sub-9162" / "ses-followup-day2" / "T1w" / "run-01" / "breacher+sub-9162+ses-followup-day2+T1w+run-01+mmwide.csv").touch()
    (pymm / "breacher" / "sub-9162" / "ses-followup-day2" / "T1wHierarchical" / "run-01" / "breacher+sub-9162+ses-followup-day2+T1wHierarchical+run-01+mmwide.csv").touch()

    results = validate_project(bids_proj, pymm)
    res = results["sub-9162/ses-followup-day2"]
    assert res.missing == []
    assert res.unexpected == []
    assert res.missing_mmwide_files == []
    assert res.missing_modalities == []
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
        (pymm / "breacher" / "sub-0102" / "ses-initial-day1" / modality / "run-01" / f"breacher+sub-0102+ses-initial-day1+{modality}+run-01+mmwide.csv").touch()

    results = validate_project(bids_proj, pymm, participant_labels=["sub-0102"])
    assert list(results.keys()) == ["sub-0102/ses-initial-day1"]
    summary = summarize_results(results)
    assert summary.session_count == 1
    assert summary.clean_session_count == 1
    rows = build_summary_table(results)
    assert rows[0].status == "clean"


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
    (pymm / "breacher" / "sub-0102" / "ses-initial-day1" / "T1w" / "run-01" / "breacher+sub-0102+ses-initial-day1+T1w+run-01+mmwide.csv").touch()

    results = validate_project(bids_proj, pymm)
    pct_rows = build_missing_percentage_table(results)
    t1w_row = next(row for row in pct_rows if row.modality == "T1w")
    assert t1w_row.expected_count == 2
    assert t1w_row.missing_dir_count == 1
    assert t1w_row.missing_dir_pct == 50.0
    assert t1w_row.missing_mmwide_count == 1
    assert t1w_row.missing_mmwide_pct == 50.0


def test_validate_cli_prints_summary_and_tables(tmp_path):
    bids_proj = tmp_path / "BIDS" / "breacher"
    subj = bids_proj / "sub-9162" / "ses-followup-day2"
    (subj / "anat").mkdir(parents=True)
    (subj / "anat" / "sub-9162_ses-followup-day2_T1w.nii.gz").touch()

    output_dir = tmp_path / "pymm"
    _mk_expected_outputs(output_dir, "breacher", "sub-9162", "ses-followup-day2", [("T1w", "run-01")])

    runner = CliRunner()
    result = runner.invoke(main, ["validate", str(bids_proj), str(output_dir)])
    assert result.exit_code == 0
    assert "Validation summary" in result.output
    assert "Per-session table" in result.output
    assert "Missing percentage table" in result.output
    assert "Missing modalities" in result.output
    assert "sub-9162/ses-followup-day2" in result.output
    assert "T1w" in result.output
    assert "100.0%" in result.output
