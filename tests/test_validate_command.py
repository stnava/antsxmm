from pathlib import Path

from antsxmm.validate import validate_project


def _mk_expected_outputs(pymm_root: Path, project: str, subject: str, session: str, paths: list[tuple[str, str]]):
    for modality, run in paths:
        d = pymm_root / project / subject / session / modality / run
        d.mkdir(parents=True, exist_ok=True)


def _touch_mmwide_files(pymm_root: Path, project: str, subject: str, session: str, paths: list[tuple[str, str]]):
    for modality, run in paths:
        d = pymm_root / project / subject / session / modality / run
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{project}+{subject}+{session}+{modality}+{run}+mmwide.csv").write_text("ok\n")


def test_validate_perfect_tree(tmp_path):
    bids_proj = tmp_path / "BIDS" / "breacher"
    subj = bids_proj / "sub-9162" / "ses-followup-day2"
    (subj / "anat").mkdir(parents=True)
    (subj / "anat" / "sub-9162_ses-followup-day2_T1w.nii.gz").touch()

    pymm = tmp_path / "pymm"
    _mk_expected_outputs(pymm, "breacher", "sub-9162", "ses-followup-day2", [("T1w", "run-01"), ("T1wHierarchical", "run-01")])
    _touch_mmwide_files(pymm, "breacher", "sub-9162", "ses-followup-day2", [("T1w", "run-01"), ("T1wHierarchical", "run-01")])

    results = validate_project(bids_proj, pymm_dir=pymm)
    res = results["sub-9162/ses-followup-day2"]
    assert res.missing == []
    assert res.unexpected == []
    assert "breacher/sub-9162/ses-followup-day2/T1w/run-01" in res.ok


def test_validate_missing_modality(tmp_path):
    bids_proj = tmp_path / "BIDS" / "breacher"
    subj = bids_proj / "sub-9162" / "ses-followup-day2"
    (subj / "perf").mkdir(parents=True)
    (subj / "perf" / "sub-9162_ses-followup-day2_asl.nii.gz").touch()

    pymm = tmp_path / "pymm"
    # Do not create expected perf/run-01
    results = validate_project(bids_proj, pymm_dir=pymm)
    res = results["sub-9162/ses-followup-day2"]
    assert any("perf/run-01" in m for m in res.missing)


def test_validate_unexpected_directory(tmp_path):
    bids_proj = tmp_path / "BIDS" / "breacher"
    subj = bids_proj / "sub-9162" / "ses-followup-day2"
    (subj / "anat").mkdir(parents=True)
    (subj / "anat" / "sub-9162_ses-followup-day2_T1w.nii.gz").touch()

    pymm = tmp_path / "pymm"
    # Create expected
    _mk_expected_outputs(pymm, "breacher", "sub-9162", "ses-followup-day2", [("T1w", "run-01"), ("T1wHierarchical", "run-01")])
    # Create unexpected legacy directory
    (pymm / "breacher" / "sub-9162" / "ses-followup-day2" / "perf" / "sub-9162_perf_000").mkdir(parents=True, exist_ok=True)

    results = validate_project(bids_proj, pymm_dir=pymm)
    res = results["sub-9162/ses-followup-day2"]
    assert any("perf/sub-9162_perf_000" in u for u in res.unexpected)


from click.testing import CliRunner
from antsxmm.pipeline import main


def test_validate_cli_accepts_input_and_output_dirs(tmp_path):
    bids_proj = tmp_path / "BIDS" / "breacher"
    subj = bids_proj / "sub-9162" / "ses-followup-day2"
    (subj / "anat").mkdir(parents=True)
    (subj / "anat" / "sub-9162_ses-followup-day2_T1w.nii.gz").touch()

    output_root = tmp_path / "Processed"
    _mk_expected_outputs(output_root, "breacher", "sub-9162", "ses-followup-day2", [("T1w", "run-01"), ("T1wHierarchical", "run-01")])
    _touch_mmwide_files(output_root, "breacher", "sub-9162", "ses-followup-day2", [("T1w", "run-01"), ("T1wHierarchical", "run-01")])

    runner = CliRunner()
    result = runner.invoke(main, ["validate", str(bids_proj), str(output_root)])

    assert result.exit_code == 0
    assert "Session: sub-9162/ses-followup-day2" in result.output
    assert "OK: 2" in result.output
    assert "Missing: 0" in result.output
    assert "Unexpected: 0" in result.output
    assert f"output_root={output_root}" in result.output


def test_validate_cli_help_describes_input_and_output_dirs():
    runner = CliRunner()
    result = runner.invoke(main, ["validate", "--help"])

    assert result.exit_code == 0
    assert "INPUT_BIDS_PROJECT" in result.output
    assert "OUTPUT_DIR" in result.output
    assert "BIDS/breacher" in result.output
    assert "pymm or Processed" in result.output
    assert "missing expected outputs" in result.output
    assert "unexpected output directories present" in result.output


def test_validate_cli_legacy_pymm_dir_still_supported(tmp_path):
    bids_proj = tmp_path / "BIDS" / "breacher"
    subj = bids_proj / "sub-9162" / "ses-followup-day2"
    (subj / "anat").mkdir(parents=True)
    (subj / "anat" / "sub-9162_ses-followup-day2_T1w.nii.gz").touch()

    output_root = tmp_path / "pymm"
    _mk_expected_outputs(output_root, "breacher", "sub-9162", "ses-followup-day2", [("T1w", "run-01"), ("T1wHierarchical", "run-01")])
    _touch_mmwide_files(output_root, "breacher", "sub-9162", "ses-followup-day2", [("T1w", "run-01"), ("T1wHierarchical", "run-01")])

    runner = CliRunner()
    result = runner.invoke(main, ["validate", str(bids_proj), "--pymm-dir", str(output_root)])

    assert result.exit_code == 0
    assert "[deprecated] --pymm-dir" in result.output


def test_validate_reports_missing_mmwide_csv_files(tmp_path):
    bids_proj = tmp_path / "BIDS" / "breacher"
    subj = bids_proj / "sub-9162" / "ses-followup-day2"
    (subj / "anat").mkdir(parents=True)
    (subj / "anat" / "sub-9162_ses-followup-day2_T1w.nii.gz").touch()

    output_root = tmp_path / "Processed"
    _mk_expected_outputs(output_root, "breacher", "sub-9162", "ses-followup-day2", [("T1w", "run-01"), ("T1wHierarchical", "run-01")])
    _touch_mmwide_files(output_root, "breacher", "sub-9162", "ses-followup-day2", [("T1w", "run-01")])

    runner = CliRunner()
    result = runner.invoke(main, ["validate", str(bids_proj), str(output_root)])

    assert result.exit_code == 1
    assert "Missing mmwide.csv: 1" in result.output
    assert "breacher/sub-9162/ses-followup-day2/T1wHierarchical/run-01/breacher+sub-9162+ses-followup-day2+T1wHierarchical+run-01+mmwide.csv" in result.output


def test_validate_participant_label_filters_subjects(tmp_path):
    bids_proj = tmp_path / "BIDS" / "breacher"
    subj1 = bids_proj / "sub-1111" / "ses-01"
    (subj1 / "anat").mkdir(parents=True)
    (subj1 / "anat" / "sub-1111_ses-01_T1w.nii.gz").touch()
    subj2 = bids_proj / "sub-2222" / "ses-01"
    (subj2 / "anat").mkdir(parents=True)
    (subj2 / "anat" / "sub-2222_ses-01_T1w.nii.gz").touch()

    output_root = tmp_path / "Processed"
    _mk_expected_outputs(output_root, "breacher", "sub-1111", "ses-01", [("T1w", "run-01"), ("T1wHierarchical", "run-01")])
    _touch_mmwide_files(output_root, "breacher", "sub-1111", "ses-01", [("T1w", "run-01"), ("T1wHierarchical", "run-01")])

    runner = CliRunner()
    result = runner.invoke(main, ["validate", str(bids_proj), str(output_root), "--participant-label", "sub-1111"])

    assert result.exit_code == 0
    assert "Session: sub-1111/ses-01" in result.output
    assert "Session: sub-2222/ses-01" not in result.output
    assert "participant_filter=sub-1111" in result.output
