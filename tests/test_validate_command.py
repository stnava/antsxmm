from pathlib import Path

from antsxmm.validate import validate_project


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
