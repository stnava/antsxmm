import pytest
from click.testing import CliRunner
from unittest.mock import patch

from antsxmm.pipeline import main


def test_cli_dry_run_does_not_create_output_dir(mock_bids_structure, tmp_path):
    outdir = tmp_path / "OUT_NOT_CREATED"
    assert not outdir.exists()

    runner = CliRunner()

    with patch("antsxmm.pipeline.process_session", side_effect=AssertionError("process_session should not run in --dry-run")):
        result = runner.invoke(main, ["run", str(mock_bids_structure), str(outdir), "--dry-run"])

    assert result.exit_code == 0
    # Contract: dry-run should not create outputs
    assert not outdir.exists()
    assert "Plan summary: sessions=" in result.output
    assert "PLAN " in result.output


def test_cli_run_failure_prints_single_summary_line(mock_bids_structure, tmp_path):
    outdir = tmp_path / "OUT"
    runner = CliRunner()

    with patch("antsxmm.pipeline.process_session", return_value={"success": False, "wide_df": None}):
        result = runner.invoke(main, ["run", str(mock_bids_structure), str(outdir)])

    assert result.exit_code == 1
    assert sum(1 for ln in result.output.splitlines() if ln.strip() == "Finished with 1 errors") == 1


def test_cli_tree_contract(mock_bids_structure):
    runner = CliRunner()
    subject_dir = mock_bids_structure / "sub-001"

    result = runner.invoke(main, ["tree", str(subject_dir)])

    assert result.exit_code == 0
    out = result.output
    assert "pymm/" in out
    assert "BIDS_TEST/" in out
    assert "sub-001/" in out
    assert "ses-20230101/" in out
    # Key modalities from the mock tree
    assert "T1w/" in out
    assert "T2Flair/" in out
    assert "DTI/" in out
    assert "rsfMRI/" in out
    assert "perf/" in out
    assert "pet3d/" in out
