from pathlib import Path

from click.testing import CliRunner

from antsxmm.pipeline import main


def test_tree_prediction_prints_expected_runs(tmp_path):
    bids = tmp_path / "BIDS" / "breacher" / "sub-9162" / "ses-followup-day2" / "anat"
    bids.mkdir(parents=True)
    (bids / "sub-9162_ses-followup-day2_T1w.nii.gz").touch()

    runner = CliRunner()
    result = runner.invoke(main, ["tree", str(bids.parent.parent)])
    assert result.exit_code == 0
    assert "pymm/" in result.output
    assert "breacher/" in result.output
    assert "sub-9162/" in result.output
    assert "ses-followup-day2/" in result.output
    assert "T1w/" in result.output
    assert "run-01/" in result.output
