import pytest
from click.testing import CliRunner
from antsxmm.pipeline import main

def test_pipeline_valid(mock_bids_structure, tmp_path, mocker):
    mocker.patch("antsxmm.pipeline.process_session", return_value=True)
    mocker.patch("antspymm.get_data")
    mocker.patch("antspyt1w.get_data")

    runner = CliRunner()
    result = runner.invoke(main, [str(mock_bids_structure), str(tmp_path)])
    assert result.exit_code == 0
