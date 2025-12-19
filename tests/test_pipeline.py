import pytest
from click.testing import CliRunner
from antsxmm.pipeline import main

def test_pipeline_valid(mock_bids_structure, tmp_path, mocker):
    # Mock return value must be a DICT, not a bool
    mocker.patch("antsxmm.pipeline.process_session", return_value={'success': True, 'wide_df': None})
    mocker.patch("antspymm.get_data")
    mocker.patch("antspyt1w.get_data")

    runner = CliRunner()
    result = runner.invoke(main, [str(mock_bids_structure), str(tmp_path)])
    
    # Print output if it fails to help debugging
    if result.exit_code != 0:
        print(result.output)
        print(result.exception)
        
    assert result.exit_code == 0