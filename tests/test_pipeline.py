import pytest
from click.testing import CliRunner
from antsxmm.pipeline import main
from unittest.mock import patch

def test_pipeline_valid(mock_bids_structure, tmp_path):
  # Mock return value must be a DICT, not a bool
  with patch("antsxmm.pipeline.process_session", return_value={'success': True, 'wide_df': None}), \
       patch("antsxmm.pipeline.antspymm.get_data", create=True), \
       patch("antsxmm.pipeline.antspyt1w.get_data", create=True):

    runner = CliRunner()
    result = runner.invoke(main, ["run", str(mock_bids_structure), str(tmp_path)])
   
  # Print output if it fails to help debugging
  if result.exit_code != 0:
    print(result.output)
    print(result.exception)
     
  assert result.exit_code == 0

def test_pipeline_failure(mock_bids_structure, tmp_path):
  # Simulate a processing failure
  with patch("antsxmm.pipeline.process_session", return_value={'success': False, 'wide_df': None}), \
       patch("antsxmm.pipeline.antspymm.get_data", create=True), \
       patch("antsxmm.pipeline.antspyt1w.get_data", create=True):

    runner = CliRunner()
    result = runner.invoke(main, ["run", str(mock_bids_structure), str(tmp_path)])
  
  # Assert that the CLI exits with failure code
  assert result.exit_code == 1