import pytest
from unittest.mock import MagicMock
from antsxmm.core import process_session, sanitize_filename, extract_image_id
import os
import pandas as pd
import antspymm 

def test_sanitize_filename(tmp_path):
    # Test 1: File is already good
    good_file = tmp_path / "good_lair.nii.gz"
    good_file.touch()
    res = sanitize_filename(str(good_file), ["lair"])
    assert res == str(good_file)
    
    # Test 2: File matches case mismatch (FLAIR -> lair)
    bad_case = tmp_path / "bad_FLAIR.nii.gz"
    bad_case.touch()
    res = sanitize_filename(str(bad_case), ["lair"])
    
    # Assertion fix: Check that the required string is present (case-sensitive as required by antspymm)
    # The sanitization function ensures 'lair' is injected or replaced.
    assert "lair" in os.path.basename(res)
    assert os.path.islink(res)
    
    # Test 3: File needs arbitrary injection (bold -> func)
    bold_file = tmp_path / "image_bold.nii.gz"
    bold_file.touch()
    res = sanitize_filename(str(bold_file), ["fMRI", "func"])
    assert "func" in res or "fMRI" in res
    assert os.path.islink(res)

def test_extract_image_id():
    # Test valid BIDS with run
    assert extract_image_id("sub-01_ses-01_r0001_T1w.nii.gz") == "r0001"
    assert extract_image_id("sub-01_ses-01_run-02_T1w.nii.gz") == "run-02"
    
    # Test path input
    assert extract_image_id("/path/to/sub-01_ses-01_r0001_T1w.nii.gz") == "r0001"
    
    # Test fallback
    assert extract_image_id("sub-01_ses-01_T1w.nii.gz") == "000"
    
    # Test middle of string
    assert extract_image_id("T1w_r123_something.nii") == "r123"

def test_process_session_success(mock_session_data, tmp_path, mocker):
    mocker.patch("antspymm.generate_mm_dataframe", return_value=pd.DataFrame({'A': [1]}))
    mocker.patch("antspymm.get_data", return_value=None) 
    mocker.patch("ants.image_read", return_value=MagicMock())
    mocker.patch("ants.crop_image", return_value=MagicMock())
    mocker.patch("ants.iMath", return_value=MagicMock())
    mock_mm_csv = mocker.patch("antspymm.mm_csv")

    output_dir = tmp_path / "processed"
    
    # Mock t1 filename to ensure it has a run ID to test extraction logic
    mock_session_data['t1_filename'] = "/tmp/sub-01_ses-01_r999_T1w.nii.gz"

    result = process_session(mock_session_data, str(output_dir), verbose=True, build_wide_table=False)
    
    assert result['success'] is True
    
    # Verify generate_mm_dataframe was called with the extracted run ID (r999) NOT '000'
    gen_call_args = antspymm.generate_mm_dataframe.call_args
    assert gen_call_args.kwargs['imageUniqueID'] == 'r999'

def test_process_session_error(mock_session_data, tmp_path, mocker):
    mocker.patch("antspymm.generate_mm_dataframe", side_effect=Exception("Boom"))
    result = process_session(mock_session_data, str(tmp_path))
    assert result['success'] is False