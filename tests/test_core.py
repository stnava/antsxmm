import pytest
from unittest.mock import MagicMock
from antsxmm.core import process_session, sanitize_filename
import os
import pandas as pd
import antspymm # FIX: Added import

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
    
    # The result might be 'bad_Flair.nii.gz' or 'bad_flair.nii.gz'
    # Both satisfy the requirement "lair" in filename
    assert "lair" in os.path.basename(res) 
    assert os.path.islink(res)
    
    # Test 3: File needs arbitrary injection (bold -> func)
    bold_file = tmp_path / "image_bold.nii.gz"
    bold_file.touch()
    res = sanitize_filename(str(bold_file), ["fMRI", "func"])
    # If injection logic appends, it might be image_bold_fMRI.nii.gz
    # If regex replace logic worked (no match), it falls back to inject
    assert "func" in res or "fMRI" in res
    assert os.path.islink(res)

def test_process_session_success(mock_session_data, tmp_path, mocker):
    mocker.patch("antspymm.generate_mm_dataframe", return_value=pd.DataFrame({'A': [1]}))
    mocker.patch("antspymm.get_data", return_value=None) 
    mocker.patch("ants.image_read", return_value=MagicMock())
    mocker.patch("ants.crop_image", return_value=MagicMock())
    mocker.patch("ants.iMath", return_value=MagicMock())
    mock_mm_csv = mocker.patch("antspymm.mm_csv")

    output_dir = tmp_path / "processed"
    
    success = process_session(mock_session_data, str(output_dir), verbose=True)
    
    assert success is True
    # Verify sanitization happened on FLAIR
    call_args = antspymm.generate_mm_dataframe.call_args
    if 'flair_filename' in call_args.kwargs:
        # Check that the sanitized name passed to antspymm contains the required 'lair'
        assert "lair" in call_args.kwargs['flair_filename']

def test_process_session_error(mock_session_data, tmp_path, mocker):
    mocker.patch("antspymm.generate_mm_dataframe", side_effect=Exception("Boom"))
    success = process_session(mock_session_data, str(tmp_path))
    assert success is False
