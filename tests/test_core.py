import pytest
from unittest.mock import MagicMock
# Updated import name
from antsxmm.core import process_session, sanitize_and_stage_file, extract_image_id
import os
import pandas as pd
import antspymm 

def test_sanitize_and_stage_file(tmp_path):
    src_dir = tmp_path / "source"
    src_dir.mkdir()
    good_file = src_dir / "good_lair.nii.gz"
    good_file.touch()
    
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    
    # Test 1: Simple staging
    # Returns: path, modality, id
    path, mod, uid = sanitize_and_stage_file(str(good_file), "PROJ", "SUB", "DATE", "T2Flair", "r001", "_", str(staging_root))
    
    assert os.path.exists(path)
    assert os.path.islink(path)
    assert "T2Flair" in path
    assert uid == "r001"
    
    # Test 2: Direction variant (rsfMRI + LR)
    bold_file = src_dir / "task-rest_dir-LR_bold.nii.gz"
    bold_file.touch()
    
    path, mod, uid = sanitize_and_stage_file(str(bold_file), "PROJ", "SUB", "DATE", "rsfMRI", "r001", "_", str(staging_root))
    
    # Modality should catch the variant
    assert mod == "rsfMRILR" # No underscore due to sep='_'
    assert "rsfMRILR" in path

def test_extract_image_id():
    assert extract_image_id("sub-01_ses-01_r0001_T1w.nii.gz") == "r0001"
    assert extract_image_id("sub-01_ses-01_run-02_T1w.nii.gz") == "run-02"
    assert extract_image_id("sub-01_ses-01_T1w.nii.gz") == "000"

def test_process_session_success(mock_session_data, tmp_path, mocker):
    mocker.patch("antspymm.generate_mm_dataframe", return_value=pd.DataFrame({'A': [1]}))
    mocker.patch("antspymm.get_data", return_value=None) 
    mocker.patch("ants.image_read", return_value=MagicMock())
    mocker.patch("ants.crop_image", return_value=MagicMock())
    mocker.patch("ants.iMath", return_value=MagicMock())
    mock_mm_csv = mocker.patch("antspymm.mm_csv")

    output_dir = tmp_path / "processed"
    
    # Test default run (r0001)
    result = process_session(mock_session_data, str(output_dir), verbose=True, build_wide_table=False)
    assert result['success'] is True
    
    # Verify ID was extracted from default first file (r0001)
    gen_call_args = antspymm.generate_mm_dataframe.call_args
    assert gen_call_args.kwargs['imageUniqueID'] == 'r0001'

def test_process_session_specific_run(mock_session_data, tmp_path, mocker):
    mocker.patch("antspymm.generate_mm_dataframe", return_value=pd.DataFrame({'A': [1]}))
    mocker.patch("antspymm.get_data", return_value=None) 
    mocker.patch("ants.image_read", return_value=MagicMock())
    mocker.patch("ants.crop_image", return_value=MagicMock())
    mocker.patch("ants.iMath", return_value=MagicMock())
    mock_mm_csv = mocker.patch("antspymm.mm_csv")

    output_dir = tmp_path / "processed"
    
    # Test selecting r0002
    result = process_session(mock_session_data, str(output_dir), verbose=True, build_wide_table=False, t1_run_match="r0002")
    assert result['success'] is True
    
    # Verify we picked up r0002
    gen_call_args = antspymm.generate_mm_dataframe.call_args
    assert gen_call_args.kwargs['imageUniqueID'] == 'r0002'

def test_process_session_error(mock_session_data, tmp_path, mocker):
    mocker.patch("antspymm.generate_mm_dataframe", side_effect=Exception("Boom"))
    result = process_session(mock_session_data, str(tmp_path))
    assert result['success'] is False