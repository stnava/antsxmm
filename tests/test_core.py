import pytest
from unittest.mock import MagicMock
from antsxmm.core import process_session, sanitize_and_stage_file, extract_image_id, bind_mm_rows, check_modality_order
import os
import pandas as pd
import antspymm 
from pathlib import Path

def test_sanitize_and_stage_file(tmp_path):
    src_dir = tmp_path / "source"
    src_dir.mkdir()
    good_file = src_dir / "good_lair.nii.gz"
    good_file.touch()
    
    # Sidecar
    (src_dir / "good_lair.json").touch()
    
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    
    # Test 1: Simple staging
    # Returns: path, modality, id
    path, mod, uid = sanitize_and_stage_file(str(good_file), "PROJ", "SUB", "DATE", "T2Flair", "r001", "_", str(staging_root))
    
    assert os.path.exists(path)
    assert os.path.islink(path)
    assert "T2Flair" in path
    assert uid == "r001"
    
    # Check sidecar (Robust path check)
    p = Path(path)
    # Reconstruct expected json path based on the symlinked file name
    # file.nii.gz -> file.json
    base_name = p.name.replace("".join(p.suffixes), "") # Removes .nii.gz
    # If suffixes are .nii.gz, join(suffixes) is .nii.gz
    # If base name has dots, this might be fragile, but consistent with core.py logic
    
    # core.py logic for new_filename_base:
    # new_filename_base = f"{project}{sep}{subject}{sep}{date}{sep}{filename_modality}{sep}{image_id}"
    # expected_json = dest_dir / (new_filename_base + ".json")
    
    expected_json = p.parent / (p.name.replace(".nii.gz", ".json").replace(".nii", ".json"))
    assert expected_json.exists()
    assert expected_json.is_symlink()

def test_extract_image_id():
    assert extract_image_id("sub-01_ses-01_r0001_T1w.nii.gz") == "r0001"
    assert extract_image_id("sub-01_ses-01_run-02_T1w.nii.gz") == "run-02"
    assert extract_image_id("sub-01_ses-01_T1w.nii.gz") == "000"

def test_bind_mm_rows():
    df1 = pd.DataFrame({'u_hier_id': ['s1'], 'vol': [100]})
    df2 = pd.DataFrame({'u_hier_id': ['s1'], 'fa': [0.5]})
    
    res = bind_mm_rows([('T1', df1), ('DTI', df2)])
    assert 'T1_vol' in res.columns
    assert 'DTI_fa' in res.columns
    assert res.iloc[0]['subject_id'] == 's1'

def test_check_modality_order():
    # Should warn but pass
    assert check_modality_order([('DTI', None), ('T1w', None)], ['T1w', 'DTI']) is True

def test_process_session_multirun_behavior(mock_session_data, tmp_path, mocker):
    """
    Ensure that selecting a specific T1 run does NOT filter out other modalities.
    """
    mocker.patch("antspymm.generate_mm_dataframe", return_value=pd.DataFrame({'A': [1]}))
    mocker.patch("antspymm.get_data", return_value=None) 
    mocker.patch("ants.image_read", return_value=MagicMock())
    mocker.patch("ants.crop_image", return_value=MagicMock())
    mocker.patch("ants.iMath", return_value=MagicMock())
    mock_mm_csv = mocker.patch("antspymm.mm_csv")

    output_dir = tmp_path / "processed"
    
    # Case 1: Select r0002 T1
    # DTI has r0001 and r0002 in mock data. Both should be passed.
    process_session(mock_session_data, str(output_dir), verbose=True, build_wide_table=False, t1_run_match="r0002")
    
    args, kwargs = antspymm.generate_mm_dataframe.call_args
    
    # T1 should be r0002
    assert "r0002" in kwargs['t1_filename']
    assert kwargs['imageUniqueID'] == 'r0002'
    
    # DTI should have 2 files (all available)
    assert len(kwargs['dti_filenames']) == 2
    assert "r0001" in kwargs['dti_filenames'][0]
    assert "r0002" in kwargs['dti_filenames'][1]

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
    
    # Check Perf/PET passed
    args, kwargs = antspymm.generate_mm_dataframe.call_args
    assert kwargs['perf_filename'] is not None
    assert kwargs['pet3d_filename'] is not None

def test_process_session_error(mock_session_data, tmp_path, mocker):
    mocker.patch("antspymm.generate_mm_dataframe", side_effect=Exception("Boom"))
    result = process_session(mock_session_data, str(tmp_path))
    assert result['success'] is False