import pytest
from unittest.mock import MagicMock, patch, call
from antsxmm.core import process_session, sanitize_and_stage_file, extract_image_id, get_modality_variant, bind_mm_rows, check_modality_order, build_wide_table_from_mmwide
import os
import pandas as pd
import antspymm 
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. Unit Tests for Utilities
# -----------------------------------------------------------------------------

def test_extract_image_id_variations():
    """Test regex extraction for various run ID formats."""
    assert extract_image_id("sub-01_run-001_T1w.nii.gz") == "run-001"
    assert extract_image_id("sub-01_r123_T1w.nii.gz") == "r123"
    assert extract_image_id("sub-01_no_run_info.nii.gz") == "000"
    assert extract_image_id("sub-01_ses-01_desc-preproc_T1w.nii.gz") == "000"

def test_get_modality_variant():
    # Test DTI direction appending
    assert get_modality_variant("dir-RL_dwi.nii.gz", "DTI", "_") == "DTIRL"
    assert get_modality_variant("dir-PA_dwi.nii.gz", "DTI", "_") == "DTIRL"
    assert get_modality_variant("dir-LR_dwi.nii.gz", "DTI", "+") == "DTI+LR"
    
    # Test rsfMRI
    assert get_modality_variant("task-rest_dir-RL_bold.nii.gz", "rsfMRI", "_") == "rsfMRIRL"
    
    # Test default mappings
    assert get_modality_variant("sub-01_dwi.nii.gz", "dwi", "_") == "DTI"
    assert get_modality_variant("sub-01_func.nii.gz", "func", "_") == "rsfMRI"

def test_sanitize_and_stage_file_fix(tmp_path):
    """
    Verify the fix for empty filename logic error.
    It should now construct a valid filename and successfully symlink.
    """
    src_dir = tmp_path / "source"
    src_dir.mkdir()
    good_file = src_dir / "sub-01_ses-01_r001_T1w.nii.gz"
    good_file.touch()
    
    # Create dummy sidecar
    (src_dir / "sub-01_ses-01_r001_T1w.json").touch()
    
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    
    # Call the function
    path, mod, uid = sanitize_and_stage_file(
        filepath=str(good_file), 
        project="PROJ", 
        subject="sub-01", 
        date="ses-01", 
        base_modality="T1w", 
        image_id="r001", 
        sep="_", 
        staging_root=str(staging_root),
        verbose=True
    )
    
    # Assertions
    assert path is not None
    assert os.path.exists(path)
    assert os.path.islink(path)
    
    # Verify the created filename is NOT empty/directory, but a file
    # Pattern: sub-01_ses-01_T1w_r001.nii.gz
    expected_name = "sub-01_ses-01_T1w_r001.nii.gz"
    assert os.path.basename(path) == expected_name
    
    # Verify sidecar was staged with the same base name
    sidecar_path = Path(path).parent / "sub-01_ses-01_T1w_r001.json"
    assert sidecar_path.exists()
    assert sidecar_path.is_symlink()

# -----------------------------------------------------------------------------
# 2. Process Session: Run Filtering & ID Preservation
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_multi_run_data(tmp_path):
    # T1s
    t1_r1 = tmp_path / "sub_r001_T1w.nii.gz"
    t1_r2 = tmp_path / "sub_r002_T1w.nii.gz"
    t1_r1.touch(); t1_r2.touch()
    
    # DWIs (Logic: Should keep BOTH even if T1 is filtered)
    dti_r1 = tmp_path / "sub_r001_dwi.nii.gz"
    dti_r2 = tmp_path / "sub_r002_dwi.nii.gz"
    dti_r1.touch(); dti_r2.touch()
    
    return {
        'subjectID': 'sub-01',
        'date': 'ses-01',
        't1_filenames': [str(t1_r1), str(t1_r2)],
        'dti_filenames': [str(dti_r1), str(dti_r2)],
        'rsf_filenames': [],
        'nm_filenames': [],
        'flair_filename': None,
        'perf_filename': None,
        'pet3d_filename': None
    }

def test_process_session_t1_filter_and_dwi_persistence(mock_multi_run_data, tmp_path):
    """
    REQUIREMENT CHECK: 
    1. Filter T1 to 'r002' -> should select T1w r002.
    2. Do NOT filter DWI -> should pass BOTH r001 and r002 DWIs to antspymm.
    3. Ensure IDs for DWI are preserved (r001 and r002) in the generated dataframe calls.
    """
    
    with patch("antspymm.generate_mm_dataframe") as mock_gen, \
         patch("antspymm.mm_csv"), \
         patch("antspymm.get_data", return_value=None):
         
        # Mock dataframe return for generate_mm_dataframe
        # We simulate what antspymm would return (empty columns usually filled by csv)
        mock_df = pd.DataFrame({
            'dtid1': ['placeholder'], 
            'dtid2': ['placeholder']
        })
        mock_gen.return_value = mock_df
        
        output_dir = tmp_path / "output"
        
        # ACT: Run processing with T1 filter for 'r002'
        process_session(mock_multi_run_data, str(output_dir), t1_run_match="r002", verbose=True)
        
        # ASSERT 1: generate_mm_dataframe was called
        assert mock_gen.called
        args, kwargs = mock_gen.call_args
        
        # ASSERT 2: T1 filename passed is the r002 file
        t1_passed = kwargs['t1_filename']
        assert "r002" in os.path.basename(t1_passed)
        
        # ASSERT 3: DWI filenames passed include BOTH files
        dti_passed = kwargs['dti_filenames']
        assert len(dti_passed) == 2
        
        # ASSERT 4: Check ID overwrite logic in dataframe
        # The logic in process_session writes back the staged IDs to the dataframe
        # verifying that we didn't lose the 'r001' ID for the first DTI
        
        # We inspect the args passed to mm_csv (which receives the modified dataframe)
        mock_mm_csv = antspymm.mm_csv
        call_args = mock_mm_csv.call_args[0]
        final_df = call_args[0]
        
        assert final_df['dtid1'].iloc[0] == "r001"
        assert final_df['dtid2'].iloc[0] == "r002"

# -----------------------------------------------------------------------------
# 3. Wide Table & Helpers
# -----------------------------------------------------------------------------

def test_bind_mm_rows_logic():
    df1 = pd.DataFrame({'u_hier_id': ['s1'], 'vol': [100]})
    df2 = pd.DataFrame({'u_hier_id': ['s1'], 'fa': [0.5]})
    res = bind_mm_rows([('T1', df1), ('DTI', df2)])
    assert 'T1_vol' in res.columns
    assert 'DTI_fa' in res.columns
    assert res.iloc[0]['subject_id'] == 's1'

def test_check_modality_order(capsys):
    check_modality_order([('B', None), ('A', None)], ['A', 'B'])
    assert "Warning: Modality order mismatch" in capsys.readouterr().out

def test_build_wide_table_from_mmwide(tmp_path):
    # Setup dummy structure
    sess_dir = tmp_path / "proj" / "sub-1" / "ses-1"
    
    # T1Hierarchical
    d1 = sess_dir / "T1wHierarchical" / "id1"
    d1.mkdir(parents=True)
    (d1 / "f_mmwide.csv").write_text("bids_subject,vol\nsub-1_ses-1,100")
    
    # DTI
    d2 = sess_dir / "DTI" / "id1"
    d2.mkdir(parents=True)
    (d2 / "f_mmwide.csv").write_text("bids_subject,fa\nsub-1_ses-1,0.5")
    
    df = build_wide_table_from_mmwide(str(sess_dir))
    
    assert len(df) == 1
    assert "T1Hier_vol" in df.columns
    assert "DTI_fa" in df.columns

def test_process_session_errors(mock_multi_run_data, tmp_path):
    # Test no T1 error
    data = mock_multi_run_data.copy()
    data['t1_filenames'] = []
    res = process_session(data, str(tmp_path))
    assert res['success'] is False