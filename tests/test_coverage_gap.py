import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import os
from antsxmm.core import sanitize_and_stage_file, build_wide_table_from_mmwide, process_session
from antsxmm.pipeline import run_study

def test_sanitize_extensions(tmp_path):
    # Test .nii
    f = tmp_path / "test.nii"
    f.touch()
    path, _, _ = sanitize_and_stage_file(str(f), "P", "S", "D", "T1w", "r1", "_", str(tmp_path))
    assert path.endswith(".nii")

    # Test unknown
    f2 = tmp_path / "test.img"
    f2.touch()
    path, _, _ = sanitize_and_stage_file(str(f2), "P", "S", "D", "T1w", "r1", "_", str(tmp_path))
    assert path.endswith(".img")

def test_build_wide_table_verbose_and_errors(tmp_path, capsys):
    # Create a structure that triggers verbose messages
    d = tmp_path / "P" / "S" / "D" / "T1wHierarchical" / "id"
    d.mkdir(parents=True)
    (d / "f_mmwide.csv").write_text("bids_subject,vol,thick\nx,1,2")
    
    # Create another to trigger overlap drop
    d2 = tmp_path / "P" / "S" / "D" / "T1w" / "id"
    d2.mkdir(parents=True)
    # 'vol' overlaps with T1Hier
    (d2 / "f_mmwide.csv").write_text("bids_subject,vol,noise\nx,1,3")

    build_wide_table_from_mmwide(str(tmp_path), verbose=True)
    out = capsys.readouterr().out
    assert "Excluding" in out

def test_process_session_wide_table_failure(tmp_path):
    # Mock build_wide_table to raise exception
    data = {'subjectID': 's', 'date': 'd', 't1_filenames': ['/tmp/fake.nii.gz']}
    
    with patch('antsxmm.core.sanitize_and_stage_file', return_value=('/tmp/f', 'm', 'i')), \
         patch('antspymm.generate_mm_dataframe'), \
         patch('antspymm.mm_csv'), \
         patch('antspymm.get_data'), \
         patch('antsxmm.core.build_wide_table_from_mmwide', side_effect=Exception("Fail")):
        
        # Should not crash, just log warning
        res = process_session(data, str(tmp_path))
        assert res['success'] is True # The main process succeeded, only wide table failed
        assert res['wide_df'] is None

def test_pipeline_failures(tmp_path, capsys):
    # Test run_study with a failing session
    with patch('antsxmm.pipeline.parse_antsxbids_layout') as mock_parse, \
         patch('antsxmm.pipeline.process_session') as mock_proc:
        
        mock_parse.return_value = pd.DataFrame([{'subjectID': 's1', 'date': 'd1'}])
        mock_proc.return_value = {'success': False}
        
        run_study("bids", "out", "proj")
        out = capsys.readouterr().out
        assert "Finished with 1 errors" in out