import pytest
from pathlib import Path
import pandas as pd

@pytest.fixture
def mock_bids_structure(tmp_path):
    """Creates a temporary antsxbids-like directory structure."""
    bids_root = tmp_path / "BIDS_TEST"
    bids_root.mkdir()
    
    sub1 = bids_root / "sub-001"
    ses1 = sub1 / "ses-20230101"
    
    (ses1 / "anat").mkdir(parents=True)
    (ses1 / "anat" / "sub-001_ses-20230101_r001_T1w.nii.gz").touch()
    # Note: Uppercase FLAIR to test sanitization
    (ses1 / "anat" / "sub-001_ses-20230101_r001_FLAIR.nii.gz").touch()
    
    (ses1 / "dwi").mkdir()
    (ses1 / "dwi" / "sub-001_ses-20230101_dir-LR_dwi.nii.gz").touch()
    (ses1 / "dwi" / "sub-001_ses-20230101_dir-RL_dwi.nii.gz").touch()
    
    (ses1 / "func").mkdir()
    # Note: 'bold' not 'func' to test sanitization
    (ses1 / "func" / "sub-001_ses-20230101_task-rest_bold.nii.gz").touch()
    
    (ses1 / "melanin").mkdir()
    (ses1 / "melanin" / "sub-001_ses-20230101_NM.nii.gz").touch()

    return bids_root

@pytest.fixture
def mock_session_data(mock_bids_structure):
    return {
        'subjectID': '001',
        'date': '20230101',
        't1_filename': str(mock_bids_structure / "sub-001/ses-20230101/anat/sub-001_ses-20230101_r001_T1w.nii.gz"),
        'flair_filename': str(mock_bids_structure / "sub-001/ses-20230101/anat/sub-001_ses-20230101_r001_FLAIR.nii.gz"),
        'rsf_filenames': [str(mock_bids_structure / "sub-001/ses-20230101/func/sub-001_ses-20230101_task-rest_bold.nii.gz")],
        'dti_filenames': [
            str(mock_bids_structure / "sub-001/ses-20230101/dwi/sub-001_ses-20230101_dir-LR_dwi.nii.gz"),
            str(mock_bids_structure / "sub-001/ses-20230101/dwi/sub-001_ses-20230101_dir-RL_dwi.nii.gz")
        ],
        'nm_filenames': [str(mock_bids_structure / "sub-001/ses-20230101/melanin/sub-001_ses-20230101_NM.nii.gz")]
    }
