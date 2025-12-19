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
    
    # Anat: Create TWO runs for T1w
    (ses1 / "anat").mkdir(parents=True)
    (ses1 / "anat" / "sub-001_ses-20230101_r0001_T1w.nii.gz").touch()
    (ses1 / "anat" / "sub-001_ses-20230101_r0001_T1w.json").touch()
    (ses1 / "anat" / "sub-001_ses-20230101_r0002_T1w.nii.gz").touch()
    
    # FLAIR
    (ses1 / "anat" / "sub-001_ses-20230101_r0001_FLAIR.nii.gz").touch()
    
    # DWI
    (ses1 / "dwi").mkdir()
    (ses1 / "dwi" / "sub-001_ses-20230101_dir-LR_r0001_dwi.nii.gz").touch()
    (ses1 / "dwi" / "sub-001_ses-20230101_dir-LR_r0001_dwi.bval").touch()
    (ses1 / "dwi" / "sub-001_ses-20230101_dir-LR_r0001_dwi.bvec").touch()
    # Second DWI run
    (ses1 / "dwi" / "sub-001_ses-20230101_dir-LR_r0002_dwi.nii.gz").touch()
    
    # Func
    (ses1 / "func").mkdir()
    (ses1 / "func" / "sub-001_ses-20230101_task-rest_bold.nii.gz").touch()
    
    # Melanin
    (ses1 / "melanin").mkdir()
    (ses1 / "melanin" / "sub-001_ses-20230101_NM.nii.gz").touch()

    # Perf
    (ses1 / "perf").mkdir()
    (ses1 / "perf" / "sub-001_ses-20230101_asl.nii.gz").touch()

    # PET
    (ses1 / "pet").mkdir()
    (ses1 / "pet" / "sub-001_ses-20230101_pet.nii.gz").touch()

    return bids_root

@pytest.fixture
def mock_session_data(mock_bids_structure):
    t1_r1 = str(mock_bids_structure / "sub-001/ses-20230101/anat/sub-001_ses-20230101_r0001_T1w.nii.gz")
    t1_r2 = str(mock_bids_structure / "sub-001/ses-20230101/anat/sub-001_ses-20230101_r0002_T1w.nii.gz")
    return {
        'subjectID': '001',
        'date': '20230101',
        't1_filename': t1_r1,
        't1_filenames': [t1_r1, t1_r2],
        'flair_filename': str(mock_bids_structure / "sub-001/ses-20230101/anat/sub-001_ses-20230101_r0001_FLAIR.nii.gz"),
        'rsf_filenames': [str(mock_bids_structure / "sub-001/ses-20230101/func/sub-001_ses-20230101_task-rest_bold.nii.gz")],
        'dti_filenames': [
            str(mock_bids_structure / "sub-001/ses-20230101/dwi/sub-001_ses-20230101_dir-LR_r0001_dwi.nii.gz"),
            str(mock_bids_structure / "sub-001/ses-20230101/dwi/sub-001_ses-20230101_dir-LR_r0002_dwi.nii.gz")
        ],
        'nm_filenames': [str(mock_bids_structure / "sub-001/ses-20230101/melanin/sub-001_ses-20230101_NM.nii.gz")],
        'perf_filename': str(mock_bids_structure / "sub-001/ses-20230101/perf/sub-001_ses-20230101_asl.nii.gz"),
        'pet3d_filename': str(mock_bids_structure / "sub-001/ses-20230101/pet/sub-001_ses-20230101_pet.nii.gz")
    }