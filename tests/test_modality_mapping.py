from antsxmm.pymm_execution import modality_from_path


def test_modality_mapping_is_explicit():
    assert modality_from_path('sub-1_ses-1_run-01_asl.nii.gz') == 'perf'
    assert modality_from_path('sub-1_ses-1_task-rest_run-01_bold.nii.gz') == 'rsfMRI'
    assert modality_from_path('sub-1_ses-1_run-01_dwi.nii.gz') == 'DTI'
