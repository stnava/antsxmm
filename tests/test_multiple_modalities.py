from pathlib import Path

from antsxmm.pymm_execution import generate_xmm_dataframe


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'')
    return str(path)


def test_multiple_modalities_share_canonical_run(tmp_path):
    base = tmp_path / 'sub-1' / 'ses-1'
    t1 = _touch(base / 'anat' / 'sub-1_ses-1_run-001_T1w.nii.gz')
    asl = _touch(base / 'perf' / 'sub-1_ses-1_run-001_asl.nii.gz')
    dwi_ap = _touch(base / 'dwi' / 'sub-1_ses-1_run-001_dir-AP_dwi.nii.gz')
    dwi_pa = _touch(base / 'dwi' / 'sub-1_ses-1_run-001_dir-PA_dwi.nii.gz')
    bold = _touch(base / 'func' / 'sub-1_ses-1_task-rest_run-001_bold.nii.gz')
    df = generate_xmm_dataframe({'subjectID':'sub-1','sessionID':'ses-1','session_path':str(base),'t1_filenames':[t1],'perf_filenames':[asl],'dti_filenames':[dwi_ap,dwi_pa],'rsf_filenames':[bold]}, str(tmp_path/'out'), 'Proj')
    row = df.iloc[0]
    assert '/perf/run-01/' in row['xmm_prefix_perf']
    assert '/DTI/run-01/' in row['xmm_prefix_DTI']
    assert '/rsfMRI/run-01/' in row['xmm_prefix_rsfMRI']
