import re
from pathlib import Path

from antsxmm.pymm_execution import generate_xmm_dataframe


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'')
    return str(path)


def test_layout_regex_contract(tmp_path):
    t1 = _touch(tmp_path / 'sub-1' / 'ses-1' / 'anat' / 'sub-1_ses-1_run-001_T1w.nii.gz')
    dwi = _touch(tmp_path / 'sub-1' / 'ses-1' / 'dwi' / 'sub-1_ses-1_run-001_dir-AP_dwi.nii.gz')
    bold = _touch(tmp_path / 'sub-1' / 'ses-1' / 'func' / 'sub-1_ses-1_task-rest_run-001_bold.nii.gz')
    df = generate_xmm_dataframe({'subjectID':'sub-1','sessionID':'ses-1','session_path':str(tmp_path/'sub-1'/'ses-1'),'t1_filenames':[t1],'dti_filenames':[dwi],'rsf_filenames':[bold]}, str(tmp_path/'out'), 'Proj')
    row = df.iloc[0]
    assert re.search(r'/DTI/run-\d{2}/', row['xmm_prefix_DTI'])
    assert re.search(r'/rsfMRI/run-\d{2}/', row['xmm_prefix_rsfMRI'])
    assert re.search(r'/T1w/run-\d{2}/', row['xmm_prefix_T1w'])
