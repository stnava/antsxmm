from pathlib import Path

from antsxmm.pymm_execution import generate_xmm_dataframe


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'')
    return str(path)


def test_missing_run_defaults_to_run01(tmp_path):
    t1 = _touch(tmp_path / 'sub-1' / 'ses-1' / 'anat' / 'sub-1_ses-1_T1w.nii.gz')
    df = generate_xmm_dataframe({'subjectID':'sub-1','sessionID':'ses-1','session_path':str(tmp_path/'sub-1'/'ses-1'),'t1_filenames':[t1]}, str(tmp_path/'out'), 'Proj')
    assert df.iloc[0]['xmm_run'] == 'run-01'
    assert '/run-01/' in df.iloc[0]['xmm_prefix_T1w']
