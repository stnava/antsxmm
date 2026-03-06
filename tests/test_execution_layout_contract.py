import os
from pathlib import Path

from antsxmm.pymm_execution import generate_xmm_dataframe


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'')
    return str(path)


def test_perf_layout_contract_no_filename_derived_directory(tmp_path):
    asl = _touch(tmp_path / 'sub-EAS002' / 'ses-01' / 'perf' / 'sub-EAS002_ses-01_run-001_asl.nii.gz')
    t1 = _touch(tmp_path / 'sub-EAS002' / 'ses-01' / 'anat' / 'sub-EAS002_ses-01_run-001_T1w.nii.gz')
    df = generate_xmm_dataframe({
        'subjectID': 'sub-EAS002',
        'sessionID': 'ses-01',
        'session_path': str(tmp_path / 'sub-EAS002' / 'ses-01'),
        't1_filenames': [t1],
        'perf_filenames': [asl],
    }, output_root=str(tmp_path / 'pymm'), project_id='ExpArt')
    prefix = df.iloc[0]['xmm_prefix_perf']
    assert os.path.normpath('/perf/run-01/') in os.path.normpath(prefix + os.sep)
    assert 'sub-EAS002_ses-01_perf_run-01' not in prefix
