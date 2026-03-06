from pathlib import Path

from antsxmm.pymm_execution import generate_xmm_dataframe, run_xmm_mm_csv


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'')
    return str(path)


class DummyAntsPyMM:
    def __init__(self):
        self.seen = None

    def mm_csv(self, df, **kwargs):
        ds = self.docsamson('perf', df, kwargs.get('outputdir',''), 'Proj', 'sub-1', 'ses-1', '+', t1iid='run-01', verbose=False)
        self.seen = ds
        return None


def test_run_xmm_mm_csv_patches_docsamson_to_deterministic_prefix(tmp_path):
    base = tmp_path / 'sub-1' / 'ses-1'
    t1 = _touch(base / 'anat' / 'sub-1_ses-1_run-001_T1w.nii.gz')
    asl = _touch(base / 'perf' / 'sub-1_ses-1_run-001_asl.nii.gz')
    df = generate_xmm_dataframe({'subjectID':'sub-1','sessionID':'ses-1','session_path':str(base),'t1_filenames':[t1],'perf_filenames':[asl]}, str(tmp_path/'out'), 'Proj')
    dummy = DummyAntsPyMM()
    run_xmm_mm_csv(df, dummy)
    assert dummy.seen is not None
    assert '/perf/run-01/' in dummy.seen['outprefix']
    assert 'sub-1_ses-1_perf_run-01' not in dummy.seen['outprefix']
