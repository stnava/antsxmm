import pandas as pd

from antsxmm.pymm_execution import run_xmm_mm_csv


class DummyNM:
    def __init__(self):
        self.captured = None

        def mm_csv(studycsv, **kwargs):
            doc = docsamson(
                'NM2DMT',
                studycsv=studycsv,
                outputdir='ignored',
                projid='FPA',
                sid='sub-BLAST022',
                dtid='ses-01',
                mysep='+',
                t1iid='run-01',
                verbose=False,
            )
            self.captured = doc
            return doc

        self.mm_csv = mm_csv


def test_run_xmm_mm_csv_handles_nm2dmt():
    df = pd.DataFrame(
        [
            {
                'projectID': 'FPA',
                'subjectID': 'sub-BLAST022',
                'date': 'ses-01',
                'imageID': 'run-01',
                'xmm_run': 'run-01',
                'outputdir': '/tmp/out',
                'filename': '/tmp/src/sub-BLAST022_ses-01_run-001_T1w.nii.gz',
                'nmid1': '/tmp/src/sub-BLAST022_ses-01_run-001_NM.nii.gz',
                'xmm_prefix_NM2DMT': '/tmp/out/FPA/sub-BLAST022/ses-01/NM2DMT/run-01/FPA+sub-BLAST022+ses-01+NM2DMT+run-01',
            }
        ]
    )

    mod = DummyNM()
    result = run_xmm_mm_csv(df, mod)
    assert result['outprefix'].endswith('/FPA/sub-BLAST022/ses-01/NM2DMT/run-01/FPA+sub-BLAST022+ses-01+NM2DMT+run-01')
    assert result['images'] == ['/tmp/src/sub-BLAST022_ses-01_run-001_NM.nii.gz']
