import pandas as pd

from antsxmm.pymm_execution import run_xmm_mm_csv


class DummyAntsPyMM:
    def __init__(self):
        self.captured = None

        def legacy_docsamson(locmod, studycsv, outputdir, projid, sid, dtid, mysep, t1iid=None, verbose=True):
            return {
                "modality": locmod,
                "outprefix": "WRONG_LEGACY_PREFIX",
                "images": ["legacy_image.nii.gz"],
            }

        self.docsamson = legacy_docsamson

        def mm_csv(studycsv, **kwargs):
            # This intentionally resolves docsamson from mm_csv globals.
            doc = docsamson(
                "perf",
                studycsv=studycsv,
                outputdir="ignored",
                projid="FPA",
                sid="sub-BLAST022",
                dtid="ses-01",
                mysep="+",
                t1iid="run-01",
                verbose=False,
            )
            self.captured = doc
            return doc

        self.mm_csv = mm_csv


def test_run_xmm_mm_csv_patches_mm_csv_globals():
    df = pd.DataFrame(
        [
            {
                "projectID": "FPA",
                "subjectID": "sub-BLAST022",
                "date": "ses-01",
                "imageID": "run-01",
                "modality": "T1w",
                "sourcedir": "/tmp/src",
                "outputdir": "/tmp/out",
                "filename": "/tmp/src/sub-BLAST022_ses-01_run-001_T1w.nii.gz",
                "perfid": "/tmp/src/sub-BLAST022_ses-01_run-001_asl.nii.gz",
                "xmm_prefix_perf": "/tmp/out/FPA/sub-BLAST022/ses-01/perf/run-01/FPA+sub-BLAST022+ses-01+perf+run-01",
            }
        ]
    )

    mod = DummyAntsPyMM()
    result = run_xmm_mm_csv(df, mod)

    assert result["outprefix"].endswith(
        "/FPA/sub-BLAST022/ses-01/perf/run-01/FPA+sub-BLAST022+ses-01+perf+run-01"
    )
    assert "WRONG_LEGACY_PREFIX" not in result["outprefix"]
