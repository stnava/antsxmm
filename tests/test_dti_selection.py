import json
from pathlib import Path

from antsxmm.inputs import plan_session_inputs


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return str(path)


def _write_json(path: Path, obj: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")
    return str(path)


def test_plan_session_inputs_prefers_exact_dwi_suffix_and_reverse_direction_pair(tmp_path):
    ses = tmp_path / "sub-182341" / "ses-20230111"
    anat = ses / "anat"
    dwi = ses / "dwi"

    t1 = _touch(anat / "sub-182341_ses-20230111_r0001_T1w.nii.gz")
    lr_dwi = _touch(dwi / "sub-182341_ses-20230111_r0001_dir-LR-dwi.nii.gz")
    lr_dwia = _touch(dwi / "sub-182341_ses-20230111_r0001_dir-LR-dwia.nii.gz")
    rl_dwi = _touch(dwi / "sub-182341_ses-20230111_r0001_dir-RL-dwi.nii.gz")
    rl_dwia = _touch(dwi / "sub-182341_ses-20230111_r0001_dir-RL-dwia.nii.gz")

    plan = plan_session_inputs(
        {
            "subjectID": "sub-182341",
            "sessionID": "ses-20230111",
            "session_path": str(ses),
            "t1_filenames": [t1],
            "dti_filenames": [lr_dwia, lr_dwi, rl_dwia, rl_dwi],
        }
    )

    assert plan["used"]["dti_filenames"] == [str(Path(lr_dwi).resolve()), str(Path(rl_dwi).resolve())]


def test_plan_session_inputs_can_use_json_phase_encoding_when_filename_direction_is_ambiguous(tmp_path):
    ses = tmp_path / "sub-01" / "ses-01"
    anat = ses / "anat"
    dwi = ses / "dwi"

    t1 = _touch(anat / "sub-01_ses-01_T1w.nii.gz")
    ap = _touch(dwi / "sub-01_ses-01_acq-a_dwi.nii.gz")
    pa = _touch(dwi / "sub-01_ses-01_acq-b_dwi.nii.gz")
    extra = _touch(dwi / "sub-01_ses-01_acq-c_dwia.nii.gz")

    _write_json(Path(ap[:-7] + ".json"), {"PhaseEncodingDirection": "j"})
    _write_json(Path(pa[:-7] + ".json"), {"PhaseEncodingDirection": "j-"})
    _write_json(Path(extra[:-8] + ".json"), {"PhaseEncodingDirection": "j"})

    plan = plan_session_inputs(
        {
            "subjectID": "sub-01",
            "sessionID": "ses-01",
            "session_path": str(ses),
            "t1_filenames": [t1],
            "dti_filenames": [extra, pa, ap],
        }
    )

    assert plan["used"]["dti_filenames"] == [str(Path(ap).resolve()), str(Path(pa).resolve())]
