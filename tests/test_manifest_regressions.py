import json
import os

import pandas as pd
import pytest


def _touch(path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"")
    return path


class _DummyAntsPyMM:
    """Lightweight stand-in for antspymm to keep tests fast and deterministic."""

    @staticmethod
    def generate_mm_dataframe(**kwargs):
        # Real antspymm returns a DataFrame; keep a minimal contract.
        return pd.DataFrame([{"ok": 1}])

    @staticmethod
    def mm_csv(*args, **kwargs):
        return None

    @staticmethod
    def get_data(*args, **kwargs):
        # Avoid any template IO.
        return None


@pytest.fixture
def dummy_antspymm(monkeypatch):
    import antsxmm.core as core

    monkeypatch.setattr(core, "antspymm", _DummyAntsPyMM())
    return core


def _read_manifest(out_root: str, project: str, sub: str, ses: str, sep: str = "+") -> dict:
    p = os.path.join(out_root, project, sub, ses, f"{project}{sep}{sub}{sep}{ses}{sep}mm_inputs.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def test_manifest_does_not_iterate_over_nan_lists(tmp_path, dummy_antspymm):
    """Regression: missing list-valued fields may come through as NaN (float) from pandas.

    The manifest builder must treat these as empty lists rather than iterating a float.
    """

    core = dummy_antspymm

    bids_root = tmp_path / "BIDS"
    out_root = tmp_path / "out"

    # Minimal inputs needed for a session to proceed: at least one T1.
    t1 = _touch(str(bids_root / "sub-1" / "ses-1" / "anat" / "sub-1_ses-1_T1w.nii.gz"))
    flair = _touch(str(bids_root / "sub-1" / "ses-1" / "anat" / "sub-1_ses-1_FLAIR.nii.gz"))

    session_data = {
        "subjectID": "sub-1",
        "sessionID": "ses-1",
        "session_path": str(bids_root / "sub-1" / "ses-1"),
        "t1_filenames": [t1],
        "flair_filenames": [flair],
        "t2w_filenames": [],
        "dti_filenames": [],
        "rsf_filenames": [],
        "nm_filenames": [],
        "perf_filename": None,
        "pet3d_filename": None,
        # These list-valued columns are frequently NaN in the wild.
        "perf_filenames": float("nan"),
        "pet3d_filenames": float("nan"),
    }

    # Must not raise.
    core.process_session(
        session_data=session_data,
        output_root=str(out_root),
        project_id="proj",
        denoise=False,
        separator="+",
        t1_run_match=None,
        dti_moco=True,
        denoise_dti=True,
        verbose=False,
        write_input_manifest=True,
    )

    m = _read_manifest(str(out_root), "proj", "sub-1", "ses-1")
    assert m["discovered"]["perf_filenames"] == []
    assert m["discovered"]["pet3d_filenames"] == []


def test_excluded_is_discovered_minus_used_and_truncation_is_correct(tmp_path, dummy_antspymm):
    """Regression: excluded accounting must not list used files as excluded.

    Also ensures truncation (first 2) is reflected in excluded.
    """

    core = dummy_antspymm

    bids_root = tmp_path / "BIDS"
    out_root = tmp_path / "out"

    t1 = _touch(str(bids_root / "sub-1" / "ses-2" / "anat" / "sub-1_ses-2_T1w.nii.gz"))

    # Provide 3 rsfMRI runs; only first 2 should be used.
    rsf1 = _touch(str(bids_root / "sub-1" / "ses-2" / "func" / "sub-1_ses-2_task-rest_run-01_bold.nii.gz"))
    rsf2 = _touch(str(bids_root / "sub-1" / "ses-2" / "func" / "sub-1_ses-2_task-rest_run-02_bold.nii.gz"))
    rsf3 = _touch(str(bids_root / "sub-1" / "ses-2" / "func" / "sub-1_ses-2_task-rest_run-03_bold.nii.gz"))

    session_data = {
        "subjectID": "sub-1",
        "sessionID": "ses-2",
        "session_path": str(bids_root / "sub-1" / "ses-2"),
        "t1_filenames": [t1],
        "flair_filenames": [],
        "t2w_filenames": [],
        "dti_filenames": [],
        "rsf_filenames": [rsf1, rsf2, rsf3],
        "nm_filenames": [],
        "perf_filename": None,
        "pet3d_filename": None,
        "perf_filenames": [],
        "pet3d_filenames": [],
    }

    core.process_session(
        session_data=session_data,
        output_root=str(out_root),
        project_id="proj",
        denoise=False,
        separator="+",
        t1_run_match=None,
        dti_moco=True,
        denoise_dti=True,
        verbose=False,
        write_input_manifest=True,
    )

    m = _read_manifest(str(out_root), "proj", "sub-1", "ses-2")

    discovered = set(map(os.path.realpath, m["discovered"]["rsf_filenames"]))
    used = set(map(os.path.realpath, m["used_inputs"]["rsf_filenames"]))
    excluded = set(map(os.path.realpath, m["excluded"]["rsf_truncated"]))

    assert len(discovered) == 3
    assert len(used) == 2
    assert len(excluded) == 1

    # Invariant: discovered = used ∪ excluded, and used ∩ excluded = ∅
    assert discovered == (used | excluded)
    assert used.isdisjoint(excluded)

    # T1 selection should not incorrectly mark the selected T1 as excluded.
    assert m["excluded"]["t1_not_selected"] == []


def test_verbose_mode_prints_manifest_path_and_does_not_crash(tmp_path, dummy_antspymm, capsys):
    """Smoke test: verbose should be safe even with missing optional modalities."""

    core = dummy_antspymm

    bids_root = tmp_path / "BIDS"
    out_root = tmp_path / "out"
    t1 = _touch(str(bids_root / "sub-2" / "ses-1" / "anat" / "sub-2_ses-1_T1w.nii.gz"))

    session_data = {
        "subjectID": "sub-2",
        "sessionID": "ses-1",
        "session_path": str(bids_root / "sub-2" / "ses-1"),
        "t1_filenames": [t1],
        "flair_filenames": [],
        "t2w_filenames": [],
        "dti_filenames": [],
        "rsf_filenames": [],
        "nm_filenames": [],
        "perf_filename": None,
        "pet3d_filename": None,
        # NaN again, to ensure verbose paths don't trigger float iteration.
        "perf_filenames": float("nan"),
        "pet3d_filenames": float("nan"),
    }

    core.process_session(
        session_data=session_data,
        output_root=str(out_root),
        project_id="proj",
        denoise=False,
        separator="+",
        t1_run_match=None,
        dti_moco=True,
        denoise_dti=True,
        verbose=True,
        write_input_manifest=True,
    )

    out = capsys.readouterr().out
    assert "mm_inputs.json" in out
