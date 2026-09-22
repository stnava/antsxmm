import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from antsxmm.core import process_session
from antsxmm.pipeline import run_study
from antsxmm.modalities.dispatch import SessionContext


@pytest.fixture
def mock_session_context(tmp_path):
    ctx = MagicMock(spec=SessionContext)
    ctx.t1 = MagicMock()
    ctx.hierarchical = {"brain_mask": MagicMock()}
    ctx.t1_prefix = str(tmp_path / "t1_prefix")
    ctx.cit168_tx = {"fwdtransforms": [], "invtransforms": []}
    ctx.ppmi_tx = {"fwdtransforms": [], "invtransforms": []}
    return ctx


def test_process_session_native_execution(mock_session_data, mock_session_context, tmp_path):
    """Test process_session with native_execution=True runs without touching legacy mm_csv."""
    def fake_execute_unit(unit, ctx, **kwargs):
        # Create output directory and mock mmwide file as expected by serializer
        prefix = Path(unit.output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        marker = prefix.parent / f"{prefix.name}+mmwide.csv"
        df = pd.DataFrame({"metric_a": [1.0], "metric_b": [2.0]})
        df.to_csv(marker, index=False)
        return df

    with patch("antsxmm.modalities.dispatch.initialize_session_context", return_value=mock_session_context) as mock_init, \
         patch("antsxmm.modalities.dispatch.execute_unit", side_effect=fake_execute_unit) as mock_exec, \
         patch("antsxmm.core.antspymm.mm_csv", create=True) as mock_mm_csv:

        result = process_session(
            mock_session_data,
            output_root=str(tmp_path),
            project_id="TESTPROJ",
            native_execution=True,
            write_input_manifest=True,
        )

        assert result["success"] is True
        assert result["error"] is None
        assert mock_mm_csv.call_count == 0
        assert mock_init.call_count == 1
        assert mock_exec.call_count > 0

        # Verify status file
        sub = mock_session_data["subjectID"]
        ses = mock_session_data["date"]
        status_file = tmp_path / "TESTPROJ" / sub / ses / ".antsxmm_status.json"
        assert status_file.exists()
        status_data = json.loads(status_file.read_text(encoding="utf-8"))
        assert status_data["success"] is True
        assert status_data["execution_engine"] == "antsxmm_native"
        assert status_data["args"]["native_execution"] is True


def test_run_study_native_execution(mock_bids_structure, mock_session_context, tmp_path):
    """Test run_study with native_execution=True completes cleanly."""
    def fake_execute_unit(unit, ctx, **kwargs):
        prefix = Path(unit.output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        marker = prefix.parent / f"{prefix.name}+mmwide.csv"
        df = pd.DataFrame({"metric_x": [42.0]})
        df.to_csv(marker, index=False)
        return df

    with patch("antsxmm.modalities.dispatch.initialize_session_context", return_value=mock_session_context), \
         patch("antsxmm.modalities.dispatch.execute_unit", side_effect=fake_execute_unit), \
         patch("antsxmm.core.antspymm.mm_csv", create=True) as mock_mm_csv:

        failures = run_study(
            str(mock_bids_structure),
            str(tmp_path / "output"),
            "TESTPROJ",
            native_execution=True,
        )

        assert failures == []
        assert mock_mm_csv.call_count == 0


def test_process_session_native_execution_handles_failure(mock_session_data, tmp_path):
    """Test process_session with native_execution=True properly traps and reports errors."""
    with patch("antsxmm.modalities.dispatch.initialize_session_context", side_effect=RuntimeError("T1 init failed")), \
         patch("antsxmm.core.antspymm.mm_csv", create=True) as mock_mm_csv:

        result = process_session(
            mock_session_data,
            output_root=str(tmp_path),
            project_id="TESTPROJ",
            native_execution=True,
        )

        assert result["success"] is False
        assert "T1 init failed" in str(result["error"])
        assert mock_mm_csv.call_count == 0

        # Verify status file records failure
        sub = mock_session_data["subjectID"]
        ses = mock_session_data["date"]
        status_file = tmp_path / "TESTPROJ" / sub / ses / ".antsxmm_status.json"
        assert status_file.exists()
        status_data = json.loads(status_file.read_text(encoding="utf-8"))
        assert status_data["success"] is False
        assert "T1 init failed" in status_data["error"]
