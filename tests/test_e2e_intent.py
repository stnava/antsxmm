"""End-to-end integration test verifying that antsxmm does everything it is intended to do:

1. BIDS Layout Parsing & Execution Planning
2. Native Multi-Modality Processing Pipeline Execution
3. Artifact & Manifest Provenance Generation
4. Resumability & Fingerprinting Guarantees
5. Aggregation CLI (antsxmm aggregate)
6. Directory Tree Prediction (antsxmm tree)
7. Output Tree & Modality Validation (antsxmm validate)
8. Public API & Subpackage Namespace Completeness
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from click.testing import CliRunner

import antsxmm
from antsxmm.pipeline import main, run_study
from antsxmm.modalities.dispatch import SessionContext


def test_public_api_and_namespace_completeness():
    """Verify that antsxmm exposes all expected functions and modular subpackages."""
    # Top-level exports
    assert hasattr(antsxmm, "parse_antsxbids_layout")
    assert hasattr(antsxmm, "build_wide_table_from_mmwide")
    assert hasattr(antsxmm, "bind_mm_rows")
    assert hasattr(antsxmm, "process_session")
    assert hasattr(antsxmm, "run_study")
    assert hasattr(antsxmm, "check_modality_order")
    assert hasattr(antsxmm, "modalities")
    assert hasattr(antsxmm, "registration")
    assert hasattr(antsxmm, "segmentation")

    # Modular subpackages
    from antsxmm import registration, segmentation, modalities
    from antsxmm.registration import timeseries_reg, dti_reg, get_average_rsf, transform_and_reorient_dti
    from antsxmm.segmentation import wmh, map_scalar_to_labels, warn_if_small_mask, trim_dti_mask
    from antsxmm.modalities import dti, fmri, perfusion, neuromelanin, wmh as wmh_mod, dispatch, io

    assert callable(timeseries_reg)
    assert callable(dti_reg)
    assert callable(get_average_rsf)
    assert callable(transform_and_reorient_dti)
    assert callable(wmh)
    assert callable(map_scalar_to_labels)
    assert callable(warn_if_small_mask)
    assert callable(trim_dti_mask)
    assert hasattr(dti, "joint_dti_recon")
    assert hasattr(fmri, "resting_state_fmri_networks")
    assert hasattr(perfusion, "bold_perfusion")
    assert callable(neuromelanin)
    assert callable(wmh_mod)
    assert hasattr(dispatch, "run_session_plan_natively")
    assert hasattr(io, "write_modality_mmwide")


def test_e2e_full_lifecycle(mock_bids_structure, tmp_path):
    """Verify full end-to-end lifecycle: run study, verify outputs, validate, aggregate, and resume."""
    runner = CliRunner()
    project = "ProjectE2E"
    output_dir = tmp_path / "output"

    # Mock SessionContext and execute_unit to produce realistic modality outputs
    mock_ctx = MagicMock(spec=SessionContext)
    mock_ctx.separator = "+"
    mock_ctx.t1_image = MagicMock()
    mock_ctx.hier = {
        "brain_mask": MagicMock(),
        "brain_n4_dnz": MagicMock(),
        "dkt_parc": {"tissue_segmentation": MagicMock()},
    }
    mock_ctx.hier_dir = str(output_dir / project / "001" / "20230101" / "T1wHierarchical" / "run-01")

    def fake_execute_unit(unit, context, **kwargs):
        prefix = Path(unit.output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        mmwide_file = prefix.parent / f"{prefix.name}+mmwide.csv"
        # Create deterministic metrics for each modality
        data = {
            f"{unit.modality}_metric_mean": [10.5],
            f"{unit.modality}_metric_sd": [1.2],
        }
        df = pd.DataFrame(data)
        df.to_csv(mmwide_file, index=False)
        return df

    # 1. Native Execution via run_study API
    with patch("antsxmm.modalities.dispatch.initialize_session_context", return_value=mock_ctx), \
         patch("antsxmm.modalities.dispatch.execute_unit", side_effect=fake_execute_unit) as mock_exec:

        failures = run_study(
            bids_dir=str(mock_bids_structure),
            output_dir=str(output_dir),
            project=project,
            native_execution=True,
            verbose=True,
        )

        assert failures == []
        assert mock_exec.call_count > 0

    # 2. Verify Output Directory Hierarchy & Manifests
    ses_dir = output_dir / project / "sub-001" / "ses-20230101"
    assert ses_dir.exists()

    status_file = ses_dir / ".antsxmm_status.json"
    assert status_file.exists()
    status_data = json.loads(status_file.read_text(encoding="utf-8"))
    assert status_data["success"] is True
    assert status_data["execution_engine"] == "antsxmm_native"
    assert status_data["args"]["native_execution"] is True

    manifest_file = ses_dir / f"{project}+sub-001+ses-20230101+mm_inputs.json"
    assert manifest_file.exists()

    diag_file = ses_dir / f"{project}+sub-001+ses-20230101+input_diagnostics.json"
    assert diag_file.exists()

    merged_table = ses_dir / "T1wHierarchical" / "run-01" / f"{project}+sub-001+ses-20230101+T1wHierarchical+run-01+mmwidemerged.csv"
    assert merged_table.exists()

    # 3. Test Resumability Guarantees: Re-running skips without doing work
    with patch("antsxmm.modalities.dispatch.initialize_session_context") as mock_init, \
         patch("antsxmm.modalities.dispatch.execute_unit") as mock_exec_resume:

        failures_resume = run_study(
            bids_dir=str(mock_bids_structure),
            output_dir=str(output_dir),
            project=project,
            resume=True,
            native_execution=True,
        )
        assert failures_resume == []
        assert mock_init.call_count == 0
        assert mock_exec_resume.call_count == 0

    # 4. CLI Subcommand: antsxmm tree
    # Construct a BIDS subject path for tree command: <bids>/<project>/<subject>
    bids_sub_dir = mock_bids_structure / "sub-001"
    # To satisfy len(parts) >= 3 for BIDS tree prediction:
    project_bids_dir = tmp_path / "BIDS" / project
    project_bids_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copytree(bids_sub_dir, project_bids_dir / "sub-001")

    res_tree = runner.invoke(main, ["tree", str(project_bids_dir / "sub-001")])
    assert res_tree.exit_code == 0
    assert "pymm/" in res_tree.output
    assert "T1wHierarchical" in res_tree.output

    # 5. CLI Subcommand: antsxmm validate
    report_json = tmp_path / "validation_report.json"
    res_val = runner.invoke(main, ["validate", str(project_bids_dir), str(output_dir), "--report-json", str(report_json)])
    assert res_val.exit_code == 0
    assert report_json.exists()

    # 6. CLI Subcommand: antsxmm aggregate
    study_csv = tmp_path / "study_aggregated.csv"
    res_agg = runner.invoke(main, ["aggregate", str(output_dir), "--output", str(study_csv)])
    assert res_agg.exit_code == 0
    assert study_csv.exists()
    agg_df = pd.read_csv(study_csv)
    assert len(agg_df) >= 1
    assert "subjectID" in agg_df.columns or any("metric" in col for col in agg_df.columns)
