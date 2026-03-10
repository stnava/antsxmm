from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from antsxmm.pipeline import main



def _write_merged_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)



def test_aggregate_command_merges_and_prefers_processed(tmp_path):
    root = tmp_path / "study"
    pymm_file = root / "pymm" / "breacher" / "sub-0102" / "ses-initial-day1" / "T1wHierarchical" / "run-01" / "breacher+sub-0102+ses-initial-day1+T1wHierarchical+run-01+mmwidemerged.csv"
    processed_file = root / "Processed" / "breacher" / "sub-0102" / "ses-initial-day1" / "T1wHierarchical" / "run-01" / "breacher+sub-0102+ses-initial-day1+T1wHierarchical+run-01+mmwidemerged.csv"
    second_file = root / "pymm" / "SOCOM" / "sub-Blast-01" / "ses-01" / "T1wHierarchical" / "run-01" / "SOCOM+sub-Blast-01+ses-01+T1wHierarchical+run-01+mmwidemerged.csv"

    _write_merged_csv(pymm_file, [{"metric": 1.0, "score": 10}])
    _write_merged_csv(processed_file, [{"metric": 2.0, "score": 20}])
    _write_merged_csv(second_file, [{"metric": 3.0, "score": 30}])

    out = tmp_path / "aggregate.csv"
    runner = CliRunner()
    result = runner.invoke(main, ["aggregate", str(root), "--output", str(out)])

    assert result.exit_code == 0, result.output
    df = pd.read_csv(out)
    assert df.shape[0] == 2
    assert sorted(df["entity_id"].tolist()) == [
        "SOCOM|sub-Blast-01|ses-01|T1wHierarchical|run-01",
        "breacher|sub-0102|ses-initial-day1|T1wHierarchical|run-01",
    ]
    breacher_row = df[df["project_id"] == "breacher"].iloc[0]
    assert breacher_row["source_root"] == "Processed"
    assert float(breacher_row["metric"]) == 2.0
    assert "aggregate scanned=3" in result.output



def test_aggregate_incremental_reads_only_new_or_changed_entities(tmp_path):
    root = tmp_path / "study"
    first = root / "pymm" / "breacher" / "sub-0102" / "ses-initial-day1" / "T1wHierarchical" / "run-01" / "breacher+sub-0102+ses-initial-day1+T1wHierarchical+run-01+mmwidemerged.csv"
    second = root / "pymm" / "SOCOM" / "sub-Blast-01" / "ses-01" / "T1wHierarchical" / "run-01" / "SOCOM+sub-Blast-01+ses-01+T1wHierarchical+run-01+mmwidemerged.csv"
    third = root / "Processed" / "SOCOM" / "sub-Blast-02" / "ses-01" / "T1wHierarchical" / "run-01" / "SOCOM+sub-Blast-02+ses-01+T1wHierarchical+run-01+mmwidemerged.csv"

    _write_merged_csv(first, [{"metric": 1.0}])
    _write_merged_csv(second, [{"metric": 3.0}])

    out = tmp_path / "aggregate.csv"
    state = tmp_path / "aggregate.state.json"
    runner = CliRunner()

    first_run = runner.invoke(main, ["aggregate", str(root), "--output", str(out), "--state", str(state)])
    assert first_run.exit_code == 0, first_run.output
    assert "read=2" in first_run.output

    _write_merged_csv(third, [{"metric": 9.0}])
    second_run = runner.invoke(main, ["aggregate", str(root), "--output", str(out), "--state", str(state)])
    assert second_run.exit_code == 0, second_run.output
    assert "read=1" in second_run.output
    assert "reused_existing=yes" not in second_run.output

    df = pd.read_csv(out)
    assert df.shape[0] == 3
    assert set(df["project_id"].tolist()) == {"breacher", "SOCOM"}
    assert "SOCOM|sub-Blast-02|ses-01|T1wHierarchical|run-01" in set(df["entity_id"].tolist())


def test_aggregate_command_accepts_flat_wide_dropzone_and_duplicate_copy_names(tmp_path):
    root = tmp_path / "study"
    wide_dir = root / "pymms" / "wides"
    first = wide_dir / "breacher+sub-2221+ses-initial-day1+T1wHierarchical+run-01+mmwidemerged.csv"
    duplicate_copy = wide_dir / "breacher+sub-2221+ses-initial-day1+T1wHierarchical+run-01+mmwidemerged 2.csv"
    second = wide_dir / "ExpArt+sub-EAS009+ses-01+T1wHierarchical+run-01+mmwidemerged.csv"

    _write_merged_csv(first, [{"metric": 1.0}])
    _write_merged_csv(duplicate_copy, [{"metric": 9.0}])
    _write_merged_csv(second, [{"metric": 3.0}])

    out = tmp_path / "aggregate.csv"
    runner = CliRunner()
    result = runner.invoke(main, ["aggregate", str(root), "--output", str(out), "--prefer", "newest"])

    assert result.exit_code == 0, result.output
    df = pd.read_csv(out)
    assert df.shape[0] == 2
    assert set(df["entity_id"].tolist()) == {
        "breacher|sub-2221|ses-initial-day1|T1wHierarchical|run-01",
        "ExpArt|sub-EAS009|ses-01|T1wHierarchical|run-01",
    }
    breacher_row = df[df["project_id"] == "breacher"].iloc[0]
    assert float(breacher_row["metric"]) == 9.0



def test_aggregate_command_accommodates_mixed_flat_and_nested_inputs(tmp_path):
    root = tmp_path / "study"
    flat = root / "pymms" / "wides" / "FPA+sub-BLAST023+ses-01+T1wHierarchical+run-01+mmwidemerged.csv"
    nested = root / "Processed" / "breacher" / "sub-4178" / "ses-initial-day1" / "T1wHierarchical" / "run-01" / "breacher+sub-4178+ses-initial-day1+T1wHierarchical+run-01+mmwidemerged.csv"

    _write_merged_csv(flat, [{"metric": 4.0}])
    _write_merged_csv(nested, [{"metric": 7.0}])

    out = tmp_path / "aggregate.csv"
    runner = CliRunner()
    result = runner.invoke(main, ["aggregate", str(root), "--output", str(out)])

    assert result.exit_code == 0, result.output
    df = pd.read_csv(out)
    assert df.shape[0] == 2
    assert set(df["entity_id"].tolist()) == {
        "FPA|sub-BLAST023|ses-01|T1wHierarchical|run-01",
        "breacher|sub-4178|ses-initial-day1|T1wHierarchical|run-01",
    }
    nested_row = df[df["project_id"] == "breacher"].iloc[0]
    flat_row = df[df["project_id"] == "FPA"].iloc[0]
    assert nested_row["source_root"] == "Processed"
    assert flat_row["source_root"] == "pymms"
