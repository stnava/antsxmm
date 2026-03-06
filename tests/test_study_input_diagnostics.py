import json
from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from antsxmm.diagnostics import diagnose_bids_tree
from antsxmm.pipeline import main, run_study


def test_diagnose_bids_tree_detects_broken_symlink(tmp_path):
    anat = tmp_path / 'BIDS' / 'ExpArt' / 'sub-FAIL01' / 'ses-01' / 'anat'
    anat.mkdir(parents=True)
    (anat / 'sub-FAIL01_ses-01_T1w.nii.gz').symlink_to('/does/not/exist/sub-FAIL01_ses-01_T1w.nii.gz')
    (anat / 'sub-FAIL01_ses-01_T1w.json').write_text('{}')

    diag = diagnose_bids_tree(anat.parents[3])

    assert diag['counts']['broken_symlinks'] == 1
    assert diag['counts']['json_without_image'] == 1
    assert 'broken_symlink' in diag['suspicious_reasons']
    assert 'json_without_image' in diag['suspicious_reasons']


def test_run_study_empty_layout_schema_does_not_crash_and_writes_diagnostics(tmp_path, capsys, monkeypatch):
    bids_dir = tmp_path / 'BIDS' / 'ExpArt'
    bids_dir.mkdir(parents=True)
    anat = bids_dir / 'sub-FAIL01' / 'ses-01' / 'anat'
    anat.mkdir(parents=True)
    (anat / 'sub-FAIL01_ses-01_T1w.nii.gz').symlink_to('/does/not/exist/sub-FAIL01_ses-01_T1w.nii.gz')
    (anat / 'sub-FAIL01_ses-01_T1w.json').write_text('{}')

    monkeypatch.setattr('antsxmm.pipeline.parse_antsxbids_layout', lambda _: pd.DataFrame())

    failures = run_study(str(bids_dir), str(tmp_path / 'out'), 'ExpArt', participant_label='sub-FAIL01')
    assert failures == []

    out = capsys.readouterr().out
    assert 'No usable BIDS sessions were discovered' in out
    assert 'broken_symlink' in out

    diag_path = tmp_path / 'out' / 'ExpArt' / '.antsxmm_study_input_diagnostics.json'
    assert diag_path.exists()
    payload = json.loads(diag_path.read_text())
    assert payload['counts']['broken_symlinks'] == 1
    assert payload['requested_filters']['participant_label'] == 'sub-FAIL01'


def test_run_study_wrong_participant_filter_reports_diagnostics(tmp_path, capsys, monkeypatch):
    bids_dir = tmp_path / 'BIDS' / 'ExpArt'
    anat = bids_dir / 'sub-REAL01' / 'ses-01' / 'anat'
    anat.mkdir(parents=True)
    (anat / 'sub-REAL01_ses-01_T1w.nii.gz').write_text('')
    (anat / 'sub-REAL01_ses-01_T1w.json').write_text('{}')

    df = pd.DataFrame([{'subjectID': 'sub-REAL01', 'date': 'ses-01', 't1_filenames': [str(anat / 'sub-REAL01_ses-01_T1w.nii.gz')]}])
    monkeypatch.setattr('antsxmm.pipeline.parse_antsxbids_layout', lambda _: df)

    failures = run_study(str(bids_dir), str(tmp_path / 'out'), 'ExpArt', participant_label='sub-WRONG01')
    assert failures == []
    out = capsys.readouterr().out
    assert 'requested participant=sub-WRONG01' in out
    assert 'No usable BIDS sessions were discovered' in out


def test_cli_broken_symlink_dataset_reports_diagnostics_without_traceback(tmp_path):
    bids_dir = tmp_path / 'BIDS' / 'ExpArt'
    anat = bids_dir / 'sub-FAIL01' / 'ses-01' / 'anat'
    anat.mkdir(parents=True)
    (anat / 'sub-FAIL01_ses-01_T1w.nii.gz').symlink_to('/does/not/exist/sub-FAIL01_ses-01_T1w.nii.gz')
    (anat / 'sub-FAIL01_ses-01_T1w.json').write_text('{}')

    runner = CliRunner()
    result = runner.invoke(main, ['run', '--project', 'ExpArt', str(bids_dir), str(tmp_path / 'out'), '--participant-label', 'sub-FAIL01'])

    assert result.exit_code == 0
    assert 'No usable BIDS sessions were discovered' in result.output
    assert 'broken_symlink' in result.output
    assert 'KeyError: \"subjectID\"' not in result.output
