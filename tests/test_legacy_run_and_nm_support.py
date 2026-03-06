from click.testing import CliRunner

from antsxmm.bids import parse_antsxbids_layout
from antsxmm.bids_entities import parse_entities
from antsxmm.execution_plan import modality_from_path
from antsxmm.pipeline import main
from antsxmm.run_id import normalize_run_id


def test_parse_entities_accepts_legacy_r_token():
    ent = parse_entities('sub-182341_ses-20230111_r0001_NM.nii.gz')
    assert ent['sub'] == '182341'
    assert ent['ses'] == '20230111'
    assert ent['run'] == '0001'
    assert ent['suffix'] == 'NM'
    assert normalize_run_id(ent['run']) == 'run-01'


def test_parse_layout_finds_nm_in_anat(tmp_path):
    ses = tmp_path / 'BIDS' / 'PPMI' / 'sub-182341' / 'ses-20230111'
    anat = ses / 'anat'
    anat.mkdir(parents=True)
    (anat / 'sub-182341_ses-20230111_r0001_T1w.nii.gz').touch()
    (anat / 'sub-182341_ses-20230111_r0001_NM.nii.gz').touch()

    df = parse_antsxbids_layout(tmp_path / 'BIDS' / 'PPMI')
    assert len(df) == 1
    row = df.iloc[0].to_dict()
    assert row['subjectID'] == 'sub-182341'
    assert any(p.endswith('_NM.nii.gz') for p in row['nm_filenames'])


def test_tree_accepts_legacy_run_variant_for_nm_dwi_and_bold(tmp_path):
    ses = tmp_path / 'BIDS' / 'PPMI' / 'sub-182341' / 'ses-20230111'
    anat = ses / 'anat'
    dwi = ses / 'dwi'
    func = ses / 'func'
    anat.mkdir(parents=True)
    dwi.mkdir()
    func.mkdir()

    (anat / 'sub-182341_ses-20230111_r0001_T1w.nii.gz').touch()
    (anat / 'sub-182341_ses-20230111_r0001_NM.nii.gz').touch()
    (dwi / 'sub-182341_ses-20230111_r0001_dir-LR-dwi.nii.gz').touch()
    (func / 'sub-182341_ses-20230111_r0001_task-rest-dir-RL-bold.nii.gz').touch()

    runner = CliRunner()
    result = runner.invoke(main, ['tree', str(tmp_path / 'BIDS' / 'PPMI' / 'sub-182341')])
    assert result.exit_code == 0
    out = result.output
    assert 'NM2DMT/' in out
    assert 'DTI/' in out
    assert 'rsfMRI/' in out
    assert 'run-01/' in out


def test_modality_mapping_nm_suffix_with_legacy_run_token():
    assert modality_from_path('sub-182341_ses-20230111_r0001_NM.nii.gz') == 'NM2DMT'


def test_run_accepts_participant_label_with_trailing_slash(mock_bids_structure, tmp_path):
    outdir = tmp_path / 'OUT'
    runner = CliRunner()
    result = runner.invoke(
        main,
        ['run', str(mock_bids_structure), str(outdir), '--dry-run', '--participant-label', 'sub-001/'],
    )
    assert result.exit_code == 0
    assert 'Filtering for subject: sub-001' in result.output
    assert 'No valid subjects/sessions found' not in result.output
