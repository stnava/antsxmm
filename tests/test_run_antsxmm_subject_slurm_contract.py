from pathlib import Path


def test_run_antsxmm_subject_slurm_passes_project_and_subject_filters():
    script = (Path(__file__).resolve().parents[1] / 'run_antsxmm_subject.slurm').read_text(encoding='utf-8')
    assert 'antsxmm run' in script
    assert '--project "$PROJECT_ID"' in script
    assert '--participant-label "$SUBJECT_LABEL"' in script
    assert 'PROJECT_ID="$(basename_no_trailing "$BIDS_PROJECT_DIR")"' in script


def test_run_antsxmm_subject_slurm_accepts_subject_dir_and_derives_project_dir():
    script = (Path(__file__).resolve().parents[1] / 'run_antsxmm_subject.slurm').read_text(encoding='utf-8')
    assert 'if [[ "$leaf" == sub-* ]]; then' in script
    assert 'BIDS_PROJECT_DIR="$(cd "$(dirname "$INPUT_PATH")" && pwd)"' in script
    assert ': "${SUBJECT_LABEL:=$leaf}"' in script
