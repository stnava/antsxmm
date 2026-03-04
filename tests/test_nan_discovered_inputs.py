import math
from pathlib import Path

import numpy as np

from antsxmm.core import _as_path_list, _collect_discovered_inputs


def test_as_path_list_handles_nan_and_scalars():
    assert _as_path_list(None) == []
    assert _as_path_list(float('nan')) == []
    assert _as_path_list(np.nan) == []
    assert _as_path_list(1.0) == []
    assert _as_path_list(True) == []
    assert _as_path_list('x.nii.gz') == ['x.nii.gz']
    assert _as_path_list(['a.nii.gz', np.nan, None]) == ['a.nii.gz']


def test_collect_discovered_inputs_does_not_iterate_over_nan(tmp_path: Path):
    # create a real nifti-like file so os.path.exists passes
    t1 = tmp_path / 'sub-1_ses-1_T1w.nii.gz'
    t1.write_bytes(b'')

    session_data = {
        't1_filenames': [str(t1)],
        'flair_filenames': np.nan,
        't2w_filenames': float('nan'),
        'dti_filenames': np.nan,
        'rsf_filenames': np.nan,
        'nm_filenames': np.nan,
        'perf_filenames': np.nan,
        'pet3d_filenames': np.nan,
    }

    discovered = _collect_discovered_inputs(session_data)
    assert discovered['t1_filenames'] == [str(t1)]
    # all the NaN fields normalize to empty lists
    assert discovered['perf_filenames'] == []
    assert discovered['pet3d_filenames'] == []
