from __future__ import annotations

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import ants

import antsxmm
from antsxmm.modalities import (
    alffmap,
    ants_to_nibabel_affine,
    augment_image,
    calculate_CBF,
    compute_PerAF_voxel,
    convert_np_in_dict,
    crop_mcimage,
    despike_time_series,
    despike_time_series_afni,
    dict_to_dataframe,
    distortion_correct_bvecs,
    down2iso,
    dti_numpy_to_image,
    dvars,
    get_antsimage_keys,
    get_data,
    repair_bvecs,
    segment_timeseries_by_bvalue,
    segment_timeseries_by_meanvalue,
    shorten_pymm_names,
    shorten_pymm_names2,
    slice_snr,
    spec_pgram,
    triangular_to_tensor,
    tsnr,
    write_bvals_bvecs,
)


def test_modalities_reexport_on_package():
    """Verify antsxmm.modalities is accessible through package root."""
    assert hasattr(antsxmm, "modalities")
    assert callable(antsxmm.modalities.get_data)
    assert callable(antsxmm.modalities.joint_dti_recon)
    assert callable(antsxmm.modalities.resting_state_fmri_networks)
    assert callable(antsxmm.modalities.neuromelanin)
    assert callable(antsxmm.modalities.bold_perfusion)
    assert callable(antsxmm.modalities.wmh)


def test_metrics_shorten_names():
    """Test shortening functions for long neuroimaging labels."""
    short1 = shorten_pymm_names("cit168_description_left_superior_temporal_gyrus")
    assert isinstance(short1, str)
    assert len(short1) <= 18

    short2 = shorten_pymm_names2("anterior.limb.of.internal.capsule")
    assert "alintcap" in short2


def test_metrics_dict_to_dataframe():
    """Test converting nested/scalar dictionaries to DataFrames."""
    data = {
        "scalar_int": 42,
        "scalar_float": 3.14,
        "scalar_str": "test",
        "array_data": np.array([1.0, 2.0, 3.0]),
        "list_data": [10.0, 20.0, 30.0],
    }
    df = dict_to_dataframe(data)
    assert isinstance(df, pd.DataFrame)
    assert "scalar_int" in df.columns
    assert "array_data_mean" in df.columns
    assert df["array_data_mean"].iloc[0] == pytest.approx(2.0)
    assert df["list_data_mean"].iloc[0] == pytest.approx(20.0)


def test_metrics_convert_np_in_dict():
    """Test converting numpy numbers to standard python types in dictionaries."""
    data = {
        "a": np.float32(1.5),
        "b": np.int64(42),
        "c": "already_str",
    }
    converted = convert_np_in_dict(data)
    assert type(converted["a"]) is float
    assert type(converted["b"]) is int
    assert converted["c"] == "already_str"


def test_segment_timeseries_by_mean_and_bvalue():
    """Test time-series segmentation by mean and by b-values."""
    # Synthetic 4D image with 4 volumes: 2 bright (B0) and 2 dark (DWI)
    arr = np.ones((8, 8, 8, 4), dtype=np.float32)
    arr[..., 0] *= 100.0
    arr[..., 1] *= 100.0
    arr[..., 2] *= 10.0
    arr[..., 3] *= 10.0
    img4d = ants.from_numpy(arr)

    seg_means = segment_timeseries_by_meanvalue(img4d)
    assert set(seg_means["highermeans"]) == {0, 1}
    assert set(seg_means["lowermeans"]) == {2, 3}

    bvals = np.array([0.0, 0.0, 1000.0, 1000.0])
    seg_bval = segment_timeseries_by_bvalue(bvals)
    assert seg_bval["lowbvals"] == [0, 1]
    assert seg_bval["largerbvals"] == [2, 3]


def test_tsnr_and_dvars():
    """Test temporal SNR and DVARS calculation on synthetic 4D image."""
    np.random.seed(42)
    data = np.random.normal(100, 5, size=(10, 10, 10, 10)).astype(np.float32)
    img4d = ants.from_numpy(data)
    mask = ants.from_numpy(np.ones((10, 10, 10), dtype=np.float32))

    tsnr_img = tsnr(img4d, mask)
    assert tsnr_img.dimension == 3
    assert tsnr_img.numpy().shape == (10, 10, 10)

    dvars_vals = dvars(img4d, mask)
    assert len(dvars_vals) == 10
    assert not np.any(np.isnan(dvars_vals))


def test_dti_triangular_tensor_conversions():
    """Test round-trip conversion between 6-component tensor and 3x3 matrix tensor."""
    shape = (4, 4, 4)
    ref = ants.from_numpy(np.zeros(shape, dtype=np.float32))
    tensors = np.zeros(shape + (3, 3), dtype=np.float64)

    # Set known symmetric tensors
    for idx in np.ndindex(shape):
        tensors[idx] = np.array([
            [1.0, 0.2, 0.3],
            [0.2, 2.0, 0.4],
            [0.3, 0.4, 3.0],
        ])

    dti_img = dti_numpy_to_image(ref, tensors, upper_triangular=True)
    assert dti_img.components == 6

    reconstructed = triangular_to_tensor(dti_img, upper_triangular=True)
    assert reconstructed.shape == shape + (3, 3)
    np.testing.assert_allclose(tensors, reconstructed, atol=1e-6)


def test_dti_bvec_repair_and_io(tmp_path):
    """Test bvec normalization and writing to disk."""
    bvecs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],  # Non-unit norm
        [0.0, 0.0, 1.0],
    ])
    bvals = np.array([0.0, 1000.0, 1000.0])

    repaired = repair_bvecs(bvecs)
    norms = np.linalg.norm(repaired, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0, 1.0], atol=1e-5)

    prefix = str(tmp_path / "test_dti")
    write_bvals_bvecs(bvals, repaired, prefix)
    assert os.path.exists(f"{prefix}.bval")
    assert os.path.exists(f"{prefix}.bvec")


def test_fmri_spectral_and_peraf():
    """Test periodogram, ALFF, and PerAF calculation."""
    # Sine wave with frequency 0.05 Hz sampled at 1 Hz for 100 seconds
    t = np.arange(100)
    signal_1d = np.sin(2 * np.pi * 0.05 * t) + 10.0

    pgram = spec_pgram(signal_1d, xfreq=1.0, plot=False)
    assert "freq" in pgram
    assert "spec" in pgram
    assert len(pgram["freq"]) > 0

    alff_res = alffmap(signal_1d, flo=0.01, fhi=0.1, tr=1.0)
    assert "alff" in alff_res
    assert "falff" in alff_res
    assert alff_res["alff"] > 0
    assert 0.0 <= alff_res["falff"] <= 1.0

    peraf_val = compute_PerAF_voxel(signal_1d)
    assert peraf_val > 0


def test_fmri_despike():
    """Test time series despiking."""
    data = np.ones((6, 6, 6, 30), dtype=np.float32) * 100.0
    # Add a huge spike at index 15
    data[3, 3, 3, 15] = 10000.0
    img4d = ants.from_numpy(data)

    despiked, spike_counts = despike_time_series(img4d, threshold=3.0, replacement="median")
    assert despiked.shape == (6, 6, 6, 30)
    assert spike_counts[15] >= 1
    assert despiked.numpy()[3, 3, 3, 15] == pytest.approx(100.0)


def test_calculate_cbf():
    """Test quantified CBF calculation with pCASL equation."""
    shape = (8, 8, 8)
    delta_m = ants.from_numpy(np.ones(shape, dtype=np.float32) * 5.0)
    m0 = ants.from_numpy(np.ones(shape, dtype=np.float32) * 1000.0)
    mask = ants.from_numpy(np.ones(shape, dtype=np.float32))

    cbf = calculate_CBF(delta_m, m0, mask)
    cbf_np = cbf.numpy()
    assert cbf_np.shape == shape
    assert np.all(cbf_np >= 0.0)
    assert np.mean(cbf_np) > 0.0


def test_down2iso():
    """Test resampling anisotropic image to isotropic resolution."""
    arr = np.zeros((10, 10, 5), dtype=np.float32)
    img = ants.from_numpy(arr, spacing=[1.0, 1.0, 2.0])
    iso = down2iso(img, takemin=True)
    spc = ants.get_spacing(iso)
    assert spc[0] == pytest.approx(1.0)
    assert spc[1] == pytest.approx(1.0)
    assert spc[2] == pytest.approx(1.0)
