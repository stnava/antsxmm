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


def test_registration_module_exports():
    """Verify antsxmm.registration exports all expected registration functions."""
    import antsxmm.registration as reg

    assert callable(reg.timeseries_reg)
    assert callable(reg.mc_reg)
    assert callable(reg.dti_reg)
    assert callable(reg.dti_template)
    assert callable(reg.dewarp_imageset)
    assert callable(reg.get_average_dwi_b0)
    assert callable(reg.get_average_rsf)
    assert callable(reg.timeseries_transform)
    assert callable(reg.tra_initializer)
    assert callable(reg.bvec_reorientation)
    assert callable(reg.apply_transforms_mixed_interpolation)

    assert hasattr(antsxmm, "registration")
    assert antsxmm.registration.timeseries_reg is reg.timeseries_reg


def test_segmentation_module_exports():
    """Verify antsxmm.segmentation exports all expected segmentation functions."""
    import antsxmm.segmentation as seg

    assert callable(seg.wmh)
    assert callable(seg.boot_wmh)
    assert callable(seg.trim_dti_mask)
    assert callable(seg.crop_mcimage)
    assert callable(seg.warn_if_small_mask)
    assert callable(seg.segment_timeseries_by_meanvalue)
    assert callable(seg.segment_timeseries_by_bvalue)
    assert callable(seg.map_scalar_to_labels)
    assert callable(seg.enantiomorphic_filling_without_mask)

    assert hasattr(antsxmm, "segmentation")
    assert antsxmm.segmentation.wmh is seg.wmh


def test_get_average_rsf():
    """Test get_average_rsf computes temporal mean in range [min_t, max_t]."""
    from antsxmm.registration import get_average_rsf

    shape = (4, 4, 4, 10)
    arr = np.arange(10, dtype=np.float32).reshape(1, 1, 1, 10)
    arr = np.broadcast_to(arr, shape).copy()
    img4d = ants.from_numpy(arr)

    avg_img = get_average_rsf(img4d, min_t=2, max_t=6)
    assert avg_img.shape == (4, 4, 4)
    # Mean of [2, 3, 4, 5] is 3.5
    assert np.allclose(avg_img.numpy(), 3.5)


def test_map_scalar_to_labels():
    """Test mapping dataframe scalar column onto anatomical integer labels."""
    from antsxmm.segmentation import map_scalar_to_labels

    label_np = np.zeros((6, 6, 6), dtype=np.float32)
    label_np[1:3, 1:3, 1:3] = 1
    label_np[3:5, 3:5, 3:5] = 2
    label_img = ants.from_numpy(label_np)

    df = pd.DataFrame({"label": [1, 2], "scalar_value": [42.0, 99.0]})
    mapped = map_scalar_to_labels(df, label_img)

    mapped_np = mapped.numpy()
    assert np.allclose(mapped_np[1:3, 1:3, 1:3], 42.0)
    assert np.allclose(mapped_np[3:5, 3:5, 3:5], 99.0)
    assert np.allclose(mapped_np[0, 0, 0], 0.0)


def test_warn_if_small_mask():
    """Test warn_if_small_mask detects low-fraction masks."""
    from antsxmm.segmentation import warn_if_small_mask

    mask_np = np.zeros((10, 10, 10), dtype=np.float32)
    mask_np[0, 0, 0] = 1.0  # 1 / 1000 = 0.001 < 0.05
    mask_img = ants.from_numpy(mask_np)

    with pytest.warns(UserWarning, match="Small mask detected"):
        warn_if_small_mask(mask_img, threshold_fraction=0.05, label="TestMask")


def test_io_serialization(tmp_path):
    """Test serializing modality outputs to disk."""
    from antsxmm.modalities.io import (
        ensure_parent_dir,
        write_flair_outputs,
        write_modality_mmwide,
    )

    out_prefix = str(tmp_path / "test_proj" / "sub-01" / "ses-01" / "T2Flair" / "run-01" / "test_proj+sub-01+ses-01+T2Flair+run-01")
    ensure_parent_dir(out_prefix)

    # Test writing raw mmwide
    df = pd.DataFrame({"col_a": [1.0], "col_b": [2.0]})
    csv_path = write_modality_mmwide(out_prefix, df, separator="+")
    assert os.path.exists(csv_path)
    loaded_df = pd.read_csv(csv_path)
    assert list(loaded_df.columns) == ["col_a", "col_b"]

    # Test write_flair_outputs
    flair_mock = {
        "wmh_mass": 123.45,
        "wmh_SNR": 10.2,
        "WMH_probability_map": ants.from_numpy(np.zeros((4, 4, 4), dtype=np.float32)),
    }
    flwide = write_flair_outputs(out_prefix, flair_mock, separator="+", visualize=False)
    assert os.path.exists(csv_path)
    assert os.path.exists(f"{out_prefix}+wmh.nii.gz")
    assert "wmh_mass" in flwide.columns


def test_dispatch_bids_sidecars(tmp_path):
    """Test resolving .bval and .bvec files alongside NIfTI image."""
    from antsxmm.modalities.dispatch import resolve_bids_sidecars

    dwi_nii = tmp_path / "sub-01_ses-01_dwi.nii.gz"
    dwi_bval = tmp_path / "sub-01_ses-01_dwi.bval"
    dwi_bvec = tmp_path / "sub-01_ses-01_dwi.bvec"

    dwi_nii.touch()
    bval_out, bvec_out = resolve_bids_sidecars(str(dwi_nii))
    assert bval_out is None
    assert bvec_out is None

    dwi_bval.touch()
    dwi_bvec.touch()
    bval_out, bvec_out = resolve_bids_sidecars(str(dwi_nii))
    assert bval_out == str(dwi_bval)
    assert bvec_out == str(dwi_bvec)


def test_dispatch_execute_unit_mock(tmp_path):
    """Test dispatching an execution unit with mocked modality."""
    from antsxmm.execution_plan import ExecutionUnit
    from antsxmm.modalities.dispatch import SessionContext, execute_unit

    out_prefix = str(tmp_path / "out" / "proj" / "sub-01" / "ses-01" / "T2Flair" / "run-01" / "proj+sub-01+ses-01+T2Flair+run-01")
    flair_path = str(tmp_path / "sub-01_ses-01_flair.nii.gz")

    ants.image_write(ants.from_numpy(np.ones((4, 4, 4), dtype=np.float32)), flair_path)
    t1_img = ants.from_numpy(np.ones((4, 4, 4), dtype=np.float32))

    context = SessionContext(
        project_id="proj",
        subject_id="sub-01",
        session_id="ses-01",
        t1_image=t1_img,
        t1_path="t1.nii.gz",
        hier={"dkt_parc": {"tissue_segmentation": t1_img}},
        hier_dir=str(tmp_path / "hier"),
        separator="+",
    )

    unit = ExecutionUnit(
        project_id="proj",
        subject="sub-01",
        session="ses-01",
        modality="T2Flair",
        run="run-01",
        input_paths=(flair_path,),
        output_prefix=out_prefix,
    )

    # If file already exists, it skips execution
    mmwide_file = f"{out_prefix}+mmwide.csv"
    os.makedirs(os.path.dirname(mmwide_file), exist_ok=True)
    pd.DataFrame({"test_col": [123]}).to_csv(mmwide_file, index=False)

    res = execute_unit(unit, context, verbose=False)
    assert isinstance(res, pd.DataFrame)
    assert res["test_col"].iloc[0] == 123



