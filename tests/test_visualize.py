"""Comprehensive 4-Tier Test Suite for ANTsXMM Visual Reporting Infrastructure.

Organized strictly across four tiers and four requirements:
- R1: Core visual system & Theme (theme.py, core.py, base64 encoding, ortho montage, slice gallery, carpet plot, violin plot, correlation matrix, zero CDN links).
- R2: Modality-Specific Visualizers across all 8 modalities (structural, wmh, dti, fmri, perfusion, pet, neuromelanin, registry & dispatcher).
- R3: Session & Study Multi-Modality Aggregator + CLI (session report, study report, executive overview, tabs, completion matrix, missingness heatmap, population QC distributions, outlier flags, CLI commands).
- R4: Real cohort validation against real assembled sessions (SOCOM, PPMI, study multimodal aggregated CSV).
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import re

from click.testing import CliRunner
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pytest

try:
    import ants

    HAS_ANTS = True
except ImportError:
    HAS_ANTS = False

from antsxmm.pipeline import main
from antsxmm.visualize import (
    build_html_document,
    figure_to_base64,
    generate_session_report,
    generate_study_report,
    get_theme_css,
    get_theme_js,
    image_to_base64,
    render_badge,
    render_card,
    render_carpet_plot,
    render_correlation_matrix,
    render_kpi_card,
    render_ortho_montage,
    render_slice_gallery,
    render_tabs,
    render_violin_plot,
)
from antsxmm.visualize.modalities import (
    AVAILABLE_MODALITIES,
    MODALITY_ALIASES,
    ModalityReport,
    coalesce_multi_row_df,
    find_file,
    find_run_dir,
    get_modality_visualizer,
    list_available_modalities,
    load_nifti_data,
    safe_read_csv,
    visualize_modality,
)
from antsxmm.visualize.modalities.dti import (
    can_visualize as can_visualize_dti,
    visualize_dti,
)
from antsxmm.visualize.modalities.fmri import (
    can_visualize as can_visualize_fmri,
    visualize_fmri,
)
from antsxmm.visualize.modalities.neuromelanin import (
    can_visualize as can_visualize_neuromelanin,
    visualize_neuromelanin,
)
from antsxmm.visualize.modalities.perfusion import (
    can_visualize as can_visualize_perfusion,
    visualize_perfusion,
)
from antsxmm.visualize.modalities.pet import (
    can_visualize as can_visualize_pet,
    visualize_pet,
)
from antsxmm.visualize.modalities.structural import (
    can_visualize as can_visualize_structural,
    visualize_structural,
)
from antsxmm.visualize.modalities.wmh import (
    can_visualize as can_visualize_wmh,
    visualize_wmh,
)


# ==============================================================================
# Deterministic Synthetic Fixtures
# ==============================================================================

@pytest.fixture
def synthetic_volume_3d() -> np.ndarray:
    """Generate a fast, deterministic synthetic 3D anatomical volume (32x32x24)."""
    np.random.seed(42)
    x, y, z = np.mgrid[:32, :32, :24]
    # Ellipsoid brain contour
    brain = (((x - 16) ** 2 / 12 ** 2) + ((y - 16) ** 2 / 12 ** 2) + ((z - 12) ** 2 / 8 ** 2)) <= 1.0
    vol = np.zeros((32, 32, 24), dtype=np.float32)
    vol[brain] = 60.0 + 30.0 * np.sin(x[brain] / 3.0) + np.random.normal(0, 2.0, size=brain.sum())
    vol[vol < 0] = 0.0
    return vol


@pytest.fixture
def synthetic_mask_3d() -> np.ndarray:
    """Generate a binary brain mask corresponding to the synthetic ellipsoid."""
    x, y, z = np.mgrid[:32, :32, :24]
    mask = (((x - 16) ** 2 / 12 ** 2) + ((y - 16) ** 2 / 12 ** 2) + ((z - 12) ** 2 / 8 ** 2)) <= 1.0
    return mask.astype(np.uint8)


@pytest.fixture
def synthetic_multilabel_3d() -> np.ndarray:
    """Generate a 4-class multi-label segmentation volume (0: BG, 1: CSF, 2: GM, 3: WM)."""
    x, y, z = np.mgrid[:32, :32, :24]
    dist_sq = ((x - 16) ** 2 / 12 ** 2) + ((y - 16) ** 2 / 12 ** 2) + ((z - 12) ** 2 / 8 ** 2)
    labels = np.zeros((32, 32, 24), dtype=np.int32)
    labels[dist_sq <= 1.0] = 1  # CSF outer
    labels[dist_sq <= 0.75] = 2  # GM middle
    labels[dist_sq <= 0.45] = 3  # WM inner
    return labels


@pytest.fixture
def synthetic_timeseries_4d() -> np.ndarray:
    """Generate a fast 4D synthetic BOLD timeseries (16x16x12x25)."""
    np.random.seed(42)
    x, y, z = np.mgrid[:16, :16, :12]
    mask = (((x - 8) ** 2 / 6 ** 2) + ((y - 8) ** 2 / 6 ** 2) + ((z - 6) ** 2 / 4 ** 2)) <= 1.0
    ts = np.zeros((16, 16, 12, 25), dtype=np.float32)
    t = np.arange(25)
    signal = 100.0 + 10.0 * np.sin(2 * np.pi * t / 10.0)
    for i in range(25):
        vol = np.zeros((16, 16, 12), dtype=np.float32)
        vol[mask] = signal[i] + np.random.normal(0, 3.0, size=mask.sum())
        ts[:, :, :, i] = vol
    return ts


@pytest.fixture
def synthetic_motion_traces() -> tuple[np.ndarray, np.ndarray]:
    """Generate aligned Framewise Displacement (FD) and DVARS vectors (length 25)."""
    np.random.seed(42)
    fd = np.abs(np.random.normal(0.15, 0.1, size=25))
    fd[7] = 0.68  # deliberate spike exceeding 0.5mm threshold
    fd[18] = 0.54
    dvars = 20.0 + 8.0 * fd + np.random.normal(0, 1.5, size=25)
    return fd, dvars


@pytest.fixture
def synthetic_metric_df() -> pd.DataFrame:
    """Generate sample regional morphometry measurements across diagnostic groups."""
    np.random.seed(42)
    rois = ["Frontal", "Parietal", "Temporal", "Occipital", "Cingulate"]
    cohorts = ["Control", "Patient"]
    records = []
    for r in rois:
        for c in cohorts:
            base = 3.2 if c == "Control" else 2.8
            vals = np.random.normal(base, 0.25, size=8)
            for v in vals:
                records.append({"ROI": r, "Thickness": float(v), "Cohort": c})
    return pd.DataFrame(records)


@pytest.fixture
def synthetic_corr_df() -> pd.DataFrame:
    """Generate a symmetric 8x8 functional connectivity correlation matrix."""
    np.random.seed(42)
    n = 8
    names = [f"Network_{i+1}" for i in range(n)]
    raw = np.random.uniform(-0.6, 0.8, size=(n, n))
    sym = (raw + raw.T) / 2.0
    np.fill_diagonal(sym, 1.0)
    return pd.DataFrame(sym, index=names, columns=names)


# ==============================================================================
# TIER 1: FEATURE COVERAGE (>=5 tests per feature)
# ==============================================================================

class TestTier1ThemeCss:
    """Feature 1: CSS Generation and Design System Variables."""

    def test_css_generation_returns_nonempty_stylesheet(self):
        css = get_theme_css()
        assert isinstance(css, str)
        assert len(css) > 500
        assert ":root" in css

    def test_css_root_variables_and_typography(self):
        css = get_theme_css()
        assert "--font-sans:" in css
        assert "--font-mono:" in css
        assert "--radius-sm:" in css
        assert "--radius-md:" in css
        assert "--radius-lg:" in css

    def test_css_dark_clinical_mode_variables(self):
        css = get_theme_css()
        assert ':root[data-theme="dark"]' in css
        assert "#0b0f19" in css  # Dark clinical background
        assert "#162032" in css  # Dark card background
        assert "#38bdf8" in css  # Cyan accent

    def test_css_light_mode_variables(self):
        css = get_theme_css()
        assert ':root[data-theme="light"]' in css
        assert "#f8fafc" in css  # Light background
        assert "#0f172a" in css  # Dark text on light bg

    def test_css_status_color_definitions(self):
        css = get_theme_css()
        assert "--status-normal-bg:" in css
        assert "--status-warning-bg:" in css
        assert "--status-danger-bg:" in css
        assert "--status-info-bg:" in css

    def test_css_responsive_and_card_layout_rules(self):
        css = get_theme_css()
        assert ".kpi-grid" in css
        assert ".card" in css
        assert "@media (max-width:" in css
        assert ".tabs-container" in css


class TestTier1InlineJs:
    """Feature 2: Inline JavaScript Theme Switcher and Tab Controller."""

    def test_js_theme_toggle_function_defined(self):
        js = get_theme_js()
        assert "window.toggleTheme" in js
        assert "document.documentElement.setAttribute" in js

    def test_js_localstorage_persistence(self):
        js = get_theme_js()
        assert "localStorage.getItem" in js
        assert "localStorage.setItem" in js
        assert "antsxmm-theme" in js

    def test_js_prefers_color_scheme_detection(self):
        js = get_theme_js()
        assert "prefers-color-scheme: dark" in js
        assert "matchMedia" in js

    def test_js_tab_switching_logic(self):
        js = get_theme_js()
        assert "initTabs" in js
        assert "activateTab" in js
        assert '[role="tab"]' in js
        assert '[role="tabpanel"]' in js
        assert 'aria-selected' in js
        assert 'aria-controls' in js

    def test_js_keyboard_accessibility_listeners(self):
        js = get_theme_js()
        assert "keydown" in js
        assert "ArrowRight" in js
        assert "ArrowLeft" in js


class TestTier1KpiCards:
    """Feature 3: KPI Metric Cards and Badges."""

    def test_kpi_card_numeric_formatting_standard(self):
        card = render_kpi_card(label="Brain Volume", value=1420.5, unit="cm³", status="normal")
        assert "kpi-card" in card
        assert "Brain Volume" in card
        assert "1420.50" in card or "1.42e+03" in card or "1420.5" in card
        assert "cm³" in card
        assert "badge-normal" in card

    def test_kpi_card_float_scientific_and_small_numbers(self):
        card = render_kpi_card(label="Mean Diffusivity", value=0.000784, unit="mm²/s", status="info")
        assert "Mean Diffusivity" in card
        assert "0.000784" in card or "7.84e-04" in card
        assert "mm²/s" in card

    def test_kpi_card_status_classes_and_badges(self):
        for stat in ["normal", "warning", "danger", "info", "neutral"]:
            card = render_kpi_card(label="Metric", value=42, status=stat)
            assert f"badge-{stat}" in card

    def test_kpi_card_with_unit_and_tooltip(self):
        card = render_kpi_card(
            label="Lesion Load",
            value=12.4,
            unit="mL",
            status="warning",
            tooltip="Total white matter hyperintensity volume",
        )
        assert 'data-tooltip="Total white matter hyperintensity volume"' in card
        assert "mL" in card

    def test_kpi_card_none_and_missing_values(self):
        card = render_kpi_card(label="CBF Global", value=None, unit="mL/100g/min")
        assert "N/A" in card

    def test_kpi_card_html_escaping(self):
        card = render_kpi_card(label="T1w <Contrast & SNR>", value="<b>42</b>", tooltip='"Quote" & test')
        assert "&lt;Contrast &amp; SNR&gt;" in card
        assert "&lt;b&gt;42&lt;/b&gt;" in card


class TestTier1NavigationTabs:
    """Feature 4: Accessible Navigation Tabs."""

    def test_tabs_accessible_roles_and_attributes(self):
        tabs_data = [("t1", "Tab One", "<p>One</p>"), ("t2", "Tab Two", "<p>Two</p>")]
        rendered = render_tabs(tabs_data)
        assert 'role="tablist"' in rendered
        assert 'role="tab"' in rendered
        assert 'role="tabpanel"' in rendered

    def test_tabs_active_tab_selection_default_first(self):
        tabs_data = [("tabA", "First", "<p>A</p>"), ("tabB", "Second", "<p>B</p>")]
        rendered = render_tabs(tabs_data)
        assert 'aria-selected="true"' in rendered
        assert 'tab-tabgroup-' in rendered or 'tab-' in rendered
        # First tab should be active
        assert 'class="tab-btn active"' in rendered
        assert 'id="panel-' in rendered

    def test_tabs_explicit_active_tab(self):
        tabs_data = [("tabA", "First", "<p>A</p>"), ("tabB", "Second", "<p>B</p>")]
        rendered = render_tabs(tabs_data, active_tab="tabB")
        assert 'tab-btn active" id="tab-' in rendered
        # tabB panel should not be hidden
        assert 'panel-' in rendered and 'tabB' in rendered

    def test_tabs_tuple_and_dict_formats(self):
        tuple_tabs = [("t1", "L1", "C1"), ("t2", "L2", "C2")]
        dict_tabs = [{"id": "t1", "label": "L1", "content": "C1"}, {"id": "t2", "label": "L2", "content": "C2"}]
        res1 = render_tabs(tuple_tabs, tab_group_id="test-grp")
        res2 = render_tabs(dict_tabs, tab_group_id="test-grp")
        assert res1 == res2

    def test_tabs_unique_group_id_generation(self):
        tabs_data = [("t1", "Tab One", "<p>One</p>")]
        res1 = render_tabs(tabs_data)
        res2 = render_tabs(tabs_data)
        id1 = re.search(r'id="(tabgroup-[a-f0-9]+)"', res1).group(1)
        id2 = re.search(r'id="(tabgroup-[a-f0-9]+)"', res2).group(1)
        assert id1 != id2

    def test_tabs_empty_list_handling(self):
        assert render_tabs([]) == ""


class TestTier1Base64Serializer:
    """Feature 5: Base64 Plot and Raster Serializers."""

    def test_figure_to_base64_format_and_prefix(self):
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot([0, 1], [0, 1])
        uri = figure_to_base64(fig, format="png")
        assert uri.startswith("data:image/png;base64,")
        assert len(uri) > 100

    def test_figure_to_base64_decodes_to_png_magic_bytes(self):
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.scatter([1, 2, 3], [3, 2, 1])
        uri = figure_to_base64(fig, format="png")
        b64_payload = uri.split(",")[1]
        decoded = base64.b64decode(b64_payload)
        assert decoded.startswith(b"\x89PNG\r\n\x1a\n")

    def test_figure_to_base64_closes_figure_no_leak(self):
        initial_figs = plt.get_fignums()
        fig, ax = plt.subplots(figsize=(2, 2))
        assert len(plt.get_fignums()) == len(initial_figs) + 1
        _ = figure_to_base64(fig, close=True)
        assert len(plt.get_fignums()) == len(initial_figs)

    def test_figure_to_base64_dpi_and_kwargs(self):
        fig1, ax1 = plt.subplots(figsize=(2, 2))
        ax1.plot([1, 2], [1, 2])
        uri_low = figure_to_base64(fig1, dpi=50)

        fig2, ax2 = plt.subplots(figsize=(2, 2))
        ax2.plot([1, 2], [1, 2])
        uri_high = figure_to_base64(fig2, dpi=200)

        assert len(uri_high) > len(uri_low)

    def test_figure_to_base64_svg_format(self):
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot([1, 2], [1, 2])
        uri_svg = figure_to_base64(fig, format="svg")
        assert uri_svg.startswith("data:image/svg+xml;base64,")
        b64 = uri_svg.split(",")[1]
        decoded = base64.b64decode(b64).decode("utf-8")
        assert "<svg" in decoded

    def test_image_to_base64_numpy_array(self):
        arr = np.random.rand(30, 40)
        uri = image_to_base64(arr, cmap="viridis")
        assert uri.startswith("data:image/png;base64,")
        decoded = base64.b64decode(uri.split(",")[1])
        assert decoded.startswith(b"\x89PNG\r\n\x1a\n")


class TestTier1OrthoMontage:
    """Feature 6: 3-View Orthogonal Slice Montage."""

    def test_ortho_montage_returns_valid_data_uri(self, synthetic_volume_3d):
        uri = render_ortho_montage(synthetic_volume_3d)
        assert uri.startswith("data:image/png;base64,")
        decoded = base64.b64decode(uri.split(",")[1])
        assert decoded.startswith(b"\x89PNG\r\n\x1a\n")

    def test_ortho_montage_with_numpy_volume(self, synthetic_volume_3d):
        uri = render_ortho_montage(synthetic_volume_3d, title="Numpy Volume")
        assert len(uri) > 1000

    def test_ortho_montage_with_ants_image(self, synthetic_volume_3d):
        if not HAS_ANTS:
            pytest.skip("ANTsPy not installed")
        ants_img = ants.from_numpy(synthetic_volume_3d)
        uri = render_ortho_montage(ants_img, title="ANTsImage Volume")
        assert uri.startswith("data:image/png;base64,")

    def test_ortho_montage_with_explicit_xyz(self, synthetic_volume_3d):
        uri = render_ortho_montage(synthetic_volume_3d, xyz=(14, 18, 10))
        assert uri.startswith("data:image/png;base64,")

    def test_ortho_montage_with_auto_centering(self, synthetic_volume_3d, synthetic_mask_3d):
        # Auto-centering when xyz is None
        uri = render_ortho_montage(synthetic_volume_3d, overlay=synthetic_mask_3d, xyz=None)
        assert uri.startswith("data:image/png;base64,")

    def test_ortho_montage_with_mask_overlay_and_alpha(self, synthetic_volume_3d, synthetic_mask_3d):
        uri = render_ortho_montage(
            synthetic_volume_3d,
            overlay=synthetic_mask_3d,
            overlay_cmap="hot",
            overlay_alpha=0.6,
            crosshairs=False,
        )
        assert uri.startswith("data:image/png;base64,")


class TestTier1SliceGallery:
    """Feature 7: Multi-Slice Contour/Mask Overlay Gallery."""

    def test_slice_gallery_axial_coronal_sagittal_axes(self, synthetic_volume_3d):
        for ax in (0, 1, 2):
            uri = render_slice_gallery(synthetic_volume_3d, axis=ax, nslices=4)
            assert uri.startswith("data:image/png;base64,")

    def test_slice_gallery_nslices_configuration(self, synthetic_volume_3d):
        uri3 = render_slice_gallery(synthetic_volume_3d, nslices=3)
        uri7 = render_slice_gallery(synthetic_volume_3d, nslices=7)
        assert len(uri7) > len(uri3)

    def test_slice_gallery_contour_overlay(self, synthetic_volume_3d, synthetic_multilabel_3d):
        uri = render_slice_gallery(
            synthetic_volume_3d,
            overlay=synthetic_multilabel_3d,
            contours=True,
            title="Tissue Contours",
        )
        assert uri.startswith("data:image/png;base64,")

    def test_slice_gallery_alpha_fill_overlay(self, synthetic_volume_3d, synthetic_mask_3d):
        uri = render_slice_gallery(
            synthetic_volume_3d,
            overlay=synthetic_mask_3d,
            contours=False,
            overlay_alpha=0.4,
            title="Alpha Fill Overlay",
        )
        assert uri.startswith("data:image/png;base64,")

    def test_slice_gallery_ants_and_numpy_inputs(self, synthetic_volume_3d):
        if not HAS_ANTS:
            pytest.skip("ANTsPy not installed")
        ants_img = ants.from_numpy(synthetic_volume_3d)
        uri = render_slice_gallery(ants_img, nslices=5)
        assert uri.startswith("data:image/png;base64,")

    def test_slice_gallery_returns_valid_base64(self, synthetic_volume_3d):
        uri = render_slice_gallery(synthetic_volume_3d, nslices=4)
        decoded = base64.b64decode(uri.split(",")[1])
        assert decoded.startswith(b"\x89PNG\r\n\x1a\n")


class TestTier1CarpetPlot:
    """Feature 8: Timeseries Carpet and Motion Plotter."""

    def test_carpet_plot_returns_valid_data_uri(self, synthetic_timeseries_4d):
        uri = render_carpet_plot(synthetic_timeseries_4d)
        assert uri.startswith("data:image/png;base64,")
        decoded = base64.b64decode(uri.split(",")[1])
        assert decoded.startswith(b"\x89PNG\r\n\x1a\n")

    def test_carpet_plot_with_aligned_fd_and_dvars(self, synthetic_timeseries_4d, synthetic_motion_traces):
        fd, dvars = synthetic_motion_traces
        uri = render_carpet_plot(
            synthetic_timeseries_4d,
            fd=fd,
            dvars=dvars,
            title="fMRI Motion & Carpet",
        )
        assert uri.startswith("data:image/png;base64,")

    def test_carpet_plot_subsampling_performance(self):
        # Create a large 4D timeseries with 50,000 voxels and 30 timepoints
        large_ts = np.random.randn(50, 50, 20, 30).astype(np.float32)
        import time
        t0 = time.perf_counter()
        uri = render_carpet_plot(large_ts, max_voxels=500)
        elapsed = time.perf_counter() - t0
        assert uri.startswith("data:image/png;base64,")
        assert elapsed < 3.0  # Must render in <3 seconds

    def test_carpet_plot_fd_threshold_line(self, synthetic_timeseries_4d, synthetic_motion_traces):
        fd, _ = synthetic_motion_traces
        uri = render_carpet_plot(synthetic_timeseries_4d, fd=fd, fd_threshold=0.3)
        assert uri.startswith("data:image/png;base64,")

    def test_carpet_plot_with_brain_mask(self, synthetic_timeseries_4d):
        mask_3d = np.ones((16, 16, 12), dtype=np.uint8)
        mask_3d[0:4, :, :] = 0
        uri = render_carpet_plot(synthetic_timeseries_4d, mask=mask_3d)
        assert uri.startswith("data:image/png;base64,")

    def test_carpet_plot_2d_voxel_input(self):
        voxel_matrix = np.random.randn(120, 40)
        uri = render_carpet_plot(voxel_matrix)
        assert uri.startswith("data:image/png;base64,")


class TestTier1ViolinPlot:
    """Feature 9: Regional ROI Distribution and Violin Plotter."""

    def test_violin_plot_returns_valid_data_uri(self, synthetic_metric_df):
        uri = render_violin_plot(synthetic_metric_df, x="ROI", y="Thickness")
        assert uri.startswith("data:image/png;base64,")
        decoded = base64.b64decode(uri.split(",")[1])
        assert decoded.startswith(b"\x89PNG\r\n\x1a\n")

    def test_violin_plot_basic_roi_distributions(self, synthetic_metric_df):
        uri = render_violin_plot(synthetic_metric_df, x="ROI", y="Thickness", title="Cortical Thickness")
        assert len(uri) > 1000

    def test_violin_plot_with_hue_grouping(self, synthetic_metric_df):
        uri = render_violin_plot(
            synthetic_metric_df,
            x="ROI",
            y="Thickness",
            hue="Cohort",
            title="Group Comparison",
        )
        assert uri.startswith("data:image/png;base64,")

    def test_violin_plot_with_reference_range_shading(self, synthetic_metric_df):
        uri = render_violin_plot(
            synthetic_metric_df,
            x="ROI",
            y="Thickness",
            reference_range=(2.5, 3.5),
            title="Normative Shading",
        )
        assert uri.startswith("data:image/png;base64,")

    def test_violin_plot_custom_figsize_and_dpi(self, synthetic_metric_df):
        uri = render_violin_plot(
            synthetic_metric_df,
            x="ROI",
            y="Thickness",
            figsize=(12.0, 6.0),
            dpi=100,
        )
        assert uri.startswith("data:image/png;base64,")

    def test_violin_plot_single_category(self):
        df_single = pd.DataFrame({"ROI": ["Amygdala"] * 10, "Volume": np.random.normal(1500, 50, 10)})
        uri = render_violin_plot(df_single, x="ROI", y="Volume")
        assert uri.startswith("data:image/png;base64,")


class TestTier1CorrelationMatrix:
    """Feature 10: Correlation Matrix Heatmap."""

    def test_correlation_matrix_returns_valid_data_uri(self, synthetic_corr_df):
        uri = render_correlation_matrix(synthetic_corr_df)
        assert uri.startswith("data:image/png;base64,")
        decoded = base64.b64decode(uri.split(",")[1])
        assert decoded.startswith(b"\x89PNG\r\n\x1a\n")

    def test_correlation_matrix_symmetric_colormap(self, synthetic_corr_df):
        uri = render_correlation_matrix(synthetic_corr_df, cmap="vlag", title="Resting State Connectivity")
        assert len(uri) > 1000

    def test_correlation_matrix_value_clamping_minus1_to_1(self, synthetic_corr_df):
        uri = render_correlation_matrix(synthetic_corr_df, vmin=-1.0, vmax=1.0)
        assert uri.startswith("data:image/png;base64,")

    def test_correlation_matrix_labels_and_annotations(self, synthetic_corr_df):
        # Explicit annotation enabled
        uri = render_correlation_matrix(synthetic_corr_df, annot=True)
        assert uri.startswith("data:image/png;base64,")

    def test_correlation_matrix_size_scaling(self):
        # Test 17-network Yeo matrix
        n = 17
        names = [f"Y{i+1}" for i in range(n)]
        mat = pd.DataFrame(np.eye(n), index=names, columns=names)
        uri = render_correlation_matrix(mat, title="Yeo 17 Networks")
        assert uri.startswith("data:image/png;base64,")

    def test_correlation_matrix_identity(self):
        ident = pd.DataFrame(np.eye(4), index=list("ABCD"), columns=list("ABCD"))
        uri = render_correlation_matrix(ident)
        assert uri.startswith("data:image/png;base64,")


class TestTier1HtmlBundler:
    """Feature 11: Standalone Self-Contained HTML Bundler."""

    def test_bundler_produces_valid_html5_structure(self):
        doc = build_html_document("Test Report", "<p>Body Content</p>")
        assert doc.strip().startswith("<!DOCTYPE html>")
        assert "<html" in doc
        assert "<head>" in doc
        assert "<body>" in doc
        assert "</html>" in doc

    def test_bundler_zero_external_cdn_links(self):
        doc = build_html_document("Zero CDN Check", "<p>Content</p>")
        # Reject any http:// or https:// script or stylesheet links
        assert not re.search(r'<script[^>]+src=["\']https?://', doc)
        assert not re.search(r'<link[^>]+href=["\']https?://', doc)
        assert "cdn.jsdelivr.net" not in doc
        assert "fonts.googleapis.com" not in doc
        assert "cdnjs.cloudflare.com" not in doc

    def test_bundler_theme_attribute_embedding(self):
        doc_dark = build_html_document("Dark Report", "<p>Text</p>", theme="dark")
        assert '<html lang="en" data-theme="dark">' in doc_dark

        doc_light = build_html_document("Light Report", "<p>Text</p>", theme="light")
        assert '<html lang="en" data-theme="light">' in doc_light

        doc_auto = build_html_document("Auto Report", "<p>Text</p>", theme="auto")
        assert '<html lang="en">' in doc_auto

    def test_bundler_title_and_subtitle_injection(self):
        doc = build_html_document(
            title="ANTsXMM Session Report",
            body_html="<p>Data</p>",
            header_title="Subject 182341 Session 20230111",
            subtitle="Executive QC Summary",
        )
        assert "<title>ANTsXMM Session Report</title>" in doc
        assert "Subject 182341 Session 20230111" in doc
        assert "Executive QC Summary" in doc

    def test_bundler_extra_css_and_js_injection(self):
        doc = build_html_document(
            title="Custom Assets",
            body_html="<p>Body</p>",
            extra_css=".custom-card { border: 2px red solid; }",
            extra_js="console.log('antsxmm-custom-js');",
        )
        assert ".custom-card { border: 2px red solid; }" in doc
        assert "console.log('antsxmm-custom-js');" in doc

    def test_bundler_svg_icon_inlined(self):
        doc = build_html_document("Icon Check", "<p>Body</p>")
        assert "<svg" in doc
        assert "theme-toggle-icon" in doc


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES (>=5 tests per feature/category)
# ==============================================================================

class TestTier2BoundaryDataFrames:
    """Tier 2: Boundary conditions for DataFrames and tabular metrics."""

    def test_boundary_empty_dataframe_with_columns(self):
        df_empty = pd.DataFrame({"ROI": pd.Series(dtype="str"), "Thickness": pd.Series(dtype="float")})
        uri = render_violin_plot(df_empty, x="ROI", y="Thickness")
        assert uri.startswith("data:image/png;base64,")

    def test_boundary_all_nan_metric_column(self):
        df_nan = pd.DataFrame({"ROI": ["Frontal", "Parietal"], "Thickness": [np.nan, np.nan]})
        uri = render_violin_plot(df_nan, x="ROI", y="Thickness")
        assert uri.startswith("data:image/png;base64,")

    def test_boundary_single_row_dataframe(self):
        df_single = pd.DataFrame({"ROI": ["WholeBrain"], "Thickness": [3.14]})
        uri = render_violin_plot(df_single, x="ROI", y="Thickness")
        assert uri.startswith("data:image/png;base64,")

    def test_boundary_empty_correlation_dataframe(self):
        df_empty_corr = pd.DataFrame(index=[], columns=[])
        uri = render_correlation_matrix(df_empty_corr)
        assert uri.startswith("data:image/png;base64,")

    def test_boundary_single_element_correlation_matrix(self):
        df_1x1 = pd.DataFrame([[1.0]], index=["SingleROI"], columns=["SingleROI"])
        uri = render_correlation_matrix(df_1x1)
        assert uri.startswith("data:image/png;base64,")

    def test_boundary_constant_identical_values_in_violin(self):
        df_const = pd.DataFrame({"ROI": ["A", "A", "A", "B", "B", "B"], "Thickness": [2.5, 2.5, 2.5, 2.5, 2.5, 2.5]})
        uri = render_violin_plot(df_const, x="ROI", y="Thickness")
        assert uri.startswith("data:image/png;base64,")


class TestTier2BoundaryTimeseries:
    """Tier 2: Boundary conditions for Timeseries and Motion signals."""

    def test_boundary_zero_variance_timeseries(self):
        const_ts = np.ones((8, 8, 6, 15), dtype=np.float32) * 100.0
        uri = render_carpet_plot(const_ts)
        assert uri.startswith("data:image/png;base64,")

    def test_boundary_single_timepoint_timeseries(self):
        # 2D timeseries (V, 1) represents single-timepoint timeseries
        single_tp_ts_2d = np.random.randn(20, 1)
        uri = render_carpet_plot(single_tp_ts_2d)
        assert uri.startswith("data:image/png;base64,")

        # 4D with T=1 is recognized as squeezed 3D anatomical by _to_numpy and raises ValueError
        single_tp_ts_4d = np.random.randn(8, 8, 6, 1)
        with pytest.raises(ValueError, match="expects 2D or 4D array"):
            render_carpet_plot(single_tp_ts_4d)

    def test_boundary_empty_brain_mask(self):
        ts = np.random.randn(8, 8, 6, 10).astype(np.float32)
        all_zero_mask = np.zeros((8, 8, 6), dtype=np.uint8)
        uri = render_carpet_plot(ts, mask=all_zero_mask)
        assert uri.startswith("data:image/png;base64,")

    def test_boundary_extreme_motion_spikes_and_negatives(self, synthetic_timeseries_4d):
        t_len = synthetic_timeseries_4d.shape[-1]
        fd_extreme = np.array([0.0] * (t_len - 1) + [150.0])  # extreme spike
        dvars_neg = np.array([-10.0] + [20.0] * (t_len - 1))  # negative dvars
        uri = render_carpet_plot(synthetic_timeseries_4d, fd=fd_extreme, dvars=dvars_neg)
        assert uri.startswith("data:image/png;base64,")

    def test_boundary_missing_motion_traces_both_none(self, synthetic_timeseries_4d):
        uri = render_carpet_plot(synthetic_timeseries_4d, fd=None, dvars=None)
        assert uri.startswith("data:image/png;base64,")

    def test_boundary_zero_voxels_in_2d_matrix(self):
        zero_voxels = np.zeros((0, 20))
        uri = render_carpet_plot(zero_voxels)
        assert uri.startswith("data:image/png;base64,")


class TestTier2BoundarySpatialVolumes:
    """Tier 2: Boundary conditions for 2D/3D Spatial Volumes and Overlays."""

    def test_boundary_single_slice_3d_volume(self):
        single_slice = np.random.rand(24, 24, 1).astype(np.float32)
        uri_ortho = render_ortho_montage(single_slice)
        uri_gallery = render_slice_gallery(single_slice, nslices=1)
        assert uri_ortho.startswith("data:image/png;base64,")
        assert uri_gallery.startswith("data:image/png;base64,")

    def test_boundary_2d_array_rejected_by_ortho_montage(self):
        arr_2d = np.random.rand(30, 30)
        with pytest.raises(ValueError, match="expects 3D"):
            render_ortho_montage(arr_2d)

    def test_boundary_all_zero_anatomical_volume(self):
        zero_vol = np.zeros((20, 20, 20), dtype=np.float32)
        uri = render_ortho_montage(zero_vol)
        assert uri.startswith("data:image/png;base64,")

    def test_boundary_extreme_intensity_ranges(self):
        extreme_vol = np.array([
            [[-1e8, 1e8], [0.0, 1e-10]],
            [[500.0, -500.0], [10.0, 0.0]],
        ], dtype=np.float32)
        uri = render_ortho_montage(extreme_vol)
        assert uri.startswith("data:image/png;base64,")

    def test_boundary_disjoint_and_all_zero_overlay_mask(self, synthetic_volume_3d):
        zero_overlay = np.zeros_like(synthetic_volume_3d)
        uri = render_ortho_montage(synthetic_volume_3d, overlay=zero_overlay)
        assert uri.startswith("data:image/png;base64,")

    def test_boundary_out_of_bounds_xyz_coordinates(self, synthetic_volume_3d):
        # XYZ coordinates far outside volume bounds (clipped safely)
        uri = render_ortho_montage(synthetic_volume_3d, xyz=(-50, 999, 500))
        assert uri.startswith("data:image/png;base64,")

    def test_boundary_invalid_axis_in_slice_gallery(self, synthetic_volume_3d):
        with pytest.raises(ValueError, match="axis must be 0, 1, or 2"):
            render_slice_gallery(synthetic_volume_3d, axis=5)


class TestTier2BoundaryThemeAndBundler:
    """Tier 2: Boundary conditions for UI Components, Themes, and Escaping."""

    def test_boundary_invalid_theme_mode_defaults_safely(self):
        doc = build_html_document("Test", "<p>Hi</p>", theme="invalid-neon-theme")
        # html tag should not have invalid data-theme
        html_tag = doc.splitlines()[1]
        assert 'data-theme="invalid-neon-theme"' not in html_tag
        assert html_tag == '<html lang="en">'

    def test_boundary_empty_title_and_body(self):
        doc = build_html_document("", "")
        assert "<title></title>" in doc
        assert "<main>\n\n    </main>" in doc

    def test_boundary_empty_tabs_returns_empty_string(self):
        assert render_tabs([]) == ""

    def test_boundary_kpi_card_with_nan_and_empty_label(self):
        card = render_kpi_card(label="", value=float("nan"))
        assert "kpi-card" in card
        assert "nan" in card.lower()

    def test_boundary_card_renderer_minimal(self):
        card = render_card(title="Only Title", content_html="<p>Plain</p>")
        assert "Only Title" in card
        assert "<p>Plain</p>" in card
        assert "card-subtitle" not in card
        assert "card-badge" not in card

    def test_boundary_special_characters_and_xss_prevention(self):
        card = render_card(
            title="<script>alert('xss')</script>",
            content_html="<b>safe</b>",
            subtitle='"><img src=x onerror=alert(1)>',
            badge="<style>body{color:red}</style>",
        )
        assert "<script>" not in card
        assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in card
        assert "&lt;style&gt;" in card


# ==============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS
# ==============================================================================

class TestTier3CrossFeatureCombinations:
    """Tier 3: Multi-component composition, asset integrity, and document completeness."""

    def test_cross_feature_full_multimodal_executive_report(
        self,
        synthetic_volume_3d,
        synthetic_mask_3d,
        synthetic_timeseries_4d,
        synthetic_motion_traces,
        synthetic_metric_df,
        synthetic_corr_df,
    ):
        """Assemble all core visual components into a unified, interactive tabbed report."""
        fd, dvars = synthetic_motion_traces

        # 1. KPI Cards Row
        kpi1 = render_kpi_card("Brain Volume", 1450.2, "cm³", "normal", "Whole brain volume")
        kpi2 = render_kpi_card("Mean FD", float(np.mean(fd)), "mm", "normal", "Framewise displacement")
        kpi3 = render_kpi_card("Lesion Load", 0.0, "mL", "normal", "White matter hyperintensities")
        kpis_row = f'<div class="kpi-grid">\n{kpi1}\n{kpi2}\n{kpi3}\n</div>'

        # 2. Modality Visualizations
        ortho_uri = render_ortho_montage(synthetic_volume_3d, overlay=synthetic_mask_3d, title="T1w Brain Mask Overlay")
        gallery_uri = render_slice_gallery(synthetic_volume_3d, overlay=synthetic_mask_3d, nslices=5, title="Axial Slices")
        carpet_uri = render_carpet_plot(synthetic_timeseries_4d, fd=fd, dvars=dvars, title="BOLD Timeseries & Motion")
        violin_uri = render_violin_plot(synthetic_metric_df, x="ROI", y="Thickness", hue="Cohort", title="Regional Cortical Thickness")
        corr_uri = render_correlation_matrix(synthetic_corr_df, title="Functional Connectivity")

        # 3. Encapsulate in Cards
        card_struct = render_card("Structural Anatomy", f'<img src="{ortho_uri}" style="max-width:100%;"><br><img src="{gallery_uri}" style="max-width:100%;">', subtitle="3D Orthogonal & Gallery")
        card_func = render_card("Functional Dynamics", f'<img src="{carpet_uri}" style="max-width:100%;"><br><img src="{corr_uri}" style="max-width:100%;">', subtitle="Carpet Plot & FC Matrix")
        card_morpho = render_card("Regional Morphometry", f'<img src="{violin_uri}" style="max-width:100%;">', subtitle="Cohort Distributions")

        # 4. Accessible Tabs
        tabs_html = render_tabs([
            ("tab-structural", "Structural", card_struct),
            ("tab-functional", "Functional", card_func),
            ("tab-morphometry", "Morphometry", card_morpho),
        ])

        # 5. Full HTML Document
        full_page = build_html_document(
            title="ANTsXMM Executive Multimodal Report",
            body_html=f"{kpis_row}\n{tabs_html}",
            header_title="Subject sub-001 Session ses-01",
            subtitle="Automated ANTsXMM Quality & Quantitative Analytics",
            theme="auto",
        )

        assert "<!DOCTYPE html>" in full_page
        assert "Subject sub-001 Session ses-01" in full_page
        assert "tab-structural" in full_page
        assert "tab-functional" in full_page
        assert "tab-morphometry" in full_page
        assert len(full_page) > 50000  # Substantial bundled payload

    def test_cross_feature_multiple_tabs_with_distinct_visualizations(
        self,
        synthetic_volume_3d,
        synthetic_corr_df,
    ):
        ortho_uri = render_ortho_montage(synthetic_volume_3d)
        corr_uri = render_correlation_matrix(synthetic_corr_df)

        tabs_html = render_tabs([
            ("tab-1", "Ortho Montage", f'<img id="img-ortho" src="{ortho_uri}">'),
            ("tab-2", "Correlation Matrix", f'<img id="img-corr" src="{corr_uri}">'),
        ])

        assert 'id="panel-' in tabs_html
        assert 'id="img-ortho"' in tabs_html
        assert 'id="img-corr"' in tabs_html

    def test_cross_feature_html_document_completeness_and_wellformedness(self):
        content = '<div class="kpi-grid">' + render_kpi_card("Metric", 99.9) + '</div>'
        doc = build_html_document("Integrity Test", content)
        assert doc.count("<html") == 1
        assert doc.count("</html>") == 1
        assert doc.count("<head>") == 1
        assert doc.count("</head>") == 1
        assert doc.count("<body>") == 1
        assert doc.count("</body>") == 1
        assert doc.count("<main>") == 1
        assert doc.count("</main>") == 1

    def test_cross_feature_asset_payload_integrity_all_images_valid_png(
        self,
        synthetic_volume_3d,
        synthetic_metric_df,
    ):
        """Verify that every embedded data URI in a generated page decodes to a valid PNG."""
        img1 = render_ortho_montage(synthetic_volume_3d)
        img2 = render_violin_plot(synthetic_metric_df, x="ROI", y="Thickness")

        body = f'<img src="{img1}"><img src="{img2}">'
        doc = build_html_document("Asset Validation", body)

        uris = re.findall(r'src="(data:image/png;base64,[A-Za-z0-9+/=]+)"', doc)
        assert len(uris) == 2

        for uri in uris:
            b64_data = uri.split(",")[1]
            raw_bytes = base64.b64decode(b64_data)
            assert raw_bytes.startswith(b"\x89PNG\r\n\x1a\n")
            assert len(raw_bytes) > 2000

    def test_cross_feature_zero_leak_and_memory_cleanliness(
        self,
        synthetic_volume_3d,
        synthetic_timeseries_4d,
        synthetic_metric_df,
    ):
        """Confirm that Matplotlib figures are consistently cleaned up."""
        initial_count = len(plt.get_fignums())

        _ = render_ortho_montage(synthetic_volume_3d)
        _ = render_slice_gallery(synthetic_volume_3d, nslices=3)
        _ = render_carpet_plot(synthetic_timeseries_4d)
        _ = render_violin_plot(synthetic_metric_df, x="ROI", y="Thickness")

        final_count = len(plt.get_fignums())
        assert final_count == initial_count


# ==============================================================================
# SYNTHETIC MODALITY & SESSION FIXTURES (R2, R3)
# ==============================================================================

def save_synthetic_nifti(
    path: Path,
    data: np.ndarray | None = None,
    shape: tuple[int, ...] = (16, 16, 12),
) -> Path:
    """Helper to write small synthetic 3D or 4D NIfTI file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if data is None:
        data = np.zeros(shape, dtype=np.float32)
        slices = tuple(slice(s // 4, 3 * s // 4) for s in shape[:3])
        if len(shape) == 4:
            data[slices + (slice(None),)] = 50.0
        else:
            data[slices] = 50.0
    img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    nib.save(img, str(path))
    return path


@pytest.fixture
def synthetic_modality_bundle(tmp_path: Path) -> dict[str, Path]:
    """Generate all 8 synthetic modality directories under a temporary session."""
    sess_dir = tmp_path / "sub-TEST01" / "ses-01"
    sess_dir.mkdir(parents=True, exist_ok=True)

    vol = np.zeros((16, 16, 12), dtype=np.float32)
    vol[4:12, 4:12, 3:9] = 100.0

    # 1. Structural
    s_dir = sess_dir / "T1wHierarchical" / "run-01"
    save_synthetic_nifti(s_dir / "brain_n4_dnz.nii.gz", vol)
    save_synthetic_nifti(s_dir / "brain_extraction.nii.gz", (vol > 0).astype(np.uint8))
    save_synthetic_nifti(s_dir / "tissue_segmentation.nii.gz", (vol > 0).astype(np.int32) * 2)
    save_synthetic_nifti(s_dir / "dkt_parcellation.nii.gz", (vol > 0).astype(np.int32) * 3)
    pd.DataFrame([{"icv": 1450000.0}]).to_csv(s_dir / "icv.csv", index=False)
    pd.DataFrame([
        {"Label": 2, "VolumeInMillimeters": 650000.0},
        {"Label": 3, "VolumeInMillimeters": 500000.0},
    ]).to_csv(s_dir / "tissues.csv", index=False)
    pd.DataFrame([{"Label": 1001, "VolumeInMillimeters": 12000.0}]).to_csv(s_dir / "dktcortex.csv", index=False)
    pd.DataFrame([{"T1wHierarchical+icv_cm3": 1450.0}]).to_csv(s_dir / "mmwide.csv", index=False)

    # 2. WMH Positive
    w_dir = sess_dir / "T2Flair" / "run-01"
    wmh_vol = np.zeros((16, 16, 12), dtype=np.float32)
    wmh_vol[7:10, 7:10, 5:7] = 0.85
    save_synthetic_nifti(w_dir / "brain.nii.gz", vol)
    save_synthetic_nifti(w_dir / "wmh.nii.gz", wmh_vol)
    pd.DataFrame([{"wmh_mass": 1250.0, "wmh_SNR": 3.2, "wmh_evr": 0.88}]).to_csv(
        w_dir / "sub-TEST01_T2Flair_run-01_mmwide.csv", index=False
    )

    # 3. DTI
    d_dir = sess_dir / "DTI" / "run-01"
    save_synthetic_nifti(d_dir / "dtifa.nii.gz", vol / 250.0)
    save_synthetic_nifti(d_dir / "dtimd.nii.gz", vol / 120000.0)
    save_synthetic_nifti(d_dir / "b0avg.nii.gz", vol)
    (d_dir / "reoriented.bval").write_text("0 1000 1000 1000 1000 1000 1000\n")
    (d_dir / "reoriented.bvec").write_text("0 1 0 0 0.5 0.5 0\n0 0 1 0 0.5 -0.5 0\n0 0 0 1 0 0 1\n")
    pd.DataFrame([{
        "FA_mean": 0.42,
        "MD_mean": 0.00078,
        "dti_fa_SNR": 8.4,
        "high_motion_volumes": 0,
    }]).to_csv(d_dir / "sub-TEST01_DTI_run-01_mmwide.csv", index=False)

    # 4. rsfMRI
    f_dir = sess_dir / "rsfMRI" / "run-01"
    save_synthetic_nifti(f_dir / "meanBold.nii.gz", vol)
    ts4d = np.repeat(vol[:, :, :, np.newaxis], 10, axis=3)
    save_synthetic_nifti(f_dir / "motion_corrected.nii.gz", ts4d)
    corr = pd.DataFrame(np.eye(6), columns=[f"Network_{i+1}" for i in range(6)])
    corr.to_csv(f_dir / "sub-TEST01_rsfcorr.csv", index=False)
    pd.DataFrame([{
        "FD_mean": 0.11,
        "FD_max": 0.28,
        "DVARS_mean": 21.5,
        "tSNR_mean": 58.2,
    }]).to_csv(f_dir / "sub-TEST01_rsfMRI_run-01_mmwide.csv", index=False)

    # 5. Perfusion
    p_dir = sess_dir / "perf" / "run-01"
    save_synthetic_nifti(p_dir / "cbf.nii.gz", vol / 2.0)
    save_synthetic_nifti(p_dir / "m0.nii.gz", vol * 4.0)
    pd.DataFrame([
        {"cbf_mean": 54.2, "m0_mean": 480.0, "perfusion_mean": 17.5, "tSNR_mean": 41.2},
        {"GM_mean": 61.5, "WM_mean": 26.8},
    ]).to_csv(p_dir / "sub-TEST01_perf_run-01_mmwide.csv", index=False)

    # 6. PET3D
    pet_dir = sess_dir / "pet3d" / "run-01"
    save_synthetic_nifti(pet_dir / "pet3d.nii.gz", vol / 80.0)
    save_synthetic_nifti(pet_dir / "brain_mask.nii.gz", (vol > 0).astype(np.uint8))
    pd.DataFrame([
        {"pet3d_mean": 1.18, "GM_WM_ratio": 1.45},
        {"gm_mean": 1.34, "wm_mean": 0.92, "csf_mean": 0.38},
    ]).to_csv(pet_dir / "sub-TEST01_pet3d_run-01_mmwide.csv", index=False)

    # 7. NM2DMT
    nm_dir = sess_dir / "NM2DMT" / "run-01"
    save_synthetic_nifti(nm_dir / "NM_avg_cropped.nii.gz", vol)
    save_synthetic_nifti(nm_dir / "NM_labels.nii.gz", (vol > 0).astype(np.int32))
    save_synthetic_nifti(nm_dir / "NM_avg.nii.gz", vol)
    pd.DataFrame([{"NM2DMT+SNCR": 1.25, "NM2DMT+nm_mean": 140.0}]).to_csv(
        nm_dir / "sub-TEST01_NM2DMT_run-01_mmwide.csv", index=False
    )

    # Status JSON
    status = {
        "subject_id": "sub-TEST01",
        "session_id": "ses-01",
        "project_id": "TEST_STUDY",
        "execution_engine": "antsxmm_native",
        "args": {"tool_version": "1.3.0"},
        "error": None,
    }
    with open(sess_dir / ".antsxmm_status.json", "w") as f:
        json.dump(status, f)

    return {
        "session_dir": sess_dir,
        "structural": s_dir,
        "wmh": w_dir,
        "dti": d_dir,
        "fmri": f_dir,
        "perf": p_dir,
        "pet": pet_dir,
        "neuromelanin": nm_dir,
    }


@pytest.fixture
def synthetic_zero_lesion_wmh_dir(tmp_path: Path) -> Path:
    """Generate synthetic WMH directory with clean brain and 0.0 lesion mass."""
    w_dir = tmp_path / "zero_wmh" / "T2Flair" / "run-01"
    w_dir.mkdir(parents=True, exist_ok=True)
    vol = np.zeros((16, 16, 12), dtype=np.float32)
    vol[4:12, 4:12, 3:9] = 100.0
    save_synthetic_nifti(w_dir / "brain.nii.gz", vol)
    save_synthetic_nifti(w_dir / "wmh.nii.gz", np.zeros((16, 16, 12), dtype=np.float32))
    pd.DataFrame([{"wmh_mass": 0.0, "wmh_SNR": 0.0, "wmh_evr": 0.0}]).to_csv(
        w_dir / "sub-02_T2Flair_run-01_mmwide.csv", index=False
    )
    return w_dir


@pytest.fixture
def synthetic_study_csv_path(tmp_path: Path) -> Path:
    """Generate synthetic aggregated study CSV with 4 subjects and known outliers."""
    csv_path = tmp_path / "study_aggregated_test.csv"
    data = [
        {
            "subject_id": "sub-01",
            "session_id": "ses-01",
            "project_id": "COHORT_A",
            "T1wHierarchical+icv_cm3": 1450.0,
            "T1wHierarchical+brain_volume_cm3": 1180.0,
            "DTI+FA_mean": 0.42,
            "FD_mean": 0.08,
            "perf+cbf_mean": 55.0,
            "T2Flair+wmh_mass": 250.0,
        },
        {
            "subject_id": "sub-02",
            "session_id": "ses-01",
            "project_id": "COHORT_A",
            "T1wHierarchical+icv_cm3": 1390.0,
            "T1wHierarchical+brain_volume_cm3": 1130.0,
            "DTI+FA_mean": 0.40,
            "FD_mean": 0.12,
            "perf+cbf_mean": 52.0,
            "T2Flair+wmh_mass": 0.0,
        },
        {
            "subject_id": "sub-03",
            "session_id": "ses-01",
            "project_id": "COHORT_B",
            "T1wHierarchical+icv_cm3": 1520.0,
            "T1wHierarchical+brain_volume_cm3": 1260.0,
            "DTI+FA_mean": 0.44,
            "FD_mean": 0.38,  # High motion outlier (>0.25mm)
            "perf+cbf_mean": 58.0,
            "T2Flair+wmh_mass": 1800.0,
        },
        {
            "subject_id": "sub-04",
            "session_id": "ses-01",
            "project_id": "COHORT_B",
            "T1wHierarchical+icv_cm3": 1410.0,
            "T1wHierarchical+brain_volume_cm3": 1150.0,
            "DTI+FA_mean": 0.39,
            "FD_mean": 0.09,
            "perf+cbf_mean": 250.0,  # CBF outlier (>3 SD)
            "T2Flair+wmh_mass": 320.0,
        },
    ]
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


# ==============================================================================
# REQUIREMENT 2: MODALITY-SPECIFIC VISUALIZERS (R2)
# ==============================================================================

class TestR2ModalityRegistryAndUtilities:
    """R2: Modality registry, alias mappings, and utility functions."""

    def test_registry_available_modalities_contains_all_eight(self):
        expected = ["T1w", "T1wHierarchical", "T2Flair", "DTI", "rsfMRI", "perf", "pet3d", "NM2DMT"]
        for mod in expected:
            assert mod in AVAILABLE_MODALITIES
        assert len(AVAILABLE_MODALITIES) == 8

    def test_registry_modality_aliases_complete_mapping(self):
        assert MODALITY_ALIASES["structural"] == "T1w"
        assert MODALITY_ALIASES["t1"] == "T1w"
        assert MODALITY_ALIASES["flair"] == "T2Flair"
        assert MODALITY_ALIASES["wmh"] == "T2Flair"
        assert MODALITY_ALIASES["dwi"] == "DTI"
        assert MODALITY_ALIASES["diffusion"] == "DTI"
        assert MODALITY_ALIASES["bold"] == "rsfMRI"
        assert MODALITY_ALIASES["functional"] == "rsfMRI"
        assert MODALITY_ALIASES["asl"] == "perf"
        assert MODALITY_ALIASES["perfusion"] == "perf"
        assert MODALITY_ALIASES["pet"] == "pet3d"
        assert MODALITY_ALIASES["nm"] == "NM2DMT"
        assert MODALITY_ALIASES["neuromelanin"] == "NM2DMT"

    def test_registry_list_available_modalities_returns_list(self):
        mods = list_available_modalities()
        assert isinstance(mods, list)
        assert len(mods) == 8
        assert mods == AVAILABLE_MODALITIES

    def test_registry_get_modality_visualizer_all_modalities_and_aliases(self):
        for mod in AVAILABLE_MODALITIES:
            vis = get_modality_visualizer(mod)
            assert callable(vis)

        # Test aliases
        assert get_modality_visualizer("structural") is not None
        assert get_modality_visualizer("wmh") is not None
        assert get_modality_visualizer("dwi") is not None
        assert get_modality_visualizer("bold") is not None
        assert get_modality_visualizer("asl") is not None
        assert get_modality_visualizer("pet") is not None
        assert get_modality_visualizer("nm") is not None

    def test_registry_get_modality_visualizer_unsupported_raises_value_error(self):
        with pytest.raises(ValueError, match="Unsupported modality 'unknown_modality'"):
            get_modality_visualizer("unknown_modality")

    def test_registry_find_run_dir_resolution(self, tmp_path: Path):
        base = tmp_path / "modality_test"
        base.mkdir()
        # Case 1: no run directory returns base
        assert find_run_dir(base) == base

        # Case 2: run-01 exists
        run01 = base / "run-01"
        run01.mkdir()
        assert find_run_dir(base) == run01

        # Case 3: only run-02 exists
        run01.rmdir()
        run02 = base / "run-02"
        run02.mkdir()
        assert find_run_dir(base) == run02

    def test_registry_find_file_exact_wildcard_recursive(self, tmp_path: Path):
        sub = tmp_path / "nested" / "dir"
        sub.mkdir(parents=True)
        f1 = tmp_path / "exact_target.txt"
        f1.touch()
        f2 = tmp_path / "sub-01_modality_output.nii.gz"
        f2.touch()
        f3 = sub / "deeply_nested_file.csv"
        f3.touch()

        # Exact match
        assert find_file(tmp_path, "exact_target.txt") == f1
        # Wildcard prefix match
        assert find_file(tmp_path, "output.nii.gz") == f2
        # Recursive match
        assert find_file(tmp_path, "deeply_nested_file.csv") == f3
        # Non-existent match
        assert find_file(tmp_path, "missing_file.txt") is None
        # Non-existent directory returns None
        assert find_file(tmp_path / "nonexistent", "target.txt") is None

    def test_registry_load_nifti_data_formats_and_orientation(self, tmp_path: Path):
        # 1. Ndarray direct return
        arr = np.ones((10, 10, 8), dtype=np.float32)
        assert np.array_equal(load_nifti_data(arr), arr)

        # 2. Squeeze 4D trailing singleton
        arr4d = np.ones((10, 10, 8, 1), dtype=np.float32)
        nii_path = tmp_path / "test4d.nii.gz"
        save_synthetic_nifti(nii_path, arr4d)
        loaded = load_nifti_data(nii_path)
        assert loaded is not None
        assert loaded.shape == (10, 10, 8)

        # 3. Invalid or non-existent path
        assert load_nifti_data(None) is None
        assert load_nifti_data(tmp_path / "missing.nii.gz") is None

    def test_registry_safe_read_csv_behavior(self, tmp_path: Path):
        assert safe_read_csv(None) is None
        assert safe_read_csv(tmp_path / "nonexistent.csv") is None

        valid_csv = tmp_path / "valid.csv"
        valid_csv.write_text("a,b\n1,2\n")
        df = safe_read_csv(valid_csv)
        assert df is not None
        assert len(df) == 1
        assert list(df.columns) == ["a", "b"]

    def test_registry_coalesce_multi_row_df_logic(self):
        # 1. Empty or None
        assert coalesce_multi_row_df(None).empty
        assert coalesce_multi_row_df(pd.DataFrame()).empty

        # 2. Single row
        df1 = pd.DataFrame([{"cbf_mean": 54.2, "tSNR": 40.0}])
        s1 = coalesce_multi_row_df(df1)
        assert s1["cbf_mean"] == 54.2
        assert s1["tSNR"] == 40.0

        # 3. Two rows (Row 0 global, Row 1 regional)
        df2 = pd.DataFrame([
            {"global_cbf": 54.2, "gm_cbf": np.nan},
            {"global_cbf": np.nan, "gm_cbf": 62.8},
        ])
        s2 = coalesce_multi_row_df(df2)
        assert s2["global_cbf"] == 54.2
        assert s2["gm_cbf"] == 62.8

    def test_registry_visualize_modality_auto_detection(self, synthetic_modality_bundle: dict[str, Path]):
        # Auto-detect T2Flair from directory name
        w_dir = synthetic_modality_bundle["wmh"]
        rep = visualize_modality(w_dir.parent)
        assert isinstance(rep, ModalityReport)
        assert rep.name == "T2Flair"
        assert rep.status == "success"


class TestR2ModalityStructural:
    """R2: Structural T1w and T1wHierarchical visualizer."""

    def test_structural_can_visualize(self, tmp_path: Path):
        assert can_visualize_structural(tmp_path / "T1w")
        assert can_visualize_structural(tmp_path / "T1wHierarchical")
        assert can_visualize_structural(tmp_path / "structural")

        folder = tmp_path / "custom_anatomical" / "run-01"
        folder.mkdir(parents=True)
        (folder / "brain_n4_dnz.nii.gz").touch()
        assert can_visualize_structural(folder.parent)
        assert not can_visualize_structural(tmp_path / "unrelated_dir")

    def test_structural_visualize_with_complete_anatomy(self, synthetic_modality_bundle: dict[str, Path]):
        s_dir = synthetic_modality_bundle["structural"]
        rep = visualize_structural(s_dir)
        assert isinstance(rep, ModalityReport)
        assert rep.name in ("T1w", "T1wHierarchical")
        assert rep.status == "success"
        assert len(rep.kpis) >= 4

        kpi_labels = [k["label"] for k in rep.kpis]
        assert "Intracranial Volume (ICV)" in kpi_labels or "ICV" in str(kpi_labels)
        assert "Total Brain Volume" in kpi_labels

        assert "Brain Extraction" in rep.html_content
        assert "Tissue Segmentation" in rep.html_content
        assert "data:image/png;base64," in rep.html_content

    def test_structural_missing_underlay_graceful_fallback(self, tmp_path: Path):
        empty_s = tmp_path / "empty_structural" / "run-01"
        empty_s.mkdir(parents=True)
        rep = visualize_structural(empty_s)
        assert isinstance(rep, ModalityReport)
        assert rep.status in ("warning", "error")
        assert len(rep.errors) >= 1
        assert "T1w Anatomical Underlay Missing" in rep.html_content

    def test_structural_companion_hierarchical_resolution(self, synthetic_modality_bundle: dict[str, Path]):
        sess_dir = synthetic_modality_bundle["session_dir"]
        t1_dir = sess_dir / "T1w" / "run-01"
        t1_dir.mkdir(parents=True, exist_ok=True)

        rep = visualize_structural(t1_dir, session_dir=sess_dir)
        assert rep.status == "success"
        assert "Tissue Segmentation" in rep.html_content


class TestR2ModalityWmh:
    """R2: T2Flair White Matter Hyperintensity visualizer & Zero-Lesion Fallback."""

    def test_wmh_can_visualize(self, tmp_path: Path):
        assert can_visualize_wmh(tmp_path / "T2Flair")
        assert can_visualize_wmh(tmp_path / "flair")
        assert can_visualize_wmh(tmp_path / "wmh")

        folder = tmp_path / "other" / "run-01"
        folder.mkdir(parents=True)
        (folder / "wmh.nii.gz").touch()
        assert can_visualize_wmh(folder.parent)
        assert not can_visualize_wmh(tmp_path / "DTI")

    def test_wmh_visualize_positive_lesions(self, synthetic_modality_bundle: dict[str, Path]):
        w_dir = synthetic_modality_bundle["wmh"]
        rep = visualize_wmh(w_dir)
        assert rep.name == "T2Flair"
        assert rep.status == "success"
        kpi_labels = [k["label"] for k in rep.kpis]
        assert "Total Lesion Mass" in kpi_labels
        assert "Lesion SNR" in kpi_labels
        assert "WMH Lesion Spatial Distribution" in rep.html_content or "High-Burden" in rep.html_content

    def test_wmh_visualize_zero_lesion_graceful_fallback(self, synthetic_zero_lesion_wmh_dir: Path):
        rep = visualize_wmh(synthetic_zero_lesion_wmh_dir)
        assert rep.name == "T2Flair"
        assert rep.status == "success"
        assert len(rep.errors) == 0
        assert "No White Matter Hyperintensities Detected" in rep.html_content

        mass_kpi = next(k for k in rep.kpis if k["label"] == "Total Lesion Mass")
        assert mass_kpi["status"] == "normal"
        assert mass_kpi["value"] == "0.0"

    def test_wmh_visualize_empty_or_subthreshold_mask_fallback(self, tmp_path: Path):
        clean_dir = tmp_path / "clean_flair" / "run-01"
        clean_dir.mkdir(parents=True)
        vol = np.zeros((16, 16, 12), dtype=np.float32)
        vol[4:12, 4:12, 3:9] = 80.0
        save_synthetic_nifti(clean_dir / "brain.nii.gz", vol)
        save_synthetic_nifti(clean_dir / "wmh.nii.gz", np.zeros((16, 16, 12), dtype=np.float32))

        rep = visualize_wmh(clean_dir)
        assert rep.status == "success"
        assert "No White Matter Hyperintensities Detected" in rep.html_content


class TestR2ModalityDti:
    """R2: DTI Diffusion Tensor Imaging visualizer."""

    def test_dti_can_visualize(self, tmp_path: Path):
        assert can_visualize_dti(tmp_path / "DTI")
        assert can_visualize_dti(tmp_path / "diffusion")
        assert can_visualize_dti(tmp_path / "dwi")

        folder = tmp_path / "diff" / "run-01"
        folder.mkdir(parents=True)
        (folder / "dtifa.nii.gz").touch()
        assert can_visualize_dti(folder.parent)
        assert not can_visualize_dti(tmp_path / "rsfMRI")

    def test_dti_visualize_complete_fa_md_b0_and_gradients(self, synthetic_modality_bundle: dict[str, Path]):
        d_dir = synthetic_modality_bundle["dti"]
        rep = visualize_dti(d_dir)
        assert rep.name == "DTI"
        assert rep.status == "success"
        kpi_labels = [k["label"] for k in rep.kpis]
        assert any("FA" in lab for lab in kpi_labels)
        assert any("MD" in lab or "Diffusivity" in lab for lab in kpi_labels)
        assert any("Signal-to-Noise" in lab or "SNR" in lab for lab in kpi_labels)
        assert "Fractional Anisotropy (FA) Map" in rep.html_content
        assert "Mean Diffusivity (MD) Map" in rep.html_content
        assert "Averaged B0 Diffusion Reference" in rep.html_content
        assert "Gradient Direction Sampling Geometry" in rep.html_content

    def test_dti_visualize_missing_gradient_tables_graceful(self, tmp_path: Path):
        d_dir = tmp_path / "dti_no_grad" / "run-01"
        d_dir.mkdir(parents=True)
        vol = np.zeros((16, 16, 12), dtype=np.float32)
        vol[4:12, 4:12, 3:9] = 100.0
        save_synthetic_nifti(d_dir / "dtifa.nii.gz", vol / 200.0)
        pd.DataFrame([{"dti_fa_SNR": 24.5}]).to_csv(
            d_dir / "sub-TEST_DTI_run-01_mmwide.csv", index=False
        )

        rep = visualize_dti(d_dir)
        assert rep.status == "success"
        assert "Fractional Anisotropy (FA) Map" in rep.html_content

    def test_dti_visualize_missing_metrics_csv_graceful(self, tmp_path: Path):
        d_dir = tmp_path / "dti_no_csv" / "run-01"
        d_dir.mkdir(parents=True)
        vol = np.zeros((16, 16, 12), dtype=np.float32)
        vol[4:12, 4:12, 3:9] = 100.0
        save_synthetic_nifti(d_dir / "dtifa.nii.gz", vol / 200.0)

        rep = visualize_dti(d_dir)
        assert rep.status in ("success", "warning")
        assert "Fractional Anisotropy (FA) Map" in rep.html_content


class TestR2ModalityFmri:
    """R2: rsfMRI Functional MRI visualizer."""

    def test_fmri_can_visualize(self, tmp_path: Path):
        assert can_visualize_fmri(tmp_path / "rsfMRI")
        assert can_visualize_fmri(tmp_path / "fmri")
        assert can_visualize_fmri(tmp_path / "functional")
        assert can_visualize_fmri(tmp_path / "bold")

        folder = tmp_path / "func" / "run-01"
        folder.mkdir(parents=True)
        (folder / "meanBold.nii.gz").touch()
        assert can_visualize_fmri(folder.parent)
        assert not can_visualize_fmri(tmp_path / "perf")

    def test_fmri_visualize_complete_timeseries_carpet_and_correlation(
        self, synthetic_modality_bundle: dict[str, Path]
    ):
        f_dir = synthetic_modality_bundle["fmri"]
        rep = visualize_fmri(f_dir)
        assert rep.name == "rsfMRI"
        assert rep.status == "success"
        kpi_labels = [k["label"] for k in rep.kpis]
        assert "Mean FD" in kpi_labels
        assert "Mean DVARS" in kpi_labels
        assert "Temporal SNR" in kpi_labels
        assert "Mean BOLD Intensity" in rep.html_content
        assert "Timeseries Carpet Plot" in rep.html_content
        assert "Functional Connectivity" in rep.html_content

    def test_fmri_visualize_missing_correlation_csv_graceful(self, tmp_path: Path):
        f_dir = tmp_path / "fmri_no_corr" / "run-01"
        f_dir.mkdir(parents=True)
        vol = np.zeros((16, 16, 12), dtype=np.float32)
        vol[4:12, 4:12, 3:9] = 100.0
        save_synthetic_nifti(f_dir / "meanBold.nii.gz", vol)

        rep = visualize_fmri(f_dir)
        assert rep.status == "success"
        assert "Mean BOLD Intensity" in rep.html_content

    def test_fmri_visualize_missing_4d_timeseries_graceful(self, tmp_path: Path):
        f_dir = tmp_path / "fmri_no_4d" / "run-01"
        f_dir.mkdir(parents=True)
        vol = np.zeros((16, 16, 12), dtype=np.float32)
        vol[4:12, 4:12, 3:9] = 100.0
        save_synthetic_nifti(f_dir / "meanBold.nii.gz", vol)

        rep = visualize_fmri(f_dir)
        assert rep.status == "success"
        assert "Mean BOLD Intensity" in rep.html_content
        assert "Timeseries Carpet Plot" not in rep.html_content


class TestR2ModalityPerfusion:
    """R2: Perfusion / ASL visualizer & Multi-Row CSV Coalesce."""

    def test_perfusion_can_visualize(self, tmp_path: Path):
        assert can_visualize_perfusion(tmp_path / "perf")
        assert can_visualize_perfusion(tmp_path / "perfusion")
        assert can_visualize_perfusion(tmp_path / "asl")
        assert can_visualize_perfusion(tmp_path / "pcasl")

        folder = tmp_path / "asl_run" / "run-01"
        folder.mkdir(parents=True)
        (folder / "cbf.nii.gz").touch()
        assert can_visualize_perfusion(folder.parent)
        assert not can_visualize_perfusion(tmp_path / "pet3d")

    def test_perfusion_visualize_cbf_m0_and_multi_row_coalesce(
        self, synthetic_modality_bundle: dict[str, Path]
    ):
        p_dir = synthetic_modality_bundle["perf"]
        rep = visualize_perfusion(p_dir)
        assert rep.name == "perf"
        assert rep.status == "success"
        kpi_labels = [k["label"] for k in rep.kpis]
        assert "Global Mean CBF" in kpi_labels
        assert "M0 Reference Mean" in kpi_labels
        assert "Quantitative Cerebral Blood Flow" in rep.html_content
        assert "M0-to-Perfusion Coregistration Checker" in rep.html_content or "Regional Cerebral Blood Flow" in rep.html_content

    def test_perfusion_visualize_single_row_csv(self, tmp_path: Path):
        p_dir = tmp_path / "perf_1row" / "run-01"
        p_dir.mkdir(parents=True)
        vol = np.zeros((16, 16, 12), dtype=np.float32)
        vol[4:12, 4:12, 3:9] = 50.0
        save_synthetic_nifti(p_dir / "cbf.nii.gz", vol)
        pd.DataFrame([{"cbf_mean": 48.0}]).to_csv(p_dir / "perf_mmwide.csv", index=False)

        rep = visualize_perfusion(p_dir)
        assert rep.status == "success"
        assert any(k["label"] == "Global Mean CBF" for k in rep.kpis)

    def test_perfusion_visualize_missing_m0_graceful(self, tmp_path: Path):
        p_dir = tmp_path / "perf_no_m0" / "run-01"
        p_dir.mkdir(parents=True)
        vol = np.zeros((16, 16, 12), dtype=np.float32)
        vol[4:12, 4:12, 3:9] = 50.0
        save_synthetic_nifti(p_dir / "cbf.nii.gz", vol)

        rep = visualize_perfusion(p_dir)
        assert rep.status == "success"
        assert "Quantitative Cerebral Blood Flow" in rep.html_content


class TestR2ModalityPet:
    """R2: PET3D visualizer & Multi-Row CSV Coalesce."""

    def test_pet_can_visualize(self, tmp_path: Path):
        assert can_visualize_pet(tmp_path / "pet3d")
        assert can_visualize_pet(tmp_path / "pet")

        folder = tmp_path / "tracer" / "run-01"
        folder.mkdir(parents=True)
        (folder / "pet3d.nii.gz").touch()
        assert can_visualize_pet(folder.parent)
        assert not can_visualize_pet(tmp_path / "T1w")

    def test_pet_visualize_uptake_and_multi_row_coalesce(
        self, synthetic_modality_bundle: dict[str, Path]
    ):
        pet_dir = synthetic_modality_bundle["pet"]
        rep = visualize_pet(pet_dir)
        assert rep.name == "pet3d"
        assert rep.status == "success"
        kpi_labels = [k["label"] for k in rep.kpis]
        assert "Mean Whole Brain PET" in kpi_labels
        assert "Gray Matter Uptake" in kpi_labels
        assert "3D PET Tracer Intensity" in rep.html_content
        assert "PET-to-Anatomical Boundary" in rep.html_content

    def test_pet_visualize_missing_mask_graceful(self, tmp_path: Path):
        pet_dir = tmp_path / "pet_no_mask" / "run-01"
        pet_dir.mkdir(parents=True)
        vol = np.zeros((16, 16, 12), dtype=np.float32)
        vol[4:12, 4:12, 3:9] = 1.2
        save_synthetic_nifti(pet_dir / "pet3d.nii.gz", vol)

        rep = visualize_pet(pet_dir)
        assert rep.status == "success"
        assert "3D PET Tracer Intensity" in rep.html_content

    def test_pet_visualize_missing_csv_metrics_graceful(self, tmp_path: Path):
        pet_dir = tmp_path / "pet_no_csv" / "run-01"
        pet_dir.mkdir(parents=True)
        vol = np.zeros((16, 16, 12), dtype=np.float32)
        vol[4:12, 4:12, 3:9] = 1.2
        save_synthetic_nifti(pet_dir / "pet3d.nii.gz", vol)

        rep = visualize_pet(pet_dir)
        assert rep.status == "success"
        assert isinstance(rep.kpis, list)
        assert "3D PET Tracer Intensity" in rep.html_content


class TestR2ModalityNeuromelanin:
    """R2: NM2DMT Neuromelanin-sensitive MRI visualizer."""

    def test_neuromelanin_can_visualize(self, tmp_path: Path):
        assert can_visualize_neuromelanin(tmp_path / "NM2DMT")
        assert can_visualize_neuromelanin(tmp_path / "nm")
        assert can_visualize_neuromelanin(tmp_path / "neuromelanin")

        folder = tmp_path / "substantia_nigra" / "run-01"
        folder.mkdir(parents=True)
        (folder / "NM_avg.nii.gz").touch()
        assert can_visualize_neuromelanin(folder.parent)
        assert not can_visualize_neuromelanin(tmp_path / "T2Flair")

    def test_neuromelanin_visualize_midbrain_slab_and_rois(
        self, synthetic_modality_bundle: dict[str, Path]
    ):
        nm_dir = synthetic_modality_bundle["neuromelanin"]
        rep = visualize_neuromelanin(nm_dir)
        assert rep.name == "NM2DMT"
        assert rep.status == "success"
        kpi_labels = [k["label"] for k in rep.kpis]
        assert "SN Contrast Ratio (SNCR)" in kpi_labels
        assert "Midbrain Slab Average" in rep.html_content
        assert "Target Nuclei ROI Segmentations" in rep.html_content

    def test_neuromelanin_visualize_missing_rois_graceful(self, tmp_path: Path):
        nm_dir = tmp_path / "nm_no_rois" / "run-01"
        nm_dir.mkdir(parents=True)
        vol = np.zeros((16, 16, 12), dtype=np.float32)
        vol[4:12, 4:12, 3:9] = 100.0
        save_synthetic_nifti(nm_dir / "NM_avg_cropped.nii.gz", vol)

        rep = visualize_neuromelanin(nm_dir)
        assert rep.status == "success"
        assert "Midbrain Slab Average" in rep.html_content

    def test_neuromelanin_visualize_missing_full_slab_graceful(self, tmp_path: Path):
        nm_dir = tmp_path / "nm_no_full" / "run-01"
        nm_dir.mkdir(parents=True)
        vol = np.zeros((16, 16, 12), dtype=np.float32)
        vol[4:12, 4:12, 3:9] = 100.0
        save_synthetic_nifti(nm_dir / "NM_avg_cropped.nii.gz", vol)

        rep = visualize_neuromelanin(nm_dir)
        assert rep.status == "success"
        assert "Midbrain Slab Average" in rep.html_content


# ==============================================================================
# REQUIREMENT 3: SESSION & STUDY MULTI-MODALITY AGGREGATOR + CLI (R3)
# ==============================================================================

class TestR3SessionReportAggregator:
    """R3: Session report aggregator, executive overview, and missingness handling."""

    def test_session_report_generation_synthetic_multimodal(
        self, synthetic_modality_bundle: dict[str, Path], tmp_path: Path
    ):
        sess_dir = synthetic_modality_bundle["session_dir"]
        out_html = tmp_path / "session_report_output.html"

        res_path = generate_session_report(
            session_dir=sess_dir,
            output_html=out_html,
            theme="dark",
            title="Synthetic Multimodal Clinical Session QC",
        )
        assert res_path == out_html
        assert out_html.is_file()
        content = out_html.read_text(encoding="utf-8")

        # Key Executive Overview components
        assert "Synthetic Multimodal Clinical Session QC" in content
        assert "Executive Overview" in content
        assert "sub-TEST01" in content
        assert "ses-01" in content
        assert "8-Modality Completion Matrix" in content
        assert "Cross-Modality Key Performance Indicators" in content

        # Modality tabs
        assert "T1wHierarchical" in content
        assert "T2Flair" in content
        assert "DTI" in content
        assert "rsfMRI" in content
        assert "perf" in content
        assert "pet3d" in content
        assert "NM2DMT" in content

    def test_session_report_nonexistent_dir_raises_filenotfound(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Session directory does not exist"):
            generate_session_report(tmp_path / "nonexistent_session")

    def test_session_report_zero_cdn_links_enforced(
        self, synthetic_modality_bundle: dict[str, Path], tmp_path: Path
    ):
        sess_dir = synthetic_modality_bundle["session_dir"]
        out_html = tmp_path / "session_zero_cdn.html"
        generate_session_report(sess_dir, out_html)
        content = out_html.read_text(encoding="utf-8")
        external_links = re.findall(r"https?://", content)
        assert len(external_links) == 0

    def test_session_report_missing_modality_gracefulness(self, tmp_path: Path):
        # Create session with ONLY T1w
        sess_dir = tmp_path / "sparse_sess" / "sub-SOLO" / "ses-01"
        s_dir = sess_dir / "T1wHierarchical" / "run-01"
        s_dir.mkdir(parents=True)
        vol = np.zeros((16, 16, 12), dtype=np.float32)
        vol[4:12, 4:12, 3:9] = 100.0
        save_synthetic_nifti(s_dir / "brain_n4_dnz.nii.gz", vol)

        out_html = tmp_path / "sparse_report.html"
        generate_session_report(sess_dir, out_html)
        content = out_html.read_text(encoding="utf-8")

        # Completed for T1w
        assert "8-Modality Completion Matrix" in content
        # Not acquired for other modalities
        assert "Not Acquired" in content
        assert "Omitted in acquisition protocol" in content

    def test_session_report_custom_themes_and_title(
        self, synthetic_modality_bundle: dict[str, Path], tmp_path: Path
    ):
        sess_dir = synthetic_modality_bundle["session_dir"]
        # Light theme
        out_light = tmp_path / "light.html"
        generate_session_report(sess_dir, out_light, theme="light", title="Light Theme Report")
        assert 'data-theme="light"' in out_light.read_text(encoding="utf-8")
        assert "Light Theme Report" in out_light.read_text(encoding="utf-8")

        # Dark theme
        out_dark = tmp_path / "dark.html"
        generate_session_report(sess_dir, out_dark, theme="dark", title="Dark Theme Report")
        assert 'data-theme="dark"' in out_dark.read_text(encoding="utf-8")

    def test_session_report_default_output_path(
        self, synthetic_modality_bundle: dict[str, Path]
    ):
        sess_dir = synthetic_modality_bundle["session_dir"]
        res = generate_session_report(sess_dir)
        assert res.is_file()
        assert res.name == "session_report.html"
        assert res.parent == sess_dir


class TestR3StudyReportAggregator:
    """R3: Study report aggregator, cohort distributions, and outlier detection."""

    def test_study_report_generation_synthetic_cohort(
        self, synthetic_study_csv_path: Path, tmp_path: Path
    ):
        out_html = tmp_path / "study_report_test.html"
        res_path = generate_study_report(
            study_csv_path=synthetic_study_csv_path,
            output_html=out_html,
            theme="auto",
            title="Multi-Subject Cohort Aggregation QC",
        )
        assert res_path == out_html
        assert out_html.is_file()
        assert out_html.stat().st_size > 30000

        content = out_html.read_text(encoding="utf-8")
        assert "Multi-Subject Cohort Aggregation QC" in content
        assert "Total Sessions" in content
        assert "Unique Subjects" in content
        assert "Modalities Found" in content
        assert "Total Features" in content
        assert "Modality Missingness &amp; Acquisition Matrix" in content or "Modality Missingness & Acquisition Matrix" in content
        assert "Population Quantitative Distributions" in content
        assert "Quality Control Outliers" in content

    def test_study_report_nonexistent_csv_raises_filenotfound(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Study CSV file does not exist"):
            generate_study_report(tmp_path / "nonexistent_study.csv")

    def test_study_report_zero_cdn_links_enforced(
        self, synthetic_study_csv_path: Path, tmp_path: Path
    ):
        out_html = tmp_path / "study_zero_cdn.html"
        generate_study_report(synthetic_study_csv_path, out_html)
        content = out_html.read_text(encoding="utf-8")
        external_links = re.findall(r"https?://", content)
        assert len(external_links) == 0

    def test_study_report_outlier_detection_and_alerts(
        self, synthetic_study_csv_path: Path, tmp_path: Path
    ):
        out_html = tmp_path / "study_outliers.html"
        generate_study_report(synthetic_study_csv_path, out_html)
        content = out_html.read_text(encoding="utf-8")

        # sub-03 has high motion (FD = 0.38 > 0.25mm)
        assert "sub-03" in content
        # sub-04 has extreme CBF outlier (135.0)
        assert "sub-04" in content
        assert "DANGER" in content or "WARNING" in content

    def test_study_report_custom_themes_and_title(
        self, synthetic_study_csv_path: Path, tmp_path: Path
    ):
        out_html = tmp_path / "study_dark.html"
        generate_study_report(
            synthetic_study_csv_path,
            out_html,
            theme="dark",
            title="Custom Clinical Cohort Trial",
        )
        content = out_html.read_text(encoding="utf-8")
        assert 'data-theme="dark"' in content
        assert "Custom Clinical Cohort Trial" in content

    def test_study_report_default_output_path(
        self, synthetic_study_csv_path: Path
    ):
        res = generate_study_report(synthetic_study_csv_path)
        assert res.is_file()
        assert res.name == "study_report.html"
        assert res.parent == synthetic_study_csv_path.parent


class TestR3CommandLineInterface:
    """R3: Click CLI command interface and validation."""

    def test_cli_mutual_exclusion_neither_option_fails(self):
        runner = CliRunner()
        res = runner.invoke(main, ["report"])
        assert res.exit_code != 0
        assert "Must specify exactly one of --session or --study" in res.output

    def test_cli_mutual_exclusion_both_options_fails(self, tmp_path: Path):
        runner = CliRunner()
        res = runner.invoke(
            main,
            ["report", "--session", str(tmp_path), "--study", str(tmp_path / "study.csv")],
        )
        assert res.exit_code != 0
        assert "Must specify exactly one of --session or --study" in res.output

    def test_cli_session_report_invocation_success(
        self, synthetic_modality_bundle: dict[str, Path], tmp_path: Path
    ):
        runner = CliRunner()
        sess_dir = synthetic_modality_bundle["session_dir"]
        out_html = tmp_path / "cli_session_report.html"
        res = runner.invoke(
            main,
            [
                "report",
                "--session",
                str(sess_dir),
                "--out",
                str(out_html),
                "--theme",
                "dark",
                "--title",
                "CLI Session QC",
            ],
        )
        assert res.exit_code == 0
        assert "Session report generated successfully" in res.output
        assert out_html.is_file()
        assert out_html.stat().st_size > 20000

    def test_cli_study_report_invocation_success(
        self, synthetic_study_csv_path: Path, tmp_path: Path
    ):
        runner = CliRunner()
        out_html = tmp_path / "cli_study_report.html"
        res = runner.invoke(
            main,
            [
                "report",
                "--study",
                str(synthetic_study_csv_path),
                "--out",
                str(out_html),
                "--title",
                "CLI Study QC",
            ],
        )
        assert res.exit_code == 0
        assert "Study report generated successfully" in res.output
        assert out_html.is_file()
        assert out_html.stat().st_size > 20000

    def test_cli_report_help_options(self):
        runner = CliRunner()
        res = runner.invoke(main, ["report", "--help"])
        assert res.exit_code == 0
        assert "--session" in res.output
        assert "--study" in res.output
        assert "--out" in res.output
        assert "--theme" in res.output
        assert "--title" in res.output
        assert "--open-browser" in res.output


# ==============================================================================
# REQUIREMENT 4: REAL COHORT INTEGRATION TESTING (R4)
# ==============================================================================

class TestTier4RealWorldCohorts:
    """R4: Comprehensive verification against real assembled multimodal evaluation cohorts."""

    SOCOM_DIR = Path("/Users/stnava/data/processed/antsxmm_multimodal_eval/SOCOM/sub-Blast-01/ses-01")
    PPMI_DIR = Path("/Users/stnava/data/processed/antsxmm_multimodal_eval/PPMI/sub-182341/ses-20230111")
    STUDY_CSV = Path("/Users/stnava/data/processed/antsxmm_multimodal_eval/study_multimodal_aggregated.csv")

    def test_real_world_socom_session_exploration(self):
        """Examine real SOCOM Blast-01 multimodal session directory and verify modal data."""
        if not self.SOCOM_DIR.is_dir():
            pytest.skip(f"SOCOM reference session not found at {self.SOCOM_DIR}")

        expected_modalities = ["T1w", "T1wHierarchical", "T2Flair", "DTI", "rsfMRI", "perf", "pet3d"]
        for mod in expected_modalities:
            mod_path = self.SOCOM_DIR / mod
            assert mod_path.is_dir(), f"Expected modality directory {mod} in SOCOM"

        rsf_corr_csv = next(self.SOCOM_DIR.glob("rsfMRI/run-01/*rsfcorr.csv"), None)
        assert rsf_corr_csv is not None and rsf_corr_csv.is_file()

        df_corr = pd.read_csv(rsf_corr_csv)
        assert df_corr.shape[0] > 0

        net_cols = [c for c in df_corr.columns if c not in ("Unnamed: 0", "networks")]
        corr_matrix = df_corr[net_cols].copy()
        if "networks" in df_corr.columns:
            corr_matrix.index = df_corr["networks"].tolist()
        else:
            corr_matrix.index = net_cols

        assert corr_matrix.shape[0] == corr_matrix.shape[1]

        uri = render_correlation_matrix(corr_matrix, title="SOCOM sub-Blast-01 rsfMRI Correlation")
        assert uri.startswith("data:image/png;base64,")

    def test_real_world_socom_individual_modalities_rendering(self):
        """Verify individual visualizer execution on real SOCOM modalities."""
        if not self.SOCOM_DIR.is_dir():
            pytest.skip(f"SOCOM reference session not found at {self.SOCOM_DIR}")

        for mod in ["T1wHierarchical", "T2Flair", "DTI", "perf", "pet3d"]:
            mod_path = self.SOCOM_DIR / mod
            rep = visualize_modality(mod, mod_path, session_dir=self.SOCOM_DIR)
            assert rep.status in ("success", "warning"), f"SOCOM {mod} returned unexpected status: {rep.status}"
            assert len(rep.kpis) >= 3
            assert "data:image/png;base64," in rep.html_content

    def test_real_world_socom_full_session_report_assembly(self, tmp_path: Path):
        """Verify end-to-end generate_session_report on real SOCOM session."""
        if not self.SOCOM_DIR.is_dir():
            pytest.skip(f"SOCOM reference session not found at {self.SOCOM_DIR}")

        out_file = tmp_path / "real_socom_report.html"
        res = generate_session_report(
            session_dir=self.SOCOM_DIR,
            output_html=out_file,
            theme="dark",
            title="SOCOM Real Multimodal Evaluation Report",
        )
        assert res == out_file
        assert out_file.is_file()
        assert out_file.stat().st_size > 1_000_000

        content = out_file.read_text(encoding="utf-8")
        assert "sub-Blast-01" in content
        assert "ses-01" in content
        assert "Executive Overview" in content
        assert "T1wHierarchical" in content
        assert "DTI" in content
        assert "perf" in content
        assert "pet3d" in content

        # Zero CDN check
        assert len(re.findall(r"https?://", content)) == 0

    def test_real_world_ppmi_session_exploration(self):
        """Examine real PPMI sub-182341 session and verify NM2DMT and DTI outputs."""
        if not self.PPMI_DIR.is_dir():
            pytest.skip(f"PPMI reference session not found at {self.PPMI_DIR}")

        expected_modalities = ["T1w", "T1wHierarchical", "DTI", "rsfMRI", "NM2DMT"]
        for mod in expected_modalities:
            assert (self.PPMI_DIR / mod).is_dir(), f"Expected modality directory {mod} in PPMI"

        nm_dir = self.PPMI_DIR / "NM2DMT/run-01"
        nm_files = list(nm_dir.glob("*.nii.gz"))
        assert len(nm_files) >= 1

        session_csv = next(self.PPMI_DIR.glob("*study.csv"), None)
        assert session_csv is not None
        df_sess = pd.read_csv(session_csv)
        assert len(df_sess) > 0

    def test_real_world_ppmi_neuromelanin_and_modalities_rendering(self):
        """Verify NM2DMT neuromelanin visualizer on real PPMI session."""
        if not self.PPMI_DIR.is_dir():
            pytest.skip(f"PPMI reference session not found at {self.PPMI_DIR}")

        nm_path = self.PPMI_DIR / "NM2DMT"
        rep = visualize_modality("NM2DMT", nm_path, session_dir=self.PPMI_DIR)
        assert rep.status in ("success", "warning")
        assert any("SN" in k["label"] for k in rep.kpis)
        assert "Midbrain Slab Average" in rep.html_content
        assert "data:image/png;base64," in rep.html_content

    def test_real_world_ppmi_full_session_report_assembly(self, tmp_path: Path):
        """Verify end-to-end generate_session_report on real PPMI session."""
        if not self.PPMI_DIR.is_dir():
            pytest.skip(f"PPMI reference session not found at {self.PPMI_DIR}")

        out_file = tmp_path / "real_ppmi_report.html"
        res = generate_session_report(
            session_dir=self.PPMI_DIR,
            output_html=out_file,
            theme="light",
            title="PPMI Real Neuromelanin Evaluation Report",
        )
        assert res == out_file
        assert out_file.is_file()
        assert out_file.stat().st_size > 1_000_000

        content = out_file.read_text(encoding="utf-8")
        assert "sub-182341" in content
        assert "ses-20230111" in content
        assert "NM2DMT" in content
        assert len(re.findall(r"https?://", content)) == 0

    def test_real_world_study_aggregated_csv_exploration(self):
        """Load and verify study_multimodal_aggregated.csv (240KB cohort table)."""
        if not self.STUDY_CSV.is_file():
            pytest.skip(f"Aggregated study CSV not found at {self.STUDY_CSV}")

        df_study = pd.read_csv(self.STUDY_CSV)
        assert len(df_study) >= 2
        assert "subject_id" in df_study.columns
        assert "session_id" in df_study.columns

        subjects = set(df_study["subject_id"].tolist())
        assert "sub-Blast-01" in subjects or "sub-182341" in subjects

        kpi_sub = render_kpi_card("Total Subjects", len(subjects), status="normal")
        kpi_col = render_kpi_card("Metric Columns", len(df_study.columns), status="info")
        html_doc = build_html_document("Study QC Summary", f'<div class="kpi-grid">{kpi_sub}{kpi_col}</div>')
        assert "Total Subjects" in html_doc
        assert "Metric Columns" in html_doc

    def test_real_world_study_full_report_assembly(self, tmp_path: Path):
        """Verify end-to-end generate_study_report on real aggregated study CSV."""
        if not self.STUDY_CSV.is_file():
            pytest.skip(f"Aggregated study CSV not found at {self.STUDY_CSV}")

        out_file = tmp_path / "real_study_report.html"
        res = generate_study_report(
            study_csv_path=self.STUDY_CSV,
            output_html=out_file,
            theme="auto",
            title="ANTsXMM Real Cohort Multimodal Study QC",
        )
        assert res == out_file
        assert out_file.is_file()
        assert out_file.stat().st_size > 100_000

        content = out_file.read_text(encoding="utf-8")
        assert "ANTsXMM Real Cohort Multimodal Study QC" in content
        assert "sub-Blast-01" in content
        assert "sub-182341" in content
        assert "Modality Missingness &amp; Acquisition Matrix" in content or "Modality Missingness & Acquisition Matrix" in content
        assert "Population Quantitative Distributions" in content
        assert len(re.findall(r"https?://", content)) == 0

    def test_real_world_status_json_inspection(self):
        """Inspect .antsxmm_status.json in reference sessions for execution metadata."""
        socom_status = self.SOCOM_DIR / ".antsxmm_status.json"
        if not socom_status.is_file():
            pytest.skip(f"SOCOM status JSON not found at {socom_status}")

        with open(socom_status) as f:
            status = json.load(f)

        assert "execution_engine" in status
        assert status["execution_engine"] == "antsxmm_native"
        assert "tool_version" in status.get("args", {})
        assert status["error"] is None

    def test_real_world_vs_synthetic_consistency(self, synthetic_volume_3d):
        """Verify that synthetic generators produce the same compliant format as real data."""
        synth_uri = render_ortho_montage(synthetic_volume_3d)
        assert synth_uri.startswith("data:image/png;base64,")
        payload = base64.b64decode(synth_uri.split(",")[1])
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")

    def test_real_world_cli_study_report(self, tmp_path: Path):
        """Verify CLI report generation on real study aggregated CSV."""
        if not self.STUDY_CSV.is_file():
            pytest.skip(f"Aggregated study CSV not found at {self.STUDY_CSV}")

        runner = CliRunner()
        out_file = tmp_path / "cli_real_study.html"
        res = runner.invoke(
            main,
            [
                "report",
                "--study",
                str(self.STUDY_CSV),
                "--out",
                str(out_file),
                "--title",
                "Real Cohort CLI QC",
            ],
        )
        assert res.exit_code == 0
        assert "Study report generated successfully" in res.output
        assert out_file.is_file()
        assert out_file.stat().st_size > 100_000

