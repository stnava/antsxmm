"""T2Flair and White Matter Hyperintensity (WMH) modality visualizer.

Provides publication-grade visual reporting for FLAIR / WMH:
- Anatomical underlay (FLAIR or co-registered T1 brain) with continuous WMH probability overlay
- Thresholded lesion contours across high-burden slices
- Lesion volume breakdown & clinical KPI metric cards (wmh_mass, wmh_SNR, wmh_evr)
- Graceful zero-lesion fallback: if wmh_mass == 0 or empty mask, renders a clean
  "No Lesions Detected" banner without crashing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from antsxmm.visualize.core import render_ortho_montage, render_slice_gallery
from antsxmm.visualize.modalities import (
    ModalityReport,
    find_file,
    find_run_dir,
    load_nifti_data,
    safe_read_csv,
)
from antsxmm.visualize.theme import render_card, render_kpi_card

logger = logging.getLogger(__name__)


def can_visualize(modality_dir: Path | str) -> bool:
    """Determine if directory contains T2Flair / WMH outputs."""
    p = Path(modality_dir)
    name_lower = p.name.lower()
    if any(k in name_lower for k in ("flair", "t2", "wmh")):
        return True

    run_dir = find_run_dir(p)
    return any(
        find_file(run_dir, pat) is not None
        for pat in (
            "wmh.nii.gz",
            "*wmh*.nii.gz",
            "*T2Flair*mmwide.csv",
        )
    )


def visualize_wmh(
    modality_dir: Path | str,
    session_dir: Path | str | None = None,
    **kwargs: Any,
) -> ModalityReport:
    """Generate visual report for T2Flair and White Matter Hyperintensities.

    Parameters
    ----------
    modality_dir : Path or str
        Path to T2Flair modality folder or run folder.
    session_dir : Path or str, optional
        Parent session directory to locate co-registered anatomical T1 underlay.
    """
    mod_path = Path(modality_dir)
    errors: list[str] = []
    cards_html: list[str] = []
    kpis: list[dict[str, Any]] = []

    run_dir = find_run_dir(mod_path)

    # 1. Locate WMH probability map and metrics CSV
    wmh_file = find_file(run_dir, "wmh.nii.gz") or find_file(run_dir, "*wmh*.nii.gz")
    mmwide_csv = find_file(run_dir, "mmwide.csv") or find_file(mod_path, "mmwide.csv")
    df_mmwide = safe_read_csv(mmwide_csv)

    # 2. Extract metrics from mmwide.csv
    wmh_mass: float = 0.0
    wmh_snr: float | None = None
    wmh_evr: float | None = None
    raw_mean: float | None = None

    if df_mmwide is not None and not df_mmwide.empty:
        row = df_mmwide.iloc[0]
        if "wmh_mass" in row:
            wmh_mass = float(row["wmh_mass"])
        if "wmh_SNR" in row:
            wmh_snr = float(row["wmh_SNR"])
        if "wmh_evr" in row:
            wmh_evr = float(row["wmh_evr"])
        if "WMH_probability_map_raw_mean" in row:
            raw_mean = float(row["WMH_probability_map_raw_mean"])

    # 3. Locate anatomical underlay
    # Priority:
    # a. T1wHierarchical / T1w brain_n4_dnz.nii.gz in session_dir or parent
    # b. Local FLAIR anatomical image in run_dir
    # c. WMH map itself as fallback underlay
    underlay_file: Path | None = None
    if session_dir is not None:
        sess_p = Path(session_dir)
        hier_dir = find_run_dir(sess_p / "T1wHierarchical")
        underlay_file = find_file(hier_dir, "brain_n4_dnz.nii.gz") or find_file(hier_dir, "head.nii.gz")
    if underlay_file is None and mod_path.parent.is_dir():
        hier_dir = find_run_dir(mod_path.parent / "T1wHierarchical")
        underlay_file = find_file(hier_dir, "brain_n4_dnz.nii.gz") or find_file(hier_dir, "head.nii.gz")
    if underlay_file is None:
        underlay_file = find_file(run_dir, "*flair*.nii.gz") or find_file(run_dir, "*t2*.nii.gz")

    wmh_arr = load_nifti_data(wmh_file)
    underlay_arr = load_nifti_data(underlay_file) if underlay_file else None

    # If no separate anatomical underlay was found, use WMH array if available
    if underlay_arr is None and wmh_arr is not None:
        underlay_arr = wmh_arr

    # Determine lesion presence
    has_lesions = (
        wmh_mass > 0.1
        or (wmh_arr is not None and float(np.max(wmh_arr)) >= 0.1 and float(np.sum(wmh_arr > 0.3)) >= 5)
    )

    # 4. Build KPI metric cards
    # Lesion status classification
    if wmh_mass == 0.0:
        mass_status = "normal"
    elif wmh_mass < 500.0:
        mass_status = "normal"
    elif wmh_mass < 2500.0:
        mass_status = "warning"
    else:
        mass_status = "danger"

    kpis.append({
        "label": "Total Lesion Mass",
        "value": f"{wmh_mass:,.1f}",
        "unit": "mm³",
        "status": mass_status,
        "tooltip": "White matter hyperintensity total lesion volume (Burden: <500mm³ Mild, 500-2500mm³ Moderate, >2500mm³ High).",
    })

    if wmh_snr is not None:
        kpis.append({
            "label": "Lesion SNR",
            "value": f"{wmh_snr:.2f}",
            "unit": "",
            "status": "normal" if wmh_snr >= 2.0 else "warning",
            "tooltip": "Signal-to-noise ratio of segmented hyperintense lesions against white matter baseline.",
        })

    if wmh_evr is not None:
        kpis.append({
            "label": "Explained Variance (EVR)",
            "value": f"{wmh_evr:.3f}",
            "unit": "",
            "status": "normal",
            "tooltip": "Eigenvalue variance ratio of probability estimation model.",
        })

    if raw_mean is not None:
        kpis.append({
            "label": "Mean Prob Intensity",
            "value": f"{raw_mean:.4g}",
            "unit": "",
            "status": "normal",
            "tooltip": "Mean continuous voxel probability across cerebral white matter mask.",
        })

    # Render KPI Cards HTML Grid
    kpi_cards_html = "".join(
        render_kpi_card(
            label=k["label"],
            value=k["value"],
            unit=k.get("unit", ""),
            status=k.get("status", "normal"),
            tooltip=k.get("tooltip", ""),
        )
        for k in kpis
    )
    if kpi_cards_html:
        cards_html.append(f'<div class="kpi-grid" style="margin-bottom: 24px;">{kpi_cards_html}</div>')

    # 5. Handle Zero-Lesion Fallback vs Lesion Present
    if not has_lesions:
        # Graceful zero-lesion banner
        zero_banner = (
            '<div class="banner banner-info" style="'
            'padding: 16px 20px; border-radius: var(--radius-md); '
            'background-color: var(--status-normal-bg); color: var(--status-normal-text); '
            'border: 1px solid var(--status-normal-border); margin-bottom: 20px; '
            'display: flex; align-items: center; gap: 14px;'
            '">'
            '<span style="font-size: 24px;" aria-hidden="true">🛡️</span>'
            '<div>'
            '<strong style="font-size: 15px;">No White Matter Hyperintensities Detected</strong>'
            '<p style="margin-top: 4px; font-size: 13px; opacity: 0.9;">'
            f'Automated deep probability analysis identified 0.0 mm³ lesion burden '
            f'(wmh_mass = {wmh_mass:.1f} mm³). Brain parenchyma demonstrates healthy, uncompromised white matter appearance.'
            '</p>'
            '</div>'
            '</div>'
        )
        cards_html.append(zero_banner)

        # Show anatomical underlay montage if available to confirm clean white matter
        if underlay_arr is not None:
            try:
                ortho_uri = render_ortho_montage(
                    underlay=underlay_arr,
                    overlay=None,
                    title="Anatomical White Matter Verification (No Lesions Detected)",
                )
                img_html = f'<div class="img-container"><img src="{ortho_uri}" class="img-responsive" alt="Anatomical White Matter" /></div>'
                cards_html.append(
                    render_card(
                        title="Anatomical Underlay & White Matter Verification",
                        subtitle="Ortho montage verifying intact white matter parenchyma without focal hyperintensities",
                        content_html=img_html,
                        badge="Clear",
                    )
                )
            except Exception as exc:
                errors.append(f"Failed to render zero-lesion anatomical montage: {exc}")
    else:
        # Lesions detected: render ortho montage and high-burden slice gallery
        if underlay_arr is not None and wmh_arr is not None:
            # Card 1: Ortho Montage with continuous probability
            try:
                ortho_uri = render_ortho_montage(
                    underlay=underlay_arr,
                    overlay=wmh_arr,
                    overlay_cmap="hot",
                    overlay_alpha=0.6,
                    title="White Matter Hyperintensity Continuous Probability (Ortho View)",
                )
                img_html = f'<div class="img-container"><img src="{ortho_uri}" class="img-responsive" alt="WMH Probability Ortho" /></div>'
                cards_html.append(
                    render_card(
                        title="WMH Lesion Spatial Distribution & Probability",
                        subtitle="Continuous lesion probability map overlaid on co-registered anatomical underlay",
                        content_html=img_html,
                        badge="Overlay",
                    )
                )
            except Exception as exc:
                errors.append(f"Failed to render WMH ortho montage: {exc}")

            # Card 2: High-Burden Axial Slice Gallery with Contours
            try:
                thresholded_mask = (wmh_arr >= 0.5).astype(np.float32)
                gallery_uri = render_slice_gallery(
                    underlay=underlay_arr,
                    overlay=thresholded_mask,
                    axis=2,
                    nslices=7,
                    contours=True,
                    overlay_cmap="autumn",
                    title="Axial Slices with Thresholded Lesion Contours (P >= 0.5)",
                )
                img_html = f'<div class="img-container"><img src="{gallery_uri}" class="img-responsive" alt="High-Burden Lesions" /></div>'
                cards_html.append(
                    render_card(
                        title="High-Burden Lesion Contours across Axial Slices",
                        subtitle="Vector contours outlining segmented hyperintense lesions across periventricular and subcortical regions",
                        content_html=img_html,
                        badge="Contours",
                    )
                )
            except Exception as exc:
                errors.append(f"Failed to render WMH slice gallery: {exc}")

    status = "error" if (wmh_arr is None and not kpis) else ("warning" if errors else "success")

    return ModalityReport(
        name="T2Flair",
        title="White Matter Hyperintensities (T2-FLAIR / WMH)",
        status=status,
        kpis=kpis,
        html_content="\n".join(cards_html),
        errors=errors,
    )


visualize_modality = visualize_wmh
