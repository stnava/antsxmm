"""Perfusion / ASL (Arterial Spin Labeling) modality visualizer.

Provides publication-grade visual reporting for Perfusion / ASL:
- Mean Cerebral Blood Flow (CBF) map with physiological scaling (0-150 ml/100g/min)
- Gray matter / tissue segmentation alignment gallery
- Temporal Signal-to-Noise Ratio (tSNR) map
- M0-to-perfusion coregistration checker (m0.nii.gz vs cbf.nii.gz)
- Robust multi-row CSV handling coalescing global (row 0) and regional (row 1) metrics
- Clinical KPI metric cards (CBF mean, GM CBF mean, tSNR mean, M0 mean, Perfusion mean).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from antsxmm.visualize.core import (
    DARK_BG,
    DARK_BORDER,
    DARK_MUTED,
    DARK_PANEL_BG,
    DARK_TEXT,
    figure_to_base64,
    render_ortho_montage,
    render_slice_gallery,
)
from antsxmm.visualize.modalities import (
    ModalityReport,
    coalesce_multi_row_df,
    find_file,
    find_run_dir,
    load_nifti_data,
    safe_read_csv,
)
from antsxmm.visualize.theme import render_card, render_kpi_card

logger = logging.getLogger(__name__)


def can_visualize(modality_dir: Path | str) -> bool:
    """Determine if directory contains Perfusion / ASL outputs."""
    p = Path(modality_dir)
    name_lower = p.name.lower()
    if any(k in name_lower for k in ("perf", "perfusion", "asl", "pcasl")):
        return True

    run_dir = find_run_dir(p)
    return any(
        find_file(run_dir, pat) is not None
        for pat in (
            "cbf.nii.gz",
            "perfusion.nii.gz",
            "m0.nii.gz",
            "*perf*mmwide.csv",
        )
    )


def _render_regional_cbf_bars(
    series: pd.Series,
    title: str = "Regional Mean Cerebral Blood Flow (ml/100g/min)",
    top_n: int = 12,
) -> str:
    """Render horizontal bar chart of top regional CBF values."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150, facecolor=DARK_BG)
    try:
        ax.set_facecolor(DARK_PANEL_BG)

        # Extract regional mean entries
        regional: dict[str, float] = {}
        for k, v in series.items():
            k_str = str(k)
            if (k_str.startswith("mean_") or k_str.startswith("cbf_mean_")) and isinstance(v, (int, float)) and not np.isnan(v):
                clean_name = k_str.replace("cbf_mean_", "").replace("mean_", "").replace("_", " ").title()
                regional[clean_name] = float(v)

        if not regional:
            # Fallback dummy bar if no regional columns
            regional = {"Global CBF": float(series.get("cbf_mean", 50.0))}

        # Sort and select top regions
        sorted_items = sorted(regional.items(), key=lambda item: item[1], reverse=True)[:top_n]
        labels = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values, color="#38bdf8", edgecolor="none", height=0.6)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, color=DARK_TEXT, fontsize=9)
        ax.set_xlabel("Mean CBF (ml/100g/min)", color=DARK_TEXT, fontsize=9)
        ax.tick_params(colors=DARK_MUTED, labelsize=8)
        ax.grid(True, axis="x", linestyle=":", alpha=0.3, color=DARK_BORDER)

        for spine in ax.spines.values():
            spine.set_color(DARK_BORDER)

        max_val = max(values) if values else 100.0
        for bar, val in zip(bars, values):
            ax.text(
                val + max_val * 0.02,
                bar.get_y() + bar.get_height() / 2.0,
                f"{val:.1f}",
                va="center",
                ha="left",
                color=DARK_TEXT,
                fontsize=8,
                fontweight="bold",
            )

        ax.set_xlim(0, max_val * 1.22)
        ax.set_title(title, color=DARK_TEXT, fontsize=11, fontweight="bold", pad=10)
        fig.tight_layout()
        return figure_to_base64(fig, format="png", dpi=150, close=True)
    finally:
        plt.close(fig)


def visualize_perfusion(
    modality_dir: Path | str,
    session_dir: Path | str | None = None,
    **kwargs: Any,
) -> ModalityReport:
    """Generate visual report for Arterial Spin Labeling / Perfusion."""
    mod_path = Path(modality_dir)
    errors: list[str] = []
    cards_html: list[str] = []
    kpis: list[dict[str, Any]] = []

    run_dir = find_run_dir(mod_path)

    # 1. Locate files
    cbf_file = find_file(run_dir, "cbf.nii.gz") or find_file(run_dir, "*cbf*.nii.gz")
    m0_file = find_file(run_dir, "m0.nii.gz") or find_file(run_dir, "*m0*.nii.gz")
    perf_file = find_file(run_dir, "perfusion.nii.gz") or find_file(run_dir, "*perfusion*.nii.gz")
    gm_file = find_file(run_dir, "gmseg.nii.gz") or find_file(run_dir, "*gmseg*.nii.gz")
    tsnr_file = find_file(run_dir, "tsnr.nii.gz") or find_file(run_dir, "*tsnr*.nii.gz")
    mmwide_csv = find_file(run_dir, "mmwide.csv") or find_file(mod_path, "mmwide.csv")

    df_mmwide = safe_read_csv(mmwide_csv)
    # Multi-row CSV handling: coalesce row 0 (global) and row 1 (regional)
    metrics_series = coalesce_multi_row_df(df_mmwide)

    # 2. Extract metrics
    cbf_mean = metrics_series.get("cbf_mean")
    cbf_gm_mean = metrics_series.get("cbf_gm_mean")
    m0_mean = metrics_series.get("m0_mean")
    perf_mean = metrics_series.get("perfusion_mean")
    tsnr_mean = metrics_series.get("tsnr_mean")

    # 3. Build KPI metric cards
    if cbf_mean is not None and not np.isnan(float(cbf_mean)):
        cbf_val = float(cbf_mean)
        kpis.append({
            "label": "Global Mean CBF",
            "value": f"{cbf_val:.1f}",
            "unit": "ml/100g/min",
            "status": "normal" if (20.0 <= cbf_val <= 120.0) else "warning",
            "tooltip": "Quantitative whole-brain Cerebral Blood Flow (Nominal: 40 - 80 ml/100g/min).",
        })

    if cbf_gm_mean is not None and not np.isnan(float(cbf_gm_mean)):
        gm_cbf_val = float(cbf_gm_mean)
        kpis.append({
            "label": "Gray Matter CBF",
            "value": f"{gm_cbf_val:.1f}",
            "unit": "ml/100g/min",
            "status": "normal",
            "tooltip": "Mean perfusion restricted to segmented cerebral gray matter cortex.",
        })

    if m0_mean is not None and not np.isnan(float(m0_mean)):
        kpis.append({
            "label": "M0 Reference Mean",
            "value": f"{float(m0_mean):.1f}",
            "unit": "",
            "status": "normal",
            "tooltip": "Equilibrium blood/tissue magnetization intensity used for quantitative scaling.",
        })

    if perf_mean is not None and not np.isnan(float(perf_mean)):
        kpis.append({
            "label": "Perfusion Difference",
            "value": f"{float(perf_mean):.2f}",
            "unit": "",
            "status": "normal",
            "tooltip": "Raw label-control subtraction signal amplitude.",
        })

    if tsnr_mean is not None and not np.isnan(float(tsnr_mean)):
        kpis.append({
            "label": "Perfusion tSNR",
            "value": f"{float(tsnr_mean):.4g}",
            "unit": "",
            "status": "normal",
            "tooltip": "Temporal signal-to-noise ratio across raw ASL timeframes.",
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

    # 4. Load NIfTI Images
    cbf_arr = load_nifti_data(cbf_file)
    m0_arr = load_nifti_data(m0_file)
    gm_arr = load_nifti_data(gm_file)
    tsnr_arr = load_nifti_data(tsnr_file)

    # Card 1: Quantitative CBF Ortho Montage with Physiological Scaling (0-150 ml/100g/min)
    if cbf_arr is not None:
        try:
            cbf_uri = render_ortho_montage(
                underlay=cbf_arr,
                overlay=None,
                cmap="turbo",
                vmin=0.0,
                vmax=150.0,
                title="Quantitative Cerebral Blood Flow (CBF: 0 - 150 ml/100g/min)",
            )
            img_html = f'<div class="img-container"><img src="{cbf_uri}" class="img-responsive" alt="Quantitative CBF" /></div>'
            cards_html.append(
                render_card(
                    title="Quantitative Cerebral Blood Flow (CBF) Map",
                    subtitle="Physiologically calibrated perfusion map scaled from 0 to 150 ml/100g/min",
                    content_html=img_html,
                    badge="Physiological Scale",
                )
            )
        except Exception as exc:
            errors.append(f"Failed to render CBF ortho montage: {exc}")

    # Card 2: GM/WM Tissue Alignment Gallery
    if cbf_arr is not None:
        try:
            gm_gallery_uri = render_slice_gallery(
                underlay=cbf_arr,
                overlay=gm_arr if (gm_arr is not None and gm_arr.shape == cbf_arr.shape) else None,
                axis=2,
                nslices=7,
                contours=True,
                overlay_cmap="spring",
                title="CBF with Gray Matter Segmentation Alignment Contours",
            )
            img_html = f'<div class="img-container"><img src="{gm_gallery_uri}" class="img-responsive" alt="GM Alignment" /></div>'
            cards_html.append(
                render_card(
                    title="Gray Matter Segmentation Alignment & Spatial Verification",
                    subtitle="Multi-slice axial gallery verifying co-registration of cortical gray matter onto perfusion space",
                    content_html=img_html,
                    badge="Alignment QC",
                )
            )
        except Exception as exc:
            errors.append(f"Failed to render GM alignment gallery: {exc}")

    # Card 3: M0-to-Perfusion Coregistration Checker
    if m0_arr is not None:
        try:
            m0_uri = render_ortho_montage(
                underlay=m0_arr,
                overlay=(cbf_arr > 10.0).astype(float) if (cbf_arr is not None and cbf_arr.shape == m0_arr.shape) else None,
                overlay_cmap="hot",
                overlay_alpha=0.45,
                title="M0 Equilibrium Magnetization & Perfusion Coregistration Check",
            )
            img_html = f'<div class="img-container"><img src="{m0_uri}" class="img-responsive" alt="M0 Coregistration" /></div>'
            cards_html.append(
                render_card(
                    title="M0-to-Perfusion Coregistration Checker",
                    subtitle="M0 calibration underlay with active perfusion boundary overlay verifying geometric congruency",
                    content_html=img_html,
                    badge="Calibration QC",
                )
            )
        except Exception as exc:
            errors.append(f"Failed to render M0 coregistration montage: {exc}")

    # Card 4: Perfusion tSNR Map Gallery
    if tsnr_arr is not None:
        try:
            tsnr_uri = render_slice_gallery(
                underlay=tsnr_arr,
                overlay=None,
                axis=2,
                nslices=7,
                contours=False,
                cmap="magma",
                title="Perfusion Temporal Signal-to-Noise Ratio (tSNR)",
            )
            img_html = f'<div class="img-container"><img src="{tsnr_uri}" class="img-responsive" alt="Perfusion tSNR" /></div>'
            cards_html.append(
                render_card(
                    title="Temporal Signal-to-Noise Ratio (tSNR) Map",
                    subtitle="Axial gallery assessing temporal stability across label and control acquisition frames",
                    content_html=img_html,
                    badge="tSNR",
                )
            )
        except Exception as exc:
            errors.append(f"Failed to render tSNR gallery: {exc}")
    elif tsnr_mean is not None and not np.isnan(float(tsnr_mean)):
        tsnr_info_html = (
            f'<div class="kpi-value-container" style="padding: 16px;">'
            f'<span class="kpi-value">{float(tsnr_mean):.1f}</span> '
            f'<span class="kpi-unit">dB</span>'
            f'<p style="color: var(--muted); margin-top: 8px;">Temporal Signal-to-Noise Ratio across ASL label and control acquisition series.</p>'
            f'</div>'
        )
        cards_html.append(
            render_card(
                title="Temporal Signal-to-Noise Ratio (tSNR)",
                subtitle="Whole-brain temporal stability metric",
                content_html=tsnr_info_html,
                badge="tSNR",
            )
        )

    # Card 5: Regional CBF Breakdown Bar Chart
    if len(metrics_series) > 5:
        try:
            bar_uri = _render_regional_cbf_bars(metrics_series, title="Regional Mean CBF Distribution")
            img_html = f'<div class="img-container"><img src="{bar_uri}" class="img-responsive" alt="Regional CBF" /></div>'
            cards_html.append(
                render_card(
                    title="Regional Cerebral Blood Flow Breakdown",
                    subtitle="Quantitative perfusion values across key cortical and subcortical anatomical parcels",
                    content_html=img_html,
                    badge="Regional",
                )
            )
        except Exception as exc:
            errors.append(f"Failed to render regional CBF bars: {exc}")

    status = "error" if (cbf_arr is None and not kpis) else ("warning" if errors else "success")

    return ModalityReport(
        name="perf",
        title="Arterial Spin Labeling Perfusion (ASL / CBF)",
        status=status,
        kpis=kpis,
        html_content="\n".join(cards_html),
        errors=errors,
    )


visualize_modality = visualize_perfusion
