"""PET3D Positron Emission Tomography modality visualizer.

Provides publication-grade visual reporting for 3D PET:
- High-resolution resampled PET tracer intensity underlay with brain mask overlay
- PET-to-anatomical coregistration and boundary alignment checks
- Multi-slice axial tracer distribution gallery
- Regional tracer uptake bar charts across DKT and subcortical parcels
- Multi-row CSV handling coalescing global summary metrics (row 0) and regional values (row 1)
- Clinical KPI metric cards (PET mean, GM uptake, WM uptake, CSF uptake, GM/WM ratio).
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
    """Determine if directory contains PET3D outputs."""
    p = Path(modality_dir)
    name_lower = p.name.lower()
    if any(k in name_lower for k in ("pet3d", "pet")):
        return True

    run_dir = find_run_dir(p)
    return any(
        find_file(run_dir, pat) is not None
        for pat in (
            "pet3d.nii.gz",
            "*pet*.nii.gz",
            "*pet3d*mmwide.csv",
        )
    )


def _render_regional_uptake_bars(
    series: pd.Series,
    title: str = "Regional Mean PET Tracer Uptake Intensity",
    top_n: int = 14,
) -> str:
    """Render horizontal bar chart of top regional tracer uptake values."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=150, facecolor=DARK_BG)
    try:
        ax.set_facecolor(DARK_PANEL_BG)

        regional: dict[str, float] = {}
        for k, v in series.items():
            k_str = str(k)
            if k_str.startswith("mean_") and isinstance(v, (int, float)) and not np.isnan(v):
                clean_name = k_str.replace("mean_", "").replace("_", " ").title()
                regional[clean_name] = float(v)

        if not regional:
            regional = {"Whole Brain PET": float(series.get("pet3d_mean", 2.5))}

        sorted_items = sorted(regional.items(), key=lambda item: item[1], reverse=True)[:top_n]
        labels = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values, color="#f43f5e", edgecolor="none", height=0.6)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, color=DARK_TEXT, fontsize=9)
        ax.set_xlabel("Tracer Uptake Intensity (SUV / normalized)", color=DARK_TEXT, fontsize=9)
        ax.tick_params(colors=DARK_MUTED, labelsize=8)
        ax.grid(True, axis="x", linestyle=":", alpha=0.3, color=DARK_BORDER)

        for spine in ax.spines.values():
            spine.set_color(DARK_BORDER)

        max_val = max(values) if values else 10.0
        for bar, val in zip(bars, values):
            ax.text(
                val + max_val * 0.02,
                bar.get_y() + bar.get_height() / 2.0,
                f"{val:.2f}",
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


def visualize_pet(
    modality_dir: Path | str,
    session_dir: Path | str | None = None,
    **kwargs: Any,
) -> ModalityReport:
    """Generate visual report for PET3D Positron Emission Tomography."""
    mod_path = Path(modality_dir)
    errors: list[str] = []
    cards_html: list[str] = []
    kpis: list[dict[str, Any]] = []

    run_dir = find_run_dir(mod_path)

    # 1. Locate files
    pet_file = find_file(run_dir, "pet3d.nii.gz") or find_file(run_dir, "*pet*.nii.gz")
    mask_file = find_file(run_dir, "brainmask.nii.gz") or find_file(run_dir, "*brainmask*.nii.gz")
    mmwide_csv = find_file(run_dir, "mmwide.csv") or find_file(mod_path, "mmwide.csv")

    df_mmwide = safe_read_csv(mmwide_csv)
    # Multi-row CSV handling: coalesce row 0 (global) and row 1 (regional)
    metrics_series = coalesce_multi_row_df(df_mmwide)

    # 2. Load Images early for fallback calculations
    pet_arr = load_nifti_data(pet_file)
    mask_arr = load_nifti_data(mask_file)

    # 3. Extract metrics
    pet_mean = metrics_series.get("pet3d_mean") or metrics_series.get("pet_mean")
    gm_mean = metrics_series.get("gm_mean") or metrics_series.get("GM_mean")
    wm_mean = metrics_series.get("wm_mean") or metrics_series.get("WM_mean")
    csf_mean = metrics_series.get("csf_mean") or metrics_series.get("CSF_mean")

    if pet_mean is None and (df_mmwide is not None and not df_mmwide.empty) and pet_arr is not None:
        pos_pet = pet_arr[pet_arr > 0]
        if len(pos_pet) > 0:
            pet_mean = float(np.mean(pos_pet))

    gm_wm_ratio: float | None = None
    if gm_mean is not None and wm_mean is not None and not np.isnan(float(gm_mean)) and not np.isnan(float(wm_mean)):
        if float(wm_mean) > 0:
            gm_wm_ratio = float(gm_mean) / float(wm_mean)

    # 4. Build KPI metric cards
    if pet_mean is not None and not np.isnan(float(pet_mean)):
        kpis.append({
            "label": "Mean Whole Brain PET",
            "value": f"{float(pet_mean):.2f}",
            "unit": "",
            "status": "normal",
            "tooltip": "Volumetric average of PET tracer uptake intensity across whole-brain mask.",
        })

    if gm_mean is not None and not np.isnan(float(gm_mean)):
        kpis.append({
            "label": "Gray Matter Uptake",
            "value": f"{float(gm_mean):.2f}",
            "unit": "",
            "status": "normal",
            "tooltip": "Mean tracer intensity within segmented cerebral cortical gray matter.",
        })

    if wm_mean is not None and not np.isnan(float(wm_mean)):
        kpis.append({
            "label": "White Matter Uptake",
            "value": f"{float(wm_mean):.2f}",
            "unit": "",
            "status": "normal",
            "tooltip": "Mean tracer intensity within white matter reference parenchyma.",
        })

    if csf_mean is not None and not np.isnan(float(csf_mean)):
        kpis.append({
            "label": "CSF Non-Specific Background",
            "value": f"{float(csf_mean):.2f}",
            "unit": "",
            "status": "normal",
            "tooltip": "Tracer concentration in ventricular CSF (non-specific reference).",
        })

    if gm_wm_ratio is not None:
        kpis.append({
            "label": "GM / WM Ratio",
            "value": f"{gm_wm_ratio:.2f}",
            "unit": "ratio",
            "status": "normal" if gm_wm_ratio >= 1.0 else "warning",
            "tooltip": "Cortical gray matter contrast relative to subcortical white matter.",
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

    # Card 1: 3D PET Intensity Ortho Montage
    if pet_arr is not None:
        try:
            pet_vmax = float(np.percentile(pet_arr[pet_arr > 0], 99)) if np.any(pet_arr > 0) else 50.0
            ortho_uri = render_ortho_montage(
                underlay=pet_arr,
                overlay=mask_arr if (mask_arr is not None and mask_arr.shape == pet_arr.shape) else None,
                overlay_cmap="cool",
                overlay_alpha=0.3,
                cmap="inferno",
                vmin=0.0,
                vmax=pet_vmax,
                title="3D PET Tracer Intensity Distribution (Inferno Palette)",
            )
            img_html = f'<div class="img-container"><img src="{ortho_uri}" class="img-responsive" alt="3D PET Ortho Montage" /></div>'
            cards_html.append(
                render_card(
                    title="3D PET Tracer Intensity & Spatial Extent",
                    subtitle="3-view orthogonal montage displaying molecular tracer concentration with brain boundary overlay",
                    content_html=img_html,
                    badge="PET3D",
                )
            )
        except Exception as exc:
            errors.append(f"Failed to render PET ortho montage: {exc}")

    # Card 2: Multi-Slice PET Intensity & Alignment Gallery
    if pet_arr is not None:
        try:
            gallery_uri = render_slice_gallery(
                underlay=pet_arr,
                overlay=mask_arr if (mask_arr is not None and mask_arr.shape == pet_arr.shape) else None,
                axis=2,
                nslices=7,
                contours=True,
                overlay_cmap="cool",
                cmap="inferno",
                title="Axial PET Tracer Slices with Brain Boundary Contours",
            )
            img_html = f'<div class="img-container"><img src="{gallery_uri}" class="img-responsive" alt="PET Slice Gallery" /></div>'
            cards_html.append(
                render_card(
                    title="PET-to-Anatomical Boundary Verification",
                    subtitle="Multi-slice axial gallery confirming skull stripping and spatial normalization accuracy",
                    content_html=img_html,
                    badge="Alignment QC",
                )
            )
        except Exception as exc:
            errors.append(f"Failed to render PET slice gallery: {exc}")

    # Card 3: Regional Tracer Uptake Bar Graph
    if len(metrics_series) > 5:
        try:
            bars_uri = _render_regional_uptake_bars(metrics_series, title="Regional Tracer Uptake Across Brain Parcels")
            img_html = f'<div class="img-container"><img src="{bars_uri}" class="img-responsive" alt="Regional Uptake Bars" /></div>'
            cards_html.append(
                render_card(
                    title="Regional Tracer Uptake Breakdown",
                    subtitle="Target binding concentrations across cortical and deep subcortical structures",
                    content_html=img_html,
                    badge="Regional",
                )
            )
        except Exception as exc:
            errors.append(f"Failed to render regional uptake bars: {exc}")

    status = "error" if (pet_arr is None and not kpis) else ("warning" if errors else "success")

    return ModalityReport(
        name="pet3d",
        title="Positron Emission Tomography (PET3D)",
        status=status,
        kpis=kpis,
        html_content="\n".join(cards_html),
        errors=errors,
    )


visualize_modality = visualize_pet
