"""DTI Diffusion Tensor Imaging modality visualizer.

Provides publication-grade visual reporting for DTI:
- B0 reference average underlay multi-slice gallery
- Fractional Anisotropy (FA) and Mean Diffusivity (MD) parametric maps
- Direction-encoded gradient sampling geometry and b-value shell verification
- Motion and diffusion quality metrics & clinical KPI cards (FA_mean, MD_mean, dti_fa_SNR, motion count).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

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
    find_file,
    find_run_dir,
    load_nifti_data,
    safe_read_csv,
)
from antsxmm.visualize.theme import render_card, render_kpi_card

logger = logging.getLogger(__name__)


def can_visualize(modality_dir: Path | str) -> bool:
    """Determine if directory contains DTI diffusion outputs."""
    p = Path(modality_dir)
    name_lower = p.name.lower()
    if any(k in name_lower for k in ("dti", "diffusion", "dwi")):
        return True

    run_dir = find_run_dir(p)
    return any(
        find_file(run_dir, pat) is not None
        for pat in (
            "dtifa.nii.gz",
            "dtimd.nii.gz",
            "b0avg.nii.gz",
            "reoriented.bval",
            "*DTI*mmwide.csv",
        )
    )


def _render_gradient_sampling_plot(
    bvals: np.ndarray,
    bvecs: np.ndarray,
    title: str = "Diffusion Gradient Direction Sampling & Shells",
) -> str:
    """Render 3D gradient vector distribution on the unit sphere alongside shell counts."""
    fig = plt.figure(figsize=(10.0, 4.2), dpi=150, facecolor=DARK_BG)
    try:
        # If bvecs is shape (3, N), transpose to (N, 3)
        if bvecs.shape[0] == 3 and bvecs.ndim == 2:
            vecs = bvecs.T
        else:
            vecs = bvecs

        # Subplot 1: 3D Direction Sphere
        ax3d = fig.add_subplot(1, 2, 1, projection="3d", facecolor=DARK_PANEL_BG)
        ax3d.set_facecolor(DARK_PANEL_BG)

        # Draw wireframe unit sphere for spatial reference
        u = np.linspace(0, 2 * np.pi, 24)
        v = np.linspace(0, np.pi, 16)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        ax3d.plot_wireframe(x_sphere, y_sphere, z_sphere, color=DARK_BORDER, alpha=0.25, linewidth=0.6)

        # Plot b0 vs diffusion directions
        unique_shells = np.unique(np.round(bvals, -2))
        colors = ["#94a3b8", "#38bdf8", "#34d399", "#f472b6", "#fbbf24"]

        for idx, shell in enumerate(unique_shells):
            shell_mask = np.abs(bvals - shell) < 150
            sub_vecs = vecs[shell_mask]
            c = colors[idx % len(colors)]
            label = f"b = {int(shell)}" if shell > 0 else "b = 0 (Ref)"
            ax3d.scatter(
                sub_vecs[:, 0],
                sub_vecs[:, 1],
                sub_vecs[:, 2],
                c=c,
                s=25,
                alpha=0.85,
                label=label,
                edgecolors="none",
            )
            # Plot antipodal directions for diffusion symmetry
            ax3d.scatter(
                -sub_vecs[:, 0],
                -sub_vecs[:, 1],
                -sub_vecs[:, 2],
                c=c,
                s=12,
                alpha=0.35,
                edgecolors="none",
            )

        ax3d.set_xlim([-1.1, 1.1])
        ax3d.set_ylim([-1.1, 1.1])
        ax3d.set_zlim([-1.1, 1.1])
        ax3d.set_xlabel("X (R-L)", color=DARK_TEXT, fontsize=8)
        ax3d.set_ylabel("Y (A-P)", color=DARK_TEXT, fontsize=8)
        ax3d.set_zlabel("Z (I-S)", color=DARK_TEXT, fontsize=8)
        ax3d.tick_params(colors=DARK_MUTED, labelsize=7)
        ax3d.legend(loc="upper left", fontsize=8, facecolor=DARK_PANEL_BG, edgecolor=DARK_BORDER, labelcolor=DARK_TEXT)
        ax3d.set_title("Gradient Angular Distribution", color=DARK_TEXT, fontsize=10, fontweight="bold")

        # Subplot 2: Shell histogram
        ax_bar = fig.add_subplot(1, 2, 2, facecolor=DARK_PANEL_BG)
        shell_names = []
        shell_counts = []
        bar_colors = []
        for idx, shell in enumerate(unique_shells):
            count = int(np.sum(np.abs(bvals - shell) < 150))
            name = f"b={int(shell)}" if shell > 0 else "b=0"
            shell_names.append(name)
            shell_counts.append(count)
            bar_colors.append(colors[idx % len(colors)])

        y_pos = np.arange(len(shell_names))
        bars = ax_bar.bar(y_pos, shell_counts, color=bar_colors, width=0.5, edgecolor="none")
        ax_bar.set_xticks(y_pos)
        ax_bar.set_xticklabels(shell_names, color=DARK_TEXT, fontsize=9, fontweight="bold")
        ax_bar.set_ylabel("Number of Directions", color=DARK_TEXT, fontsize=9)
        ax_bar.tick_params(colors=DARK_MUTED, labelsize=8)
        ax_bar.grid(True, axis="y", linestyle=":", alpha=0.3, color=DARK_BORDER)

        for spine in ax_bar.spines.values():
            spine.set_color(DARK_BORDER)

        max_c = max(shell_counts) if shell_counts else 10
        for bar, cnt in zip(bars, shell_counts):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2.0,
                cnt + max_c * 0.03,
                str(cnt),
                ha="center",
                va="bottom",
                color=DARK_TEXT,
                fontsize=9,
                fontweight="bold",
            )
        ax_bar.set_ylim(0, max_c * 1.25)
        ax_bar.set_title("Shell Acquisition Counts", color=DARK_TEXT, fontsize=10, fontweight="bold")

        fig.suptitle(title, color=DARK_TEXT, fontsize=12, fontweight="bold", y=0.98)
        fig.tight_layout()
        return figure_to_base64(fig, format="png", dpi=150, close=True)
    finally:
        plt.close(fig)


def visualize_dti(
    modality_dir: Path | str,
    session_dir: Path | str | None = None,
    **kwargs: Any,
) -> ModalityReport:
    """Generate visual report for DTI Diffusion Tensor Imaging."""
    mod_path = Path(modality_dir)
    errors: list[str] = []
    cards_html: list[str] = []
    kpis: list[dict[str, Any]] = []

    run_dir = find_run_dir(mod_path)

    # 1. Locate files
    fa_file = find_file(run_dir, "dtifa.nii.gz")
    md_file = find_file(run_dir, "dtimd.nii.gz")
    b0_file = find_file(run_dir, "b0avg.nii.gz") or find_file(run_dir, "dwiavg.nii.gz")
    bval_file = find_file(run_dir, "reoriented.bval") or find_file(run_dir, "*.bval")
    bvec_file = find_file(run_dir, "reoriented.bvec") or find_file(run_dir, "*.bvec")
    mmwide_csv = find_file(run_dir, "mmwide.csv") or find_file(mod_path, "mmwide.csv")

    fa_arr = load_nifti_data(fa_file)
    md_arr = load_nifti_data(md_file)
    b0_arr = load_nifti_data(b0_file)

    df_mmwide = safe_read_csv(mmwide_csv)

    # 2. Extract metrics
    fa_mean: float | None = None
    md_mean: float | None = None
    fa_snr: float | None = None
    motion_count: float | None = None
    n_dirs: int | None = None

    if df_mmwide is not None and not df_mmwide.empty:
        row = df_mmwide.iloc[0]
        if "FA_mean" in row and not np.isnan(row["FA_mean"]):
            fa_mean = float(row["FA_mean"])
        if "MD_mean" in row and not np.isnan(row["MD_mean"]):
            md_mean = float(row["MD_mean"])
        if "dti_fa_SNR" in row and not np.isnan(row["dti_fa_SNR"]):
            fa_snr = float(row["dti_fa_SNR"])
        if "dti_high_motion_count" in row and not np.isnan(row["dti_high_motion_count"]):
            motion_count = float(row["dti_high_motion_count"])

    # Load bvals/bvecs if available
    bvals: np.ndarray | None = None
    bvecs: np.ndarray | None = None
    if bval_file and bvec_file:
        try:
            bvals = np.loadtxt(bval_file)
            bvecs = np.loadtxt(bvec_file)
            n_dirs = len(bvals)
        except Exception as exc:
            errors.append(f"Failed to parse gradient tables: {exc}")

    # 3. KPI Metric Cards
    if fa_mean is not None:
        kpis.append({
            "label": "Mean FA (Whole Brain)",
            "value": f"{fa_mean:.3f}",
            "unit": "",
            "status": "normal" if fa_mean >= 0.01 else "warning",
            "tooltip": "Fractional Anisotropy whole-brain voxel mean (directional white matter integrity).",
        })

    if md_mean is not None:
        md_display = f"{md_mean * 1e3:.2f}e-3" if md_mean < 0.01 else f"{md_mean:.4f}"
        kpis.append({
            "label": "Mean Diffusivity (MD)",
            "value": md_display,
            "unit": "mm²/s",
            "status": "normal",
            "tooltip": "Mean Diffusivity (Trace(D)/3), quantifying isotropic molecular displacement.",
        })

    if fa_snr is not None:
        kpis.append({
            "label": "FA SNR",
            "value": f"{fa_snr:.1f}",
            "unit": "dB",
            "status": "normal" if fa_snr >= 20.0 else "warning",
            "tooltip": "Signal-to-noise ratio of reconstructed diffusion fractional anisotropy tensor.",
        })
    elif fa_arr is not None:
        pos_fa = fa_arr[fa_arr > 0]
        if len(pos_fa) > 0:
            std_v = float(np.std(pos_fa))
            est_snr = float(np.mean(pos_fa)) / (std_v + 1e-6)
            kpis.append({
                "label": "FA SNR",
                "value": f"{est_snr:.1f}",
                "unit": "dB",
                "status": "normal",
                "tooltip": "Estimated whole-brain diffusion FA signal-to-noise ratio.",
            })

    if n_dirs is not None:
        kpis.append({
            "label": "Gradient Directions",
            "value": str(n_dirs),
            "unit": "dirs",
            "status": "normal" if n_dirs >= 30 else "warning",
            "tooltip": "Total number of diffusion-weighted gradient orientations in acquisition shell.",
        })

    if motion_count is not None:
        kpis.append({
            "label": "High Motion Volumes",
            "value": f"{int(motion_count)}",
            "unit": "vols",
            "status": "normal" if motion_count < 20 else "warning",
            "tooltip": "Diffusion volumes flagged with elevated framewise displacement or signal dropout.",
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

    # 4. Load NIfTIs
    fa_arr = load_nifti_data(fa_file)
    md_arr = load_nifti_data(md_file)
    b0_arr = load_nifti_data(b0_file)

    # Card 1: FA Map Ortho Montage
    if fa_arr is not None:
        try:
            fa_uri = render_ortho_montage(
                underlay=fa_arr,
                overlay=None,
                cmap="inferno",
                vmin=0.0,
                vmax=float(np.percentile(fa_arr[fa_arr > 0], 98)) if np.any(fa_arr > 0) else 0.8,
                title="Fractional Anisotropy (FA) Scalar Field (Inferno Palette)",
            )
            img_html = f'<div class="img-container"><img src="{fa_uri}" class="img-responsive" alt="FA Ortho Montage" /></div>'
            cards_html.append(
                render_card(
                    title="Fractional Anisotropy (FA) Map",
                    subtitle="3-view orthogonal montage of directional anisotropy highlighting major tract pathways",
                    content_html=img_html,
                    badge="FA Map",
                )
            )
        except Exception as exc:
            errors.append(f"Failed to render FA ortho montage: {exc}")

    # Card 2: MD Map Ortho Montage
    if md_arr is not None:
        try:
            md_uri = render_ortho_montage(
                underlay=md_arr,
                overlay=None,
                cmap="viridis",
                vmin=0.0,
                vmax=float(np.percentile(md_arr[md_arr > 0], 95)) if np.any(md_arr > 0) else 0.003,
                title="Mean Diffusivity (MD) Scalar Field (Viridis Palette)",
            )
            img_html = f'<div class="img-container"><img src="{md_uri}" class="img-responsive" alt="MD Ortho Montage" /></div>'
            cards_html.append(
                render_card(
                    title="Mean Diffusivity (MD) Map",
                    subtitle="3-view orthogonal montage quantifying cellular microstructural density",
                    content_html=img_html,
                    badge="MD Map",
                )
            )
        except Exception as exc:
            errors.append(f"Failed to render MD ortho montage: {exc}")

    # Card 3: B0 Average Reference Underlay Gallery
    if b0_arr is not None:
        try:
            b0_gallery_uri = render_slice_gallery(
                underlay=b0_arr,
                overlay=None,
                axis=2,
                nslices=7,
                title="Averaged B0 Diffusion Reference Slices (Axial)",
            )
            img_html = f'<div class="img-container"><img src="{b0_gallery_uri}" class="img-responsive" alt="B0 Slice Gallery" /></div>'
            cards_html.append(
                render_card(
                    title="Averaged B0 Diffusion Reference Anatomical Gallery",
                    subtitle="Multi-slice gallery of unweighted diffusion volumes verifying SNR and susceptibility alignment",
                    content_html=img_html,
                    badge="B0 Reference",
                )
            )
        except Exception as exc:
            errors.append(f"Failed to render B0 slice gallery: {exc}")

    # Card 4: Gradient Vector Distribution & Shell Verification
    if bvals is not None and bvecs is not None and len(bvals) > 0:
        try:
            grad_uri = _render_gradient_sampling_plot(
                bvals=bvals,
                bvecs=bvecs,
                title=f"Diffusion Gradient Sampling Geometry ({len(bvals)} Directions)",
            )
            img_html = f'<div class="img-container"><img src="{grad_uri}" class="img-responsive" alt="Gradient Sampling Sphere" /></div>'
            cards_html.append(
                render_card(
                    title="Gradient Direction Sampling Geometry & Shells",
                    subtitle="3D spherical vector distribution verifying angular coverage, b-value shells, and symmetry",
                    content_html=img_html,
                    badge="Acquisition QC",
                )
            )
        except Exception as exc:
            errors.append(f"Failed to render gradient sampling plot: {exc}")

    status = "error" if (fa_arr is None and not kpis) else ("warning" if errors else "success")

    return ModalityReport(
        name="DTI",
        title="Diffusion Tensor Imaging (DTI)",
        status=status,
        kpis=kpis,
        html_content="\n".join(cards_html),
        errors=errors,
    )


visualize_modality = visualize_dti
