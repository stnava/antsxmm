"""Core visual generation components for neuroimaging reporting.

Provides headless matplotlib plotting utilities, base64 PNG/SVG serializers,
orthogonal slice montages, multi-slice contour galleries, 4D carpet plots,
distribution violins, and correlation matrix heatmaps.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Sequence

import matplotlib

# Force headless Agg backend prior to importing pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

# Clinical dark theme palette constants
DARK_BG = "#0b0f19"
DARK_PANEL_BG = "#162032"
DARK_TEXT = "#f8fafc"
DARK_MUTED = "#94a3b8"
DARK_BORDER = "#27354a"
ACCENT_CYAN = "#38bdf8"
ACCENT_GREEN = "#34d399"
ACCENT_RED = "#f87171"


def figure_to_base64(
    fig: matplotlib.figure.Figure,
    format: str = "png",
    dpi: int = 150,
    close: bool = True,
) -> str:
    """Serialize a Matplotlib figure into a base64 Data URI string.

    Guarantees closing the figure in a finally block to prevent memory leaks.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to serialize.
    format : str, optional
        Image format: "png", "svg", "jpeg". Default is "png".
    dpi : int, optional
        Resolution for raster export. Default is 150.
    close : bool, optional
        Whether to close the figure after serialization. Default is True.

    Returns
    -------
    str
        Base64 data URI string (e.g. "data:image/png;base64,...").
    """
    buf = io.BytesIO()
    try:
        fig.savefig(
            buf,
            format=format,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        mime = "image/svg+xml" if format.lower() == "svg" else f"image/{format.lower()}"
        return f"data:{mime};base64,{encoded}"
    finally:
        buf.close()
        if close:
            plt.close(fig)


def image_to_base64(
    img_array: np.ndarray,
    format: str = "png",
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    dpi: int = 150,
) -> str:
    """Serialize a 2D numpy array directly to a base64 Data URI string.

    Parameters
    ----------
    img_array : np.ndarray
        2D image array.
    format : str, optional
        Output format. Default is "png".
    cmap : str, optional
        Colormap name. Default is "gray".
    vmin, vmax : float, optional
        Intensity scale bounds.
    dpi : int, optional
        Rendering DPI.

    Returns
    -------
    str
        Base64 data URI string.
    """
    arr = np.asarray(img_array)
    if arr.ndim != 2:
        raise ValueError(f"image_to_base64 expects a 2D array, got shape {arr.shape}")

    h, w = arr.shape
    figsize = (max(2.0, w / 80.0), max(2.0, h / 80.0))
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=DARK_BG)
    try:
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        return figure_to_base64(fig, format=format, dpi=dpi, close=True)
    finally:
        plt.close(fig)


def _to_numpy(img: Any) -> np.ndarray:
    """Convert an ANTsImage or array-like object to a 3D or 4D numpy ndarray."""
    if img is None:
        raise ValueError("Image input cannot be None")

    if hasattr(img, "numpy"):
        arr = img.numpy()
    elif isinstance(img, np.ndarray):
        arr = img
    else:
        arr = np.asarray(img)

    # Squeeze out singleton dimensions if 4D with 1 slice
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)
    elif arr.ndim == 4 and arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)

    return arr


def _find_center_coords(
    underlay: np.ndarray,
    overlay: np.ndarray | None = None,
) -> tuple[int, int, int]:
    """Find appropriate center coordinates for 3-view orthogonal montage.

    Uses overlay center of mass if available, else underlay brain center of mass.
    """
    nx, ny, nz = underlay.shape[:3]

    # Prioritize center of non-zero voxels in overlay
    if overlay is not None and overlay.shape[:3] == (nx, ny, nz):
        non_zero = np.argwhere(np.abs(overlay) > 1e-4)
        if len(non_zero) > 0:
            com = np.mean(non_zero, axis=0)
            return (
                int(np.clip(round(com[0]), 0, nx - 1)),
                int(np.clip(round(com[1]), 0, ny - 1)),
                int(np.clip(round(com[2]), 0, nz - 1)),
            )

    # Use underlay brain tissue center of mass
    non_zero_u = underlay[underlay > 0]
    if len(non_zero_u) > 0:
        thresh = float(np.percentile(non_zero_u, 30))
        brain_vox = np.argwhere(underlay > thresh)
        if len(brain_vox) > 0:
            com = np.mean(brain_vox, axis=0)
            return (
                int(np.clip(round(com[0]), 0, nx - 1)),
                int(np.clip(round(com[1]), 0, ny - 1)),
                int(np.clip(round(com[2]), 0, nz - 1)),
            )

    return nx // 2, ny // 2, nz // 2


def render_ortho_montage(
    underlay: Any,
    overlay: Any = None,
    xyz: tuple[int, int, int] | None = None,
    crosshairs: bool = True,
    title: str = "",
    cmap: str = "gray",
    overlay_cmap: str = "hot",
    overlay_alpha: float = 0.5,
    vmin: float | None = None,
    vmax: float | None = None,
    overlay_vmin: float | None = None,
    overlay_vmax: float | None = None,
    crosshair_color: str = ACCENT_CYAN,
    figsize: tuple[float, float] = (12.0, 4.2),
    dpi: int = 150,
) -> str:
    """Generate a 3-view orthogonal montage (Axial, Coronal, Sagittal).

    Includes crosshairs, coordinates, and optional alpha-blended overlay.
    Handles both ANTsImage and numpy ndarray objects.

    Parameters
    ----------
    underlay : ANTsImage or np.ndarray
        3D anatomical image.
    overlay : ANTsImage or np.ndarray, optional
        3D functional, lesion, or statistical overlay.
    xyz : tuple of (int, int, int), optional
        Slice coordinates (X, Y, Z). If None, auto-centers on brain or overlay mass.
    crosshairs : bool, optional
        Whether to draw intersecting crosshair lines. Default is True.
    title : str, optional
        Super-title for the montage.
    cmap : str, optional
        Underlay colormap. Default is "gray".
    overlay_cmap : str, optional
        Overlay colormap. Default is "hot".
    overlay_alpha : float, optional
        Overlay transparency in [0.0, 1.0]. Default is 0.5.
    vmin, vmax : float, optional
        Underlay display window limits.
    overlay_vmin, overlay_vmax : float, optional
        Overlay display window limits.
    crosshair_color : str, optional
        Color of crosshair lines.
    figsize : tuple, optional
        Figure width and height in inches.
    dpi : int, optional
        Rendering DPI.

    Returns
    -------
    str
        Base64 PNG data URI.
    """
    und_arr = _to_numpy(underlay)
    if und_arr.ndim != 3:
        raise ValueError(f"render_ortho_montage expects 3D underlay, got shape {und_arr.shape}")

    ov_arr = _to_numpy(overlay) if overlay is not None else None
    if ov_arr is not None and ov_arr.ndim != 3:
        raise ValueError(f"render_ortho_montage expects 3D overlay, got shape {ov_arr.shape}")

    nx, ny, nz = und_arr.shape
    if xyz is None:
        x, y, z = _find_center_coords(und_arr, ov_arr)
    else:
        x, y, z = xyz
        x = int(np.clip(x, 0, nx - 1))
        y = int(np.clip(y, 0, ny - 1))
        z = int(np.clip(z, 0, nz - 1))

    # Determine automatic contrast scaling for underlay
    if vmin is None:
        vmin = float(np.percentile(und_arr, 1.0))
    if vmax is None:
        vmax = float(np.percentile(und_arr, 99.5))

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi, facecolor=DARK_BG)

    views = [
        # (axis_obj, underlay_slice, overlay_slice, title, h_coord, v_coord, h_line, v_line)
        ("Axial", und_arr[:, :, z].T, ov_arr[:, :, z].T if ov_arr is not None else None, f"Axial (Z={z})", x, y),
        ("Coronal", und_arr[:, y, :].T, ov_arr[:, y, :].T if ov_arr is not None else None, f"Coronal (Y={y})", x, z),
        ("Sagittal", und_arr[x, :, :].T, ov_arr[x, :, :].T if ov_arr is not None else None, f"Sagittal (X={x})", y, z),
    ]

    try:
        for idx, (view_name, und_slice, ov_slice, view_title, ch_x, ch_y) in enumerate(views):
            ax = axes[idx]
            ax.set_facecolor(DARK_BG)
            ax.imshow(und_slice, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)

            if ov_slice is not None:
                # Mask out background/zeros
                mask = (ov_slice == 0) | np.isnan(ov_slice) | (np.abs(ov_slice) < 1e-6)
                masked_ov = np.ma.masked_where(mask, ov_slice)
                if not mask.all():
                    ax.imshow(
                        masked_ov,
                        cmap=overlay_cmap,
                        origin="lower",
                        alpha=overlay_alpha,
                        vmin=overlay_vmin,
                        vmax=overlay_vmax,
                    )

            if crosshairs:
                ax.axvline(x=ch_x, color=crosshair_color, linestyle="--", linewidth=0.8, alpha=0.75)
                ax.axhline(y=ch_y, color=crosshair_color, linestyle="--", linewidth=0.8, alpha=0.75)

            ax.set_title(view_title, color=DARK_TEXT, fontsize=11, pad=6, fontweight="bold")
            ax.axis("off")

        if title:
            fig.suptitle(title, color=DARK_TEXT, fontsize=13, fontweight="bold", y=0.98)

        fig.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.88 if title else 0.95, bottom=0.05)
        return figure_to_base64(fig, format="png", dpi=dpi, close=True)
    finally:
        plt.close(fig)


def render_slice_gallery(
    underlay: Any,
    overlay: Any = None,
    axis: int = 2,
    nslices: int = 7,
    contours: bool = True,
    title: str = "",
    cmap: str = "gray",
    overlay_cmap: str = "tab10",
    overlay_alpha: float = 0.5,
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
) -> str:
    """Render a multi-slice gallery along a specified axis.

    Supports vector contours or alpha-blended fills for masks/labels.

    Parameters
    ----------
    underlay : ANTsImage or np.ndarray
        3D anatomical underlay.
    overlay : ANTsImage or np.ndarray, optional
        3D mask, segmentation, or probability map.
    axis : int, optional
        Slicing axis: 0 (Sagittal), 1 (Coronal), 2 (Axial). Default is 2.
    nslices : int, optional
        Number of evenly spaced slices to render. Default is 7.
    contours : bool, optional
        If True, render overlay as vector contour lines. If False, render alpha fill.
    title : str, optional
        Gallery header title.
    cmap : str, optional
        Underlay colormap.
    overlay_cmap : str, optional
        Overlay colormap.
    overlay_alpha : float, optional
        Alpha transparency when contours=False.
    figsize : tuple, optional
        Figure size. If None, auto-calculated from nslices.
    dpi : int, optional
        Rendering DPI.

    Returns
    -------
    str
        Base64 PNG data URI.
    """
    und_arr = _to_numpy(underlay)
    ov_arr = _to_numpy(overlay) if overlay is not None else None

    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}")

    total_slices = und_arr.shape[axis]
    nslices = max(1, min(nslices, total_slices))

    # Identify slice index bounds: focus on non-zero overlay region if available
    if ov_arr is not None:
        slice_sums = np.sum(np.abs(ov_arr) > 1e-4, axis=tuple(i for i in range(3) if i != axis))
        active_slices = np.where(slice_sums > 0)[0]
        if len(active_slices) >= nslices:
            min_s, max_s = active_slices[0], active_slices[-1]
            slice_indices = np.linspace(min_s, max_s, nslices, dtype=int)
        else:
            slice_indices = np.linspace(int(total_slices * 0.15), int(total_slices * 0.85), nslices, dtype=int)
    else:
        slice_indices = np.linspace(int(total_slices * 0.15), int(total_slices * 0.85), nslices, dtype=int)

    vmin = float(np.percentile(und_arr, 1.0))
    vmax = float(np.percentile(und_arr, 99.5))

    if figsize is None:
        figsize = (2.2 * nslices, 2.8)

    fig, axes = plt.subplots(1, nslices, figsize=figsize, dpi=dpi, facecolor=DARK_BG)
    if nslices == 1:
        axes = [axes]

    axis_labels = {0: "x", 1: "y", 2: "z"}
    coord_label = axis_labels.get(axis, "s")

    try:
        for idx, s_idx in enumerate(slice_indices):
            ax = axes[idx]
            ax.set_facecolor(DARK_BG)

            if axis == 0:
                und_slice = und_arr[s_idx, :, :].T
                ov_slice = ov_arr[s_idx, :, :].T if ov_arr is not None else None
            elif axis == 1:
                und_slice = und_arr[:, s_idx, :].T
                ov_slice = ov_arr[:, s_idx, :].T if ov_arr is not None else None
            else:
                und_slice = und_arr[:, :, s_idx].T
                ov_slice = ov_arr[:, :, s_idx].T if ov_arr is not None else None

            ax.imshow(und_slice, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)

            if ov_slice is not None and np.any(ov_slice > 0):
                if contours:
                    # Discrete labels or continuous scalar
                    unique_labels = np.unique(ov_slice[ov_slice > 0])
                    if len(unique_labels) <= 15:
                        # Draw individual label contours
                        color_map = plt.get_cmap(overlay_cmap)
                        for l_i, lab in enumerate(unique_labels):
                            lab_color = color_map(l_i % 10)
                            binary_mask = (ov_slice == lab).astype(float)
                            if binary_mask.max() > 0:
                                ax.contour(
                                    binary_mask,
                                    levels=[0.5],
                                    colors=[lab_color],
                                    linewidths=1.2,
                                    origin="lower",
                                )
                    else:
                        # Continuous contour
                        ax.contour(
                            ov_slice,
                            levels=3,
                            cmap=overlay_cmap,
                            linewidths=1.0,
                            origin="lower",
                        )
                else:
                    masked_ov = np.ma.masked_where(ov_slice <= 0, ov_slice)
                    ax.imshow(masked_ov, cmap=overlay_cmap, alpha=overlay_alpha, origin="lower")

            ax.set_title(f"{coord_label}={s_idx}", color=DARK_MUTED, fontsize=9, pad=4)
            ax.axis("off")

        if title:
            fig.suptitle(title, color=DARK_TEXT, fontsize=12, fontweight="bold", y=0.98)

        fig.subplots_adjust(wspace=0.03, left=0.01, right=0.99, top=0.82 if title else 0.92, bottom=0.05)
        return figure_to_base64(fig, format="png", dpi=dpi, close=True)
    finally:
        plt.close(fig)


def render_carpet_plot(
    timeseries_4d: Any,
    mask: Any = None,
    fd: np.ndarray | None = None,
    dvars: np.ndarray | None = None,
    fd_threshold: float = 0.5,
    title: str = "",
    max_voxels: int = 1000,
    figsize: tuple[float, float] = (11.0, 6.5),
    dpi: int = 150,
) -> str:
    """Render a 4D BOLD timeseries carpet plot with aligned FD/DVARS traces.

    Guarantees fast rendering (<3s) by subsampling voxels inside the brain mask
    to at most `max_voxels` (default 1000).

    Parameters
    ----------
    timeseries_4d : ANTsImage or np.ndarray
        4D timeseries with shape (X, Y, Z, T) or 2D (Voxels, T).
    mask : ANTsImage or np.ndarray, optional
        Brain mask. If None, voxels above non-zero 25th percentile are used.
    fd : np.ndarray, optional
        1D Framewise Displacement trace across time points.
    dvars : np.ndarray, optional
        1D DVARS trace across time points.
    fd_threshold : float, optional
        Horizontal threshold line for FD (default 0.5mm).
    title : str, optional
        Plot super-title.
    max_voxels : int, optional
        Maximum voxels to display in carpet plot (default 1000).
    figsize : tuple, optional
        Figure size.
    dpi : int, optional
        Rendering DPI.

    Returns
    -------
    str
        Base64 PNG data URI.
    """
    ts_arr = _to_numpy(timeseries_4d)

    if ts_arr.ndim == 4:
        t_len = ts_arr.shape[-1]
        if mask is not None:
            m_arr = _to_numpy(mask) > 0
        else:
            mean_vol = np.mean(ts_arr, axis=-1)
            pos = mean_vol[mean_vol > 0]
            thresh = np.percentile(pos, 25) if len(pos) > 0 else 0
            m_arr = mean_vol > thresh

        voxel_data = ts_arr[m_arr, :]  # Shape: (V, T)
    elif ts_arr.ndim == 2:
        voxel_data = ts_arr
        t_len = ts_arr.shape[1]
    else:
        raise ValueError(f"render_carpet_plot expects 2D or 4D array, got shape {ts_arr.shape}")

    n_vox = voxel_data.shape[0]
    if n_vox == 0:
        voxel_data = np.zeros((10, t_len))
        n_vox = 10

    # Subsample to <= max_voxels for performance
    if n_vox > max_voxels:
        step = max(1, n_vox // max_voxels)
        voxel_data = voxel_data[::step][:max_voxels]

    # Z-score normalize each voxel timeseries across time
    means = np.mean(voxel_data, axis=1, keepdims=True)
    stds = np.std(voxel_data, axis=1, keepdims=True)
    stds[stds < 1e-6] = 1.0
    carpet = (voxel_data - means) / stds

    # Determine layout of subplots
    plot_fd = fd is not None and len(fd) > 0
    plot_dvars = dvars is not None and len(dvars) > 0

    height_ratios = []
    if plot_fd:
        height_ratios.append(1.0)
    if plot_dvars:
        height_ratios.append(1.0)
    height_ratios.append(3.5)  # Carpet

    n_rows = len(height_ratios)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=figsize,
        dpi=dpi,
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
        facecolor=DARK_BG,
    )

    if n_rows == 1:
        axes = [axes]

    try:
        curr_ax_idx = 0

        # FD Plot
        if plot_fd:
            ax_fd = axes[curr_ax_idx]
            curr_ax_idx += 1
            ax_fd.set_facecolor(DARK_PANEL_BG)
            fd_arr = np.asarray(fd)
            time_axis = np.arange(len(fd_arr))
            ax_fd.plot(time_axis, fd_arr, color=ACCENT_CYAN, linewidth=1.2, label="FD")
            ax_fd.axhline(
                y=fd_threshold,
                color=ACCENT_RED,
                linestyle="--",
                linewidth=1.0,
                label=f"Threshold ({fd_threshold}mm)",
            )
            ax_fd.fill_between(
                time_axis,
                fd_threshold,
                fd_arr,
                where=(fd_arr > fd_threshold),
                color=ACCENT_RED,
                alpha=0.3,
            )
            ax_fd.set_ylabel("FD (mm)", color=DARK_TEXT, fontsize=9)
            ax_fd.tick_params(colors=DARK_MUTED, labelsize=8)
            ax_fd.set_xlim(0, t_len - 1)
            ax_fd.grid(True, linestyle=":", alpha=0.3, color=DARK_BORDER)
            ax_fd.legend(loc="upper right", fontsize=8, facecolor=DARK_PANEL_BG, edgecolor=DARK_BORDER, labelcolor=DARK_TEXT)

        # DVARS Plot
        if plot_dvars:
            ax_dv = axes[curr_ax_idx]
            curr_ax_idx += 1
            ax_dv.set_facecolor(DARK_PANEL_BG)
            dv_arr = np.asarray(dvars)
            ax_dv.plot(np.arange(len(dv_arr)), dv_arr, color=ACCENT_GREEN, linewidth=1.2, label="DVARS")
            mean_dv = float(np.mean(dv_arr))
            ax_dv.axhline(
                y=mean_dv,
                color=DARK_MUTED,
                linestyle=":",
                linewidth=1.0,
                label=f"Mean ({mean_dv:.1f})",
            )
            ax_dv.set_ylabel("DVARS", color=DARK_TEXT, fontsize=9)
            ax_dv.tick_params(colors=DARK_MUTED, labelsize=8)
            ax_dv.set_xlim(0, t_len - 1)
            ax_dv.grid(True, linestyle=":", alpha=0.3, color=DARK_BORDER)
            ax_dv.legend(loc="upper right", fontsize=8, facecolor=DARK_PANEL_BG, edgecolor=DARK_BORDER, labelcolor=DARK_TEXT)

        # Carpet Plot
        ax_cp = axes[curr_ax_idx]
        ax_cp.set_facecolor(DARK_PANEL_BG)
        ax_cp.imshow(
            carpet,
            aspect="auto",
            cmap="gray",
            vmin=-2.0,
            vmax=2.0,
            origin="lower",
            interpolation="nearest",
        )
        ax_cp.set_ylabel(f"Voxels (N={carpet.shape[0]})", color=DARK_TEXT, fontsize=9)
        ax_cp.set_xlabel("Time Point (Volume)", color=DARK_TEXT, fontsize=10)
        ax_cp.tick_params(colors=DARK_MUTED, labelsize=8)
        ax_cp.set_xlim(0, t_len - 1)

        for spine in ax_cp.spines.values():
            spine.set_color(DARK_BORDER)

        if title:
            fig.suptitle(title, color=DARK_TEXT, fontsize=12, fontweight="bold", y=0.98)

        fig.tight_layout()
        return figure_to_base64(fig, format="png", dpi=dpi, close=True)
    finally:
        plt.close(fig)


def render_violin_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    title: str = "",
    reference_range: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (9.5, 4.5),
    dpi: int = 150,
) -> str:
    """Render a regional ROI distribution violin and strip plot.

    Includes optional normative reference range shading.

    Parameters
    ----------
    data : pd.DataFrame
        Input data table.
    x : str
        Column name for x-axis categories (e.g. ROI or Region).
    y : str
        Column name for y-axis quantitative measurements.
    hue : str, optional
        Grouping column for color differentiation.
    title : str, optional
        Chart title.
    reference_range : tuple of (float, float), optional
        (min_val, max_val) normative boundary for green shading.
    figsize : tuple, optional
        Figure size.
    dpi : int, optional
        Rendering DPI.

    Returns
    -------
    str
        Base64 PNG data URI.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=DARK_BG)
    try:
        ax.set_facecolor(DARK_PANEL_BG)

        # Normative range shading behind data
        if reference_range is not None:
            ax.axhspan(
                reference_range[0],
                reference_range[1],
                color=ACCENT_GREEN,
                alpha=0.15,
                label=f"Normative ({reference_range[0]}-{reference_range[1]})",
                zorder=0,
            )

        violin_hue = hue if hue is not None else x
        violin_legend = bool(hue)
        sns.violinplot(
            data=data,
            x=x,
            y=y,
            hue=violin_hue,
            inner="quartile",
            cut=0,
            palette="mako",
            legend=violin_legend,
            ax=ax,
            zorder=2,
        )

        strip_kwargs: dict[str, Any] = {
            "data": data,
            "x": x,
            "y": y,
            "hue": hue,
            "dodge": bool(hue),
            "jitter": 0.2,
            "size": 3.5,
            "alpha": 0.4,
            "ax": ax,
            "zorder": 3,
            "legend": False,
        }
        if hue:
            strip_kwargs["palette"] = f"dark:{DARK_TEXT}"
        else:
            strip_kwargs["color"] = DARK_TEXT

        sns.stripplot(**strip_kwargs)

        # Remove duplicate strip plot legend entries if hue is set
        if hue:
            handles, labels = ax.get_legend_handles_labels()
            # Keep only the first len(data[hue].unique()) + reference range
            n_unique = len(data[hue].unique())
            ax.legend(
                handles[:n_unique] + (handles[-1:] if reference_range else []),
                labels[:n_unique] + (labels[-1:] if reference_range else []),
                loc="upper right",
                facecolor=DARK_PANEL_BG,
                edgecolor=DARK_BORDER,
                labelcolor=DARK_TEXT,
                fontsize=9,
            )
        elif reference_range:
            ax.legend(loc="upper right", facecolor=DARK_PANEL_BG, edgecolor=DARK_BORDER, labelcolor=DARK_TEXT, fontsize=9)

        ax.set_title(title, color=DARK_TEXT, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel(x, color=DARK_TEXT, fontsize=10)
        ax.set_ylabel(y, color=DARK_TEXT, fontsize=10)
        ax.tick_params(colors=DARK_MUTED, labelsize=9)

        # Rotate x ticks if many labels or long strings
        x_labels = [str(t.get_text()) for t in ax.get_xticklabels()]
        if len(x_labels) > 4 or any(len(lab) > 8 for lab in x_labels):
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

        ax.grid(True, linestyle=":", alpha=0.3, color=DARK_BORDER, zorder=1)
        for spine in ax.spines.values():
            spine.set_color(DARK_BORDER)

        fig.tight_layout()
        return figure_to_base64(fig, format="png", dpi=dpi, close=True)
    finally:
        plt.close(fig)


def render_correlation_matrix(
    corr_df: pd.DataFrame,
    title: str = "",
    cmap: str = "coolwarm",
    vmin: float = -1.0,
    vmax: float = 1.0,
    annot: bool = False,
    figsize: tuple[float, float] = (8.0, 7.0),
    dpi: int = 150,
) -> str:
    """Render a symmetric diverging correlation matrix heatmap.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Square correlation matrix dataframe.
    title : str, optional
        Heatmap title.
    cmap : str, optional
        Diverging colormap name (default "coolwarm").
    vmin, vmax : float, optional
        Colorbar scale limits (default -1.0 to 1.0).
    annot : bool, optional
        Whether to annotate numerical values in each cell.
    figsize : tuple, optional
        Figure size.
    dpi : int, optional
        Rendering DPI.

    Returns
    -------
    str
        Base64 PNG data URI.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor=DARK_BG)
    try:
        ax.set_facecolor(DARK_PANEL_BG)

        # Auto-annotate only if matrix is small and not explicitly disabled
        show_annot = annot or (corr_df.shape[0] <= 12 and annot is not False)

        cbar_kws = {
            "label": "Correlation (r)",
            "shrink": 0.8,
        }

        sns.heatmap(
            corr_df,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=0,
            annot=show_annot,
            fmt=".2f" if show_annot else "",
            square=True,
            linewidths=0.5,
            linecolor=DARK_BG,
            cbar_kws=cbar_kws,
            ax=ax,
        )

        ax.set_title(title, color=DARK_TEXT, fontsize=12, fontweight="bold", pad=12)
        ax.tick_params(colors=DARK_MUTED, labelsize=8)

        # Style colorbar text
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.tick_params(colors=DARK_MUTED, labelsize=8)
            cbar.ax.yaxis.label.set_color(DARK_TEXT)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

        fig.tight_layout()
        return figure_to_base64(fig, format="png", dpi=dpi, close=True)
    finally:
        plt.close(fig)
