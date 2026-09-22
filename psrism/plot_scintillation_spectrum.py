"""Plotting for scintillation/secondary spectra."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_scintillation_spectrum(
    spectrum,
    fringe_frequency,
    delay,
    title: str | None = None,
    output_path: str | None = None,
    arc_fit_result=None,
    central_mask_bins: int = 2,
):
    import matplotlib.pyplot as plt

    arr = np.asarray(spectrum, dtype=float)
    display = _mask_central_axes(
        arr,
        fringe_frequency,
        delay,
        central_mask_bins,
    )
    fringe_proj = np.mean(display, axis=1)
    delay_proj = np.mean(display, axis=0)

    fig = plt.figure(figsize=(8.5, 6.2))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1, 4],
        width_ratios=[4, 1, 0.16],
        hspace=0.0,
        wspace=0.0,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_cbar = fig.add_subplot(gs[1, 2])
    ax_blank = fig.add_subplot(gs[0, 1:])
    ax_blank.axis("off")

    # PSRISM display choice: use the median-to-maximum range for image limits.
    im = ax_main.imshow(
        display.T,
        vmin=np.median(display),
        vmax=np.max(display),
        extent=(fringe_frequency.min(), fringe_frequency.max(), delay.min(), delay.max()),
        origin="lower",
        aspect="auto",
        cmap="afmhot",
    )
    if arc_fit_result is not None:
        _plot_arc_overlay(ax_main, fringe_frequency, delay, arc_fit_result)
    ax_main.set_xlabel("Fringe frequency (Hz)")
    ax_main.set_ylabel("Delay (s)")
    if title:
        fig.suptitle(Path(title).name)

    ax_top.plot(fringe_frequency, fringe_proj, color="black")
    ax_top.set_ylabel("Power")
    ax_top.tick_params(labelbottom=False)

    ax_right.plot(delay_proj, delay, color="black")
    ax_right.set_xlabel("Power")
    ax_right.tick_params(labelleft=False)
    fig.colorbar(im, cax=ax_cbar, label="Power")
    fig.subplots_adjust(top=0.90)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_arc_search(arc_fit_result, output_path: str | None = None):
    """Plot arc strength over the searched curvature magnitudes."""
    import matplotlib.pyplot as plt

    curvature = np.abs(np.asarray(arc_fit_result.trial_curvatures, dtype=float))
    strength = np.asarray(arc_fit_result.arc_strength, dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(curvature, strength, color="black", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel(r"Curvature magnitude $|\eta|$ (s$^3$)")
    ax.set_ylabel("Mean power along trial arc")

    best_color = "tab:blue" if arc_fit_result.accepted else "tab:red"
    ax.axvline(
        abs(arc_fit_result.curvature),
        color=best_color,
        linestyle="--",
        linewidth=1.2,
        label="accepted" if arc_fit_result.accepted else "rejected",
    )
    if (
        arc_fit_result.curvature_lower is not None
        and arc_fit_result.curvature_upper is not None
    ):
        limits = sorted(
            [
                abs(arc_fit_result.curvature_lower),
                abs(arc_fit_result.curvature_upper),
            ]
        )
        ax.axvspan(limits[0], limits[1], color=best_color, alpha=0.15)
    if arc_fit_result.score_baseline is not None:
        ax.axhline(
            arc_fit_result.score_baseline,
            color="0.45",
            linestyle=":",
            linewidth=1.0,
            label="median baseline",
        )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def _mask_central_axes(spectrum, fringe_frequency, delay, mask_bins: int) -> np.ndarray:
    """Return a display copy with central FFT-axis leakage set to its floor."""
    arr = np.asarray(spectrum, dtype=float).copy()
    if mask_bins < 0:
        raise ValueError("central_mask_bins cannot be negative")
    if mask_bins == 0:
        return arr
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    floor = float(np.min(finite))
    fringe = np.asarray(fringe_frequency, dtype=float)
    delay = np.asarray(delay, dtype=float)
    fringe_step = _axis_step(fringe)
    delay_step = _axis_step(delay)
    # PSRISM display choice: hide the central FFT cross where residual DC and
    # axis leakage can dominate. The underlying linear spectrum is unchanged.
    arr[np.abs(fringe) <= mask_bins * fringe_step, :] = floor
    arr[:, np.abs(delay) <= mask_bins * delay_step] = floor
    return arr


def _axis_step(axis) -> float:
    values = np.sort(np.unique(np.asarray(axis, dtype=float)))
    positive = np.diff(values)
    positive = positive[positive > 0]
    if not len(positive):
        return 1.0
    return float(np.median(positive))


def _plot_arc_overlay(ax, fringe_frequency, delay, arc_fit_result) -> None:
    # Reference: Lorimer & Kramer (2005), psrhandbook.pdf, Section 7.4.4.3,
    # Eq. 7.45, for the parabolic secondary-spectrum arc relation.
    fringe_frequency = np.asarray(fringe_frequency, dtype=float)
    delay = np.asarray(delay, dtype=float)
    fringe = np.linspace(fringe_frequency.min(), fringe_frequency.max(), 800)
    branches = []
    if arc_fit_result.half in {"positive", "both"}:
        branches.append(
            arc_fit_result.delay_offset
            + abs(arc_fit_result.curvature) * (fringe - arc_fit_result.fringe_offset) ** 2
        )
    if arc_fit_result.half in {"negative", "both"}:
        branches.append(
            arc_fit_result.delay_offset
            - abs(arc_fit_result.curvature) * (fringe - arc_fit_result.fringe_offset) ** 2
        )

    for branch in branches:
        valid = (branch >= delay.min()) & (branch <= delay.max())
        if np.any(valid):
            ax.plot(
                fringe[valid],
                branch[valid],
                color="cyan" if arc_fit_result.accepted else "0.7",
                linestyle="--",
                linewidth=1.2,
            )
