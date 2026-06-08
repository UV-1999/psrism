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
):
    import matplotlib.pyplot as plt

    arr = np.asarray(spectrum)
    fringe_proj = np.mean(arr, axis=1)
    delay_proj = np.mean(arr, axis=0)

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

    im = ax_main.imshow(
        arr.T,
        vmin=np.median(arr),
        vmax=np.max(arr),
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


def _plot_arc_overlay(ax, fringe_frequency, delay, arc_fit_result) -> None:
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
                color="cyan",
                linestyle="--",
                linewidth=1.2,
            )
