"""Plotting for autocorrelation spectra."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .autocorrelation_spectrum import autocorrelation_axes


def plot_autocorrelation_spectrum(acf2d, metadata, output_path: str | None = None):
    import matplotlib.pyplot as plt

    arr = np.asarray(acf2d)
    nsub = metadata.processed_shape[0]
    nchan = metadata.processed_shape[2]
    time_lag, freq_lag = autocorrelation_axes(
        nsub,
        nchan,
        metadata.observation_time_s,
        metadata.bandwidth_mhz,
    )
    if arr.shape != (len(time_lag), len(freq_lag)):
        time_lag, freq_lag = autocorrelation_axes(
            arr.shape[0],
            arr.shape[1],
            metadata.observation_time_s,
            metadata.bandwidth_mhz,
        )

    time_proj = np.mean(arr, axis=1)
    freq_proj = np.mean(arr, axis=0)

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
        extent=(time_lag.min(), time_lag.max(), freq_lag.min(), freq_lag.max()),
        origin="lower",
        aspect="auto",
        cmap="afmhot",
    )
    ax_main.set_xlabel("Time lag (s)")
    ax_main.set_ylabel("Frequency lag (MHz)")
    fig.suptitle(Path(metadata.filename).name)

    ax_top.plot(time_lag, time_proj, color="black")
    ax_top.set_ylabel("ACF")
    ax_top.tick_params(labelbottom=False)

    ax_right.plot(freq_proj, freq_lag, color="black")
    ax_right.set_xlabel("ACF")
    ax_right.tick_params(labelleft=False)
    fig.colorbar(im, cax=ax_cbar, label="ACF")
    fig.subplots_adjust(top=0.90)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig
