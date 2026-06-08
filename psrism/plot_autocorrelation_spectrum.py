"""Plotting for autocorrelation spectra."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .autocorrelation_spectrum import autocorrelation_axes


def plot_autocorrelation_spectrum(
    acf2d,
    metadata,
    output_path: str | None = None,
    acf_fit_result=None,
    zoom_fit: bool = False,
):
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
    ax_main.axvline(0.0, color="deepskyblue", linewidth=1.0, alpha=0.9)
    ax_main.axhline(0.0, color="deepskyblue", linewidth=1.0, alpha=0.9)
    if acf_fit_result is not None:
        from .fit_autocorrelation_spectrum import evaluate_acf_fit

        model = evaluate_acf_fit(acf_fit_result, time_lag, freq_lag)
        contour_level = acf_fit_result.offset + 0.5 * acf_fit_result.amplitude
        ax_main.contour(
            time_lag,
            freq_lag,
            model.T,
            levels=[contour_level],
            colors="cyan",
            linewidths=1.2,
        )
        slope = acf_fit_result.drift_slope_s_per_mhz
        if slope is not None and slope != 0:
            ridge_freq = np.asarray(freq_lag, dtype=float)
            ridge_time = slope * ridge_freq
            valid = (ridge_time >= time_lag.min()) & (ridge_time <= time_lag.max())
            ax_main.plot(
                ridge_time[valid],
                ridge_freq[valid],
                color="white",
                linestyle="--",
                linewidth=1.1,
            )
        if zoom_fit:
            xlim, ylim = _acf_zoom_limits(time_lag, freq_lag, acf_fit_result)
            ax_main.set_xlim(*xlim)
            ax_main.set_ylim(*ylim)
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


def _acf_zoom_limits(time_lag, freq_lag, acf_fit_result) -> tuple[tuple[float, float], tuple[float, float]]:
    time_lag = np.asarray(time_lag, dtype=float)
    freq_lag = np.asarray(freq_lag, dtype=float)
    time_half_span = _zoom_half_span(acf_fit_result.delta_t_diss, time_lag)
    freq_half_span = _zoom_half_span(acf_fit_result.delta_f_diss, freq_lag)
    return (
        _clamped_symmetric_limits(time_half_span, time_lag),
        _clamped_symmetric_limits(freq_half_span, freq_lag),
    )


def _zoom_half_span(width: float, axis: np.ndarray) -> float:
    """Choose a window where the fitted half width takes about 50 percent."""
    step = _axis_step(axis)
    if not np.isfinite(width) or width <= 0:
        return float(np.nanmax(np.abs(axis)))
    return max(2.0 * float(width), 3.0 * step)


def _clamped_symmetric_limits(half_span: float, axis: np.ndarray) -> tuple[float, float]:
    max_half = float(np.nanmax(np.abs(axis)))
    half_span = min(max(float(half_span), 0.0), max_half)
    if half_span == 0:
        half_span = max_half
    return -half_span, half_span


def _axis_step(axis: np.ndarray) -> float:
    diffs = np.diff(np.sort(np.unique(np.asarray(axis, dtype=float))))
    positive = diffs[diffs > 0]
    if len(positive) == 0:
        return 1.0
    return float(np.median(positive))
