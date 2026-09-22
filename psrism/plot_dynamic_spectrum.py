"""Plotting for dynamic spectra and pulse profiles."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .archive_io import (
    frequency_phase_array,
    integrated_profile_array,
    integrated_profile_shift,
    time_phase_array,
)


def plot_dynamic_spectrum(
    dynspec,
    metadata,
    output_path: str | None = None,
    valid_mask: np.ndarray | None = None,
):
    import matplotlib.pyplot as plt

    arr = np.asarray(dynspec, dtype=float).copy()
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != arr.shape:
            raise ValueError("valid_mask must have the same shape as dynspec")
        arr[~mask] = np.nan
    nsub, nchan = arr.shape
    freq_proj = _finite_mean(arr, axis=0)
    time_proj = _finite_mean(arr, axis=1)

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

    cmap = plt.get_cmap("afmhot").copy()
    cmap.set_bad(color="0.75")
    im = ax_main.imshow(
        arr.T,
        extent=(
            0,
            metadata.observation_time_s / 60.0,
            metadata.frequency_low_mhz,
            metadata.frequency_high_mhz,
        ),
        aspect="auto",
        cmap=cmap,
        origin="lower",
    )
    ax_main.set_xlabel("Time (min)")
    ax_main.set_ylabel("Frequency (MHz)")
    fig.suptitle(_plot_title(metadata.filename))

    ax_top.plot(np.linspace(0, metadata.observation_time_s / 60.0, nsub), time_proj, color="black")
    ax_top.set_ylabel("Flux")
    ax_top.tick_params(labelbottom=False)

    ax_right.plot(
        freq_proj,
        np.linspace(metadata.frequency_low_mhz, metadata.frequency_high_mhz, nchan),
        color="black",
    )
    ax_right.set_xlabel("Flux")
    ax_right.tick_params(labelleft=False)
    fig.colorbar(im, cax=ax_cbar, label="Flux")
    fig.subplots_adjust(top=0.90)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def _finite_mean(values: np.ndarray, axis: int) -> np.ndarray:
    finite = np.isfinite(values)
    count = np.sum(finite, axis=axis)
    total = np.sum(np.where(finite, values, 0.0), axis=axis)
    result = np.full(np.asarray(count).shape, np.nan, dtype=float)
    np.divide(total, count, out=result, where=count > 0)
    return result


def plot_integrated_profile(
    archive,
    metadata,
    width_factor: int = 10,
    tau_fit_result=None,
    output_path: str | None = None,
):
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    phase_shift = integrated_profile_shift(archive)
    mat_time = time_phase_array(archive, phase_shift=phase_shift)
    mat_freq = frequency_phase_array(archive, phase_shift=phase_shift)
    nsub, nbin = mat_time.shape
    nchan = mat_freq.shape[0]

    phase_axis = np.linspace(0.0, 1.0, nbin, endpoint=False)
    time_axis = np.arange(nsub) * (metadata.observation_time_s / nsub) / 60.0
    freq_axis = np.linspace(metadata.frequency_low_mhz, metadata.frequency_high_mhz, nchan)
    prof_int = integrated_profile_array(
        archive,
        normalize=tau_fit_result is None,
        center_peak=True,
        remove_baseline=tau_fit_result is None,
    )

    fig = plt.figure(figsize=(8.5, 9.2))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])
    ax_top = fig.add_subplot(gs[0])
    ax_freq = fig.add_subplot(gs[1], sharex=ax_top)
    ax_time = fig.add_subplot(gs[2], sharex=ax_top)

    fit_curve = None
    if tau_fit_result is not None:
        from .fit_tau import evaluate_tau_fit, profile_model_for_plot

        fit_curve = evaluate_tau_fit(tau_fit_result, nbin)
        prof_int, fit_curve = profile_model_for_plot(prof_int, fit_curve)
    ax_top.scatter(phase_axis, prof_int, color="black", s=8, label="Profile")
    if fit_curve is not None:
        ax_top.plot(phase_axis, fit_curve, color="tab:red", linewidth=2, label="tau fit")
    mx = np.max(prof_int)
    # PSRISM display choices: these profile fractions visualize the automatic
    # pulse window; they are not literature-derived physical thresholds.
    ax_top.axhline(mx, color="teal", label="max")
    ax_top.axhline(mx / width_factor, color="green", label=f"1/{width_factor} max")
    ax_top.axhline(mx / 10.0, color="red", label="1/10 max")
    ax_top.axhline(mx / 2.0, color="blue", label="1/2 max")
    ax_top.set_ylabel("Normalized flux")
    ax_top.legend(loc="best")

    ax_freq.imshow(
        mat_freq,
        aspect="auto",
        origin="lower",
        cmap="afmhot",
        extent=(0, 1, freq_axis.min(), freq_axis.max()),
    )
    ax_freq.set_ylabel("Frequency (MHz)")

    ax_time.imshow(
        mat_time,
        aspect="auto",
        origin="lower",
        cmap="afmhot",
        extent=(0, 1, time_axis.min(), time_axis.max()),
    )
    ax_time.set_ylabel("Time (min)")
    ax_time.set_xlabel("Pulse phase")
    if tau_fit_result is not None:
        fig.text(
            0.5,
            0.018,
            _tau_caption(tau_fit_result),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle(_plot_title(metadata.filename))
    fig.tight_layout(rect=(0, 0.06 if tau_fit_result is not None else 0, 1, 0.96))
    if output_path:
        fig.savefig(output_path)
    return fig


def _normalise_profile(profile) -> np.ndarray:
    prof = np.asarray(profile, dtype=float)
    prof = prof - np.min(prof)
    mx = np.max(prof)
    if mx != 0:
        prof = prof / mx
    return prof


def _plot_title(filename: str) -> str:
    return Path(filename).name


def _tau_caption(result) -> str:
    tau_seconds = result.tau_seconds
    tau_seconds_error = result.tau_seconds_error
    if tau_seconds is None or tau_seconds_error is None:
        tau_text = f"τ = {result.tau_bins:.4g} ± {result.tau_bins_error:.2g} bins"
    else:
        tau_text = f"τ = {tau_seconds * 1e3:.4g} ± {tau_seconds_error * 1e3:.2g} ms"
    reduced = _format_optional_float(result.reduced_chi_square)
    rms = _format_optional_float(result.rms_residual)
    return f"{tau_text}; unweighted reduced χ²={reduced}; RMS residual={rms}"


def _format_optional_float(value) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.3g}"
