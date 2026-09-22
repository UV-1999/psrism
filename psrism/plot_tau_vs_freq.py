"""Plot tau versus frequency and optional power-law fits."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .conventions import TAU_REFERENCE_FREQUENCY_MHZ
from .fit_alpha import tau_power_law


def plot_tau_vs_frequency(
    freq_mhz,
    tau,
    tau_error=None,
    alpha_result=None,
    accepted_mask=None,
    measurement_label: str = "Subband τ",
    reference_freq_mhz: float = TAU_REFERENCE_FREQUENCY_MHZ,
    title: str | None = None,
    output_path: str | None = None,
):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter

    freq = np.asarray(freq_mhz, dtype=float)
    values = np.asarray(tau, dtype=float)
    errors = None if tau_error is None else np.asarray(tau_error, dtype=float)
    accepted = (
        np.ones(freq.shape, dtype=bool)
        if accepted_mask is None
        else np.asarray(accepted_mask, dtype=bool)
    )
    if accepted.shape != freq.shape:
        raise ValueError("accepted_mask and freq_mhz must have the same shape")
    fig, ax = plt.subplots()
    accepted_error = None if errors is None else errors[..., accepted]
    ax.errorbar(
        freq[accepted],
        values[accepted],
        yerr=accepted_error,
        fmt="o",
        color="black",
        label=f"Accepted {measurement_label.lower()}",
    )
    if np.any(~accepted):
        rejected_error = None if errors is None else errors[..., ~accepted]
        ax.errorbar(
            freq[~accepted],
            values[~accepted],
            yerr=rejected_error,
            fmt="x",
            color="tab:gray",
            label=f"Rejected {measurement_label.lower()}",
        )

    if alpha_result is not None:
        grid = np.geomspace(np.min(freq), np.max(freq), 300)
        # Reference: Lorimer & Kramer (2005), psrhandbook.pdf, Section 4.2.3,
        # gives alpha=4.4 for a Kolmogorov fluctuation spectrum.
        ax.plot(
            grid,
            tau_power_law(grid, alpha_result.tau0, alpha_result.alpha, reference_freq_mhz),
            label=f"fit α = {alpha_result.alpha:.3f}",
        )
        ax.plot(
            grid,
            tau_power_law(grid, alpha_result.tau0, 4.4, reference_freq_mhz),
            linestyle="--",
            color="tab:red",
            label="α = 4.4",
        )
        if (
            alpha_result.monte_carlo_alpha is not None
            and alpha_result.monte_carlo_tau0 is not None
        ):
            sampled_curves = tau_power_law(
                grid[np.newaxis, :],
                alpha_result.monte_carlo_tau0[:, np.newaxis],
                alpha_result.monte_carlo_alpha[:, np.newaxis],
                reference_freq_mhz,
            )
            lower, upper = np.percentile(sampled_curves, [16.0, 84.0], axis=0)
            ax.fill_between(
                grid,
                lower,
                upper,
                color="tab:blue",
                alpha=0.2,
                linewidth=0,
                label="Monte Carlo 68% interval",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Scattering timescale τ (s)")
    if title:
        ax.set_title(Path(title).name)
    if len(freq) <= 12:
        ax.set_xticks(freq)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend()
    if alpha_result is not None:
        caption = (
            "Power-law fit goodness: "
            f"weighted χ²={alpha_result.chi_square:.4g}, "
            f"red. χ²={_format_optional_float(alpha_result.reduced_chi_square)}, "
            f"dof={alpha_result.dof}, "
            f"RMS log10 residual={alpha_result.rms_log_residual:.4g}, "
            f"N={alpha_result.n_points}, "
            f"MC={alpha_result.monte_carlo_samples}"
        )
        fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=9, wrap=True)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
    else:
        fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig


def _format_optional_float(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.4g}"
