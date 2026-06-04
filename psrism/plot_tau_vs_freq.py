"""Plot tau versus frequency and optional power-law fits."""

from __future__ import annotations

import numpy as np

from .fit_alpha import tau_power_law


def plot_tau_vs_frequency(
    freq_mhz,
    tau,
    tau_error=None,
    alpha_result=None,
    reference_freq_mhz: float = 1000.0,
    output_path: str | None = None,
):
    import matplotlib.pyplot as plt

    freq = np.asarray(freq_mhz, dtype=float)
    values = np.asarray(tau, dtype=float)
    fig, ax = plt.subplots()
    ax.errorbar(freq, values, yerr=tau_error, fmt="o", label="Tau")

    if alpha_result is not None:
        grid = np.linspace(np.min(freq), np.max(freq), 300)
        ax.plot(
            grid,
            tau_power_law(grid, alpha_result.tau0, alpha_result.alpha, reference_freq_mhz),
            label=f"alpha = {alpha_result.alpha:.3f}",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Tau")
    ax.set_title("Tau vs Frequency")
    ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig
