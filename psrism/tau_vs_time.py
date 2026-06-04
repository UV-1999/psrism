"""Tau versus time helpers."""

from __future__ import annotations

import numpy as np

from .dm_vs_time import TimeSeries


def tau_vs_time(times, tau_values, tau_errors=None) -> TimeSeries:
    """Package tau measurements as a validated time series."""
    time = np.asarray(times, dtype=float)
    values = np.asarray(tau_values, dtype=float)
    if time.shape != values.shape:
        raise ValueError("times and tau_values must have the same shape")
    errors = None if tau_errors is None else np.asarray(tau_errors, dtype=float)
    return TimeSeries(time=time, values=values, errors=errors)


def plot_tau_vs_time(series: TimeSeries, output_path: str | None = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.errorbar(series.time, series.values, yerr=series.errors, fmt="o-")
    ax.set_xlabel("Time")
    ax.set_ylabel("Tau")
    ax.set_title("Tau vs Time")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig
