"""DM versus time helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimeSeries:
    time: np.ndarray
    values: np.ndarray
    errors: np.ndarray | None = None


def dm_vs_time(times, dm_values, dm_errors=None) -> TimeSeries:
    """Package DM measurements as a validated time series."""
    time = np.asarray(times, dtype=float)
    values = np.asarray(dm_values, dtype=float)
    if time.shape != values.shape:
        raise ValueError("times and dm_values must have the same shape")
    errors = None if dm_errors is None else np.asarray(dm_errors, dtype=float)
    return TimeSeries(time=time, values=values, errors=errors)


def plot_dm_vs_time(series: TimeSeries, output_path: str | None = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.errorbar(series.time, series.values, yerr=series.errors, fmt="o-")
    ax.set_xlabel("Time")
    ax.set_ylabel("DM (pc cm^-3)")
    ax.set_title("DM vs Time")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig
