"""Batch time-series analysis for pulsar archive directories."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import csv
import re

import numpy as np


MJD0 = datetime(1858, 11, 17, tzinfo=timezone.utc)


@dataclass(frozen=True)
class EpochMeasurement:
    archive_path: str
    archive_name: str
    utc: datetime
    mjd: float
    dm: float | None = None
    tau_s: float | None = None
    tau_error_s: float | None = None
    alpha: float | None = None
    alpha_error: float | None = None
    tau0_s: float | None = None
    tau0_error_s: float | None = None
    decorrelation_bandwidth_mhz: float | None = None
    diffractive_timescale_s: float | None = None
    refractive_timescale_days: float | None = None


def parse_time_params(value: str | None) -> set[str]:
    aliases = {
        "dm": "dm",
        "tau": "tau",
        "alpha": "alpha",
        "dnu": "dnu_d",
        "dnu_d": "dnu_d",
        "bandwidth": "dnu_d",
        "decorrelation_bandwidth": "dnu_d",
        "diffractive_bandwidth": "dnu_d",
        "dt": "dt_d",
        "dt_d": "dt_d",
        "diffractive_timescale": "dt_d",
        "tr": "t_r",
        "t_r": "t_r",
        "refractive": "t_r",
        "refractive_timescale": "t_r",
        "refractive_timescale_days": "t_r",
        "all": "all",
    }
    if value is None or value.strip().lower() == "all":
        return {"dm", "tau", "alpha", "dnu_d", "dt_d", "t_r"}

    params: set[str] = set()
    for item in value.split(","):
        key = item.strip().lower().replace("-", "_")
        if not key:
            continue
        if key not in aliases:
            valid = ", ".join(sorted(k for k in aliases if k != "all"))
            raise ValueError(f"unknown time-series parameter '{item}'. Choose from: {valid}, all")
        mapped = aliases[key]
        if mapped == "all":
            return {"dm", "tau", "alpha", "dnu_d", "dt_d", "t_r"}
        params.add(mapped)
    return params or {"dm", "tau", "alpha", "dnu_d", "dt_d", "t_r"}


def archive_epoch_datetime(archive, archive_path: str) -> tuple[datetime, float]:
    """Return archive epoch as UTC datetime and MJD."""
    try:
        mjd = float(archive.get_Integration(0).get_epoch().in_days())
        return mjd_to_datetime(mjd), mjd
    except Exception:
        dt = _datetime_from_filename(archive_path)
        return dt, datetime_to_mjd(dt)


def mjd_to_datetime(mjd: float) -> datetime:
    return MJD0 + timedelta(days=float(mjd))


def datetime_to_mjd(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt.astimezone(timezone.utc) - MJD0).total_seconds() / 86400.0


def write_time_series_csv(measurements: list[EpochMeasurement], output_path: str | Path) -> None:
    fields = [
        "archive_name",
        "archive_path",
        "utc",
        "mjd",
        "dm",
        "tau_s",
        "tau_error_s",
        "alpha",
        "alpha_error",
        "tau0_s",
        "tau0_error_s",
        "decorrelation_bandwidth_mhz",
        "diffractive_timescale_s",
        "refractive_timescale_days",
    ]
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in measurements:
            writer.writerow(
                {
                    "archive_name": item.archive_name,
                    "archive_path": item.archive_path,
                    "utc": item.utc.isoformat(),
                    "mjd": _format_optional(item.mjd),
                    "dm": _format_optional(item.dm),
                    "tau_s": _format_optional(item.tau_s),
                    "tau_error_s": _format_optional(item.tau_error_s),
                    "alpha": _format_optional(item.alpha),
                    "alpha_error": _format_optional(item.alpha_error),
                    "tau0_s": _format_optional(item.tau0_s),
                    "tau0_error_s": _format_optional(item.tau0_error_s),
                    "decorrelation_bandwidth_mhz": _format_optional(item.decorrelation_bandwidth_mhz),
                    "diffractive_timescale_s": _format_optional(item.diffractive_timescale_s),
                    "refractive_timescale_days": _format_optional(item.refractive_timescale_days),
                }
            )


def plot_parameter_vs_time(
    measurements: list[EpochMeasurement],
    parameter: str,
    time_axis: str,
    title: str,
    output_path: str | Path,
) -> bool:
    """Plot one parameter versus time with UTC and MJD axes."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    y, yerr, ylabel = _parameter_arrays(measurements, parameter)
    valid = np.isfinite(y)
    if not np.any(valid):
        return False

    x_utc = np.asarray([mdates.date2num(item.utc) for item in measurements], dtype=float)
    x_mjd = np.asarray([item.mjd for item in measurements], dtype=float)
    x = x_utc
    valid &= np.isfinite(x)
    if not np.any(valid):
        return False

    yerr_plot = None
    if yerr is not None and np.any(np.isfinite(yerr[valid])):
        yerr_plot = np.where(np.isfinite(yerr[valid]), yerr[valid], 0.0)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.errorbar(x[valid], y[valid], yerr=yerr_plot, fmt="o-", color="black")
    ax.set_xlabel("UTC date")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    def utc_num_to_mjd(values):
        values = np.asarray(values, dtype=float)
        converted = [datetime_to_mjd(mdates.num2date(value)) for value in values.ravel()]
        return np.asarray(converted, dtype=float).reshape(values.shape)

    def mjd_to_utc_num(values):
        values = np.asarray(values, dtype=float)
        converted = [mdates.date2num(mjd_to_datetime(value)) for value in values.ravel()]
        return np.asarray(converted, dtype=float).reshape(values.shape)

    secax = ax.secondary_xaxis("top", functions=(utc_num_to_mjd, mjd_to_utc_num))
    secax.set_xlabel("MJD")
    secax.ticklabel_format(style="plain", useOffset=False)

    if np.count_nonzero(valid) == 1:
        pad_days = 15.0
        ax.set_xlim(x[valid][0] - pad_days, x[valid][0] + pad_days)
    else:
        ax.set_xlim(np.nanmin(x[valid]), np.nanmax(x[valid]))
        ax.margins(x=0.05)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def _parameter_arrays(measurements: list[EpochMeasurement], parameter: str):
    values = {
        "dm": ([item.dm for item in measurements], None, "DM (pc cm^-3)"),
        "tau": ([item.tau_s for item in measurements], [item.tau_error_s for item in measurements], "Scattering timescale τ (s)"),
        "alpha": ([item.alpha for item in measurements], [item.alpha_error for item in measurements], "Scattering spectral index α"),
        "dnu_d": ([item.decorrelation_bandwidth_mhz for item in measurements], None, "Decorrelation bandwidth Δν_d (MHz)"),
        "dt_d": ([item.diffractive_timescale_s for item in measurements], None, "Diffractive timescale Δt_d (s)"),
        "t_r": ([item.refractive_timescale_days for item in measurements], None, "Refractive timescale T_r (days)"),
    }
    if parameter not in values:
        raise ValueError(f"unknown plot parameter: {parameter}")
    y, yerr, ylabel = values[parameter]
    y_arr = _as_optional_float_array(y)
    yerr_arr = None if yerr is None else _as_optional_float_array(yerr)
    return y_arr, yerr_arr, ylabel


def _datetime_from_filename(path: str) -> datetime:
    name = Path(path).name
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}:\d{2}:\d{2})", name)
    if not match:
        raise ValueError(f"could not infer epoch from archive metadata or filename: {name}")
    return datetime.fromisoformat(f"{match.group(1)}T{match.group(2)}+00:00")


def _as_optional_float_array(values) -> np.ndarray:
    return np.asarray([np.nan if value is None else float(value) for value in values], dtype=float)


def _format_optional(value) -> str:
    if value is None:
        return ""
    try:
        if not np.isfinite(value):
            return ""
    except TypeError:
        pass
    return f"{float(value):.12g}"
