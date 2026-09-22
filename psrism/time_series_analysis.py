"""Batch time-series analysis for pulsar archive directories."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import csv
import re

import numpy as np


# Standards provenance: the MJD epoch is a time-coordinate definition, not a
# value drawn from the bundled pulsar references.
MJD0 = datetime(1858, 11, 17, tzinfo=timezone.utc)


@dataclass(frozen=True)
class EpochMeasurement:
    archive_path: str
    archive_name: str
    utc: datetime
    mjd: float
    start_utc: datetime | None = None
    end_utc: datetime | None = None
    start_mjd: float | None = None
    end_mjd: float | None = None
    telescope: str | None = None
    telescope_latitude_deg: float | None = None
    telescope_longitude_deg: float | None = None
    telescope_height_m: float | None = None
    source_ra_deg: float | None = None
    source_dec_deg: float | None = None
    observing_frequency_mhz: float | None = None
    period_s: float | None = None
    sun_separation_start_deg: float | None = None
    sun_separation_mid_deg: float | None = None
    sun_separation_end_deg: float | None = None
    solar_ephemeris_method: str | None = None
    dm: float | None = None
    archive_dm: float | None = None
    dm_error: float | None = None
    dm_method: str | None = None
    dm_correction: float | None = None
    pdmp_snr: float | None = None
    scattering_delta_dm: float | None = None
    scattering_delta_dm_error: float | None = None
    scattering_corrected_dm: float | None = None
    scattering_dm_reduced_chi_square: float | None = None
    tau_s: float | None = None
    tau_error_s: float | None = None
    alpha: float | None = None
    alpha_error: float | None = None
    alpha_error_lower: float | None = None
    alpha_error_upper: float | None = None
    alpha_covariance_error: float | None = None
    alpha_monte_carlo_samples: int | None = None
    alpha_subbands_accepted: int | None = None
    alpha_subbands_rejected: int | None = None
    alpha_subbands_failed: int | None = None
    tau0_s: float | None = None
    tau0_error_s: float | None = None
    tau0_error_lower_s: float | None = None
    tau0_error_upper_s: float | None = None
    tau_reference_frequency_mhz: float | None = None
    decorrelation_bandwidth_mhz: float | None = None
    decorrelation_bandwidth_error_mhz: float | None = None
    diffractive_timescale_s: float | None = None
    diffractive_timescale_error_s: float | None = None
    scintillation_measurement_method: str | None = None
    acf_fit_decorrelation_bandwidth_mhz: float | None = None
    acf_fit_decorrelation_bandwidth_error_mhz: float | None = None
    acf_fit_diffractive_timescale_s: float | None = None
    acf_fit_diffractive_timescale_error_s: float | None = None
    acf_fit_width_covariance_mhz_s: float | None = None
    acf_width_covariance_mhz_s: float | None = None
    acf_slice_decorrelation_bandwidth_mhz: float | None = None
    acf_slice_diffractive_timescale_s: float | None = None
    acf_drift_slope_s_per_mhz: float | None = None
    acf_drift_slope_error_s_per_mhz: float | None = None
    acf_correlation: float | None = None
    acf_correlation_error: float | None = None
    scintillation_frequency_resolution_mhz: float | None = None
    scintillation_time_resolution_s: float | None = None
    decorrelation_bandwidth_resolution_bins: float | None = None
    diffractive_timescale_resolution_bins: float | None = None
    decorrelation_bandwidth_resolved: bool | None = None
    diffractive_timescale_resolved: bool | None = None
    scintillation_n_scintles: float | None = None
    scintillation_finite_scintle_fraction: float | None = None
    scintillation_eta_time: float | None = None
    scintillation_eta_frequency: float | None = None
    refractive_timescale_days: float | None = None
    refractive_timescale_error_days: float | None = None
    quality_masked_fraction: float | None = None
    quality_bad_time_bins: int | None = None
    quality_bad_frequency_channels: int | None = None
    quality_isolated_flagged_cells: int | None = None
    profile_intrinsic_model: str | None = None
    profile_components: int | None = None
    intrinsic_template: str | None = None
    anisotropic_alpha: float | None = None
    anisotropic_alpha_error: float | None = None
    anisotropic_alpha_error_lower: float | None = None
    anisotropic_alpha_error_upper: float | None = None
    anisotropic_tau0_s: float | None = None
    anisotropic_tau0_error_s: float | None = None
    anisotropic_tau0_error_lower_s: float | None = None
    anisotropic_tau0_error_upper_s: float | None = None
    anisotropic_tau_reference_frequency_mhz: float | None = None
    anisotropic_subbands_accepted: int | None = None
    anisotropic_subbands_rejected: int | None = None
    anisotropic_subbands_failed: int | None = None


def parse_time_params(value: str | None) -> set[str]:
    # Reference: Filothodoros thesis (ALEX.pdf), PDF pp. 34-35, Section 2.7.4,
    # defines the fitted reference-frequency series reported as tau150.
    aliases = {
        "dm": "dm",
        "tau": "tau",
        "tau_ref": "tau_ref",
        "tau0": "tau_ref",
        "tau150": "tau_ref",
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
        return {"dm", "tau", "tau_ref", "alpha", "dnu_d", "dt_d", "t_r"}

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
            return {"dm", "tau", "tau_ref", "alpha", "dnu_d", "dt_d", "t_r"}
        params.add(mapped)
    return params or {"dm", "tau", "tau_ref", "alpha", "dnu_d", "dt_d", "t_r"}


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
        "start_utc",
        "end_utc",
        "start_mjd",
        "end_mjd",
        "telescope",
        "telescope_latitude_deg",
        "telescope_longitude_deg",
        "telescope_height_m",
        "source_ra_deg",
        "source_dec_deg",
        "observing_frequency_mhz",
        "period_s",
        "sun_separation_start_deg",
        "sun_separation_mid_deg",
        "sun_separation_end_deg",
        "solar_ephemeris_method",
        "dm",
        "archive_dm",
        "dm_error",
        "dm_method",
        "dm_correction",
        "pdmp_snr",
        "scattering_delta_dm",
        "scattering_delta_dm_error",
        "scattering_corrected_dm",
        "scattering_dm_reduced_chi_square",
        "tau_s",
        "tau_error_s",
        "alpha",
        "alpha_error",
        "alpha_error_lower",
        "alpha_error_upper",
        "alpha_covariance_error",
        "alpha_monte_carlo_samples",
        "alpha_subbands_accepted",
        "alpha_subbands_rejected",
        "alpha_subbands_failed",
        "tau0_s",
        "tau0_error_s",
        "tau0_error_lower_s",
        "tau0_error_upper_s",
        "tau_reference_frequency_mhz",
        "decorrelation_bandwidth_mhz",
        "decorrelation_bandwidth_error_mhz",
        "diffractive_timescale_s",
        "diffractive_timescale_error_s",
        "scintillation_measurement_method",
        "acf_fit_decorrelation_bandwidth_mhz",
        "acf_fit_decorrelation_bandwidth_error_mhz",
        "acf_fit_diffractive_timescale_s",
        "acf_fit_diffractive_timescale_error_s",
        "acf_fit_width_covariance_mhz_s",
        "acf_width_covariance_mhz_s",
        "acf_slice_decorrelation_bandwidth_mhz",
        "acf_slice_diffractive_timescale_s",
        "acf_drift_slope_s_per_mhz",
        "acf_drift_slope_error_s_per_mhz",
        "acf_correlation",
        "acf_correlation_error",
        "scintillation_frequency_resolution_mhz",
        "scintillation_time_resolution_s",
        "decorrelation_bandwidth_resolution_bins",
        "diffractive_timescale_resolution_bins",
        "decorrelation_bandwidth_resolved",
        "diffractive_timescale_resolved",
        "scintillation_n_scintles",
        "scintillation_finite_scintle_fraction",
        "scintillation_eta_time",
        "scintillation_eta_frequency",
        "refractive_timescale_days",
        "refractive_timescale_error_days",
        "quality_masked_fraction",
        "quality_bad_time_bins",
        "quality_bad_frequency_channels",
        "quality_isolated_flagged_cells",
        "profile_intrinsic_model",
        "profile_components",
        "intrinsic_template",
        "anisotropic_alpha",
        "anisotropic_alpha_error",
        "anisotropic_alpha_error_lower",
        "anisotropic_alpha_error_upper",
        "anisotropic_tau0_s",
        "anisotropic_tau0_error_s",
        "anisotropic_tau0_error_lower_s",
        "anisotropic_tau0_error_upper_s",
        "anisotropic_tau_reference_frequency_mhz",
        "anisotropic_subbands_accepted",
        "anisotropic_subbands_rejected",
        "anisotropic_subbands_failed",
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
                    "start_utc": "" if item.start_utc is None else item.start_utc.isoformat(),
                    "end_utc": "" if item.end_utc is None else item.end_utc.isoformat(),
                    "start_mjd": _format_optional(item.start_mjd),
                    "end_mjd": _format_optional(item.end_mjd),
                    "telescope": "" if item.telescope is None else item.telescope,
                    "telescope_latitude_deg": _format_optional(item.telescope_latitude_deg),
                    "telescope_longitude_deg": _format_optional(item.telescope_longitude_deg),
                    "telescope_height_m": _format_optional(item.telescope_height_m),
                    "source_ra_deg": _format_optional(item.source_ra_deg),
                    "source_dec_deg": _format_optional(item.source_dec_deg),
                    "observing_frequency_mhz": _format_optional(
                        item.observing_frequency_mhz
                    ),
                    "period_s": _format_optional(item.period_s),
                    "sun_separation_start_deg": _format_optional(item.sun_separation_start_deg),
                    "sun_separation_mid_deg": _format_optional(item.sun_separation_mid_deg),
                    "sun_separation_end_deg": _format_optional(item.sun_separation_end_deg),
                    "solar_ephemeris_method": item.solar_ephemeris_method or "",
                    "dm": _format_optional(item.dm),
                    "archive_dm": _format_optional(item.archive_dm),
                    "dm_error": _format_optional(item.dm_error),
                    "dm_method": item.dm_method or "",
                    "dm_correction": _format_optional(item.dm_correction),
                    "pdmp_snr": _format_optional(item.pdmp_snr),
                    "scattering_delta_dm": _format_optional(item.scattering_delta_dm),
                    "scattering_delta_dm_error": _format_optional(
                        item.scattering_delta_dm_error
                    ),
                    "scattering_corrected_dm": _format_optional(
                        item.scattering_corrected_dm
                    ),
                    "scattering_dm_reduced_chi_square": _format_optional(
                        item.scattering_dm_reduced_chi_square
                    ),
                    "tau_s": _format_optional(item.tau_s),
                    "tau_error_s": _format_optional(item.tau_error_s),
                    "alpha": _format_optional(item.alpha),
                    "alpha_error": _format_optional(item.alpha_error),
                    "alpha_error_lower": _format_optional(item.alpha_error_lower),
                    "alpha_error_upper": _format_optional(item.alpha_error_upper),
                    "alpha_covariance_error": _format_optional(item.alpha_covariance_error),
                    "alpha_monte_carlo_samples": _format_optional(
                        item.alpha_monte_carlo_samples
                    ),
                    "alpha_subbands_accepted": _format_optional(item.alpha_subbands_accepted),
                    "alpha_subbands_rejected": _format_optional(item.alpha_subbands_rejected),
                    "alpha_subbands_failed": _format_optional(item.alpha_subbands_failed),
                    "tau0_s": _format_optional(item.tau0_s),
                    "tau0_error_s": _format_optional(item.tau0_error_s),
                    "tau0_error_lower_s": _format_optional(item.tau0_error_lower_s),
                    "tau0_error_upper_s": _format_optional(item.tau0_error_upper_s),
                    "tau_reference_frequency_mhz": _format_optional(
                        item.tau_reference_frequency_mhz
                    ),
                    "decorrelation_bandwidth_mhz": _format_optional(item.decorrelation_bandwidth_mhz),
                    "decorrelation_bandwidth_error_mhz": _format_optional(
                        item.decorrelation_bandwidth_error_mhz
                    ),
                    "diffractive_timescale_s": _format_optional(item.diffractive_timescale_s),
                    "diffractive_timescale_error_s": _format_optional(
                        item.diffractive_timescale_error_s
                    ),
                    "scintillation_measurement_method": (
                        item.scintillation_measurement_method or ""
                    ),
                    "acf_fit_decorrelation_bandwidth_mhz": _format_optional(
                        item.acf_fit_decorrelation_bandwidth_mhz
                    ),
                    "acf_fit_decorrelation_bandwidth_error_mhz": _format_optional(
                        item.acf_fit_decorrelation_bandwidth_error_mhz
                    ),
                    "acf_fit_diffractive_timescale_s": _format_optional(
                        item.acf_fit_diffractive_timescale_s
                    ),
                    "acf_fit_diffractive_timescale_error_s": _format_optional(
                        item.acf_fit_diffractive_timescale_error_s
                    ),
                    "acf_fit_width_covariance_mhz_s": _format_optional(
                        item.acf_fit_width_covariance_mhz_s
                    ),
                    "acf_width_covariance_mhz_s": _format_optional(
                        item.acf_width_covariance_mhz_s
                    ),
                    "acf_slice_decorrelation_bandwidth_mhz": _format_optional(
                        item.acf_slice_decorrelation_bandwidth_mhz
                    ),
                    "acf_slice_diffractive_timescale_s": _format_optional(
                        item.acf_slice_diffractive_timescale_s
                    ),
                    "acf_drift_slope_s_per_mhz": _format_optional(
                        item.acf_drift_slope_s_per_mhz
                    ),
                    "acf_drift_slope_error_s_per_mhz": _format_optional(
                        item.acf_drift_slope_error_s_per_mhz
                    ),
                    "acf_correlation": _format_optional(item.acf_correlation),
                    "acf_correlation_error": _format_optional(item.acf_correlation_error),
                    "scintillation_frequency_resolution_mhz": _format_optional(
                        item.scintillation_frequency_resolution_mhz
                    ),
                    "scintillation_time_resolution_s": _format_optional(
                        item.scintillation_time_resolution_s
                    ),
                    "decorrelation_bandwidth_resolution_bins": _format_optional(
                        item.decorrelation_bandwidth_resolution_bins
                    ),
                    "diffractive_timescale_resolution_bins": _format_optional(
                        item.diffractive_timescale_resolution_bins
                    ),
                    "decorrelation_bandwidth_resolved": _format_optional_bool(
                        item.decorrelation_bandwidth_resolved
                    ),
                    "diffractive_timescale_resolved": _format_optional_bool(
                        item.diffractive_timescale_resolved
                    ),
                    "scintillation_n_scintles": _format_optional(
                        item.scintillation_n_scintles
                    ),
                    "scintillation_finite_scintle_fraction": _format_optional(
                        item.scintillation_finite_scintle_fraction
                    ),
                    "scintillation_eta_time": _format_optional(
                        item.scintillation_eta_time
                    ),
                    "scintillation_eta_frequency": _format_optional(
                        item.scintillation_eta_frequency
                    ),
                    "refractive_timescale_days": _format_optional(item.refractive_timescale_days),
                    "refractive_timescale_error_days": _format_optional(
                        item.refractive_timescale_error_days
                    ),
                    "quality_masked_fraction": _format_optional(item.quality_masked_fraction),
                    "quality_bad_time_bins": _format_optional(item.quality_bad_time_bins),
                    "quality_bad_frequency_channels": _format_optional(
                        item.quality_bad_frequency_channels
                    ),
                    "quality_isolated_flagged_cells": _format_optional(
                        item.quality_isolated_flagged_cells
                    ),
                    "profile_intrinsic_model": item.profile_intrinsic_model or "",
                    "profile_components": _format_optional(item.profile_components),
                    "intrinsic_template": item.intrinsic_template or "",
                    "anisotropic_alpha": _format_optional(item.anisotropic_alpha),
                    "anisotropic_alpha_error": _format_optional(
                        item.anisotropic_alpha_error
                    ),
                    "anisotropic_alpha_error_lower": _format_optional(
                        item.anisotropic_alpha_error_lower
                    ),
                    "anisotropic_alpha_error_upper": _format_optional(
                        item.anisotropic_alpha_error_upper
                    ),
                    "anisotropic_tau0_s": _format_optional(item.anisotropic_tau0_s),
                    "anisotropic_tau0_error_s": _format_optional(
                        item.anisotropic_tau0_error_s
                    ),
                    "anisotropic_tau0_error_lower_s": _format_optional(
                        item.anisotropic_tau0_error_lower_s
                    ),
                    "anisotropic_tau0_error_upper_s": _format_optional(
                        item.anisotropic_tau0_error_upper_s
                    ),
                    "anisotropic_tau_reference_frequency_mhz": _format_optional(
                        item.anisotropic_tau_reference_frequency_mhz
                    ),
                    "anisotropic_subbands_accepted": _format_optional(
                        item.anisotropic_subbands_accepted
                    ),
                    "anisotropic_subbands_rejected": _format_optional(
                        item.anisotropic_subbands_rejected
                    ),
                    "anisotropic_subbands_failed": _format_optional(
                        item.anisotropic_subbands_failed
                    ),
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
    if yerr is not None and np.any(np.isfinite(yerr[valid]) & (yerr[valid] >= 0)):
        yerr_plot = np.where(
            np.isfinite(yerr[valid]) & (yerr[valid] >= 0),
            yerr[valid],
            0.0,
        )

    sun_mid = _as_optional_float_array([item.sun_separation_mid_deg for item in measurements])
    sun_low = _as_optional_float_array([item.sun_separation_start_deg for item in measurements])
    sun_high = _as_optional_float_array([item.sun_separation_end_deg for item in measurements])
    sun_valid = np.isfinite(sun_mid) & np.isfinite(x)

    if np.any(sun_valid):
        fig, (ax_sun, ax) = plt.subplots(
            2,
            1,
            figsize=(8.5, 6.2),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 3], "hspace": 0.05},
        )
        sun_yerr = None
        if np.any(np.isfinite(sun_low[sun_valid])) and np.any(np.isfinite(sun_high[sun_valid])):
            lower = np.abs(sun_mid[sun_valid] - np.minimum(sun_low[sun_valid], sun_high[sun_valid]))
            upper = np.abs(np.maximum(sun_low[sun_valid], sun_high[sun_valid]) - sun_mid[sun_valid])
            sun_yerr = np.vstack([lower, upper])
        ax_sun.errorbar(
            x[sun_valid],
            sun_mid[sun_valid],
            yerr=sun_yerr,
            fmt="o",
            color="tab:orange",
            ecolor="tab:orange",
            capsize=2,
            markersize=4,
        )
        ax_sun.set_ylabel("Sun sep.\n(deg)")
        ax_sun.grid(True, alpha=0.25)
        ax_sun.tick_params(labelbottom=False)
    else:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))

    ax.errorbar(x[valid], y[valid], yerr=yerr_plot, fmt="o", color="black", capsize=2)
    reference_value = None
    reference_label = None
    if yerr is not None:
        weighted = valid & np.isfinite(yerr) & (yerr > 0)
        if np.count_nonzero(weighted) >= 2:
            weights = 1.0 / yerr[weighted] ** 2
            reference_value = float(np.sum(weights * y[weighted]) / np.sum(weights))
            reference_label = "inverse-variance weighted mean"
    if reference_value is None and np.count_nonzero(valid) >= 2:
        reference_value = float(np.mean(y[valid]))
        reference_label = "mean"
    if reference_value is not None:
        # Reference: Filothodoros thesis (ALEX.pdf), PDF p. 55, Section 4.2,
        # compares temporal scattering measurements with their weighted mean.
        # Falling back to an unweighted mean is a PSRISM display choice.
        ax.axhline(
            reference_value,
            color="tab:blue",
            linestyle="--",
            linewidth=1.0,
            label=reference_label,
        )
        ax.legend()
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

    top_axis_owner = ax_sun if np.any(sun_valid) else ax
    secax = top_axis_owner.secondary_xaxis("top", functions=(utc_num_to_mjd, mjd_to_utc_num))
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
        "dm": ([item.dm for item in measurements], [item.dm_error for item in measurements], "DM (pc cm^-3)"),
        "tau": ([item.tau_s for item in measurements], [item.tau_error_s for item in measurements], "Scattering timescale τ (s)"),
        # Reference: Filothodoros thesis (ALEX.pdf), PDF pp. 34-35,
        # Section 2.7.4, for the reference-frequency tau convention.
        "tau_ref": ([item.tau0_s for item in measurements], [item.tau0_error_s for item in measurements], "Reference-frequency scattering timescale τ_ref (s)"),
        "alpha": ([item.alpha for item in measurements], [item.alpha_error for item in measurements], "Scattering spectral index α"),
        "dnu_d": ([item.decorrelation_bandwidth_mhz for item in measurements], [item.decorrelation_bandwidth_error_mhz for item in measurements], "Decorrelation bandwidth Δν_d (MHz)"),
        "dt_d": ([item.diffractive_timescale_s for item in measurements], [item.diffractive_timescale_error_s for item in measurements], "Diffractive timescale Δt_d (s)"),
        "t_r": ([item.refractive_timescale_days for item in measurements], [item.refractive_timescale_error_days for item in measurements], "Refractive timescale T_r (days)"),
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


def _format_optional_bool(value: bool | None) -> str:
    if value is None:
        return ""
    return "true" if value else "false"
