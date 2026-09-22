"""Autocorrelation spectrum calculation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .conventions import ACF_DECORRELATION_LEVEL, ACF_TIMESCALE_LEVEL


@dataclass(frozen=True)
class AcfScales:
    decorrelation_bandwidth_mhz: float | None
    diffractive_timescale_s: float | None
    frequency_lag_mhz: float | None
    time_lag_s: float | None


@dataclass(frozen=True)
class ScintillationMeasurement:
    measurement_method: str
    decorrelation_bandwidth_mhz: float | None
    decorrelation_bandwidth_error_mhz: float | None
    diffractive_timescale_s: float | None
    diffractive_timescale_error_s: float | None
    fit_decorrelation_bandwidth_mhz: float | None
    fit_decorrelation_bandwidth_error_mhz: float | None
    fit_diffractive_timescale_s: float | None
    fit_diffractive_timescale_error_s: float | None
    fit_width_covariance_mhz_s: float | None
    width_covariance_mhz_s: float | None
    slice_decorrelation_bandwidth_mhz: float | None
    slice_diffractive_timescale_s: float | None
    drift_slope_s_per_mhz: float | None
    drift_slope_error_s_per_mhz: float | None
    correlation: float | None
    correlation_error: float | None
    frequency_resolution_mhz: float
    time_resolution_s: float
    decorrelation_bandwidth_resolution_bins: float | None
    diffractive_timescale_resolution_bins: float | None
    decorrelation_bandwidth_resolved: bool
    diffractive_timescale_resolved: bool
    frequency_crossing_found: bool
    time_crossing_found: bool
    minimum_resolution_bins: float
    n_scintles: float | None
    finite_scintle_fraction: float | None
    eta_time: float | None
    eta_frequency: float | None


def autocorrelation_lags(size: int) -> np.ndarray:
    """Return integer lags in the Cordes range -N/2 < lag < N/2."""
    # Reference: Lorimer & Kramer (2005), psrhandbook.pdf, Section 7.4.4.1,
    # states these integer lag ranges for the finite covariance.
    half = size / 2.0
    return np.asarray([lag for lag in range(-(size - 1), size) if -half < lag < half])


def calculate_covariance_function(
    dynspec: np.ndarray,
    subtract_mean: bool = True,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Calculate the finite-lag covariance function CF(delta_f, tau).

    Input dynamic spectra are shaped as (time, frequency). The returned covariance
    is shaped as (time lag, frequency lag).
    """
    from scipy.signal import fftconvolve

    arr, mask = _prepare_dynamic_spectrum(
        dynspec,
        subtract_mean=subtract_mean,
        valid_mask=valid_mask,
    )
    ntime, nfreq = arr.shape

    # Reference: Lorimer & Kramer (2005), psrhandbook.pdf, Section 7.4.4.1,
    # Eqs. 7.37-7.38, defines the finite-lag covariance and normalized ACF.
    full_covariance = fftconvolve(arr, arr[::-1, ::-1], mode="full")
    full_covariance = np.real(full_covariance)
    if not np.all(mask):
        # PSRISM choice: divide by valid-pair overlap so masking does not make
        # large lags artificially weak; no bundled reference fixes this rule.
        overlap = fftconvolve(
            mask.astype(float),
            mask[::-1, ::-1].astype(float),
            mode="full",
        )
        zero_overlap = float(np.count_nonzero(mask))
        with np.errstate(divide="ignore", invalid="ignore"):
            full_covariance = np.where(
                overlap > 0.5,
                full_covariance * zero_overlap / overlap,
                np.nan,
            )

    time_lags = autocorrelation_lags(ntime)
    freq_lags = autocorrelation_lags(nfreq)
    return full_covariance[np.ix_(ntime - 1 + time_lags, nfreq - 1 + freq_lags)]


def calculate_autocorrelation_spectrum(
    dynspec: np.ndarray,
    normalize: bool = True,
    subtract_mean: bool = True,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Calculate ACF = CF(delta_f, tau) / CF(0, 0) for a dynamic spectrum."""
    covariance = calculate_covariance_function(
        dynspec,
        subtract_mean=subtract_mean,
        valid_mask=valid_mask,
    )

    if normalize:
        time_lags = autocorrelation_lags(np.asarray(dynspec).shape[0])
        freq_lags = autocorrelation_lags(np.asarray(dynspec).shape[1])
        zero_time = int(np.where(time_lags == 0)[0][0])
        zero_freq = int(np.where(freq_lags == 0)[0][0])
        zero_lag = covariance[zero_time, zero_freq]
        if zero_lag != 0:
            covariance = covariance / zero_lag
    return covariance


def autocorrelation_axes(nsub: int, nchan: int, observation_time_s: float, bandwidth_mhz: float):
    """Return time-lag and frequency-lag axes for an autocorrelation spectrum."""
    dt = observation_time_s / nsub
    df = bandwidth_mhz / nchan
    time_lag = autocorrelation_lags(nsub) * dt
    freq_lag = autocorrelation_lags(nchan) * df
    return time_lag, freq_lag


def measure_acf_scales(
    acf2d: np.ndarray,
    time_lag_s: np.ndarray,
    freq_lag_mhz: np.ndarray,
) -> AcfScales:
    """Measure DISS bandwidth and timescale from normalized ACF slices."""
    arr = np.asarray(acf2d, dtype=float)
    zero_time = int(np.argmin(np.abs(time_lag_s)))
    zero_freq = int(np.argmin(np.abs(freq_lag_mhz)))

    # Reference: Lorimer & Kramer (2005), psrhandbook.pdf, Section 7.4.4.1:
    # Delta_nu_DISS is the frequency half-width at 1/2 and Delta_t_DISS is
    # the time half-width at 1/e.
    freq_width = _positive_half_width(
        freq_lag_mhz,
        arr[zero_time, :],
        threshold=ACF_DECORRELATION_LEVEL,
    )
    time_width = _positive_half_width(
        time_lag_s,
        arr[:, zero_freq],
        threshold=ACF_TIMESCALE_LEVEL,
    )

    return AcfScales(
        decorrelation_bandwidth_mhz=freq_width,
        diffractive_timescale_s=time_width,
        frequency_lag_mhz=freq_width,
        time_lag_s=time_width,
    )


def summarize_scintillation_measurement(
    scales: AcfScales,
    time_lag_s,
    freq_lag_mhz,
    fit_result=None,
    minimum_resolution_bins: float = 2.0,
    observing_duration_s: float | None = None,
    observing_bandwidth_mhz: float | None = None,
    eta_time: float = 0.2,
    eta_freq: float = 0.2,
) -> ScintillationMeasurement:
    """Select fitted ACF scales and attach resolution diagnostics."""
    if not np.isfinite(minimum_resolution_bins) or minimum_resolution_bins <= 0:
        raise ValueError("minimum_resolution_bins must be positive")
    time_resolution = _minimum_axis_step(time_lag_s, "time_lag_s")
    frequency_resolution = _minimum_axis_step(freq_lag_mhz, "freq_lag_mhz")

    if fit_result is None:
        method = "acf_slice_crossing"
        fit_dnu = None
        fit_dnu_error = None
        fit_dt = None
        fit_dt_error = None
        fit_width_covariance = None
        candidate_dnu = scales.decorrelation_bandwidth_mhz
        candidate_dnu_error = None
        candidate_dt = scales.diffractive_timescale_s
        candidate_dt_error = None
        drift = None
        drift_error = None
        correlation = None
        correlation_error = None
    else:
        method = "tilted_gaussian_2d"
        fit_dnu = float(fit_result.delta_f_diss)
        fit_dnu_error = _optional_finite(fit_result.delta_f_diss_error)
        fit_dt = float(fit_result.delta_t_diss)
        fit_dt_error = _optional_finite(fit_result.delta_t_diss_error)
        fit_width_covariance = _optional_finite(
            fit_result.delta_f_delta_t_covariance_mhz_s
        )
        candidate_dnu = fit_dnu
        candidate_dnu_error = fit_dnu_error
        candidate_dt = fit_dt
        candidate_dt_error = fit_dt_error
        drift = _optional_finite(fit_result.drift_slope_s_per_mhz)
        drift_error = _optional_finite(fit_result.drift_slope_error_s_per_mhz)
        correlation = _optional_finite(fit_result.correlation)
        correlation_error = _optional_finite(fit_result.correlation_error)

    dnu_bins = _resolution_bins(candidate_dnu, frequency_resolution)
    dt_bins = _resolution_bins(candidate_dt, time_resolution)
    frequency_crossing = scales.decorrelation_bandwidth_mhz is not None
    time_crossing = scales.diffractive_timescale_s is not None
    dnu_resolved = bool(
        frequency_crossing
        and dnu_bins is not None
        and dnu_bins >= minimum_resolution_bins
    )
    dt_resolved = bool(
        time_crossing
        and dt_bins is not None
        and dt_bins >= minimum_resolution_bins
    )
    statistics = None
    coverage_values = (observing_duration_s, observing_bandwidth_mhz)
    if any(value is not None for value in coverage_values):
        if any(value is None for value in coverage_values):
            raise ValueError(
                "observing_duration_s and observing_bandwidth_mhz must be supplied together"
            )
        if dnu_resolved and dt_resolved:
            from .refractive_scintillation import finite_scintle_statistics

            statistics = finite_scintle_statistics(
                float(observing_duration_s),
                float(observing_bandwidth_mhz),
                float(candidate_dt),
                float(candidate_dnu),
                eta_time=eta_time,
                eta_freq=eta_freq,
            )

    # Reference: Lorimer & Kramer (2005), psrhandbook.pdf, Section 7.4.4.1,
    # recommends a 2D Gaussian ACF fit and defines these two coordinate-axis
    # widths. Requiring two samples per width and retaining the slice crossing
    # as an in-window check are conservative PSRISM quality choices.
    return ScintillationMeasurement(
        measurement_method=method,
        decorrelation_bandwidth_mhz=(float(candidate_dnu) if dnu_resolved else None),
        decorrelation_bandwidth_error_mhz=(
            candidate_dnu_error if dnu_resolved else None
        ),
        diffractive_timescale_s=(float(candidate_dt) if dt_resolved else None),
        diffractive_timescale_error_s=(candidate_dt_error if dt_resolved else None),
        fit_decorrelation_bandwidth_mhz=fit_dnu,
        fit_decorrelation_bandwidth_error_mhz=fit_dnu_error,
        fit_diffractive_timescale_s=fit_dt,
        fit_diffractive_timescale_error_s=fit_dt_error,
        fit_width_covariance_mhz_s=fit_width_covariance,
        width_covariance_mhz_s=(
            fit_width_covariance if dnu_resolved and dt_resolved else None
        ),
        slice_decorrelation_bandwidth_mhz=_optional_finite(
            scales.decorrelation_bandwidth_mhz
        ),
        slice_diffractive_timescale_s=_optional_finite(scales.diffractive_timescale_s),
        drift_slope_s_per_mhz=drift,
        drift_slope_error_s_per_mhz=drift_error,
        correlation=correlation,
        correlation_error=correlation_error,
        frequency_resolution_mhz=frequency_resolution,
        time_resolution_s=time_resolution,
        decorrelation_bandwidth_resolution_bins=dnu_bins,
        diffractive_timescale_resolution_bins=dt_bins,
        decorrelation_bandwidth_resolved=dnu_resolved,
        diffractive_timescale_resolved=dt_resolved,
        frequency_crossing_found=frequency_crossing,
        time_crossing_found=time_crossing,
        minimum_resolution_bins=float(minimum_resolution_bins),
        n_scintles=None if statistics is None else statistics.n_scintles,
        finite_scintle_fraction=(
            None if statistics is None else statistics.fractional_error
        ),
        eta_time=None if statistics is None else statistics.eta_time,
        eta_frequency=None if statistics is None else statistics.eta_frequency,
    )


def write_scintillation_report(
    measurement: ScintillationMeasurement,
    output_path: str | Path,
) -> None:
    """Write one ACF scintillation measurement and its quality metadata."""
    with Path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(asdict(measurement), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _positive_half_width(axis: np.ndarray, values: np.ndarray, threshold: float) -> float | None:
    axis = np.asarray(axis, dtype=float)
    values = np.asarray(values, dtype=float)
    zero_idx = int(np.argmin(np.abs(axis)))
    positive = axis >= 0
    x = axis[positive]
    y = values[positive]

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if len(x) < 2:
        return None

    zero_local = int(np.argmin(np.abs(x)))
    x = x[zero_local:]
    y = y[zero_local:]
    if len(x) < 2 or not np.isfinite(y[0]) or y[0] <= 0:
        return None

    y = y / y[0]
    for idx in range(1, len(x)):
        if not np.isfinite(y[idx]):
            continue
        if y[idx] <= threshold:
            x0, x1 = x[idx - 1], x[idx]
            y0, y1 = y[idx - 1], y[idx]
            if y1 == y0:
                return float(x1)
            frac = (threshold - y0) / (y1 - y0)
            return float(x0 + frac * (x1 - x0))
    return None


def _minimum_axis_step(axis, name: str) -> float:
    values = np.asarray(axis, dtype=float)
    finite = np.sort(np.unique(values[np.isfinite(values)]))
    positive = np.diff(finite)
    positive = positive[positive > 0]
    if not len(positive):
        raise ValueError(f"{name} must contain at least two distinct finite values")
    return float(np.min(positive))


def _resolution_bins(value: float | None, resolution: float) -> float | None:
    if value is None or not np.isfinite(value) or value <= 0:
        return None
    return float(value / resolution)


def _optional_finite(value) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _prepare_dynamic_spectrum(
    dynspec: np.ndarray,
    subtract_mean: bool,
    valid_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(dynspec, dtype=float)
    if arr.ndim != 2:
        raise ValueError("dynspec must be a 2D array shaped as (time, frequency)")

    finite_mask = np.isfinite(arr)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != arr.shape:
            raise ValueError("valid_mask must have the same shape as dynspec")
        finite_mask &= mask

    if not np.any(finite_mask):
        raise ValueError("dynspec has no finite valid samples for ACF calculation")

    prepared = np.zeros_like(arr, dtype=float)
    values = arr[finite_mask]
    if subtract_mean:
        values = values - np.mean(values)
    prepared[finite_mask] = values
    return prepared, finite_mask
