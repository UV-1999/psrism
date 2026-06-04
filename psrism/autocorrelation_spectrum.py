"""Autocorrelation spectrum calculation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AcfScales:
    decorrelation_bandwidth_mhz: float | None
    diffractive_timescale_s: float | None
    frequency_lag_mhz: float | None
    time_lag_s: float | None


def autocorrelation_lags(size: int) -> np.ndarray:
    """Return integer lags in the Cordes range -N/2 < lag < N/2."""
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

    arr = _prepare_dynamic_spectrum(dynspec, subtract_mean=subtract_mean, valid_mask=valid_mask)
    ntime, nfreq = arr.shape

    full_covariance = fftconvolve(arr, arr[::-1, ::-1], mode="full")
    full_covariance = np.real(full_covariance)

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

    freq_width = _positive_half_width(freq_lag_mhz, arr[zero_time, :], threshold=0.5)
    time_width = _positive_half_width(time_lag_s, arr[:, zero_freq], threshold=1.0 / np.e)

    return AcfScales(
        decorrelation_bandwidth_mhz=freq_width,
        diffractive_timescale_s=time_width,
        frequency_lag_mhz=freq_width,
        time_lag_s=time_width,
    )


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


def _prepare_dynamic_spectrum(
    dynspec: np.ndarray,
    subtract_mean: bool,
    valid_mask: np.ndarray | None,
) -> np.ndarray:
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
    return prepared
