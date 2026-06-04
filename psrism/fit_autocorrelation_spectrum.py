"""Fit utilities for autocorrelation spectra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianAcfFitResult:
    amplitude: float
    time_sigma: float
    freq_sigma: float
    offset: float
    covariance: np.ndarray

    @property
    def delta_t_diss(self) -> float:
        """Half width at ACF(0, tau) = 1/e for a zero-offset Gaussian."""
        return float(np.sqrt(2.0) * self.time_sigma)

    @property
    def delta_f_diss(self) -> float:
        """Half width at ACF(delta_f, 0) = 1/2 for a zero-offset Gaussian."""
        return float(np.sqrt(2.0 * np.log(2.0)) * self.freq_sigma)


def gaussian_acf_model(coords, amplitude, time_sigma, freq_sigma, offset):
    """Centered elliptical Gaussian model for a 2D ACF."""
    time_lag, freq_lag = coords
    return offset + amplitude * np.exp(
        -0.5 * ((time_lag / time_sigma) ** 2 + (freq_lag / freq_sigma) ** 2)
    )


def fit_autocorrelation_spectrum(acf2d, time_lag, freq_lag) -> GaussianAcfFitResult:
    """Fit a centered Gaussian to a 2D autocorrelation spectrum."""
    from scipy.optimize import curve_fit

    arr = np.asarray(acf2d, dtype=float)
    tt, ff = np.meshgrid(time_lag, freq_lag, indexing="ij")
    p0 = [float(np.max(arr) - np.min(arr)), np.std(time_lag), np.std(freq_lag), float(np.min(arr))]
    bounds = ([0.0, 1e-12, 1e-12, -np.inf], [np.inf, np.inf, np.inf, np.inf])

    popt, pcov = curve_fit(
        gaussian_acf_model,
        (tt.ravel(), ff.ravel()),
        arr.ravel(),
        p0=p0,
        bounds=bounds,
        maxfev=20000,
    )
    return GaussianAcfFitResult(
        amplitude=float(popt[0]),
        time_sigma=float(popt[1]),
        freq_sigma=float(popt[2]),
        offset=float(popt[3]),
        covariance=pcov,
    )
