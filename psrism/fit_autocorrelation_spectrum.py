"""Fit utilities for autocorrelation spectra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TiltedGaussianAcfFitResult:
    amplitude: float
    time_sigma: float
    freq_sigma: float
    correlation: float
    offset: float
    covariance: np.ndarray
    n_fit_points: int
    rms_residual: float
    reduced_chi_square: float | None

    @property
    def a(self) -> float:
        """Quadratic coefficient for ``a dt^2 + b dt dnu + c dnu^2``."""
        denom = 2.0 * (1.0 - self.correlation**2) * self.time_sigma**2
        return float(1.0 / denom)

    @property
    def b(self) -> float:
        """Quadratic cross coefficient for ``a dt^2 + b dt dnu + c dnu^2``."""
        denom = (1.0 - self.correlation**2) * self.time_sigma * self.freq_sigma
        return float(-self.correlation / denom)

    @property
    def c(self) -> float:
        """Quadratic coefficient for ``a dt^2 + b dt dnu + c dnu^2``."""
        denom = 2.0 * (1.0 - self.correlation**2) * self.freq_sigma**2
        return float(1.0 / denom)

    @property
    def delta_t_diss(self) -> float:
        """Coordinate-axis half width at ACF(0, dt) = 1/e."""
        return float(1.0 / np.sqrt(self.a))

    @property
    def delta_f_diss(self) -> float:
        """Coordinate-axis half width at ACF(dnu, 0) = 1/2."""
        return float(np.sqrt(np.log(2.0) / self.c))

    @property
    def drift_slope_s_per_mhz(self) -> float | None:
        """Ridge slope ``d(dt) / d(dnu)`` in seconds per MHz."""
        if self.a == 0:
            return None
        return float(-self.b / (2.0 * self.a))

    @property
    def drift_rate_mhz_per_s(self) -> float | None:
        """Inverse ridge slope, useful as ``d(dnu) / d(dt)``."""
        slope = self.drift_slope_s_per_mhz
        if slope is None or slope == 0:
            return None
        return float(1.0 / slope)

    @property
    def rotation_angle_deg(self) -> float:
        """Rotation angle of the fitted ellipse in plot-coordinate units."""
        angle_rad = 0.5 * np.arctan2(self.b, self.a - self.c)
        return float(np.degrees(angle_rad))


GaussianAcfFitResult = TiltedGaussianAcfFitResult


def tilted_gaussian_acf_model(coords, amplitude, time_sigma, freq_sigma, correlation, offset):
    """Centered tilted Gaussian model for a 2D ACF."""
    time_lag, freq_lag = coords
    correlation = np.clip(correlation, -0.999, 0.999)
    denom = 2.0 * (1.0 - correlation**2)
    quad = (
        (time_lag / time_sigma) ** 2
        - 2.0 * correlation * (time_lag / time_sigma) * (freq_lag / freq_sigma)
        + (freq_lag / freq_sigma) ** 2
    ) / denom
    return offset + amplitude * np.exp(-quad)


def gaussian_acf_model(coords, amplitude, time_sigma, freq_sigma, offset):
    """Backward-compatible axis-aligned ACF model."""
    return tilted_gaussian_acf_model(coords, amplitude, time_sigma, freq_sigma, 0.0, offset)


def fit_autocorrelation_spectrum(
    acf2d,
    time_lag,
    freq_lag,
    max_fit_points: int = 50000,
) -> TiltedGaussianAcfFitResult:
    """Fit a centered tilted Gaussian to the central ACF lobe."""
    from scipy.optimize import curve_fit

    arr = np.asarray(acf2d, dtype=float)
    time_lag = np.asarray(time_lag, dtype=float)
    freq_lag = np.asarray(freq_lag, dtype=float)
    if arr.shape != (len(time_lag), len(freq_lag)):
        raise ValueError("acf2d shape must match time_lag and freq_lag")

    tt, ff = np.meshgrid(time_lag, freq_lag, indexing="ij")
    fit_mask = _central_fit_mask(arr, time_lag, freq_lag)
    t_fit = tt[fit_mask]
    f_fit = ff[fit_mask]
    y_fit = arr[fit_mask]
    if y_fit.size < 20:
        raise ValueError("not enough finite central ACF samples for tilted Gaussian fit")

    if y_fit.size > max_fit_points:
        keep = np.linspace(0, y_fit.size - 1, max_fit_points).astype(int)
        t_fit = t_fit[keep]
        f_fit = f_fit[keep]
        y_fit = y_fit[keep]

    time_width, freq_width = _initial_axis_widths(arr, time_lag, freq_lag)
    time_sigma0 = _safe_initial_sigma(time_width / np.sqrt(2.0), time_lag)
    freq_sigma0 = _safe_initial_sigma(freq_width / np.sqrt(2.0 * np.log(2.0)), freq_lag)
    offset0 = float(np.nanpercentile(y_fit, 5))
    amplitude0 = max(float(np.nanmax(y_fit) - offset0), 1e-6)
    p0 = [amplitude0, time_sigma0, freq_sigma0, 0.0, offset0]
    bounds = (
        [0.0, _min_positive_step(time_lag), _min_positive_step(freq_lag), -0.95, -np.inf],
        [np.inf, np.inf, np.inf, 0.95, np.inf],
    )

    popt, pcov = curve_fit(
        tilted_gaussian_acf_model,
        (t_fit, f_fit),
        y_fit,
        p0=p0,
        bounds=bounds,
        maxfev=20000,
    )
    residual = y_fit - tilted_gaussian_acf_model((t_fit, f_fit), *popt)
    dof = y_fit.size - len(popt)
    chi_square = float(np.sum(residual**2))
    reduced_chi_square = chi_square / dof if dof > 0 else None
    return TiltedGaussianAcfFitResult(
        amplitude=float(popt[0]),
        time_sigma=float(popt[1]),
        freq_sigma=float(popt[2]),
        correlation=float(popt[3]),
        offset=float(popt[4]),
        covariance=pcov,
        n_fit_points=int(y_fit.size),
        rms_residual=float(np.sqrt(np.mean(residual**2))),
        reduced_chi_square=reduced_chi_square,
    )


def evaluate_acf_fit(result: TiltedGaussianAcfFitResult, time_lag, freq_lag) -> np.ndarray:
    """Evaluate a fitted ACF model on time/frequency lag axes."""
    tt, ff = np.meshgrid(time_lag, freq_lag, indexing="ij")
    return tilted_gaussian_acf_model(
        (tt, ff),
        result.amplitude,
        result.time_sigma,
        result.freq_sigma,
        result.correlation,
        result.offset,
    )


def _central_fit_mask(arr: np.ndarray, time_lag: np.ndarray, freq_lag: np.ndarray) -> np.ndarray:
    finite = np.isfinite(arr)
    time_limit = _central_axis_limit(time_lag)
    freq_limit = _central_axis_limit(freq_lag)
    tt, ff = np.meshgrid(time_lag, freq_lag, indexing="ij")
    mask = finite & (np.abs(tt) <= time_limit) & (np.abs(ff) <= freq_limit)

    positive_lobe = arr >= max(0.03, 0.1 * np.nanmax(arr[finite]))
    if np.count_nonzero(mask & positive_lobe) >= 20:
        mask &= positive_lobe
    return mask


def _central_axis_limit(axis: np.ndarray) -> float:
    axis = np.asarray(axis, dtype=float)
    max_abs = float(np.nanmax(np.abs(axis)))
    if max_abs == 0:
        return 0.0
    return 0.35 * max_abs


def _initial_axis_widths(arr: np.ndarray, time_lag: np.ndarray, freq_lag: np.ndarray) -> tuple[float, float]:
    zero_time = int(np.argmin(np.abs(time_lag)))
    zero_freq = int(np.argmin(np.abs(freq_lag)))
    time_width = _positive_width(time_lag, arr[:, zero_freq], threshold=1.0 / np.e)
    freq_width = _positive_width(freq_lag, arr[zero_time, :], threshold=0.5)
    if time_width is None:
        time_width = max(_central_axis_limit(time_lag) / 3.0, _min_positive_step(time_lag))
    if freq_width is None:
        freq_width = max(_central_axis_limit(freq_lag) / 3.0, _min_positive_step(freq_lag))
    return float(time_width), float(freq_width)


def _positive_width(axis: np.ndarray, values: np.ndarray, threshold: float) -> float | None:
    axis = np.asarray(axis, dtype=float)
    values = np.asarray(values, dtype=float)
    positive = axis >= 0
    x = axis[positive]
    y = values[positive]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
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


def _safe_initial_sigma(value: float, axis: np.ndarray) -> float:
    step = _min_positive_step(axis)
    if not np.isfinite(value) or value <= 0:
        return step
    return max(float(value), step)


def _min_positive_step(axis: np.ndarray) -> float:
    diffs = np.diff(np.sort(np.unique(np.asarray(axis, dtype=float))))
    positive = diffs[diffs > 0]
    if len(positive) == 0:
        return 1e-12
    return float(np.min(positive))
