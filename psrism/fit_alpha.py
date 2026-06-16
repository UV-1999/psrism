"""Power-law alpha fitting for tau-frequency measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlphaFitResult:
    tau0: float
    alpha: float
    tau0_error: float
    alpha_error: float
    reference_freq_mhz: float
    intercept: float
    intercept_error: float
    chi_square: float
    reduced_chi_square: float | None
    dof: int
    rms_log_residual: float
    n_points: int


def tau_power_law(freq_mhz, tau0, alpha, reference_freq_mhz=1000.0):
    """Power law tau = tau0 * (freq / reference_freq) ** alpha."""
    return tau0 * (np.asarray(freq_mhz, dtype=float) / reference_freq_mhz) ** alpha


def fit_alpha(freq_mhz, tau, tau_error=None, reference_freq_mhz: float = 1000.0) -> AlphaFitResult:
    """Fit alpha with weighted least squares in log10(tau)-log10(freq) space."""
    freq = np.asarray(freq_mhz, dtype=float)
    values = np.asarray(tau, dtype=float)
    errors = None if tau_error is None else np.asarray(tau_error, dtype=float)

    valid = np.isfinite(freq) & np.isfinite(values) & (freq > 0) & (values > 0)
    if errors is not None:
        valid &= np.isfinite(errors) & (errors > 0)

    freq = freq[valid]
    values = values[valid]
    if errors is not None:
        errors = errors[valid]

    if len(freq) < 2:
        raise ValueError("at least two valid tau-frequency points are required to fit alpha")

    x = np.log10(freq / reference_freq_mhz)
    y = np.log10(values)

    design = np.column_stack([x, np.ones_like(x)])
    if errors is None:
        sigma_y = np.ones_like(y)
    else:
        sigma_y = errors / (values * np.log(10.0))

    weights = 1.0 / np.square(sigma_y)
    normal_matrix = design.T @ (weights[:, np.newaxis] * design)
    covariance = np.linalg.inv(normal_matrix)
    coeffs = covariance @ (design.T @ (weights * y))
    residual = y - design @ coeffs
    chi_square = float(np.sum((residual / sigma_y) ** 2))
    dof = max(len(y) - 2, 0)
    reduced_chi_square = float(chi_square / dof) if dof > 0 else None
    rms_log_residual = float(np.sqrt(np.mean(residual**2)))

    if errors is None and len(y) > 2:
        covariance = covariance * reduced_chi_square

    alpha = float(coeffs[0])
    intercept = float(coeffs[1])
    alpha_error = float(np.sqrt(covariance[0, 0]))
    intercept_error = float(np.sqrt(covariance[1, 1]))
    tau0 = float(10.0**intercept)
    tau0_error = float(tau0 * np.log(10.0) * intercept_error)

    return AlphaFitResult(
        tau0=tau0,
        alpha=alpha,
        tau0_error=tau0_error,
        alpha_error=alpha_error,
        reference_freq_mhz=reference_freq_mhz,
        intercept=intercept,
        intercept_error=intercept_error,
        chi_square=chi_square,
        reduced_chi_square=reduced_chi_square,
        dof=dof,
        rms_log_residual=rms_log_residual,
        n_points=int(len(y)),
    )
