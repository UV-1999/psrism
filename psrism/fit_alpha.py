"""Power-law alpha fitting for tau-frequency measurements."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .conventions import TAU_REFERENCE_FREQUENCY_MHZ


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
    alpha_error_lower: float
    alpha_error_upper: float
    tau0_error_lower: float
    tau0_error_upper: float
    covariance_alpha_error: float
    covariance_tau0_error: float
    monte_carlo_samples: int
    uncertainty_method: str
    monte_carlo_alpha: np.ndarray | None = field(default=None, repr=False, compare=False)
    monte_carlo_tau0: np.ndarray | None = field(default=None, repr=False, compare=False)


def tau_power_law(
    freq_mhz,
    tau0,
    alpha,
    reference_freq_mhz: float = TAU_REFERENCE_FREQUENCY_MHZ,
):
    """Return ``tau_ref * (freq / reference_freq)**(-alpha)``.

    ``alpha`` follows the positive literature convention: a normal scattering
    law that decreases with frequency has ``alpha > 0``.
    """
    # Reference: Lorimer & Kramer (2005), psrhandbook.pdf, Section 4.2.3;
    # Filothodoros thesis (ALEX.pdf), PDF pp. 34-35, Eq. 2.6 and Section 2.7.4.
    if reference_freq_mhz <= 0:
        raise ValueError("reference_freq_mhz must be positive")
    return tau0 * (np.asarray(freq_mhz, dtype=float) / reference_freq_mhz) ** (-alpha)


def fit_alpha(
    freq_mhz,
    tau,
    tau_error=None,
    reference_freq_mhz: float = TAU_REFERENCE_FREQUENCY_MHZ,
    monte_carlo_samples: int = 10000,
    random_seed: int | None = 0,
) -> AlphaFitResult:
    """Fit positive-convention alpha in log10(tau)-log10(freq) space.

    ``tau_error`` may contain symmetric errors with shape ``(N,)`` or lower
    and upper errors with shape ``(2, N)``. The weighted fit uses their mean;
    Monte Carlo draws retain the asymmetry.
    """
    freq = np.asarray(freq_mhz, dtype=float)
    values = np.asarray(tau, dtype=float)
    error_lower, error_upper = _coerce_tau_errors(tau_error, values.shape)

    if freq.shape != values.shape:
        raise ValueError("freq_mhz and tau must have the same shape")
    if reference_freq_mhz <= 0:
        raise ValueError("reference_freq_mhz must be positive")
    if monte_carlo_samples < 0:
        raise ValueError("monte_carlo_samples cannot be negative")

    valid = np.isfinite(freq) & np.isfinite(values) & (freq > 0) & (values > 0)
    if error_lower is not None:
        valid &= (
            np.isfinite(error_lower)
            & np.isfinite(error_upper)
            & (error_lower > 0)
            & (error_upper > 0)
        )

    freq = freq[valid]
    values = values[valid]
    if error_lower is not None:
        error_lower = error_lower[valid]
        error_upper = error_upper[valid]

    if len(freq) < 2:
        raise ValueError("at least two valid tau-frequency points are required to fit alpha")

    x = np.log10(freq / reference_freq_mhz)
    y = np.log10(values)

    # The log-linear form follows the Filothodoros thesis (ALEX.pdf), PDF
    # pp. 34-35, Eq. 2.6 and Section 2.7.4. Using -x makes the fitted
    # coefficient equal to the positive alpha convention in Handbook 4.2.3.
    design = np.column_stack([-x, np.ones_like(x)])
    if error_lower is None:
        sigma_y = np.ones_like(y)
    else:
        symmetric_error = 0.5 * (error_lower + error_upper)
        sigma_y = symmetric_error / (values * np.log(10.0))

    # PSRISM choice: analytic weighted least squares and first-order log-error
    # propagation implement the cited power law; these are not fixed constants
    # prescribed by the bundled references.
    weights = 1.0 / np.square(sigma_y)
    normal_matrix = design.T @ (weights[:, np.newaxis] * design)
    try:
        covariance = np.linalg.inv(normal_matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("tau-frequency points do not constrain a unique power law") from exc
    coeffs = covariance @ (design.T @ (weights * y))
    residual = y - design @ coeffs
    chi_square = float(np.sum((residual / sigma_y) ** 2))
    dof = max(len(y) - 2, 0)
    reduced_chi_square = float(chi_square / dof) if dof > 0 else None
    rms_log_residual = float(np.sqrt(np.mean(residual**2)))

    if error_lower is None and len(y) > 2:
        covariance = covariance * reduced_chi_square

    alpha = float(coeffs[0])
    intercept = float(coeffs[1])
    covariance_alpha_error = float(np.sqrt(covariance[0, 0]))
    intercept_error = float(np.sqrt(covariance[1, 1]))
    tau0 = float(10.0**intercept)
    covariance_tau0_error = float(tau0 * np.log(10.0) * intercept_error)

    mc_alpha = None
    mc_tau0 = None
    if error_lower is not None and monte_carlo_samples > 0:
        # Reference: Filothodoros et al. (Alex_and_Boe.pdf), PDF p. 4,
        # Section 2.2, and the Filothodoros thesis (ALEX.pdf), PDF pp. 34-35,
        # Section 2.7.4, estimate alpha errors both from least squares and
        # Monte Carlo and retain the larger estimate. Split-normal log-space
        # draws and the fixed default seed are reproducible PSRISM choices.
        mc_alpha, mc_tau0 = _monte_carlo_power_law(
            design,
            weights,
            values,
            error_lower,
            error_upper,
            int(monte_carlo_samples),
            random_seed,
        )

    if mc_alpha is None:
        alpha_error_lower = covariance_alpha_error
        alpha_error_upper = covariance_alpha_error
        tau0_error_lower = covariance_tau0_error
        tau0_error_upper = covariance_tau0_error
        uncertainty_method = "weighted_least_squares"
        used_mc_samples = 0
    else:
        alpha_q16, alpha_q84 = np.percentile(mc_alpha, [16.0, 84.0])
        tau0_q16, tau0_q84 = np.percentile(mc_tau0, [16.0, 84.0])
        alpha_error_lower = max(float(alpha - alpha_q16), 0.0)
        alpha_error_upper = max(float(alpha_q84 - alpha), 0.0)
        tau0_error_lower = max(float(tau0 - tau0_q16), 0.0)
        tau0_error_upper = max(float(tau0_q84 - tau0), 0.0)
        uncertainty_method = "weighted_least_squares+monte_carlo"
        used_mc_samples = int(len(mc_alpha))

    # Reference procedure: retain the larger fit or Monte Carlo uncertainty.
    alpha_error = max(covariance_alpha_error, alpha_error_lower, alpha_error_upper)
    tau0_error = max(covariance_tau0_error, tau0_error_lower, tau0_error_upper)

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
        alpha_error_lower=alpha_error_lower,
        alpha_error_upper=alpha_error_upper,
        tau0_error_lower=tau0_error_lower,
        tau0_error_upper=tau0_error_upper,
        covariance_alpha_error=covariance_alpha_error,
        covariance_tau0_error=covariance_tau0_error,
        monte_carlo_samples=used_mc_samples,
        uncertainty_method=uncertainty_method,
        monte_carlo_alpha=mc_alpha,
        monte_carlo_tau0=mc_tau0,
    )


def _coerce_tau_errors(tau_error, value_shape):
    if tau_error is None:
        return None, None
    errors = np.asarray(tau_error, dtype=float)
    if errors.shape == value_shape:
        return errors, errors
    asymmetric_shape = (2,) + tuple(value_shape)
    if errors.shape == asymmetric_shape:
        return errors[0], errors[1]
    raise ValueError(
        "tau_error must match tau or have shape (2, N) for lower/upper errors"
    )


def _monte_carlo_power_law(
    design: np.ndarray,
    weights: np.ndarray,
    tau: np.ndarray,
    error_lower: np.ndarray,
    error_upper: np.ndarray,
    n_samples: int,
    random_seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    log_tau = np.log10(tau)
    # PSRISM choice: use the same first-order log-error propagation as the
    # weighted fit on each side of an asymmetric uncertainty. Sampling in log
    # space keeps every synthetic scattering time physically positive.
    sigma_lower = error_lower / (tau * np.log(10.0))
    sigma_upper = error_upper / (tau * np.log(10.0))

    standard_draws = rng.normal(size=(n_samples, len(tau)))
    sigma = np.where(standard_draws < 0.0, sigma_lower, sigma_upper)
    sampled_log_tau = log_tau + standard_draws * sigma

    normal_matrix = design.T @ (weights[:, np.newaxis] * design)
    coefficient_transform = np.linalg.inv(normal_matrix) @ (design.T * weights)
    coefficients = sampled_log_tau @ coefficient_transform.T
    return coefficients[:, 0], np.power(10.0, coefficients[:, 1])
