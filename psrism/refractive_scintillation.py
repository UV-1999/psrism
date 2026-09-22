"""Derived scintillation and refractive-timescale estimates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScintillationEstimate:
    frequency_mhz: float
    tau_s: float
    tau_fit_error_s: float
    decorrelation_bandwidth_mhz: float
    diffractive_timescale_s: float | None
    refractive_timescale_s: float | None
    refractive_timescale_days: float | None
    n_scintles: float | None
    finite_scintle_error_s: float | None
    tau_total_error_s: float | None


@dataclass(frozen=True)
class FiniteScintleStatistics:
    n_scintles: float
    fractional_error: float
    eta_time: float
    eta_frequency: float


def finite_scintle_statistics(
    observing_duration_s: float,
    observing_bandwidth_mhz: float,
    diffractive_timescale_s: float,
    decorrelation_bandwidth_mhz: float,
    eta_time: float = 0.2,
    eta_freq: float = 0.2,
) -> FiniteScintleStatistics:
    """Estimate sampled scintle count and its inverse-square-root fraction."""
    values = (
        observing_duration_s,
        observing_bandwidth_mhz,
        diffractive_timescale_s,
        decorrelation_bandwidth_mhz,
        eta_time,
        eta_freq,
    )
    if not all(np.isfinite(value) and value > 0 for value in values):
        raise ValueError("finite-scintle inputs and filling factors must be positive")
    # Reference: Lorimer & Kramer (2005), psrhandbook.pdf, Section 4.2.5.2;
    # Geiger & Lam (NANOGrav-Memo-008.pdf), PDF p. 5, Eqs. 6-7.
    n_scintles = (
        1.0 + eta_time * observing_duration_s / diffractive_timescale_s
    ) * (
        1.0 + eta_freq * observing_bandwidth_mhz / decorrelation_bandwidth_mhz
    )
    # The maximum with one follows the finite-scintle prescription cited
    # above; eta_t=eta_nu=0.2 are its adopted empirical defaults.
    n_scintles = max(float(n_scintles), 1.0)
    return FiniteScintleStatistics(
        n_scintles=n_scintles,
        fractional_error=float(1.0 / np.sqrt(n_scintles)),
        eta_time=float(eta_time),
        eta_frequency=float(eta_freq),
    )


def estimate_scintillation_from_tau(
    frequency_mhz,
    tau_s,
    tau_error_s,
    observing_bandwidth_mhz: float,
    observing_duration_s: float,
    distance_kpc: float | None = None,
    velocity_kms: float | None = None,
    c1: float = 1.16,
    eta_time: float = 0.2,
    eta_freq: float = 0.2,
) -> tuple[ScintillationEstimate, ...]:
    """Estimate scintillation parameters from fitted scattering times."""
    freq = np.asarray(frequency_mhz, dtype=float)
    tau = np.asarray(tau_s, dtype=float)
    tau_err = np.asarray(tau_error_s, dtype=float)
    if not (freq.shape == tau.shape == tau_err.shape):
        raise ValueError("frequency, tau, and tau_error arrays must have the same shape")
    if not np.isfinite(c1) or c1 <= 0:
        raise ValueError("c1 must be positive")
    if not np.isfinite(eta_time) or eta_time <= 0:
        raise ValueError("eta_time must be positive")
    if not np.isfinite(eta_freq) or eta_freq <= 0:
        raise ValueError("eta_freq must be positive")
    if (distance_kpc is None) != (velocity_kms is None):
        raise ValueError("distance_kpc and velocity_kms must be supplied together")
    if distance_kpc is not None and (
        not np.isfinite(distance_kpc)
        or not np.isfinite(velocity_kms)
        or distance_kpc <= 0
        or velocity_kms <= 0
    ):
        raise ValueError("distance_kpc and velocity_kms must be positive")

    # Defaults C1=1.16 and eta_t=eta_nu=0.2: Geiger & Lam
    # (NANOGrav-Memo-008.pdf), PDF pp. 5-6, Eqs. 6-8. Handbook Section
    # 4.2.5.2 gives the finite-scintle factors and empirical eta range 0.1-0.2.
    estimates: list[ScintillationEstimate] = []
    for nu_mhz, tau_value, tau_error in zip(freq, tau, tau_err):
        if not np.isfinite(nu_mhz) or not np.isfinite(tau_value) or tau_value <= 0:
            continue
        # Reference: Lorimer & Kramer (2005), psrhandbook.pdf, Sections 4.2.3
        # and 4.2.5.2: 2*pi*tau*Delta_nu_DISS = C1.
        decorrelation_bandwidth_mhz = c1 / (2.0 * np.pi * tau_value) / 1e6
        diffractive_timescale_s = None
        refractive_timescale_s = None
        refractive_timescale_days = None
        n_scintles = None
        finite_scintle_error_s = None
        tau_total_error_s = None

        if (
            distance_kpc is not None
            and velocity_kms is not None
            and distance_kpc > 0
            and velocity_kms > 0
            and decorrelation_bandwidth_mhz > 0
        ):
            nu_ghz = nu_mhz / 1000.0
            # Reference: Lorimer & Kramer (2005), psrhandbook.pdf, Section
            # 7.4.4.1, Eq. 7.39, including A=2.53e4 for these units.
            diffractive_timescale_s = (
                2.53e4
                * np.sqrt(distance_kpc * decorrelation_bandwidth_mhz)
                / (nu_ghz * velocity_kms)
            )
            # Reference: Geiger & Lam (NANOGrav-Memo-008.pdf), PDF p. 9,
            # Eq. 14, for the refractive timescale estimate.
            refractive_timescale_s = (
                (4.0 / np.pi)
                * (nu_mhz / decorrelation_bandwidth_mhz)
                * diffractive_timescale_s
            )
            refractive_timescale_days = refractive_timescale_s / 86400.0
            statistics = finite_scintle_statistics(
                observing_duration_s,
                observing_bandwidth_mhz,
                diffractive_timescale_s,
                decorrelation_bandwidth_mhz,
                eta_time=eta_time,
                eta_freq=eta_freq,
            )
            n_scintles = statistics.n_scintles
            # Reference: Geiger & Lam (NANOGrav-Memo-008.pdf), PDF p. 5,
            # Eq. 6, for sigma_fse=tau/sqrt(N_scintles).
            finite_scintle_error_s = tau_value * statistics.fractional_error
            if np.isfinite(tau_error) and tau_error > 0:
                # PSRISM choice: independent fit and finite-scintle errors are
                # combined in quadrature; the memo defines sigma_fse itself.
                tau_total_error_s = float(np.sqrt(tau_error**2 + finite_scintle_error_s**2))
            else:
                tau_total_error_s = float(finite_scintle_error_s)

        estimates.append(
            ScintillationEstimate(
                frequency_mhz=float(nu_mhz),
                tau_s=float(tau_value),
                tau_fit_error_s=float(tau_error),
                decorrelation_bandwidth_mhz=float(decorrelation_bandwidth_mhz),
                diffractive_timescale_s=_optional_float(diffractive_timescale_s),
                refractive_timescale_s=_optional_float(refractive_timescale_s),
                refractive_timescale_days=_optional_float(refractive_timescale_days),
                n_scintles=_optional_float(n_scintles),
                finite_scintle_error_s=_optional_float(finite_scintle_error_s),
                tau_total_error_s=_optional_float(tau_total_error_s),
            )
        )
    return tuple(estimates)


def _optional_float(value) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)
