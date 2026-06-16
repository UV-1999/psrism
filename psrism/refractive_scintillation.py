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

    estimates: list[ScintillationEstimate] = []
    for nu_mhz, tau_value, tau_error in zip(freq, tau, tau_err):
        if not np.isfinite(nu_mhz) or not np.isfinite(tau_value) or tau_value <= 0:
            continue
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
            diffractive_timescale_s = (
                2.53e4
                * np.sqrt(distance_kpc * decorrelation_bandwidth_mhz)
                / (nu_ghz * velocity_kms)
            )
            refractive_timescale_s = (
                (4.0 / np.pi)
                * (nu_mhz / decorrelation_bandwidth_mhz)
                * diffractive_timescale_s
            )
            refractive_timescale_days = refractive_timescale_s / 86400.0
            n_scintles = (
                1.0 + eta_time * observing_duration_s / diffractive_timescale_s
            ) * (
                1.0 + eta_freq * observing_bandwidth_mhz / decorrelation_bandwidth_mhz
            )
            n_scintles = max(float(n_scintles), 1.0)
            finite_scintle_error_s = tau_value / np.sqrt(n_scintles)
            if np.isfinite(tau_error) and tau_error > 0:
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
