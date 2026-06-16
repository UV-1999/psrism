"""Annual anisotropy helper equations for multi-epoch scintillation work."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnnualAnisotropyCoefficients:
    axial_ratio: float
    psi_deg: float
    r: float
    a: float
    b: float
    c: float


def anisotropy_coefficients(axial_ratio: float, psi_deg: float) -> AnnualAnisotropyCoefficients:
    """Return coefficients for the anisotropic effective-velocity quadratic."""
    axial_ratio = max(float(axial_ratio), 1.0)
    psi_rad = np.deg2rad(float(psi_deg))
    r = (axial_ratio**2 - 1.0) / (axial_ratio**2 + 1.0)
    denom = np.sqrt(max(1.0 - r**2, 1e-15))
    a = (1.0 - r * np.cos(2.0 * psi_rad)) / denom
    b = (1.0 + r * np.cos(2.0 * psi_rad)) / denom
    c = -2.0 * r * np.sin(2.0 * psi_rad) / denom
    return AnnualAnisotropyCoefficients(
        axial_ratio=axial_ratio,
        psi_deg=float(psi_deg),
        r=float(r),
        a=float(a),
        b=float(b),
        c=float(c),
    )


def effective_velocity_components(
    distance_kpc: float,
    screen_distance_kpc: float,
    earth_velocity_alpha_kms,
    earth_velocity_delta_kms,
    pulsar_velocity_alpha_kms,
    pulsar_velocity_delta_kms,
    screen_velocity_alpha_kms=0.0,
    screen_velocity_delta_kms=0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute thin-screen effective velocity components."""
    distance = float(distance_kpc)
    screen_distance = float(screen_distance_kpc)
    if distance <= 0:
        raise ValueError("distance_kpc must be positive")
    weight_earth = (distance - screen_distance) / distance
    weight_pulsar = screen_distance / distance
    v_alpha = (
        weight_earth * np.asarray(earth_velocity_alpha_kms, dtype=float)
        + weight_pulsar * np.asarray(pulsar_velocity_alpha_kms, dtype=float)
        - np.asarray(screen_velocity_alpha_kms, dtype=float)
    )
    v_delta = (
        weight_earth * np.asarray(earth_velocity_delta_kms, dtype=float)
        + weight_pulsar * np.asarray(pulsar_velocity_delta_kms, dtype=float)
        - np.asarray(screen_velocity_delta_kms, dtype=float)
    )
    return v_alpha, v_delta


def anisotropic_effective_speed(
    v_eff_alpha_kms,
    v_eff_delta_kms,
    axial_ratio: float,
    psi_deg: float,
) -> np.ndarray:
    """Evaluate the anisotropic effective speed magnitude."""
    coeff = anisotropy_coefficients(axial_ratio, psi_deg)
    v_alpha = np.asarray(v_eff_alpha_kms, dtype=float)
    v_delta = np.asarray(v_eff_delta_kms, dtype=float)
    speed2 = coeff.a * v_alpha**2 + coeff.b * v_delta**2 + coeff.c * v_alpha * v_delta
    return np.sqrt(np.clip(speed2, 0.0, np.inf))


def scintillation_velocity(
    a_iss: float,
    distance_kpc,
    decorrelation_bandwidth_mhz,
    frequency_ghz,
    diffractive_timescale_s,
) -> np.ndarray:
    """Compute V_ISS = A_ISS sqrt(D Delta_nu_d) / (f Delta_t_d)."""
    return (
        float(a_iss)
        * np.sqrt(np.asarray(distance_kpc, dtype=float) * np.asarray(decorrelation_bandwidth_mhz, dtype=float))
        / (np.asarray(frequency_ghz, dtype=float) * np.asarray(diffractive_timescale_s, dtype=float))
    )
