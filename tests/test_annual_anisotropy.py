import numpy as np
import pytest

from psrism.annual_anisotropy import (
    anisotropic_effective_speed,
    anisotropy_coefficients,
    annual_q_model,
    effective_velocity_components,
    proper_motion_velocity_kms,
    thin_screen_a_iss,
)


def test_isotropic_limit_reduces_to_euclidean_speed():
    coefficients = anisotropy_coefficients(axial_ratio=1.0, psi_deg=37.0)
    speed = anisotropic_effective_speed([3.0], [4.0], 1.0, 37.0)

    assert coefficients.r == pytest.approx(0.0)
    assert coefficients.a == pytest.approx(1.0)
    assert coefficients.b == pytest.approx(1.0)
    assert coefficients.c == pytest.approx(0.0)
    assert speed == pytest.approx([5.0])


def test_effective_velocity_uses_thin_screen_distance_weights():
    alpha, delta = effective_velocity_components(
        distance_kpc=1.0,
        screen_distance_kpc=0.25,
        earth_velocity_alpha_kms=[20.0],
        earth_velocity_delta_kms=[-10.0],
        pulsar_velocity_alpha_kms=100.0,
        pulsar_velocity_delta_kms=40.0,
        screen_velocity_alpha_kms=5.0,
        screen_velocity_delta_kms=-2.0,
    )

    assert alpha == pytest.approx([0.75 * 20.0 + 0.25 * 100.0 - 5.0])
    assert delta == pytest.approx([0.75 * -10.0 + 0.25 * 40.0 + 2.0])


def test_annual_q_model_matches_equation_four():
    distance = 1.0
    screen_distance = 0.25
    axial_ratio = 1.0
    q = annual_q_model(
        earth_velocity_alpha_kms=[20.0],
        earth_velocity_delta_kms=[0.0],
        distance_kpc=distance,
        pulsar_velocity_alpha_kms=0.0,
        pulsar_velocity_delta_kms=0.0,
        screen_velocity_alpha_kms=0.0,
        screen_velocity_delta_kms=0.0,
        axial_ratio=axial_ratio,
        psi_deg=0.0,
        screen_distance_kpc=screen_distance,
    )
    a_iss = thin_screen_a_iss(distance, screen_distance, axial_ratio)
    expected_speed = 0.75 * 20.0
    expected = expected_speed * np.sqrt(distance) / (
        a_iss * (distance - screen_distance)
    )

    assert q == pytest.approx([expected])


def test_proper_motion_conversion_uses_ra_cos_dec_convention():
    alpha, delta = proper_motion_velocity_kms(10.0, -5.0, 0.5)

    assert alpha == pytest.approx(4.74047 * 10.0 * 0.5)
    assert delta == pytest.approx(4.74047 * -5.0 * 0.5)


def test_thin_screen_a_iss_rejects_screen_outside_line_of_sight():
    with pytest.raises(ValueError, match="between Earth and pulsar"):
        thin_screen_a_iss(1.0, 1.0, 2.0)
