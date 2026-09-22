import numpy as np
import pytest

from psrism.autocorrelation_spectrum import measure_acf_scales
from psrism.conventions import (
    ACF_DECORRELATION_LEVEL,
    ACF_TIMESCALE_LEVEL,
    TAU_REFERENCE_FREQUENCY_MHZ,
)
from psrism.fit_alpha import fit_alpha, tau_power_law
from psrism.fit_autocorrelation_spectrum import TiltedGaussianAcfFitResult
from psrism.fit_secondary_spectrum import fit_parabolic_arc


def test_tau_power_law_uses_positive_scattering_index_and_tau150():
    tau = tau_power_law(np.asarray([150.0, 300.0]), tau0=0.016, alpha=4.0)

    assert TAU_REFERENCE_FREQUENCY_MHZ == 150.0
    assert tau == pytest.approx([0.016, 0.001])


def test_fit_alpha_recovers_positive_index_and_reference_tau():
    frequency = np.asarray([100.0, 125.0, 150.0, 200.0, 250.0, 300.0])
    expected_tau150 = 0.02
    tau = tau_power_law(frequency, expected_tau150, alpha=3.8)

    result = fit_alpha(frequency, tau, tau_error=0.02 * tau)

    assert result.alpha == pytest.approx(3.8, rel=1e-10)
    assert result.tau0 == pytest.approx(expected_tau150, rel=1e-10)
    assert result.reference_freq_mhz == 150.0


def test_acf_scales_use_half_power_and_one_over_e_levels():
    axis = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    acf = np.zeros((5, 5), dtype=float)
    acf[2, :] = [0.1, 0.4, 1.0, 0.6, 0.4]
    acf[:, 2] = [0.1, 0.3, 1.0, 0.5, 0.2]

    result = measure_acf_scales(acf, time_lag_s=axis, freq_lag_mhz=axis)
    expected_time = 1.0 + (ACF_TIMESCALE_LEVEL - 0.5) / (0.2 - 0.5)

    assert ACF_DECORRELATION_LEVEL == 0.5
    assert result.decorrelation_bandwidth_mhz == pytest.approx(1.5)
    assert result.diffractive_timescale_s == pytest.approx(expected_time)


def test_positive_acf_correlation_has_positive_reported_drift_slope():
    result = TiltedGaussianAcfFitResult(
        amplitude=1.0,
        time_sigma=2.0,
        freq_sigma=1.0,
        correlation=0.4,
        offset=0.0,
        covariance=np.eye(5),
        n_fit_points=100,
        rms_residual=0.01,
        reduced_chi_square=1.0,
    )

    assert result.drift_slope_s_per_mhz == pytest.approx(0.8)
    assert result.drift_rate_mhz_per_s == pytest.approx(1.25)


def test_negative_delay_arc_returns_negative_curvature():
    fringe = np.arange(-4.0, 5.0)
    delay = np.arange(-20.0, 21.0)
    spectrum = np.zeros((len(fringe), len(delay)), dtype=float)
    for row, value in enumerate(fringe):
        spectrum[row, np.where(delay == -(value**2))[0][0]] = 10.0

    result = fit_parabolic_arc(
        spectrum,
        fringe,
        delay,
        curvature_min=0.5,
        curvature_max=2.0,
        n_trials=101,
        half="negative",
        mask_bins=0,
    )

    assert result.curvature == pytest.approx(-1.0, rel=0.03)
    assert result.half == "negative"
