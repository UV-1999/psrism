import json

import numpy as np
import pytest

from psrism.autocorrelation_spectrum import (
    AcfScales,
    summarize_scintillation_measurement,
    write_scintillation_report,
)
from psrism.fit_autocorrelation_spectrum import TiltedGaussianAcfFitResult
from psrism.cli import _refractive_timescale_error_days
from psrism.refractive_scintillation import finite_scintle_statistics


def _fit_result(correlation=0.0):
    covariance = np.zeros((5, 5), dtype=float)
    covariance[1, 1] = 0.25
    covariance[2, 2] = 0.01
    covariance[3, 3] = 0.0004
    return TiltedGaussianAcfFitResult(
        amplitude=1.0,
        time_sigma=20.0,
        freq_sigma=2.0,
        correlation=correlation,
        offset=0.0,
        covariance=covariance,
        n_fit_points=100,
        rms_residual=0.02,
        reduced_chi_square=0.001,
    )


def test_acf_fit_propagates_width_and_drift_covariance():
    result = _fit_result(correlation=0.0)

    assert result.delta_t_diss == pytest.approx(20.0 * np.sqrt(2.0))
    assert result.delta_t_diss_error == pytest.approx(0.5 * np.sqrt(2.0))
    assert result.delta_f_diss == pytest.approx(2.0 * np.sqrt(2.0 * np.log(2.0)))
    assert result.delta_f_diss_error == pytest.approx(
        0.1 * np.sqrt(2.0 * np.log(2.0))
    )
    assert result.drift_slope_s_per_mhz == pytest.approx(0.0)
    assert result.drift_slope_error_s_per_mhz == pytest.approx(0.2)
    assert result.delta_f_delta_t_covariance_mhz_s == pytest.approx(0.0)


def test_scintillation_summary_accepts_resolved_2d_fit():
    scales = AcfScales(2.1, 30.0, 2.1, 30.0)
    time_lag = np.arange(-10, 11) * 10.0
    freq_lag = np.arange(-10, 11) * 0.5

    result = summarize_scintillation_measurement(
        scales,
        time_lag,
        freq_lag,
        fit_result=_fit_result(),
        minimum_resolution_bins=2.0,
        observing_duration_s=1000.0,
        observing_bandwidth_mhz=20.0,
    )

    assert result.measurement_method == "tilted_gaussian_2d"
    assert result.decorrelation_bandwidth_resolved is True
    assert result.diffractive_timescale_resolved is True
    assert result.decorrelation_bandwidth_mhz == pytest.approx(
        2.0 * np.sqrt(2.0 * np.log(2.0))
    )
    assert result.diffractive_timescale_s == pytest.approx(20.0 * np.sqrt(2.0))
    assert result.width_covariance_mhz_s == pytest.approx(0.0)
    expected_scintles = (1.0 + 0.2 * 1000.0 / result.diffractive_timescale_s) * (
        1.0 + 0.2 * 20.0 / result.decorrelation_bandwidth_mhz
    )
    assert result.n_scintles == pytest.approx(expected_scintles)
    assert result.finite_scintle_fraction == pytest.approx(1.0 / np.sqrt(expected_scintles))


def test_scintillation_summary_retains_fit_but_rejects_missing_crossing():
    scales = AcfScales(None, 30.0, None, 30.0)
    time_lag = np.arange(-10, 11) * 10.0
    freq_lag = np.arange(-10, 11) * 0.5

    result = summarize_scintillation_measurement(
        scales,
        time_lag,
        freq_lag,
        fit_result=_fit_result(),
    )

    assert result.decorrelation_bandwidth_resolved is False
    assert result.decorrelation_bandwidth_mhz is None
    assert result.fit_decorrelation_bandwidth_mhz is not None
    assert result.diffractive_timescale_resolved is True


def test_scintillation_report_serializes_quality_metadata(tmp_path):
    scales = AcfScales(2.1, 30.0, 2.1, 30.0)
    result = summarize_scintillation_measurement(
        scales,
        np.arange(-10, 11) * 10.0,
        np.arange(-10, 11) * 0.5,
        fit_result=_fit_result(),
    )
    output = tmp_path / "scintillation.json"

    write_scintillation_report(result, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["measurement_method"] == "tilted_gaussian_2d"
    assert payload["decorrelation_bandwidth_resolved"] is True
    assert payload["minimum_resolution_bins"] == pytest.approx(2.0)


def test_refractive_timescale_error_uses_width_covariance():
    result = _refractive_timescale_error_days(
        refractive_timescale_days=10.0,
        decorrelation_bandwidth_mhz=2.0,
        decorrelation_bandwidth_error_mhz=0.1,
        diffractive_timescale_s=5.0,
        diffractive_timescale_error_s=0.2,
        width_covariance_mhz_s=0.005,
    )

    expected_variance = (-5.0) ** 2 * 0.1**2 + 2.0**2 * 0.2**2
    expected_variance += 2.0 * (-5.0) * 2.0 * 0.005
    assert result == pytest.approx(np.sqrt(expected_variance))


def test_finite_scintle_statistics_uses_time_and_frequency_coverage():
    result = finite_scintle_statistics(
        observing_duration_s=1000.0,
        observing_bandwidth_mhz=20.0,
        diffractive_timescale_s=100.0,
        decorrelation_bandwidth_mhz=2.0,
    )

    assert result.n_scintles == pytest.approx(9.0)
    assert result.fractional_error == pytest.approx(1.0 / 3.0)
    assert result.eta_time == pytest.approx(0.2)
    assert result.eta_frequency == pytest.approx(0.2)
