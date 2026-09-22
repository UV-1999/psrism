from datetime import datetime, timezone
import json
from types import SimpleNamespace

import numpy as np
import pytest

from psrism.temporal_statistics import (
    analyze_temporal_statistics,
    write_temporal_statistics_report,
)
from psrism.time_series_analysis import parse_time_params


def _measurement(index, dm, dm_error, tau, tau_error, alpha=None):
    return SimpleNamespace(
        utc=datetime(2020, 1, 1, tzinfo=timezone.utc),
        mjd=59000.0 + index * 30.0,
        dm=dm,
        dm_error=dm_error,
        tau_s=tau,
        tau_error_s=tau_error,
        alpha=alpha,
        alpha_error=None,
        decorrelation_bandwidth_mhz=None,
        decorrelation_bandwidth_error_mhz=None,
        diffractive_timescale_s=None,
        diffractive_timescale_error_s=None,
        refractive_timescale_days=None,
        refractive_timescale_error_days=None,
    )


def test_temporal_statistics_summarize_selected_parameters_and_correlations():
    measurements = [
        _measurement(i, 10.0 + i, 0.2, 8.0 - i, 0.1)
        for i in range(6)
    ]

    report = analyze_temporal_statistics(
        measurements,
        {"dm", "tau"},
        maximum_lag=3,
        p_value_threshold=0.05,
    )

    by_name = {item.parameter: item for item in report.parameters}
    assert tuple(by_name) == ("dm", "tau")
    assert by_name["dm"].n_finite == 6
    assert by_name["dm"].n_with_uncertainty == 6
    assert by_name["dm"].time_span_days == pytest.approx(150.0)
    assert by_name["dm"].weighted_mean == pytest.approx(12.5)
    assert by_name["dm"].maximum_lag == 3
    assert len(by_name["dm"].autocorrelation) == 4
    assert len(by_name["dm"].ljung_box_p_values) == 3

    correlation = report.correlations[0]
    assert correlation.parameter_x == "dm"
    assert correlation.parameter_y == "tau"
    assert correlation.n_paired == 6
    assert correlation.pearson_r == pytest.approx(-1.0)
    assert correlation.p_value < 1e-10


def test_temporal_statistics_use_pairwise_complete_observations():
    measurements = [
        _measurement(0, 10.0, 0.1, 4.0, 0.2),
        _measurement(1, 11.0, 0.1, None, None),
        _measurement(2, 12.0, 0.1, 2.0, 0.2),
        _measurement(3, 13.0, 0.1, 1.0, 0.2),
    ]

    report = analyze_temporal_statistics(measurements, {"dm", "tau"})

    assert report.correlations[0].n_paired == 3
    assert report.correlations[0].pearson_r == pytest.approx(-1.0)


def test_constant_and_missing_series_write_strict_json(tmp_path):
    measurements = [
        _measurement(0, 10.0, None, None, None, alpha=4.0),
        _measurement(1, 10.0, None, None, None, alpha=4.0),
        _measurement(2, 10.0, None, None, None, alpha=4.0),
    ]
    report = analyze_temporal_statistics(measurements, {"dm", "tau", "alpha"})
    output = tmp_path / "variability.json"

    write_temporal_statistics_report(report, output)
    payload = json.loads(output.read_text())

    by_name = {item["parameter"]: item for item in payload["parameters"]}
    assert by_name["dm"]["autocorrelation"][0] == 1.0
    assert by_name["dm"]["autocorrelation"][1] is None
    assert by_name["tau"]["n_finite"] == 0
    assert payload["correlations"][0]["pearson_r"] is None


def test_all_available_lags_stop_before_zero_ljung_box_denominator():
    measurements = [
        _measurement(i, value, 0.1, None, None)
        for i, value in enumerate([1.0, 2.0, 1.5, 2.5, 2.0])
    ]

    report = analyze_temporal_statistics(measurements, {"dm"})
    result = report.parameters[0]

    assert result.maximum_lag == 4
    assert len(result.ljung_box_q) == 4
    assert np.isfinite(result.ljung_box_q[-1])


def test_reference_tau_time_parameter_aliases_are_distinct_from_full_band_tau():
    assert parse_time_params("tau") == {"tau"}
    assert parse_time_params("tau_ref") == {"tau_ref"}
    assert parse_time_params("tau150,tau0") == {"tau_ref"}
    assert "tau_ref" in parse_time_params("all")
