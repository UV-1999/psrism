import json
from types import SimpleNamespace

import numpy as np
import pytest

from psrism.dm_variability import (
    analyze_dm_solar_variability,
    dm_measurement_arrays,
    write_dm_solar_variability_report,
)


def _measurement(
    index,
    dm,
    dm_error=0.01,
    sun_separation=None,
    corrected_dm=None,
    correction_error=None,
):
    return SimpleNamespace(
        mjd=59000.0 + 365.25 * index,
        dm=dm,
        dm_error=dm_error,
        scattering_corrected_dm=corrected_dm,
        scattering_delta_dm_error=correction_error,
        sun_separation_mid_deg=sun_separation,
        solar_ephemeris_method="astropy_builtin",
    )


def test_piecewise_dm_slopes_find_same_direction_sections():
    measurements = [
        _measurement(index, dm, sun_separation=20.0 + index)
        for index, dm in enumerate([10.0, 11.0, 12.0, 11.0, 10.0])
    ]

    report = analyze_dm_solar_variability(measurements)

    assert report.global_fit.slope_pc_cm3_per_year == pytest.approx(0.0)
    assert report.accepted_segments == 2
    assert [item.direction for item in report.segments] == [
        "increasing",
        "decreasing",
    ]
    assert report.segments[0].fit.slope_pc_cm3_per_year == pytest.approx(1.0)
    assert report.segments[1].fit.slope_pc_cm3_per_year == pytest.approx(-1.0)
    assert report.mean_absolute_slope_pc_cm3_per_year == pytest.approx(1.0)


def test_dm_sun_correlation_uses_pairwise_complete_epochs():
    measurements = [
        _measurement(0, 10.0, sun_separation=50.0),
        _measurement(1, 11.0, sun_separation=40.0),
        _measurement(2, 12.0, sun_separation=None),
        _measurement(3, 13.0, sun_separation=20.0),
    ]

    report = analyze_dm_solar_variability(
        measurements,
        solar_warning_angle_deg=30.0,
    )

    assert report.n_solar_pairs == 3
    assert report.solar_pearson_r == pytest.approx(-1.0)
    assert report.solar_pearson_p_value < 1e-10
    assert report.minimum_sun_separation_deg == pytest.approx(20.0)
    assert report.epochs_below_warning_angle == 1


def test_scattering_corrected_series_combines_available_errors():
    measurements = [
        _measurement(0, 10.0, 0.3, corrected_dm=10.2, correction_error=0.4),
        _measurement(1, 11.0, None, corrected_dm=11.2, correction_error=0.2),
        _measurement(2, 12.0, 0.1, corrected_dm=12.2, correction_error=None),
    ]

    _, values, errors = dm_measurement_arrays(
        measurements,
        "scattering-corrected",
    )

    assert values.tolist() == pytest.approx([10.2, 11.2, 12.2])
    assert errors.tolist() == pytest.approx([0.5, 0.2, 0.1])


def test_missing_dm_writes_strict_json(tmp_path):
    measurements = [
        _measurement(0, None, dm_error=None, sun_separation=15.0),
        _measurement(1, None, dm_error=None, sun_separation=9.0),
    ]
    report = analyze_dm_solar_variability(measurements)
    output = tmp_path / "dm_solar.json"

    write_dm_solar_variability_report(report, output)
    payload = json.loads(output.read_text())

    assert payload["n_dm_finite"] == 0
    assert payload["solar_pearson_r"] is None
    assert payload["global_fit"] is None
    assert payload["segments"] == []
    assert np.isfinite(payload["minimum_sun_separation_deg"])
