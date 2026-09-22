import json
from types import SimpleNamespace

import numpy as np
import pytest

from psrism.intrinsic_width import (
    GAUSSIAN_FWHM_FACTOR,
    analyze_intrinsic_widths,
    write_intrinsic_width_report,
)
from psrism.profile_models import GaussianComponent


def _subband(frequency_mhz, sigma_bins, sigma_error_bins, *, model="gaussian"):
    components = (
        ()
        if model == "template"
        else (GaussianComponent(1.0, 40.0, float(sigma_bins)),)
    )
    errors = () if model == "template" else (float(sigma_error_bins),)
    fit = SimpleNamespace(
        intrinsic_model=model,
        components=components,
        component_sigma_errors=errors,
        n_bins=256,
        period_s=1.0,
    )
    return SimpleNamespace(
        frequency_mhz=float(frequency_mhz),
        central_frequency_mhz=float(frequency_mhz),
        bandwidth_mhz=20.0,
        accepted=True,
        rejection_reasons=(),
        fit=fit,
        profile=np.zeros(256),
    )


def test_intrinsic_width_converts_sigma_to_fwhm_phase_and_seconds():
    result = SimpleNamespace(
        subbands=(_subband(150.0, 4.0, 0.5),),
        alpha_fit=SimpleNamespace(reference_freq_mhz=150.0),
    )

    report = analyze_intrinsic_widths(result, "isotropic", min_trend_subbands=2)
    measurement = report.measurements[0]

    assert measurement.fwhm_bins == pytest.approx(GAUSSIAN_FWHM_FACTOR * 4.0)
    assert measurement.fwhm_phase == pytest.approx(measurement.fwhm_bins / 256.0)
    assert measurement.fwhm_percent_period == pytest.approx(100.0 * measurement.fwhm_phase)
    assert measurement.fwhm_seconds == pytest.approx(measurement.fwhm_phase)
    assert measurement.fwhm_error_bins == pytest.approx(GAUSSIAN_FWHM_FACTOR * 0.5)


def test_intrinsic_width_recovers_power_law_frequency_evolution():
    frequencies = np.asarray([100.0, 125.0, 175.0, 250.0])
    exponent = -0.6
    sigma = 6.0 * (frequencies / 150.0) ** exponent
    result = SimpleNamespace(
        subbands=tuple(
            _subband(frequency, width, 0.02 * width)
            for frequency, width in zip(frequencies, sigma)
        ),
        alpha_fit=SimpleNamespace(reference_freq_mhz=150.0),
    )

    report = analyze_intrinsic_widths(result, "isotropic")

    assert len(report.trends) == 1
    assert report.trends[0].frequency_exponent == pytest.approx(exponent, abs=1e-10)
    assert report.trends[0].significant


def test_template_report_marks_width_unavailable_and_writes_strict_json(tmp_path):
    result = SimpleNamespace(
        subbands=(_subband(150.0, 1.0, 0.1, model="template"),),
        alpha_fit=SimpleNamespace(reference_freq_mhz=150.0),
    )
    report = analyze_intrinsic_widths(result, "anisotropic")
    output = tmp_path / "width.json"

    write_intrinsic_width_report(report, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert not payload["width_available"]
    assert payload["measurements"] == []
    assert "template" in payload["warnings"][0].lower()
