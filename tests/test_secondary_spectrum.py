import json

import numpy as np
import pytest

from psrism.fit_secondary_spectrum import fit_parabolic_arc, write_arc_fit_report
from psrism.scintillation_spectrum import (
    calculate_scintillation_spectrum,
    secondary_spectrum_metadata,
)


def test_secondary_spectrum_preserves_scientific_central_axes():
    dynspec = np.zeros((8, 10), dtype=float)
    dynspec[2, 3] = 1.0

    spectrum, fringe, delay = calculate_scintillation_spectrum(
        dynspec,
        observation_time_s=80.0,
        bandwidth_mhz=20.0,
        log_scale=False,
    )

    assert spectrum.shape == (len(fringe), len(delay))
    assert np.any(spectrum[len(fringe) // 2, :] > 0)


def test_secondary_spectrum_metadata_reports_fourier_sampling():
    dynspec = np.ones((8, 10), dtype=float)
    metadata = secondary_spectrum_metadata(
        dynspec,
        observation_time_s=80.0,
        bandwidth_mhz=20.0,
        window="hann",
        centre_frequency_mhz=150.0,
    )

    assert metadata.time_resolution_s == pytest.approx(10.0)
    assert metadata.frequency_resolution_mhz == pytest.approx(2.0)
    assert metadata.fringe_frequency_resolution_hz == pytest.approx(1.0 / 80.0)
    assert metadata.delay_resolution_s == pytest.approx(1.0 / 20e6)
    assert metadata.fringe_frequency_nyquist_hz == pytest.approx(0.05)
    assert metadata.delay_nyquist_s == pytest.approx(1.0 / 4e6)
    assert metadata.window == "hann"
    assert metadata.centre_frequency_mhz == 150.0
    assert metadata.masked_fraction == 0.0


def test_hann_secondary_spectrum_is_finite_and_shape_preserving():
    rng = np.random.default_rng(4)
    dynspec = rng.normal(size=(12, 14))

    spectrum, fringe, delay = calculate_scintillation_spectrum(
        dynspec,
        observation_time_s=120.0,
        bandwidth_mhz=28.0,
        log_scale=False,
        window="hann",
    )

    assert spectrum.shape == dynspec.shape
    assert spectrum.shape == (len(fringe), len(delay))
    assert np.isfinite(spectrum).all()


def test_arc_fit_quality_and_json_report_are_auditable(tmp_path):
    rng = np.random.default_rng(8)
    fringe = np.linspace(-10.0, 10.0, 81)
    delay = np.linspace(-5.0, 80.0, 341)
    spectrum = rng.uniform(0.0, 0.1, size=(len(fringe), len(delay)))
    expected_curvature = 0.6
    for row, value in enumerate(fringe):
        target = expected_curvature * value**2
        column = int(np.argmin(np.abs(delay - target)))
        spectrum[row, column] += 20.0

    result = fit_parabolic_arc(
        spectrum,
        fringe,
        delay,
        curvature_min=0.1,
        curvature_max=2.0,
        n_trials=301,
        half="positive",
        mask_bins=1,
        minimum_samples=20,
        minimum_score_snr=3.0,
    )

    assert result.curvature == pytest.approx(expected_curvature, rel=0.04)
    assert result.n_samples >= result.minimum_samples
    assert result.score_baseline is not None
    assert result.score_noise is not None
    assert result.peak_at_search_boundary is False

    metadata = secondary_spectrum_metadata(
        np.ones((81, 341)),
        observation_time_s=810.0,
        bandwidth_mhz=34.1,
    )
    output = tmp_path / "arc_fit.json"
    write_arc_fit_report(result, output, secondary_metadata=metadata)
    payload = json.loads(output.read_text())

    assert payload["curvature"] == pytest.approx(result.curvature)
    assert payload["accepted"] == result.accepted
    assert len(payload["trial_curvatures"]) == 301
    assert payload["secondary_spectrum"]["delay_resolution_s"] > 0


def test_arc_fit_rejects_insufficient_sample_support():
    fringe = np.arange(-4.0, 5.0)
    delay = np.arange(-20.0, 21.0)
    spectrum = np.zeros((len(fringe), len(delay)), dtype=float)
    for row, value in enumerate(fringe):
        spectrum[row, np.where(delay == value**2)[0][0]] = 10.0

    result = fit_parabolic_arc(
        spectrum,
        fringe,
        delay,
        curvature_min=0.5,
        curvature_max=2.0,
        n_trials=101,
        mask_bins=0,
        minimum_samples=20,
    )

    assert result.accepted is False
    assert result.accepted_curvature is None
    assert "insufficient_arc_samples" in result.rejection_reasons
