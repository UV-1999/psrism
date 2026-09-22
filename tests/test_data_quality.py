import json

import numpy as np
import pytest

from psrism.autocorrelation_spectrum import calculate_autocorrelation_spectrum
from psrism.data_quality import (
    apply_quality_mask_to_archive,
    assess_dynamic_spectrum,
    write_quality_report,
)
from psrism.dynamic_spectrum import normalize_dynamic_spectrum
from psrism.scintillation_spectrum import calculate_scintillation_spectrum


def test_quality_without_statistical_masking_only_rejects_invalid_samples():
    dynspec = np.ones((4, 5), dtype=float)
    dynspec[1, 2] = np.nan

    result = assess_dynamic_spectrum(dynspec, detect_rfi=False)

    assert result.input_invalid_cells == 1
    assert result.masked_cells == 1
    assert result.bad_time_bins == ()
    assert result.bad_frequency_channels == ()
    assert not result.valid_mask[1, 2]


def test_robust_mask_detects_channel_time_and_isolated_outliers():
    rng = np.random.default_rng(27)
    dynspec = rng.normal(size=(64, 48))
    dynspec[:, 7] += 30.0
    dynspec[11, :] += 30.0
    dynspec[25, 30] += 50.0

    result = assess_dynamic_spectrum(dynspec, sigma_threshold=6.0, detect_rfi=True)

    assert result.bad_frequency_channels == (7,)
    assert result.bad_time_bins == (11,)
    assert result.isolated_flagged_cells == 1
    assert not result.valid_mask[25, 30]


def test_quality_mask_is_applied_to_archive_weights():
    archive = _FakeArchive([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    valid = np.asarray([[True, False, False], [True, True, False]])

    newly_masked = apply_quality_mask_to_archive(archive, valid)

    assert newly_masked == 2
    assert archive.weights == [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]


def test_masked_samples_do_not_create_nan_spectra(tmp_path):
    rng = np.random.default_rng(4)
    dynspec = rng.normal(size=(12, 10))
    valid = np.ones_like(dynspec, dtype=bool)
    valid[2:5, 4] = False
    dynspec[~valid] = np.nan

    acf = calculate_autocorrelation_spectrum(dynspec, valid_mask=valid)
    secondary, fringe, delay = calculate_scintillation_spectrum(
        dynspec,
        observation_time_s=120.0,
        bandwidth_mhz=20.0,
        log_scale=False,
        valid_mask=valid,
    )
    report = assess_dynamic_spectrum(dynspec)
    output = tmp_path / "quality.json"
    write_quality_report(report, output)

    assert np.isfinite(acf).all()
    assert np.isfinite(secondary).all()
    assert secondary.shape == (len(fringe), len(delay))
    assert json.loads(output.read_text())["masked_cells"] == 3


def test_normalization_preserves_a_fully_masked_channel_without_warnings():
    dynspec = np.asarray([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])

    normalized = normalize_dynamic_spectrum(dynspec)

    assert np.isfinite(normalized[:, 0]).all()
    assert np.isnan(normalized[:, 1]).all()
    assert np.mean(normalized[:, 0]) == pytest.approx(0.0)


class _FakeIntegration:
    def __init__(self, weights):
        self.weights = weights

    def get_weight(self, channel):
        return self.weights[channel]

    def set_weight(self, channel, value):
        self.weights[channel] = value


class _FakeArchive:
    def __init__(self, weights):
        self.weights = weights
        self.integrations = [_FakeIntegration(row) for row in weights]

    def get_nsubint(self):
        return len(self.weights)

    def get_nchan(self):
        return len(self.weights[0])

    def get_Integration(self, subintegration):
        return self.integrations[subintegration]
