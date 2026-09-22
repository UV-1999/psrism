import csv
from datetime import datetime, timezone

import numpy as np
import pytest

from psrism.fit_alpha import fit_alpha, tau_power_law
from psrism.fit_tau import (
    _tau_subband_rejection_reasons,
    effective_subband_frequency_mhz,
)
from psrism.time_series_analysis import EpochMeasurement, write_time_series_csv


def test_effective_subband_frequency_uses_finite_bandwidth_mapping():
    result = effective_subband_frequency_mhz(100.0, 40.0)

    assert result == pytest.approx(0.5 * np.sqrt(40.0**2 + 4.0 * 100.0**2))
    assert result > 100.0


def test_alpha_monte_carlo_is_reproducible_and_reports_asymmetric_errors():
    frequency = np.asarray([100.0, 125.0, 150.0, 200.0, 250.0, 300.0])
    tau = tau_power_law(frequency, tau0=0.02, alpha=3.8)
    asymmetric_error = np.vstack([0.03 * tau, 0.08 * tau])

    first = fit_alpha(
        frequency,
        tau,
        tau_error=asymmetric_error,
        monte_carlo_samples=2000,
        random_seed=17,
    )
    second = fit_alpha(
        frequency,
        tau,
        tau_error=asymmetric_error,
        monte_carlo_samples=2000,
        random_seed=17,
    )

    assert first.alpha == pytest.approx(3.8, rel=1e-10)
    assert first.alpha_error_lower > 0
    assert first.alpha_error_upper > 0
    assert first.alpha_error == max(
        first.covariance_alpha_error,
        first.alpha_error_lower,
        first.alpha_error_upper,
    )
    assert first.monte_carlo_samples == 2000
    assert first.monte_carlo_alpha == pytest.approx(second.monte_carlo_alpha)


def test_alpha_monte_carlo_can_be_disabled():
    frequency = np.asarray([100.0, 150.0, 200.0])
    tau = tau_power_law(frequency, tau0=0.02, alpha=4.0)

    result = fit_alpha(frequency, tau, 0.05 * tau, monte_carlo_samples=0)

    assert result.monte_carlo_samples == 0
    assert result.uncertainty_method == "weighted_least_squares"
    assert result.alpha_error_lower == result.covariance_alpha_error
    assert result.alpha_error_upper == result.covariance_alpha_error


def test_subband_quality_reports_each_failed_threshold():
    reasons = _tau_subband_rejection_reasons(
        tau=0.01,
        tau_error=0.02,
        profile_snr=2.0,
        valid_channel_fraction=0.25,
        min_profile_snr=5.0,
        max_relative_tau_error=1.0,
        min_valid_channel_fraction=0.5,
    )

    assert any("relative tau uncertainty" in reason for reason in reasons)
    assert any("profile fit S/N" in reason for reason in reasons)
    assert any("valid-channel fraction" in reason for reason in reasons)


def test_time_series_csv_retains_frequency_scaling_provenance(tmp_path):
    measurement = EpochMeasurement(
        archive_path="example.ar",
        archive_name="example.ar",
        utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        mjd=61041.0,
        alpha=3.9,
        alpha_error=0.2,
        alpha_error_lower=0.15,
        alpha_error_upper=0.2,
        alpha_covariance_error=0.12,
        alpha_monte_carlo_samples=10000,
        alpha_subbands_accepted=4,
        alpha_subbands_rejected=1,
        alpha_subbands_failed=1,
        tau0_s=0.02,
        tau0_error_s=0.001,
        tau0_error_lower_s=0.0008,
        tau0_error_upper_s=0.001,
        tau_reference_frequency_mhz=150.0,
        observing_frequency_mhz=149.5,
    )
    output = tmp_path / "series.csv"

    write_time_series_csv([measurement], output)

    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["alpha_error_lower"] == "0.15"
    assert row["alpha_monte_carlo_samples"] == "10000"
    assert row["alpha_subbands_accepted"] == "4"
    assert row["tau0_error_upper_s"] == "0.001"
    assert row["observing_frequency_mhz"] == "149.5"
