import numpy as np
import pytest

from psrism.anisotropic_scattering import (
    anisotropic_scattered_profile,
    anisotropic_scattered_pulse,
    fit_anisotropic_profile,
)
from psrism.fit_tau import fit_tau_from_profile, isotropic_scattered_pulse
from psrism.profile_models import (
    GaussianComponent,
    folded_isotropic_pbf,
    gaussian_intrinsic_profile,
    isotropic_scattered_profile,
    load_intrinsic_template,
    shift_template_periodic,
)


def test_folded_isotropic_pbf_is_normalized_and_retains_wraparound_power():
    kernel = folded_isotropic_pbf(64, tau_bins=35.0)

    assert np.sum(kernel) == pytest.approx(1.0)
    assert np.all(kernel >= 0.0)
    assert kernel[-1] > 0.0


@pytest.mark.parametrize("tau_bins", [2.0, 70.0])
def test_isotropic_train_fit_recovers_weak_and_severe_scattering(tau_bins):
    rng = np.random.default_rng(int(tau_bins))
    bins = np.arange(128)
    profile = isotropic_scattered_pulse(
        bins,
        amplitude=1.3,
        mu=36.0,
        sigma=4.0,
        tau=tau_bins,
        baseline=0.12,
    )
    profile += rng.normal(0.0, 0.002, size=len(profile))

    result = fit_tau_from_profile(profile, period_s=1.28)

    assert result.tau_bins == pytest.approx(tau_bins, rel=0.06)
    assert result.tau_seconds == pytest.approx(tau_bins * 0.01, rel=0.06)
    assert result.intrinsic_model == "gaussian"
    assert np.isfinite(result.baseline)


def test_multi_gaussian_train_fit_recovers_tau():
    components = [
        GaussianComponent(1.0, 30.0, 3.0),
        GaussianComponent(0.55, 68.0, 5.0),
    ]
    intrinsic = gaussian_intrinsic_profile(128, components)
    profile = isotropic_scattered_profile(intrinsic, tau_bins=18.0, baseline=0.1)

    result = fit_tau_from_profile(profile, n_components=2)

    assert result.tau_bins == pytest.approx(18.0, rel=1e-5)
    assert result.intrinsic_model == "multi_gaussian"
    assert len(result.components) == 2
    assert [item.mu for item in result.components] == pytest.approx([30.0, 68.0])


def test_tau_fit_is_invariant_to_large_archive_dc_offset():
    bins = np.arange(96)
    profile = isotropic_scattered_pulse(bins, 1.1, 27.0, 3.5, 14.0, 0.2)

    ordinary = fit_tau_from_profile(profile)
    offset = fit_tau_from_profile(profile + 5000.0)

    assert ordinary.tau_bins == pytest.approx(14.0, rel=1e-6)
    assert offset.tau_bins == pytest.approx(ordinary.tau_bins, rel=1e-6)


def test_template_train_fit_recovers_tau_and_phase_shift():
    components = [
        GaussianComponent(1.0, 25.0, 3.0),
        GaussianComponent(0.4, 62.0, 4.0),
    ]
    template = gaussian_intrinsic_profile(128, components)
    shifted = shift_template_periodic(template, 7.2)
    profile = isotropic_scattered_profile(
        shifted,
        tau_bins=42.0,
        amplitude=1.2,
        baseline=0.08,
    )

    result = fit_tau_from_profile(profile, intrinsic_template=template)

    assert result.tau_bins == pytest.approx(42.0, rel=1e-5)
    assert result.phase_shift_bins == pytest.approx(7.2, abs=1e-4)
    assert result.intrinsic_model == "template"


def test_template_file_is_loaded_and_periodically_resampled(tmp_path):
    template = gaussian_intrinsic_profile(64, [GaussianComponent(1.0, 18.0, 3.0)])
    path = tmp_path / "intrinsic.npy"
    np.save(path, template)

    loaded = load_intrinsic_template(path, target_nbin=128)

    assert loaded.shape == (128,)
    assert np.max(loaded) == pytest.approx(1.0)


def test_anisotropic_template_fit_recovers_injected_parameters():
    template = gaussian_intrinsic_profile(
        96,
        [
            GaussianComponent(1.0, 20.0, 2.5),
            GaussianComponent(0.35, 48.0, 3.0),
        ],
    )
    shifted = shift_template_periodic(template, 4.0)
    profile = anisotropic_scattered_profile(
        shifted,
        tau_eff=13.0,
        anisotropy_ratio=2.5,
        amplitude=1.1,
        baseline=0.08,
    )

    result = fit_anisotropic_profile(profile, intrinsic_template=template)

    assert result.tau_eff_bins == pytest.approx(13.0, rel=1e-4)
    assert result.anisotropy_ratio == pytest.approx(2.5, rel=1e-3)
    assert result.phase_shift_bins == pytest.approx(4.0, abs=1e-3)
    assert result.intrinsic_model == "template"


def test_anisotropic_train_fit_recovers_injected_parameters():
    rng = np.random.default_rng(44)
    bins = np.arange(96)
    profile = anisotropic_scattered_pulse(
        bins,
        amplitude=1.2,
        mu=24.0,
        sigma=3.0,
        tau_eff=18.0,
        anisotropy_ratio=3.0,
        baseline=0.1,
    )
    profile += rng.normal(0.0, 0.001, size=len(profile))

    result = fit_anisotropic_profile(profile)

    assert result.tau_eff_bins == pytest.approx(18.0, rel=0.03)
    assert result.anisotropy_ratio == pytest.approx(3.0, rel=0.08)
    assert result.intrinsic_model == "gaussian"
