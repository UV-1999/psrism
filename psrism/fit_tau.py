"""Scattering-time tau fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .archive_io import integrated_profile_array
from .fit_alpha import AlphaFitResult, fit_alpha


@dataclass(frozen=True)
class TauFitResult:
    amplitude: float
    mu: float
    sigma: float
    tau_bins: float
    tau_bins_error: float
    tau_seconds: float | None = None
    tau_seconds_error: float | None = None


@dataclass(frozen=True)
class SubbandTauResult:
    frequency_mhz: float
    tau: float
    tau_error: float
    tau_bins: float
    tau_bins_error: float
    fit: TauFitResult
    profile: np.ndarray
    channel_start: int
    channel_stop: int


@dataclass(frozen=True)
class ScatteringSpectralIndexResult:
    n_subbands: int
    subbands: tuple[SubbandTauResult, ...]
    alpha_fit: AlphaFitResult


def scattered_pulse(t, amplitude, mu, sigma, tau):
    """Exponentially modified Gaussian pulse model."""
    from scipy.special import erf

    term1 = amplitude * (sigma / tau) * np.sqrt(np.pi / 2)
    term2 = np.exp(-(t - mu) / tau)
    arg = (t - (mu + sigma**2 / tau)) / (sigma * np.sqrt(2))
    return term1 * term2 * (1 + erf(arg))


def fit_tau_from_profile(profile, period_s: float | None = None) -> TauFitResult:
    """Fit tau to a normalized pulse profile."""
    from scipy.optimize import curve_fit

    prof = np.asarray(profile, dtype=float)
    prof = prof - np.min(prof)
    mx = np.max(prof)
    if mx != 0:
        prof = prof / mx

    nbin = len(prof)
    t = np.arange(nbin)
    p0 = [1.0, float(np.argmax(prof)), nbin / 20, nbin / 50]
    bounds = ([0.0, 0.0, 1e-6, 1e-6], [np.inf, nbin, np.inf, np.inf])
    popt, pcov = curve_fit(scattered_pulse, t, prof, p0=p0, bounds=bounds, maxfev=20000)
    tau_err = float(np.sqrt(np.diag(pcov))[3])

    tau_seconds = None
    tau_seconds_error = None
    if period_s is not None:
        tau_seconds = float(popt[3] * (period_s / nbin))
        tau_seconds_error = float(tau_err * (period_s / nbin))

    return TauFitResult(
        amplitude=float(popt[0]),
        mu=float(popt[1]),
        sigma=float(popt[2]),
        tau_bins=float(popt[3]),
        tau_bins_error=tau_err,
        tau_seconds=tau_seconds,
        tau_seconds_error=tau_seconds_error,
    )


def fit_tau_from_archive(archive) -> TauFitResult:
    """Fit tau from an archive integrated profile."""
    profile = integrated_profile_array(archive)
    period_s = archive.get_Integration(0).get_folding_period()
    return fit_tau_from_profile(profile, period_s=period_s)


def fit_tau_alpha_from_archive(
    archive,
    n_subbands: int,
    reference_freq_mhz: float | None = None,
) -> ScatteringSpectralIndexResult:
    """Fit tau in contiguous frequency subbands and then fit alpha."""
    if n_subbands < 2:
        raise ValueError("n_subbands must be at least 2 to fit alpha")

    temp = archive.clone()
    temp.tscrunch()
    temp.remove_baseline()

    nchan = temp.get_nchan()
    nbin = temp.get_nbin()
    if n_subbands > nchan:
        raise ValueError("n_subbands cannot exceed the number of frequency channels")

    period_s = temp.get_Integration(0).get_folding_period()
    frequencies = _channel_frequencies_mhz(temp)
    subband_indices = np.array_split(np.arange(nchan), n_subbands)

    subbands: list[SubbandTauResult] = []
    for indices in subband_indices:
        profile, frequency = _integrated_subband_profile(temp, indices, frequencies)
        tau_fit = fit_tau_from_profile(profile, period_s=period_s)
        subbands.append(
            SubbandTauResult(
                frequency_mhz=float(frequency),
                tau=float(tau_fit.tau_seconds if tau_fit.tau_seconds is not None else tau_fit.tau_bins),
                tau_error=float(
                    tau_fit.tau_seconds_error
                    if tau_fit.tau_seconds_error is not None
                    else tau_fit.tau_bins_error
                ),
                tau_bins=tau_fit.tau_bins,
                tau_bins_error=tau_fit.tau_bins_error,
                fit=tau_fit,
                profile=profile,
                channel_start=int(indices[0]),
                channel_stop=int(indices[-1]),
            )
        )

    freq = np.asarray([item.frequency_mhz for item in subbands])
    tau = np.asarray([item.tau for item in subbands])
    tau_error = np.asarray([item.tau_error for item in subbands])
    reference = float(reference_freq_mhz if reference_freq_mhz is not None else np.median(freq))
    alpha_result = fit_alpha(freq, tau, tau_error=tau_error, reference_freq_mhz=reference)

    return ScatteringSpectralIndexResult(
        n_subbands=n_subbands,
        subbands=tuple(subbands),
        alpha_fit=alpha_result,
    )


def _channel_frequencies_mhz(archive) -> np.ndarray:
    freqs = archive.get_frequencies()
    if freqs is not None and len(freqs) == archive.get_nchan():
        return np.asarray(freqs, dtype=float)
    return np.linspace(
        archive.get_centre_frequency() - abs(archive.get_bandwidth()) / 2.0,
        archive.get_centre_frequency() + abs(archive.get_bandwidth()) / 2.0,
        archive.get_nchan(),
    )


def _integrated_subband_profile(archive, channel_indices, frequencies_mhz):
    integration = archive.get_Integration(0)
    profiles = []
    weights = []
    freqs = []

    for ichan in channel_indices:
        weight = float(integration.get_weight(int(ichan)))
        if weight <= 0:
            continue
        profile = np.asarray(integration.get_Profile(0, int(ichan)).get_amps(), dtype=float)
        profiles.append(profile * weight)
        weights.append(weight)
        freqs.append(float(frequencies_mhz[int(ichan)]) * weight)

    if not profiles:
        raise ValueError("subband has no channels with positive weights")

    weight_sum = float(np.sum(weights))
    profile = np.sum(profiles, axis=0) / weight_sum
    frequency = float(np.sum(freqs) / weight_sum)
    profile = profile - np.min(profile)
    mx = np.max(profile)
    if mx != 0:
        profile = profile / mx
    return profile, frequency


def plot_tau_fit(profile, result: TauFitResult, output_path: str | None = None):
    import matplotlib.pyplot as plt

    prof = np.asarray(profile, dtype=float)
    prof = prof - np.min(prof)
    mx = np.max(prof)
    if mx != 0:
        prof = prof / mx

    t = np.arange(len(prof))
    fig, ax = plt.subplots()
    ax.plot(t, prof, label="Data")
    ax.plot(
        t,
        scattered_pulse(t, result.amplitude, result.mu, result.sigma, result.tau_bins),
        label="Fit",
        linewidth=2,
    )
    ax.set_xlabel("Phase bins")
    ax.set_ylabel("Normalized Flux")
    ax.set_title("Integrated Profile Fit")
    ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig


def plot_subband_tau_fits(
    result: ScatteringSpectralIndexResult,
    output_path: str | None = None,
):
    """Plot observed subband profiles with fitted scattered-pulse overlays."""
    import matplotlib.pyplot as plt

    nplots = len(result.subbands)
    ncols = min(3, nplots)
    nrows = int(np.ceil(nplots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)

    for ax, subband in zip(axes.ravel(), result.subbands):
        profile = _normalise_profile(subband.profile)
        bins = np.arange(len(profile))
        fit_curve = scattered_pulse(
            bins,
            subband.fit.amplitude,
            subband.fit.mu,
            subband.fit.sigma,
            subband.fit.tau_bins,
        )
        fit_curve = _normalise_profile(fit_curve)

        ax.plot(bins, profile, color="black", label="Profile")
        ax.plot(bins, fit_curve, color="tab:red", linewidth=2, label="Fit")
        ax.set_title(f"{subband.frequency_mhz:.2f} MHz")
        ax.set_xlabel("Phase bin")
        ax.set_ylabel("Normalized flux")
        ax.legend(loc="best", fontsize=8)

    for ax in axes.ravel()[nplots:]:
        ax.axis("off")

    fig.suptitle("Subband Tau Fits")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig


def _normalise_profile(profile) -> np.ndarray:
    prof = np.asarray(profile, dtype=float)
    prof = prof - np.min(prof)
    mx = np.max(prof)
    if mx != 0:
        prof = prof / mx
    return prof
