"""Scattering-time tau fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .archive_io import integrated_profile_array
from .conventions import TAU_REFERENCE_FREQUENCY_MHZ
from .fit_alpha import AlphaFitResult, fit_alpha
from .profile_models import (
    GaussianComponent,
    gaussian_intrinsic_profile,
    isotropic_scattered_profile,
    prepare_intrinsic_template,
    prepare_observed_profile,
    resolve_intrinsic_template,
    shift_template_periodic,
    tau_seed_grid,
)


@dataclass(frozen=True)
class TauFitResult:
    amplitude: float
    mu: float
    sigma: float
    tau_bins: float
    tau_bins_error: float
    tau_seconds: float | None = None
    tau_seconds_error: float | None = None
    chi_square: float | None = None
    reduced_chi_square: float | None = None
    dof: int | None = None
    rms_residual: float | None = None
    baseline: float = 0.0
    components: tuple[GaussianComponent, ...] = ()
    intrinsic_model: str = "gaussian"
    phase_shift_bins: float | None = None
    intrinsic_phase_bins: float | None = None
    intrinsic_phase_bins_error: float | None = None
    covariance: np.ndarray | None = None
    model_profile: np.ndarray | None = None
    component_sigma_errors: tuple[float, ...] = ()
    period_s: float | None = None
    n_bins: int | None = None


@dataclass(frozen=True)
class SubbandTauResult:
    frequency_mhz: float
    central_frequency_mhz: float
    bandwidth_mhz: float
    tau: float
    tau_error: float
    tau_bins: float
    tau_bins_error: float
    fit: TauFitResult
    profile: np.ndarray
    channel_start: int
    channel_stop: int
    valid_channel_fraction: float
    profile_snr: float
    relative_tau_error: float
    accepted: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class FailedSubbandResult:
    frequency_mhz: float
    central_frequency_mhz: float
    bandwidth_mhz: float
    channel_start: int
    channel_stop: int
    rejection_reason: str


@dataclass(frozen=True)
class ScatteringSpectralIndexResult:
    n_subbands: int
    subbands: tuple[SubbandTauResult, ...]
    failed_subbands: tuple[FailedSubbandResult, ...]
    alpha_fit: AlphaFitResult

    @property
    def accepted_subbands(self) -> tuple[SubbandTauResult, ...]:
        return tuple(item for item in self.subbands if item.accepted)

    @property
    def rejected_subbands(self) -> tuple[SubbandTauResult, ...]:
        return tuple(item for item in self.subbands if not item.accepted)


def scattered_pulse(t, amplitude, mu, sigma, tau):
    """Legacy non-periodic exponentially modified Gaussian pulse model."""
    from scipy.special import erf

    # Reference: Geiger & Lam (NANOGrav-Memo-008.pdf), PDF p. 5, Eq. 5.
    # The Gaussian-intrinsic-profile assumption is also described in the
    # Filothodoros thesis (ALEX.pdf), PDF pp. 32-33, Sections 2.7.1-2.7.3.
    term1 = amplitude * (sigma / tau) * np.sqrt(np.pi / 2)
    term2 = np.exp(-(t - mu) / tau)
    arg = (t - (mu + sigma**2 / tau)) / (sigma * np.sqrt(2))
    return term1 * term2 * (1 + erf(arg))


def isotropic_scattered_pulse(
    bins,
    amplitude,
    mu,
    sigma,
    tau,
    baseline=0.0,
):
    """Evaluate the isotropic pulse-train + DC model for one Gaussian."""
    nbin = len(np.asarray(bins))
    intrinsic = gaussian_intrinsic_profile(
        nbin,
        [GaussianComponent(1.0, float(mu), float(sigma))],
    )
    # Reference: Geyer & Karastergiou (fitting.pdf), PDF pp. 7-8, Section 2.4;
    # Filothodoros thesis (ALEX.pdf), PDF p. 33, Section 2.7.2. Both describe
    # pulse-train fitting with a free DC level to retain wrap-around tails.
    return isotropic_scattered_profile(intrinsic, tau, amplitude, baseline)


def fit_tau_from_profile(
    profile,
    period_s: float | None = None,
    n_components: int = 1,
    intrinsic_template=None,
) -> TauFitResult:
    """Fit an isotropic pulse-train + DC model to one folded profile."""
    from scipy.optimize import curve_fit

    prof = prepare_observed_profile(profile)
    nbin = len(prof)
    bins = np.arange(nbin, dtype=float)
    if n_components < 1:
        raise ValueError("n_components must be at least one")
    if intrinsic_template is not None and n_components != 1:
        raise ValueError("n_components and intrinsic_template cannot be used together")

    # PSRISM choices: multistart seeds, bounds and iteration limits are
    # numerical safeguards, not hardcoded physical values from the references.
    if intrinsic_template is None:
        model_function, base_p0, lower, upper, tau_index = _gaussian_train_fit_setup(
            prof,
            n_components,
        )
        intrinsic_model = "gaussian" if n_components == 1 else "multi_gaussian"
    else:
        template = prepare_intrinsic_template(intrinsic_template, target_nbin=nbin)
        model_function, base_p0, lower, upper, tau_index = _template_train_fit_setup(prof, template)
        intrinsic_model = "template"

    best = None
    for tau_seed in tau_seed_grid(nbin):
        p0 = np.asarray(base_p0, dtype=float).copy()
        p0[tau_index] = tau_seed
        try:
            popt, pcov = curve_fit(
                model_function,
                bins,
                prof,
                p0=p0,
                bounds=(lower, upper),
                maxfev=50000,
                x_scale="jac",
            )
        except (RuntimeError, ValueError, FloatingPointError):
            continue
        model = model_function(bins, *popt)
        rss = float(np.sum((prof - model) ** 2))
        if best is None or rss < best[0]:
            best = (rss, popt, pcov, model)
    if best is None:
        raise RuntimeError("isotropic pulse-train fit did not converge")

    chi_square, popt, pcov, model = best
    tau_value = float(popt[tau_index])
    tau_err = float(np.sqrt(max(float(pcov[tau_index, tau_index]), 0.0)))
    residual = prof - model
    dof = max(nbin - len(popt), 0)
    reduced_chi_square = float(chi_square / dof) if dof > 0 else None
    rms_residual = float(np.sqrt(np.mean(residual**2)))

    if intrinsic_template is None:
        # Reference: Filothodoros et al. (Alex_and_Boe.pdf), PDF pp. 11-12,
        # Appendix B, retain fitted intrinsic Gaussian widths by frequency.
        # Using the optimizer covariance for sigma errors is a PSRISM choice.
        indexed_components = tuple(
            (
                GaussianComponent(
                    amplitude=float(popt[3 * index]),
                    mu=float(popt[3 * index + 1]),
                    sigma=float(popt[3 * index + 2]),
                ),
                float(
                    np.sqrt(
                        max(float(pcov[3 * index + 2, 3 * index + 2]), 0.0)
                    )
                ),
            )
            for index in range(n_components)
        )
        # PSRISM choice: the highest-amplitude fitted Gaussian is the phase
        # fiducial for the optional scattering-aware residual-DM diagnostic.
        # This is unambiguous for the default one-component model; users of a
        # multi-component profile should verify component identity by eye.
        primary_index = int(
            np.argmax([component.amplitude for component, _error in indexed_components])
        )
        intrinsic_phase = float(popt[3 * primary_index + 1])
        intrinsic_phase_error = float(
            np.sqrt(max(float(pcov[3 * primary_index + 1, 3 * primary_index + 1]), 0.0))
        )
        ordered_components = tuple(
            sorted(indexed_components, key=lambda item: item[0].mu)
        )
        components = tuple(item[0] for item in ordered_components)
        component_sigma_errors = tuple(item[1] for item in ordered_components)
        first = components[0]
        baseline = float(popt[-1])
        phase_shift = None
    else:
        components = ()
        component_sigma_errors = ()
        first = GaussianComponent(float(popt[0]), float(popt[1]), float("nan"))
        baseline = float(popt[3])
        phase_shift = float(popt[1])
        intrinsic_phase = phase_shift
        intrinsic_phase_error = float(np.sqrt(max(float(pcov[1, 1]), 0.0)))

    tau_seconds = None
    tau_seconds_error = None
    if period_s is not None:
        tau_seconds = float(tau_value * (period_s / nbin))
        tau_seconds_error = float(tau_err * (period_s / nbin))

    return TauFitResult(
        amplitude=first.amplitude,
        mu=first.mu,
        sigma=first.sigma,
        tau_bins=tau_value,
        tau_bins_error=tau_err,
        tau_seconds=tau_seconds,
        tau_seconds_error=tau_seconds_error,
        chi_square=chi_square,
        reduced_chi_square=reduced_chi_square,
        dof=dof,
        rms_residual=rms_residual,
        baseline=baseline,
        components=components,
        component_sigma_errors=component_sigma_errors,
        intrinsic_model=intrinsic_model,
        period_s=None if period_s is None else float(period_s),
        n_bins=int(nbin),
        phase_shift_bins=phase_shift,
        intrinsic_phase_bins=intrinsic_phase,
        intrinsic_phase_bins_error=intrinsic_phase_error,
        covariance=pcov,
        model_profile=np.asarray(model, dtype=float),
    )


def fit_tau_from_archive(
    archive,
    center_peak: bool = False,
    n_components: int = 1,
    intrinsic_template=None,
) -> TauFitResult:
    """Fit tau from an archive integrated profile."""
    # Reference: Filothodoros et al. (Alex_and_Boe.pdf), PDF p. 4, retains the
    # profile baseline so the train+DC model can fit tails from earlier pulses.
    profile = integrated_profile_array(
        archive,
        normalize=False,
        center_peak=center_peak,
        remove_baseline=False,
    )
    period_s = archive.get_Integration(0).get_folding_period()
    template = resolve_intrinsic_template(intrinsic_template, len(profile))
    return fit_tau_from_profile(
        profile,
        period_s=period_s,
        n_components=n_components,
        intrinsic_template=template,
    )


def fit_tau_alpha_from_archive(
    archive,
    n_subbands: int,
    reference_freq_mhz: float | None = None,
    n_components: int = 1,
    intrinsic_template=None,
    min_subbands: int = 3,
    min_profile_snr: float = 5.0,
    max_relative_tau_error: float = 1.0,
    min_valid_channel_fraction: float = 0.5,
    monte_carlo_samples: int = 10000,
    random_seed: int | None = 0,
) -> ScatteringSpectralIndexResult:
    """Fit tau in contiguous frequency subbands and then fit alpha."""
    # Reference: Filothodoros et al. (Alex_and_Boe.pdf), PDF p. 5, Section 3,
    # reject weak/RFI-dominated subbands and require at least three reliable
    # tau values. The S/N, relative-error, occupancy, and default cutoff values
    # here are reproducible PSRISM choices because the paper uses visual,
    # observation-dependent rejection rather than universal numeric limits.
    if n_subbands < 2:
        raise ValueError("n_subbands must be at least 2 to fit alpha")
    if min_subbands > n_subbands:
        raise ValueError("min_subbands cannot exceed n_subbands")
    _validate_frequency_scaling_quality_options(
        min_subbands,
        min_profile_snr,
        max_relative_tau_error,
        min_valid_channel_fraction,
    )

    temp = archive.clone()
    temp.tscrunch()

    nchan = temp.get_nchan()
    nbin = temp.get_nbin()
    if n_subbands > nchan:
        raise ValueError("n_subbands cannot exceed the number of frequency channels")

    period_s = temp.get_Integration(0).get_folding_period()
    template = resolve_intrinsic_template(intrinsic_template, nbin)
    frequencies = _channel_frequencies_mhz(temp)
    # Reference: Filothodoros thesis (ALEX.pdf), PDF p. 31, Section 2.6.3,
    # motivates contiguous subbands as a frequency-coverage/SNR compromise.
    subband_indices = np.array_split(np.arange(nchan), n_subbands)

    subbands: list[SubbandTauResult] = []
    failed_subbands: list[FailedSubbandResult] = []
    for indices in subband_indices:
        central_frequency, bandwidth, frequency = _subband_frequency_metadata(
            frequencies,
            indices,
            channel_width_mhz=abs(float(temp.get_bandwidth())) / nchan,
        )
        try:
            profile, _, _, _, valid_channel_fraction = _integrated_subband_profile(
                temp,
                indices,
                frequencies,
            )
            tau_fit = fit_tau_from_profile(
                profile,
                period_s=period_s,
                n_components=n_components,
                intrinsic_template=template,
            )
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            failed_subbands.append(
                FailedSubbandResult(
                    frequency_mhz=frequency,
                    central_frequency_mhz=central_frequency,
                    bandwidth_mhz=bandwidth,
                    channel_start=int(indices[0]),
                    channel_stop=int(indices[-1]),
                    rejection_reason=str(exc),
                )
            )
            continue

        tau = float(tau_fit.tau_seconds if tau_fit.tau_seconds is not None else tau_fit.tau_bins)
        tau_error = float(
            tau_fit.tau_seconds_error
            if tau_fit.tau_seconds_error is not None
            else tau_fit.tau_bins_error
        )
        profile_snr = _profile_fit_snr(profile, tau_fit.model_profile)
        relative_tau_error = float(tau_error / tau) if tau > 0 else float("inf")
        rejection_reasons = _tau_subband_rejection_reasons(
            tau,
            tau_error,
            profile_snr,
            valid_channel_fraction,
            min_profile_snr,
            max_relative_tau_error,
            min_valid_channel_fraction,
        )
        subbands.append(
            SubbandTauResult(
                frequency_mhz=float(frequency),
                central_frequency_mhz=float(central_frequency),
                bandwidth_mhz=float(bandwidth),
                tau=tau,
                tau_error=tau_error,
                tau_bins=tau_fit.tau_bins,
                tau_bins_error=tau_fit.tau_bins_error,
                fit=tau_fit,
                profile=profile,
                channel_start=int(indices[0]),
                channel_stop=int(indices[-1]),
                valid_channel_fraction=float(valid_channel_fraction),
                profile_snr=profile_snr,
                relative_tau_error=relative_tau_error,
                accepted=not rejection_reasons,
                rejection_reasons=rejection_reasons,
            )
        )

    accepted = [item for item in subbands if item.accepted]
    if len(accepted) < min_subbands:
        raise ValueError(
            f"only {len(accepted)} of {n_subbands} subbands passed alpha quality checks; "
            f"at least {min_subbands} are required"
        )

    freq = np.asarray([item.frequency_mhz for item in accepted])
    tau = np.asarray([item.tau for item in accepted])
    tau_error = np.asarray([item.tau_error for item in accepted])
    reference = float(
        reference_freq_mhz
        if reference_freq_mhz is not None
        else TAU_REFERENCE_FREQUENCY_MHZ
    )
    alpha_result = fit_alpha(
        freq,
        tau,
        tau_error=tau_error,
        reference_freq_mhz=reference,
        monte_carlo_samples=monte_carlo_samples,
        random_seed=random_seed,
    )

    return ScatteringSpectralIndexResult(
        n_subbands=n_subbands,
        subbands=tuple(subbands),
        failed_subbands=tuple(failed_subbands),
        alpha_fit=alpha_result,
    )


def evaluate_tau_fit(result: TauFitResult, nbin: int | None = None) -> np.ndarray:
    """Return the fitted profile retained in a ``TauFitResult``."""
    if result.model_profile is not None:
        model = np.asarray(result.model_profile, dtype=float)
        if nbin is None or len(model) == nbin:
            return model.copy()
    if result.intrinsic_model != "gaussian" or not np.isfinite(result.sigma):
        raise ValueError("this fit cannot be reconstructed without its retained model profile")
    target_nbin = int(nbin) if nbin is not None else 0
    if target_nbin <= 0:
        raise ValueError("nbin is required when no retained model profile is available")
    return isotropic_scattered_pulse(
        np.arange(target_nbin),
        result.amplitude,
        result.mu,
        result.sigma,
        result.tau_bins,
        result.baseline,
    )


def _gaussian_train_fit_setup(profile: np.ndarray, n_components: int):
    from scipy.signal import find_peaks

    nbin = len(profile)
    baseline0 = float(np.percentile(profile, 10.0))
    signal = profile - baseline0
    minimum_distance = max(nbin // max(2 * n_components, 1), 1)
    peaks, _ = find_peaks(signal, distance=minimum_distance)
    ranked = list(peaks[np.argsort(signal[peaks])[::-1]]) if len(peaks) else []
    if int(np.argmax(signal)) not in ranked:
        ranked.insert(0, int(np.argmax(signal)))
    while len(ranked) < n_components:
        ranked.append(int((ranked[0] + len(ranked) * nbin / n_components) % nbin))
    centers = sorted(ranked[:n_components])

    p0 = []
    lower = []
    upper = []
    for center in centers:
        p0.extend(
            [
                max(float(signal[center]), 0.1),
                float(center),
                max(nbin / (20.0 * np.sqrt(n_components)), 1.0),
            ]
        )
        lower.extend([0.0, 0.0, 0.25])
        upper.extend([np.inf, float(nbin), float(nbin) / 2.0])
    tau_index = len(p0)
    p0.extend([max(nbin / 20.0, 0.5), baseline0])
    lower.extend([0.05, -np.inf])
    upper.extend([float(2 * nbin), np.inf])

    def model(_bins, *params):
        components = tuple(
            GaussianComponent(params[3 * index], params[3 * index + 1], params[3 * index + 2])
            for index in range(n_components)
        )
        intrinsic = gaussian_intrinsic_profile(nbin, components)
        return isotropic_scattered_profile(intrinsic, params[tau_index], 1.0, params[-1])

    return model, p0, lower, upper, tau_index


def _template_train_fit_setup(profile: np.ndarray, template: np.ndarray):
    nbin = len(profile)
    baseline0 = float(np.percentile(profile, 10.0))
    amplitude0 = max(float(np.ptp(profile)), 0.1)
    correlation = np.fft.ifft(
        np.fft.fft(profile - np.mean(profile))
        * np.conj(np.fft.fft(template - np.mean(template)))
    ).real
    shift0 = float(np.argmax(correlation))
    if shift0 > nbin / 2:
        shift0 -= nbin
    tau_index = 2
    p0 = [amplitude0, shift0, max(nbin / 20.0, 0.5), baseline0]
    lower = [0.0, -float(nbin) / 2.0, 0.05, -np.inf]
    upper = [np.inf, float(nbin) / 2.0, float(2 * nbin), np.inf]

    def model(_bins, amplitude, phase_shift, tau, baseline):
        intrinsic = shift_template_periodic(template, phase_shift)
        return isotropic_scattered_profile(intrinsic, tau, amplitude, baseline)

    return model, p0, lower, upper, tau_index


def _channel_frequencies_mhz(archive) -> np.ndarray:
    freqs = archive.get_frequencies()
    if freqs is not None and len(freqs) == archive.get_nchan():
        return np.asarray(freqs, dtype=float)
    nchan = archive.get_nchan()
    bandwidth = abs(float(archive.get_bandwidth()))
    channel_width = bandwidth / nchan
    low_edge = float(archive.get_centre_frequency()) - bandwidth / 2.0
    # PSRISM fallback: construct channel centers from the archive band edges.
    return low_edge + channel_width * (np.arange(nchan, dtype=float) + 0.5)


def _integrated_subband_profile(archive, channel_indices, frequencies_mhz):
    integration = archive.get_Integration(0)
    profiles = []
    weights = []

    for ichan in channel_indices:
        weight = float(integration.get_weight(int(ichan)))
        if weight <= 0:
            continue
        profile = np.asarray(integration.get_Profile(0, int(ichan)).get_amps(), dtype=float)
        profiles.append(profile * weight)
        weights.append(weight)

    if not profiles:
        raise ValueError("subband has no channels with positive weights")

    weight_sum = float(np.sum(weights))
    profile = np.sum(profiles, axis=0) / weight_sum
    center, bandwidth, frequency = _subband_frequency_metadata(
        frequencies_mhz,
        channel_indices,
        channel_width_mhz=abs(float(archive.get_bandwidth())) / archive.get_nchan(),
    )
    valid_channel_fraction = len(weights) / len(channel_indices)
    return profile, frequency, center, bandwidth, valid_channel_fraction


def _subband_center_frequency_mhz(frequencies_mhz, channel_indices) -> float:
    subband_freq = np.asarray(frequencies_mhz, dtype=float)[channel_indices]
    return float((np.min(subband_freq) + np.max(subband_freq)) / 2.0)


def effective_subband_frequency_mhz(
    central_frequency_mhz: float,
    bandwidth_mhz: float,
) -> float:
    """Map a finite subband to the equivalent monochromatic frequency."""
    center = float(central_frequency_mhz)
    bandwidth = float(bandwidth_mhz)
    if not np.isfinite(center) or center <= 0:
        raise ValueError("central_frequency_mhz must be positive")
    if not np.isfinite(bandwidth) or bandwidth < 0:
        raise ValueError("bandwidth_mhz cannot be negative")
    # Reference: Geyer & Karastergiou (fitting.pdf), PDF p. 4, Eq. 6.
    return float(0.5 * np.sqrt(bandwidth**2 + 4.0 * center**2))


def _subband_frequency_metadata(
    frequencies_mhz,
    channel_indices,
    channel_width_mhz: float,
) -> tuple[float, float, float]:
    subband_freq = np.asarray(frequencies_mhz, dtype=float)[channel_indices]
    center = _subband_center_frequency_mhz(frequencies_mhz, channel_indices)
    bandwidth = float(np.ptp(subband_freq) + abs(channel_width_mhz))
    return center, bandwidth, effective_subband_frequency_mhz(center, bandwidth)


def _profile_fit_snr(profile, model_profile) -> float:
    if model_profile is None:
        return float("nan")
    observed = prepare_observed_profile(profile)
    model = np.asarray(model_profile, dtype=float)
    residual = observed - model
    residual_center = float(np.median(residual))
    # PSRISM choice: 1.4826 scales the residual MAD to Gaussian sigma.
    noise = 1.4826 * float(np.median(np.abs(residual - residual_center)))
    signal = float(np.ptp(model))
    if noise <= np.finfo(float).eps:
        return float("inf") if signal > 0 else 0.0
    return signal / noise


def _tau_subband_rejection_reasons(
    tau: float,
    tau_error: float,
    profile_snr: float,
    valid_channel_fraction: float,
    min_profile_snr: float,
    max_relative_tau_error: float,
    min_valid_channel_fraction: float,
) -> tuple[str, ...]:
    reasons = []
    if not np.isfinite(tau) or tau <= 0:
        reasons.append("non-positive or non-finite tau")
    if not np.isfinite(tau_error) or tau_error <= 0:
        reasons.append("non-positive or non-finite tau uncertainty")
    elif tau > 0 and tau_error / tau > max_relative_tau_error:
        reasons.append(
            f"relative tau uncertainty {tau_error / tau:.3g} exceeds {max_relative_tau_error:g}"
        )
    if not np.isfinite(profile_snr) or profile_snr < min_profile_snr:
        reasons.append(f"profile fit S/N {profile_snr:.3g} is below {min_profile_snr:g}")
    if valid_channel_fraction < min_valid_channel_fraction:
        reasons.append(
            f"valid-channel fraction {valid_channel_fraction:.3g} is below "
            f"{min_valid_channel_fraction:g}"
        )
    return tuple(reasons)


def _validate_frequency_scaling_quality_options(
    min_subbands: int,
    min_profile_snr: float,
    max_relative_tau_error: float,
    min_valid_channel_fraction: float,
) -> None:
    if min_subbands < 2:
        raise ValueError("min_subbands must be at least 2")
    if not np.isfinite(min_profile_snr) or min_profile_snr <= 0:
        raise ValueError("min_profile_snr must be positive")
    if not np.isfinite(max_relative_tau_error) or max_relative_tau_error <= 0:
        raise ValueError("max_relative_tau_error must be positive")
    if not 0 < min_valid_channel_fraction <= 1:
        raise ValueError("min_valid_channel_fraction must be in the interval (0, 1]")


def plot_tau_fit(profile, result: TauFitResult, output_path: str | None = None):
    import matplotlib.pyplot as plt

    model = evaluate_tau_fit(result, len(profile))
    prof, model = profile_model_for_plot(profile, model)

    t = np.arange(len(prof))
    fig, ax = plt.subplots()
    ax.scatter(t, prof, color="black", s=8, label="Data")
    ax.plot(
        np.arange(len(prof)),
        model,
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
    title: str | None = None,
    output_path: str | None = None,
    nrows: int | None = None,
    ncols: int | None = None,
):
    """Plot observed subband profiles with fitted scattered-pulse overlays."""
    import matplotlib.pyplot as plt

    subbands = sorted(result.subbands, key=lambda item: item.frequency_mhz)
    nplots = len(subbands)
    nrows, ncols = _subplot_grid(nplots, nrows=nrows, ncols=ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.4 * ncols, 2.8 * nrows),
        sharex=True,
        squeeze=False,
    )

    for ax, subband in zip(axes.ravel(), subbands):
        fit_curve = evaluate_tau_fit(subband.fit, len(subband.profile))
        profile, fit_curve = profile_model_for_plot(subband.profile, fit_curve)
        phase = np.linspace(0.0, 1.0, len(profile), endpoint=False)

        color = "tab:red" if subband.accepted else "tab:gray"
        ax.scatter(phase, profile, color="black", s=8, label="Profile")
        ax.plot(
            phase,
            fit_curve,
            color=color,
            linestyle="-" if subband.accepted else "--",
            linewidth=2,
            label="Accepted fit" if subband.accepted else "Rejected fit",
        )
        ax.set_ylabel("Normalized flux")
        ax.text(
            0.985,
            0.88,
            f"f_m = {subband.frequency_mhz:.2f} MHz\n"
            f"τ = {subband.tau * 1e3:.4g} ± {subband.tau_error * 1e3:.2g} ms\n"
            f"{'accepted' if subband.accepted else 'rejected'}; S/N={subband.profile_snr:.3g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        ax.legend(loc="best", fontsize=8)

    for ax in axes.ravel()[nplots:]:
        ax.axis("off")
    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("Pulse phase")
    fit_quality = "; ".join(_subband_fit_caption(item) for item in subbands)
    if result.failed_subbands:
        fit_quality += f"; {len(result.failed_subbands)} fit failure(s)"
    fig.text(
        0.5,
        0.015,
        f"Fit quality: {fit_quality}",
        ha="center",
        va="bottom",
        fontsize=9,
        wrap=True,
    )

    if title:
        fig.suptitle(Path(title).name)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig


def _subplot_grid(nplots: int, nrows: int | None = None, ncols: int | None = None) -> tuple[int, int]:
    if nplots <= 0:
        return 1, 1
    if nrows is not None and nrows <= 0:
        raise ValueError("subband plot rows must be positive")
    if ncols is not None and ncols <= 0:
        raise ValueError("subband plot columns must be positive")
    if nrows is None and ncols is None:
        ncols = int(np.ceil(np.sqrt(nplots)))
        nrows = int(np.ceil(nplots / ncols))
    elif nrows is None:
        nrows = int(np.ceil(nplots / ncols))
    elif ncols is None:
        ncols = int(np.ceil(nplots / nrows))
    if nrows * ncols < nplots:
        raise ValueError(
            f"subband plot grid {nrows}x{ncols} has {nrows * ncols} panels for {nplots} subbands"
        )
    nrows = min(nrows, int(np.ceil(nplots / ncols)))
    return int(nrows), int(ncols)


def _normalise_profile(profile) -> np.ndarray:
    prof = np.asarray(profile, dtype=float)
    prof = prof - np.min(prof)
    mx = np.max(prof)
    if mx != 0:
        prof = prof / mx
    return prof


def profile_model_for_plot(profile, model_profile) -> tuple[np.ndarray, np.ndarray]:
    """Apply one data-derived offset and scale to profile and fitted model."""
    observed = np.asarray(profile, dtype=float)
    model = np.asarray(model_profile, dtype=float)
    if observed.shape != model.shape:
        raise ValueError("profile and fitted model must have the same shape")
    span = float(np.ptp(observed))
    if span <= np.finfo(float).eps:
        raise ValueError("profile has no measurable variation")
    conditioned = (observed - np.median(observed)) / span
    offset_in_fit_units = float(np.min(conditioned))
    return conditioned - offset_in_fit_units, model - offset_in_fit_units


def _subband_fit_caption(subband: SubbandTauResult) -> str:
    reduced = _format_optional_float(subband.fit.reduced_chi_square)
    rms = _format_optional_float(subband.fit.rms_residual)
    status = "accepted" if subband.accepted else "rejected"
    return (
        f"{subband.frequency_mhz:.2f} MHz: {status}, S/N={subband.profile_snr:.3g}, "
        f"valid channels={subband.valid_channel_fraction:.3g}, "
        f"unweighted red. χ²={reduced}, RMS={rms}"
    )


def _format_optional_float(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.3g}"
