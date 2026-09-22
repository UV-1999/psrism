"""Anisotropic pulse-broadening model fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .conventions import TAU_REFERENCE_FREQUENCY_MHZ
from .fit_alpha import AlphaFitResult, fit_alpha
from .fit_tau import FailedSubbandResult, fit_tau_from_profile
from .profile_models import (
    GaussianComponent,
    gaussian_intrinsic_profile,
    periodic_convolution,
    prepare_intrinsic_template,
    prepare_observed_profile,
    resolve_intrinsic_template,
    shift_template_periodic,
    tau_seed_grid,
)


@dataclass(frozen=True)
class AnisotropicFitResult:
    amplitude: float
    mu: float
    sigma: float
    tau_eff_bins: float
    anisotropy_ratio: float
    baseline: float
    tau_x_bins: float
    tau_y_bins: float
    tau_eff_bins_error: float
    anisotropy_ratio_error: float
    tau_x_bins_error: float
    tau_y_bins_error: float
    tau_eff_seconds: float | None
    tau_eff_seconds_error: float | None
    tau_x_seconds: float | None
    tau_x_seconds_error: float | None
    tau_y_seconds: float | None
    tau_y_seconds_error: float | None
    chi_square: float
    reduced_chi_square: float | None
    dof: int
    rms_residual: float
    covariance: np.ndarray
    components: tuple[GaussianComponent, ...] = ()
    intrinsic_model: str = "gaussian"
    phase_shift_bins: float | None = None
    model_profile: np.ndarray | None = None
    component_sigma_errors: tuple[float, ...] = ()
    period_s: float | None = None
    n_bins: int | None = None


@dataclass(frozen=True)
class AnisotropicSubbandResult:
    frequency_mhz: float
    central_frequency_mhz: float
    bandwidth_mhz: float
    tau_eff: float
    tau_eff_error: float
    tau_x: float
    tau_x_error: float
    tau_y: float
    tau_y_error: float
    anisotropy_ratio: float
    anisotropy_ratio_error: float
    fit: AnisotropicFitResult
    isotropic_reduced_chi_square: float | None
    profile: np.ndarray
    channel_start: int
    channel_stop: int
    valid_channel_fraction: float
    profile_snr: float
    relative_tau_error: float
    accepted: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class AnisotropicScatteringResult:
    n_subbands: int
    subbands: tuple[AnisotropicSubbandResult, ...]
    failed_subbands: tuple[FailedSubbandResult, ...]
    alpha_fit: AlphaFitResult

    @property
    def accepted_subbands(self) -> tuple[AnisotropicSubbandResult, ...]:
        return tuple(item for item in self.subbands if item.accepted)

    @property
    def rejected_subbands(self) -> tuple[AnisotropicSubbandResult, ...]:
        return tuple(item for item in self.subbands if not item.accepted)


def anisotropic_pbf(delays, tau_eff: float, anisotropy_ratio: float) -> np.ndarray:
    """Return the anisotropic pulse broadening function in delay-bin units."""
    from scipy.special import i0e

    t = np.asarray(delays, dtype=float)
    tau_eff = max(float(tau_eff), 1e-12)
    ratio = max(float(anisotropy_ratio), 1.0)
    # Reference: Geyer & Karastergiou (fitting.pdf), PDF p. 4, Eq. 4, gives
    # the two-timescale anisotropic PBF and identifies I0 as the modified
    # Bessel function. PDF p. 10 uses sqrt(tau_x*tau_y) as tau_geo.
    tau_x = tau_eff / np.sqrt(ratio)
    tau_y = tau_eff * np.sqrt(ratio)

    z = 0.5 * t * (1.0 / tau_x - 1.0 / tau_y)
    # PSRISM choice: scipy.special.i0e plus abs(z) evaluates Eq. 4 stably and
    # does not alter the mathematical broadening function.
    exponent = -0.5 * t * (1.0 / tau_x + 1.0 / tau_y) + np.abs(z)
    pbf = np.exp(exponent) * i0e(z) / np.sqrt(tau_x * tau_y)
    pbf = np.where(t >= 0, pbf, 0.0)
    return np.where(np.isfinite(pbf), pbf, 0.0)


def anisotropic_scattered_pulse(
    bins,
    amplitude,
    mu,
    sigma,
    tau_eff,
    anisotropy_ratio,
    baseline,
):
    """Folded Gaussian profile convolved with an anisotropic PBF."""
    nbin = len(np.asarray(bins))
    intrinsic = gaussian_intrinsic_profile(
        nbin,
        [GaussianComponent(1.0, float(mu), float(sigma))],
    )

    # Reference: Geyer & Karastergiou (fitting.pdf), PDF pp. 7-8, Section 2.4,
    # motivates a pulse train plus fitted DC level when scattering wraps
    # between rotations. Circular convolution is PSRISM's folded equivalent.
    pbf = _folded_pbf(nbin, tau_eff, anisotropy_ratio)
    scattered = periodic_convolution(intrinsic, pbf)
    return baseline + amplitude * scattered


def fit_anisotropic_profile(
    profile,
    period_s: float | None = None,
    n_components: int = 1,
    intrinsic_template=None,
) -> AnisotropicFitResult:
    """Fit a folded anisotropic scattering model to a pulse profile."""
    from scipy.optimize import curve_fit

    prof = prepare_observed_profile(profile)
    nbin = len(prof)
    bins = np.arange(nbin, dtype=float)
    if n_components < 1:
        raise ValueError("n_components must be at least one")
    if intrinsic_template is not None and n_components != 1:
        raise ValueError("n_components and intrinsic_template cannot be used together")

    # PSRISM choices: parameter initial values, bounds and maxfev are numerical
    # safeguards and are not hardcoded physical values from the references.
    if intrinsic_template is None:
        model_function, base_p0, lower, upper, tau_index, ratio_index = (
            _anisotropic_gaussian_fit_setup(prof, n_components)
        )
        intrinsic_model = "gaussian" if n_components == 1 else "multi_gaussian"
    else:
        template = prepare_intrinsic_template(intrinsic_template, target_nbin=nbin)
        model_function, base_p0, lower, upper, tau_index, ratio_index = (
            _anisotropic_template_fit_setup(prof, template)
        )
        intrinsic_model = "template"

    best = None
    for tau_seed in tau_seed_grid(nbin):
        for ratio_seed in (1.0, 2.0, 5.0):
            p0 = np.asarray(base_p0, dtype=float).copy()
            p0[tau_index] = tau_seed
            p0[ratio_index] = ratio_seed
            try:
                popt, pcov = curve_fit(
                    model_function,
                    bins,
                    prof,
                    p0=p0,
                    bounds=(lower, upper),
                    maxfev=80000,
                    x_scale="jac",
                )
            except (RuntimeError, ValueError, FloatingPointError):
                continue
            model = model_function(bins, *popt)
            rss = float(np.sum((prof - model) ** 2))
            if best is None or rss < best[0]:
                best = (rss, popt, pcov, model)
    if best is None:
        raise RuntimeError("anisotropic pulse-train fit did not converge")

    chi_square, popt, pcov, model = best
    residual = prof - model
    dof = max(nbin - len(popt), 0)
    reduced_chi_square = float(chi_square / dof) if dof > 0 else None
    rms_residual = float(np.sqrt(np.mean(residual**2)))

    tau_eff = float(popt[tau_index])
    ratio = float(popt[ratio_index])
    tau_x = tau_eff / np.sqrt(ratio)
    tau_y = tau_eff * np.sqrt(ratio)
    errors = _tau_errors_from_covariance(tau_eff, ratio, pcov, tau_index, ratio_index)

    if intrinsic_template is None:
        # Reference: Filothodoros et al. (Alex_and_Boe.pdf), PDF pp. 11-12,
        # Appendix B, retain fitted intrinsic Gaussian widths by frequency.
        # Using the optimizer covariance for sigma errors is a PSRISM choice.
        indexed_components = tuple(
            (
                GaussianComponent(
                    float(popt[3 * index]),
                    float(popt[3 * index + 1]),
                    float(popt[3 * index + 2]),
                ),
                float(
                    np.sqrt(
                        max(float(pcov[3 * index + 2, 3 * index + 2]), 0.0)
                    )
                ),
            )
            for index in range(n_components)
        )
        ordered_components = tuple(
            sorted(indexed_components, key=lambda item: item[0].mu)
        )
        components = tuple(item[0] for item in ordered_components)
        component_sigma_errors = tuple(item[1] for item in ordered_components)
        first = components[0]
        phase_shift = None
    else:
        components = ()
        component_sigma_errors = ()
        first = GaussianComponent(float(popt[0]), float(popt[1]), float("nan"))
        phase_shift = float(popt[1])

    bin_to_s = None if period_s is None else float(period_s / nbin)
    return AnisotropicFitResult(
        amplitude=first.amplitude,
        mu=first.mu,
        sigma=first.sigma,
        tau_eff_bins=tau_eff,
        anisotropy_ratio=ratio,
        baseline=float(popt[-1]),
        tau_x_bins=tau_x,
        tau_y_bins=tau_y,
        tau_eff_bins_error=errors["tau_eff"],
        anisotropy_ratio_error=errors["ratio"],
        tau_x_bins_error=errors["tau_x"],
        tau_y_bins_error=errors["tau_y"],
        tau_eff_seconds=_scale_optional(tau_eff, bin_to_s),
        tau_eff_seconds_error=_scale_optional(errors["tau_eff"], bin_to_s),
        tau_x_seconds=_scale_optional(tau_x, bin_to_s),
        tau_x_seconds_error=_scale_optional(errors["tau_x"], bin_to_s),
        tau_y_seconds=_scale_optional(tau_y, bin_to_s),
        tau_y_seconds_error=_scale_optional(errors["tau_y"], bin_to_s),
        chi_square=chi_square,
        reduced_chi_square=reduced_chi_square,
        dof=dof,
        rms_residual=rms_residual,
        covariance=pcov,
        components=components,
        component_sigma_errors=component_sigma_errors,
        intrinsic_model=intrinsic_model,
        period_s=None if period_s is None else float(period_s),
        n_bins=int(nbin),
        phase_shift_bins=phase_shift,
        model_profile=np.asarray(model, dtype=float),
    )


def fit_anisotropic_scattering_from_archive(
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
) -> AnisotropicScatteringResult:
    """Fit anisotropic scattering in subbands and fit τ_eff frequency scaling."""
    from .fit_tau import (
        _channel_frequencies_mhz,
        _integrated_subband_profile,
        _profile_fit_snr,
        _subband_frequency_metadata,
        _tau_subband_rejection_reasons,
        _validate_frequency_scaling_quality_options,
    )

    if n_subbands < 2:
        raise ValueError("n_subbands must be at least 2")
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
    if n_subbands > nchan:
        raise ValueError("n_subbands cannot exceed the number of frequency channels")

    period_s = temp.get_Integration(0).get_folding_period()
    template = resolve_intrinsic_template(intrinsic_template, temp.get_nbin())
    frequencies = _channel_frequencies_mhz(temp)
    subband_indices = np.array_split(np.arange(nchan), n_subbands)

    subbands: list[AnisotropicSubbandResult] = []
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
            fit = fit_anisotropic_profile(
                profile,
                period_s=period_s,
                n_components=n_components,
                intrinsic_template=template,
            )
            isotropic = fit_tau_from_profile(
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
        tau_eff = fit.tau_eff_seconds if fit.tau_eff_seconds is not None else fit.tau_eff_bins
        tau_eff_error = (
            fit.tau_eff_seconds_error
            if fit.tau_eff_seconds_error is not None
            else fit.tau_eff_bins_error
        )
        tau_x = fit.tau_x_seconds if fit.tau_x_seconds is not None else fit.tau_x_bins
        tau_y = fit.tau_y_seconds if fit.tau_y_seconds is not None else fit.tau_y_bins
        tau_x_error = (
            fit.tau_x_seconds_error if fit.tau_x_seconds_error is not None else fit.tau_x_bins_error
        )
        tau_y_error = (
            fit.tau_y_seconds_error if fit.tau_y_seconds_error is not None else fit.tau_y_bins_error
        )
        profile_snr = _profile_fit_snr(profile, fit.model_profile)
        relative_tau_error = (
            float(tau_eff_error / tau_eff) if tau_eff > 0 else float("inf")
        )
        rejection_reasons = _tau_subband_rejection_reasons(
            float(tau_eff),
            float(tau_eff_error),
            profile_snr,
            valid_channel_fraction,
            min_profile_snr,
            max_relative_tau_error,
            min_valid_channel_fraction,
        )
        subbands.append(
            AnisotropicSubbandResult(
                frequency_mhz=float(frequency),
                central_frequency_mhz=float(central_frequency),
                bandwidth_mhz=float(bandwidth),
                tau_eff=float(tau_eff),
                tau_eff_error=float(tau_eff_error),
                tau_x=float(tau_x),
                tau_x_error=float(tau_x_error),
                tau_y=float(tau_y),
                tau_y_error=float(tau_y_error),
                anisotropy_ratio=fit.anisotropy_ratio,
                anisotropy_ratio_error=fit.anisotropy_ratio_error,
                fit=fit,
                isotropic_reduced_chi_square=isotropic.reduced_chi_square,
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
            f"only {len(accepted)} of {n_subbands} anisotropic subbands passed alpha "
            f"quality checks; at least {min_subbands} are required"
        )
    freq = np.asarray([item.frequency_mhz for item in accepted])
    tau = np.asarray([item.tau_eff for item in accepted])
    tau_error = np.asarray([item.tau_eff_error for item in accepted])
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
    return AnisotropicScatteringResult(
        n_subbands=n_subbands,
        subbands=tuple(subbands),
        failed_subbands=tuple(failed_subbands),
        alpha_fit=alpha_result,
    )


def plot_anisotropic_subband_fits(
    result: AnisotropicScatteringResult,
    title: str | None = None,
    output_path: str | None = None,
    nrows: int | None = None,
    ncols: int | None = None,
):
    """Plot anisotropic fit overlays for each fitted subband."""
    import matplotlib.pyplot as plt
    from .fit_tau import _subplot_grid, profile_model_for_plot

    subbands = sorted(result.subbands, key=lambda item: item.frequency_mhz)
    nrows, ncols = _subplot_grid(len(subbands), nrows=nrows, ncols=ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.4 * ncols, 2.8 * nrows),
        sharex=True,
        squeeze=False,
    )
    for ax, subband in zip(axes.ravel(), subbands):
        if subband.fit.model_profile is None:
            raise ValueError("anisotropic fit result has no retained model profile")
        profile, model = profile_model_for_plot(subband.profile, subband.fit.model_profile)
        phase = np.linspace(0.0, 1.0, len(profile), endpoint=False)
        ax.scatter(phase, profile, color="black", s=8, label="Profile")
        color = "tab:blue" if subband.accepted else "tab:gray"
        ax.plot(
            phase,
            model,
            color=color,
            linestyle="-" if subband.accepted else "--",
            linewidth=2,
            label="Accepted anisotropic fit" if subband.accepted else "Rejected anisotropic fit",
        )
        ax.set_ylabel("Normalized flux")
        ax.text(
            0.985,
            0.88,
            f"f_m = {subband.frequency_mhz:.2f} MHz\n"
            f"τ_eff = {subband.tau_eff * 1e3:.4g} ms\n"
            f"τ ratio = {subband.anisotropy_ratio:.3g}\n"
            f"{'accepted' if subband.accepted else 'rejected'}; S/N={subband.profile_snr:.3g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        ax.legend(loc="best", fontsize=8)

    for ax in axes.ravel()[len(subbands):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("Pulse phase")
    if title:
        fig.suptitle(Path(title).name)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig


def _anisotropic_gaussian_fit_setup(profile: np.ndarray, n_components: int):
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
    ratio_index = tau_index + 1
    p0.extend([max(nbin / 20.0, 0.5), 1.5, baseline0])
    lower.extend([0.05, 1.0, -np.inf])
    upper.extend([float(2 * nbin), 100.0, np.inf])

    def model(_bins, *params):
        components = tuple(
            GaussianComponent(params[3 * index], params[3 * index + 1], params[3 * index + 2])
            for index in range(n_components)
        )
        intrinsic = gaussian_intrinsic_profile(nbin, components)
        return anisotropic_scattered_profile(
            intrinsic,
            params[tau_index],
            params[ratio_index],
            1.0,
            params[-1],
        )

    return model, p0, lower, upper, tau_index, ratio_index


def _anisotropic_template_fit_setup(profile: np.ndarray, template: np.ndarray):
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
    ratio_index = 3
    p0 = [amplitude0, shift0, max(nbin / 20.0, 0.5), 1.5, baseline0]
    lower = [0.0, -float(nbin) / 2.0, 0.05, 1.0, -np.inf]
    upper = [np.inf, float(nbin) / 2.0, float(2 * nbin), 100.0, np.inf]

    def model(_bins, amplitude, phase_shift, tau_eff, ratio, baseline):
        intrinsic = shift_template_periodic(template, phase_shift)
        return anisotropic_scattered_profile(
            intrinsic,
            tau_eff,
            ratio,
            amplitude,
            baseline,
        )

    return model, p0, lower, upper, tau_index, ratio_index


def anisotropic_scattered_profile(
    intrinsic,
    tau_eff: float,
    anisotropy_ratio: float,
    amplitude: float,
    baseline: float,
) -> np.ndarray:
    """Convolve an arbitrary periodic intrinsic profile with the anisotropic PBF."""
    values = np.asarray(intrinsic, dtype=float).reshape(-1)
    kernel = _folded_pbf(len(values), tau_eff, anisotropy_ratio)
    return float(baseline) + float(amplitude) * periodic_convolution(values, kernel)


def _folded_pbf(nbin: int, tau_eff: float, anisotropy_ratio: float) -> np.ndarray:
    # PSRISM choices: integrate at least four periods, about ten tau_eff, and
    # cap at 64 periods. These are numerical truncation limits for the
    # train+DC wrap-around treatment, not literature constants.
    periods = int(np.clip(np.ceil(10.0 * max(tau_eff, 1e-6) / nbin) + 2, 4, 64))
    delays = np.arange(periods * nbin, dtype=float)
    pbf_long = anisotropic_pbf(delays, tau_eff, anisotropy_ratio)
    folded = pbf_long.reshape(periods, nbin).sum(axis=0)
    total = np.sum(folded)
    if total > 0:
        folded = folded / total
    return folded


def _normalise_profile(profile) -> np.ndarray:
    prof = np.asarray(profile, dtype=float)
    prof = prof - np.min(prof)
    mx = np.max(prof)
    if mx != 0:
        prof = prof / mx
    return prof


def _tau_errors_from_covariance(
    tau_eff: float,
    ratio: float,
    covariance: np.ndarray,
    tau_index: int,
    ratio_index: int,
) -> dict[str, float]:
    # PSRISM choice: first-order covariance propagation through the tau_x and
    # tau_y parameterization; the cited PBF does not prescribe this estimator.
    diag = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    tau_eff_error = float(diag[tau_index])
    ratio_error = float(diag[ratio_index])
    subcov = covariance[np.ix_([tau_index, ratio_index], [tau_index, ratio_index])]

    jac_x = np.asarray([1.0 / np.sqrt(ratio), -0.5 * tau_eff / ratio**1.5])
    jac_y = np.asarray([np.sqrt(ratio), 0.5 * tau_eff / np.sqrt(ratio)])
    tau_x_error = float(np.sqrt(max(jac_x @ subcov @ jac_x, 0.0)))
    tau_y_error = float(np.sqrt(max(jac_y @ subcov @ jac_y, 0.0)))
    return {
        "tau_eff": tau_eff_error,
        "ratio": ratio_error,
        "tau_x": tau_x_error,
        "tau_y": tau_y_error,
    }


def _scale_optional(value: float, scale: float | None) -> float | None:
    if scale is None:
        return None
    return float(value * scale)
