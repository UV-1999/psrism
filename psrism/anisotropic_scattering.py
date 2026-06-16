"""Anisotropic pulse-broadening model fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fit_alpha import AlphaFitResult, fit_alpha
from .fit_tau import fit_tau_from_profile


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


@dataclass(frozen=True)
class AnisotropicSubbandResult:
    frequency_mhz: float
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


@dataclass(frozen=True)
class AnisotropicScatteringResult:
    n_subbands: int
    subbands: tuple[AnisotropicSubbandResult, ...]
    alpha_fit: AlphaFitResult


def anisotropic_pbf(delays, tau_eff: float, anisotropy_ratio: float) -> np.ndarray:
    """Return the anisotropic pulse broadening function in delay-bin units."""
    from scipy.special import i0e

    t = np.asarray(delays, dtype=float)
    tau_eff = max(float(tau_eff), 1e-12)
    ratio = max(float(anisotropy_ratio), 1.0)
    tau_x = tau_eff / np.sqrt(ratio)
    tau_y = tau_eff * np.sqrt(ratio)

    z = 0.5 * t * (1.0 / tau_x - 1.0 / tau_y)
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
    x = np.asarray(bins, dtype=float)
    nbin = len(x)
    sigma = max(float(sigma), 1e-6)
    intrinsic = np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    pbf = _folded_pbf(nbin, tau_eff, anisotropy_ratio)
    scattered = np.fft.ifft(np.fft.fft(intrinsic) * np.fft.fft(pbf)).real
    return baseline + amplitude * scattered


def fit_anisotropic_profile(profile, period_s: float | None = None) -> AnisotropicFitResult:
    """Fit a folded anisotropic scattering model to a pulse profile."""
    from scipy.optimize import curve_fit

    prof = _normalise_profile(profile)
    nbin = len(prof)
    bins = np.arange(nbin, dtype=float)
    p0 = [
        1.0,
        float(np.argmax(prof)),
        max(nbin / 20.0, 1.0),
        max(nbin / 50.0, 1.0),
        1.5,
        0.0,
    ]
    bounds = (
        [0.0, 0.0, 1e-3, 1e-6, 1.0, -1.0],
        [np.inf, float(nbin), float(nbin), float(2 * nbin), 100.0, 1.0],
    )
    popt, pcov = curve_fit(
        anisotropic_scattered_pulse,
        bins,
        prof,
        p0=p0,
        bounds=bounds,
        maxfev=40000,
    )

    model = anisotropic_scattered_pulse(bins, *popt)
    residual = prof - model
    chi_square = float(np.sum(residual**2))
    dof = max(nbin - len(popt), 0)
    reduced_chi_square = float(chi_square / dof) if dof > 0 else None
    rms_residual = float(np.sqrt(np.mean(residual**2)))

    tau_eff = float(popt[3])
    ratio = float(popt[4])
    tau_x = tau_eff / np.sqrt(ratio)
    tau_y = tau_eff * np.sqrt(ratio)
    errors = _tau_errors_from_covariance(tau_eff, ratio, pcov)

    bin_to_s = None if period_s is None else float(period_s / nbin)
    return AnisotropicFitResult(
        amplitude=float(popt[0]),
        mu=float(popt[1]),
        sigma=float(popt[2]),
        tau_eff_bins=tau_eff,
        anisotropy_ratio=ratio,
        baseline=float(popt[5]),
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
    )


def fit_anisotropic_scattering_from_archive(
    archive,
    n_subbands: int,
    reference_freq_mhz: float | None = None,
) -> AnisotropicScatteringResult:
    """Fit anisotropic scattering in subbands and fit τ_eff frequency scaling."""
    from .fit_tau import _channel_frequencies_mhz, _integrated_subband_profile

    if n_subbands < 2:
        raise ValueError("n_subbands must be at least 2")

    temp = archive.clone()
    temp.tscrunch()
    temp.remove_baseline()

    nchan = temp.get_nchan()
    if n_subbands > nchan:
        raise ValueError("n_subbands cannot exceed the number of frequency channels")

    period_s = temp.get_Integration(0).get_folding_period()
    frequencies = _channel_frequencies_mhz(temp)
    subband_indices = np.array_split(np.arange(nchan), n_subbands)

    subbands: list[AnisotropicSubbandResult] = []
    for indices in subband_indices:
        profile, frequency = _integrated_subband_profile(temp, indices, frequencies)
        fit = fit_anisotropic_profile(profile, period_s=period_s)
        isotropic = fit_tau_from_profile(profile, period_s=period_s)
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
        subbands.append(
            AnisotropicSubbandResult(
                frequency_mhz=float(frequency),
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
            )
        )

    freq = np.asarray([item.frequency_mhz for item in subbands])
    tau = np.asarray([item.tau_eff for item in subbands])
    tau_error = np.asarray([item.tau_eff_error for item in subbands])
    reference = float(reference_freq_mhz if reference_freq_mhz is not None else np.median(freq))
    alpha_result = fit_alpha(freq, tau, tau_error=tau_error, reference_freq_mhz=reference)
    return AnisotropicScatteringResult(
        n_subbands=n_subbands,
        subbands=tuple(subbands),
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
    from .fit_tau import _subplot_grid

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
        profile = _normalise_profile(subband.profile)
        phase = np.linspace(0.0, 1.0, len(profile), endpoint=False)
        bins = np.arange(len(profile))
        model = anisotropic_scattered_pulse(
            bins,
            subband.fit.amplitude,
            subband.fit.mu,
            subband.fit.sigma,
            subband.fit.tau_eff_bins,
            subband.fit.anisotropy_ratio,
            subband.fit.baseline,
        )
        model = _normalise_profile(model)
        ax.plot(phase, profile, color="black", label="Profile")
        ax.plot(phase, model, color="tab:blue", linewidth=2, label="Anisotropic fit")
        ax.set_ylabel("Normalized flux")
        ax.text(
            0.985,
            0.88,
            f"{subband.frequency_mhz:.2f} MHz\n"
            f"τ_eff = {subband.tau_eff * 1e3:.4g} ms\n"
            f"τ ratio = {subband.anisotropy_ratio:.3g}",
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


def _folded_pbf(nbin: int, tau_eff: float, anisotropy_ratio: float) -> np.ndarray:
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


def _tau_errors_from_covariance(tau_eff: float, ratio: float, covariance: np.ndarray) -> dict[str, float]:
    diag = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    tau_eff_error = float(diag[3])
    ratio_error = float(diag[4])
    subcov = covariance[np.ix_([3, 4], [3, 4])]

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
