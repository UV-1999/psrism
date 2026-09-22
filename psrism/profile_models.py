"""Shared periodic intrinsic-profile and pulse-broadening model helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GaussianComponent:
    amplitude: float
    mu: float
    sigma: float


def prepare_observed_profile(profile) -> np.ndarray:
    """Scale a finite observed profile without removing its fitted DC level."""
    values = np.asarray(profile, dtype=float).reshape(-1)
    if values.size < 8:
        raise ValueError("profile must contain at least eight phase bins")
    if not np.all(np.isfinite(values)):
        raise ValueError("profile contains non-finite samples")
    span = float(np.ptp(values))
    if span <= np.finfo(float).eps:
        raise ValueError("profile has no measurable variation")
    # PSRISM numerical conditioning: centering by the profile median is an
    # invertible constant shift because DC remains a free model parameter. It
    # is not PSRCHIVE off-pulse baseline removal.
    return (values - np.median(values)) / span


def gaussian_intrinsic_profile(
    nbin: int,
    components: tuple[GaussianComponent, ...] | list[GaussianComponent],
) -> np.ndarray:
    """Evaluate a periodic sum of Gaussian intrinsic components."""
    # Reference: Filothodoros et al. (Alex_and_Boe.pdf), PDF p. 4, describes
    # one Gaussian per intrinsic profile component, all convolved with the
    # same PBF. The periodic-distance evaluation is a PSRISM implementation.
    bins = np.arange(nbin, dtype=float)
    intrinsic = np.zeros(nbin, dtype=float)
    for component in components:
        sigma = max(float(component.sigma), 1e-12)
        # A pulse train makes phase periodic. The wrapped distance chooses the
        # nearest copy of each Gaussian component in the train.
        distance = (bins - float(component.mu) + nbin / 2.0) % nbin - nbin / 2.0
        intrinsic += float(component.amplitude) * np.exp(-0.5 * (distance / sigma) ** 2)
    return intrinsic


def prepare_intrinsic_template(template, target_nbin: int | None = None) -> np.ndarray:
    """Baseline-remove, normalize and optionally resample a periodic template."""
    # Reference: Filothodoros et al. (Alex_and_Boe.pdf), PDF p. 4, describes
    # using a high-frequency, negligibly scattered profile as the template.
    # Normalization and periodic resampling here are PSRISM choices.
    values = np.asarray(template, dtype=float).reshape(-1)
    if values.size < 8 or not np.all(np.isfinite(values)):
        raise ValueError("intrinsic template must contain at least eight finite samples")
    values = values - np.min(values)
    peak = float(np.max(values))
    if peak <= np.finfo(float).eps:
        raise ValueError("intrinsic template has no measurable pulse structure")
    values = values / peak
    if target_nbin is not None and len(values) != int(target_nbin):
        values = periodic_resample(values, int(target_nbin))
    return values


def load_intrinsic_template(path: str | Path, target_nbin: int | None = None) -> np.ndarray:
    """Load an intrinsic template from NPY/text or a PSRCHIVE-readable archive."""
    template_path = Path(path)
    if not template_path.exists():
        raise ValueError(f"intrinsic template does not exist: {template_path}")

    if template_path.suffix.lower() == ".npy":
        values = np.load(template_path, allow_pickle=False)
    elif template_path.suffix.lower() in {".txt", ".dat", ".csv"}:
        delimiter = "," if template_path.suffix.lower() == ".csv" else None
        values = np.loadtxt(template_path, delimiter=delimiter)
    else:
        # Archive loading remains lazy so profile-only tests do not require the
        # separately installed PSRCHIVE Python module.
        from .archive_io import integrated_profile_array, load_archive

        archive = load_archive(str(template_path))
        values = integrated_profile_array(archive)
    return prepare_intrinsic_template(values, target_nbin=target_nbin)


def resolve_intrinsic_template(template, target_nbin: int) -> np.ndarray | None:
    """Resolve an optional in-memory or path-based intrinsic template."""
    if template is None:
        return None
    if isinstance(template, (str, Path)):
        return load_intrinsic_template(template, target_nbin=target_nbin)
    return prepare_intrinsic_template(template, target_nbin=target_nbin)


def tau_seed_grid(nbin: int) -> tuple[float, ...]:
    """Return multistart tau seeds spanning weak through severe scattering."""
    # PSRISM numerical choices, not literature constants.
    return tuple(max(float(nbin) * fraction, 0.1) for fraction in (0.005, 0.02, 0.08, 0.3, 0.8))


def periodic_resample(values, target_nbin: int) -> np.ndarray:
    """Resample a one-rotation profile on a periodic phase grid."""
    source = np.asarray(values, dtype=float).reshape(-1)
    if target_nbin < 8:
        raise ValueError("target_nbin must be at least eight")
    old_phase = np.arange(len(source), dtype=float) / len(source)
    new_phase = np.arange(target_nbin, dtype=float) / target_nbin
    return np.interp(new_phase, old_phase, source, period=1.0)


def shift_template_periodic(template, shift_bins: float) -> np.ndarray:
    """Shift a periodic template by a fractional number of phase bins."""
    values = np.asarray(template, dtype=float).reshape(-1)
    bins = np.arange(len(values), dtype=float)
    source_positions = (bins - float(shift_bins)) % len(values)
    return np.interp(source_positions, bins, values, period=len(values))


def folded_isotropic_pbf(nbin: int, tau_bins: float) -> np.ndarray:
    """Return the one-rotation PBF produced by an infinite exponential train."""
    tau = max(float(tau_bins), 1e-12)
    delays = np.arange(nbin, dtype=float)
    # Reference: Geyer & Karastergiou (fitting.pdf), PDF p. 3, Eq. 1, gives
    # the causal exponential PBF. Their train+DC method on PDF pp. 7-8,
    # Section 2.4, accounts for tails from preceding rotations. Folding the
    # infinite geometric train gives this normalized one-period kernel.
    kernel = np.exp(-delays / tau)
    total = float(np.sum(kernel))
    if total <= 0 or not np.isfinite(total):
        raise ValueError("could not normalize isotropic pulse-broadening function")
    return kernel / total


def periodic_convolution(intrinsic, kernel) -> np.ndarray:
    """Circularly convolve one intrinsic rotation with a folded PBF."""
    intrinsic = np.asarray(intrinsic, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    if intrinsic.shape != kernel.shape or intrinsic.ndim != 1:
        raise ValueError("intrinsic profile and PBF must be equal-length 1D arrays")
    return np.fft.ifft(np.fft.fft(intrinsic) * np.fft.fft(kernel)).real


def isotropic_scattered_profile(
    intrinsic,
    tau_bins: float,
    amplitude: float = 1.0,
    baseline: float = 0.0,
) -> np.ndarray:
    """Convolve one periodic intrinsic profile with the isotropic train PBF."""
    values = np.asarray(intrinsic, dtype=float).reshape(-1)
    kernel = folded_isotropic_pbf(len(values), tau_bins)
    return float(baseline) + float(amplitude) * periodic_convolution(values, kernel)
