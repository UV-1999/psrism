"""Secondary/scintillation spectrum calculation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SecondarySpectrumMetadata:
    n_time_bins: int
    n_frequency_channels: int
    observation_duration_s: float
    bandwidth_mhz: float
    centre_frequency_mhz: float | None
    time_resolution_s: float
    frequency_resolution_mhz: float
    fringe_frequency_resolution_hz: float
    delay_resolution_s: float
    fringe_frequency_nyquist_hz: float
    delay_nyquist_s: float
    window: str
    masked_fraction: float


def calculate_scintillation_spectrum(
    dynspec: np.ndarray,
    observation_time_s: float,
    bandwidth_mhz: float,
    log_scale: bool = True,
    valid_mask: np.ndarray | None = None,
    window: str = "hann",
):
    """Calculate the secondary spectrum and its conjugate axes.

    Invalid or masked samples are replaced by zero after subtraction of the
    valid-sample mean, preventing NaNs and rejected samples from entering the
    Fourier transform.
    """
    arr = np.asarray(dynspec, dtype=float)
    if arr.ndim != 2 or 0 in arr.shape:
        raise ValueError("dynspec must be a non-empty 2D array")
    mask = np.isfinite(arr)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != arr.shape:
            raise ValueError("valid_mask must have the same shape as dynspec")
        mask &= supplied
    if not np.any(mask):
        raise ValueError("dynspec has no finite valid samples for secondary-spectrum calculation")
    if not np.isfinite(observation_time_s) or observation_time_s <= 0:
        raise ValueError("observation_time_s must be positive")
    if not np.isfinite(bandwidth_mhz) or bandwidth_mhz == 0:
        raise ValueError("bandwidth_mhz must be finite and nonzero")
    window = window.lower()
    if window not in {"none", "hann"}:
        raise ValueError("window must be 'none' or 'hann'")

    # Reference: Lorimer & Kramer (2005), psrhandbook.pdf, Section 7.4.4.3,
    # defines the secondary spectrum as the 2D Fourier transform of the
    # dynamic spectrum in conjugate frequency and conjugate time.
    centered = np.zeros_like(arr, dtype=float)
    centered[mask] = arr[mask] - np.mean(arr[mask])
    nsub, nchan = arr.shape
    if window == "hann":
        # PSRISM choice: a separable Hann taper reduces FFT edge leakage. The
        # RMS normalization preserves the approximate scale of the linear
        # power; axes shorter than three samples remain untapered. The bundled
        # references define the transform but not these tapering details.
        time_taper = np.hanning(nsub) if nsub >= 3 else np.ones(nsub)
        frequency_taper = np.hanning(nchan) if nchan >= 3 else np.ones(nchan)
        taper = np.outer(time_taper, frequency_taper)
        taper_rms = float(np.sqrt(np.mean(taper[mask] ** 2)))
        if taper_rms <= 0:
            raise ValueError("Hann window has no nonzero valid samples")
        centered *= taper / taper_rms
    dt = observation_time_s / nsub
    df_hz = abs(bandwidth_mhz) * 1e6 / nchan

    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(centered)) ** 2)
    if log_scale:
        # PSRISM display choice: dB scaling and the 1e-12 floor are numerical
        # presentation choices, not quantities specified by the reference.
        spectrum = 10 * np.log10(spectrum + 1e-12)

    fringe_frequency = np.fft.fftshift(np.fft.fftfreq(nsub, d=dt))
    delay = np.fft.fftshift(np.fft.fftfreq(nchan, d=df_hz))
    return spectrum, fringe_frequency, delay


def secondary_spectrum_metadata(
    dynspec: np.ndarray,
    observation_time_s: float,
    bandwidth_mhz: float,
    valid_mask: np.ndarray | None = None,
    window: str = "hann",
    centre_frequency_mhz: float | None = None,
) -> SecondarySpectrumMetadata:
    """Describe sampling and Fourier resolution for a secondary spectrum."""
    arr = np.asarray(dynspec, dtype=float)
    if arr.ndim != 2 or 0 in arr.shape:
        raise ValueError("dynspec must be a non-empty 2D array")
    if not np.isfinite(observation_time_s) or observation_time_s <= 0:
        raise ValueError("observation_time_s must be positive")
    if not np.isfinite(bandwidth_mhz) or bandwidth_mhz == 0:
        raise ValueError("bandwidth_mhz must be finite and nonzero")
    if centre_frequency_mhz is not None and (
        not np.isfinite(centre_frequency_mhz) or centre_frequency_mhz <= 0
    ):
        raise ValueError("centre_frequency_mhz must be positive when supplied")
    window = window.lower()
    if window not in {"none", "hann"}:
        raise ValueError("window must be 'none' or 'hann'")

    mask = np.isfinite(arr)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != arr.shape:
            raise ValueError("valid_mask must have the same shape as dynspec")
        mask &= supplied
    if not np.any(mask):
        raise ValueError("dynspec has no finite valid samples for secondary-spectrum metadata")

    nsub, nchan = arr.shape
    duration = float(observation_time_s)
    bandwidth = abs(float(bandwidth_mhz))
    dt = duration / nsub
    df_mhz = bandwidth / nchan
    # PSRISM sampling diagnostics from the discrete Fourier grid. The
    # secondary-spectrum definition itself is Lorimer & Kramer (2005),
    # psrhandbook.pdf, Section 7.4.4.3.
    return SecondarySpectrumMetadata(
        n_time_bins=int(nsub),
        n_frequency_channels=int(nchan),
        observation_duration_s=duration,
        bandwidth_mhz=bandwidth,
        centre_frequency_mhz=(
            None if centre_frequency_mhz is None else float(centre_frequency_mhz)
        ),
        time_resolution_s=float(dt),
        frequency_resolution_mhz=float(df_mhz),
        fringe_frequency_resolution_hz=float(1.0 / duration),
        delay_resolution_s=float(1.0 / (bandwidth * 1e6)),
        fringe_frequency_nyquist_hz=float(1.0 / (2.0 * dt)),
        delay_nyquist_s=float(1.0 / (2.0 * df_mhz * 1e6)),
        window=window,
        masked_fraction=float(1.0 - np.count_nonzero(mask) / mask.size),
    )
