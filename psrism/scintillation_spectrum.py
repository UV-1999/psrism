"""Secondary/scintillation spectrum calculation."""

from __future__ import annotations

import numpy as np


def calculate_scintillation_spectrum(
    dynspec: np.ndarray,
    observation_time_s: float,
    bandwidth_mhz: float,
    log_scale: bool = True,
):
    """Calculate the secondary spectrum and its conjugate axes."""
    arr = np.asarray(dynspec, dtype=float)
    arr = arr - np.mean(arr)
    nsub, nchan = arr.shape
    dt = observation_time_s / nsub
    df_hz = abs(bandwidth_mhz) * 1e6 / nchan

    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(arr)) ** 2)
    if log_scale:
        spectrum = 10 * np.log10(spectrum + 1e-12)

    fringe_frequency = np.fft.fftshift(np.fft.fftfreq(nsub, d=dt))
    delay = np.fft.fftshift(np.fft.fftfreq(nchan, d=df_hz))
    spectrum[nsub // 2, :] = 0
    return spectrum, fringe_frequency, delay
