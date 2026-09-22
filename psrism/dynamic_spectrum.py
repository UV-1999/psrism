"""Dynamic spectrum construction."""

from __future__ import annotations

import numpy as np

from .dynamic_flux import (
    fluxes_single_peak,
    fluxes_two_peak,
    interpulse_windows,
    single_pulse_window,
)


def calculate_dynamic_spectrum(
    archive,
    width_factor: int = 10,
    interpulse: bool = False,
    normalize: bool = False,
) -> np.ndarray:
    """Build a dynamic spectrum shaped as (subintegration, channel)."""
    if interpulse:
        windows = interpulse_windows(archive, width_factor)
        dynspec = fluxes_two_peak(archive, *windows)
    else:
        window = single_pulse_window(archive, width_factor)
        dynspec = fluxes_single_peak(archive, *window)

    if normalize:
        dynspec = normalize_dynamic_spectrum(dynspec)
    return dynspec


def normalize_dynamic_spectrum(dynspec: np.ndarray) -> np.ndarray:
    """Remove mean bandpass and normalize by global standard deviation."""
    # PSRISM choice: this optional normalization is an implementation
    # transform, not the dynamic-spectrum definition in Handbook 7.4.4.1.
    arr = np.asarray(dynspec, dtype=float).copy()
    if not np.any(np.isfinite(arr)):
        raise ValueError("dynspec has no finite samples to normalize")
    finite = np.isfinite(arr)
    count = np.sum(finite, axis=0)
    total = np.sum(np.where(finite, arr, 0.0), axis=0)
    bandpass = np.full(arr.shape[1], np.nan, dtype=float)
    np.divide(total, count, out=bandpass, where=count > 0)
    arr -= bandpass[np.newaxis, :]
    std = np.nanstd(arr)
    if np.isfinite(std) and std > 0:
        arr /= std
    return arr
