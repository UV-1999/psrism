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
    arr = np.asarray(dynspec, dtype=float).copy()
    arr -= np.mean(arr, axis=0, keepdims=True)
    std = np.std(arr)
    if std != 0:
        arr /= std
    return arr
