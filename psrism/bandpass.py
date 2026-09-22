"""Bandpass scaling for dynamic spectra."""

from __future__ import annotations

import numpy as np


def bandpass_from_dynamic_spectrum(dynspec: np.ndarray) -> np.ndarray:
    """Estimate pulsar power-frequency bandpass from a dynamic spectrum."""
    # PSRISM choice: estimate the bandpass by the time-axis arithmetic mean.
    return np.mean(np.asarray(dynspec, dtype=float), axis=0)


def scale_dynamic_spectrum_by_bandpass(
    dynspec: np.ndarray,
    bandpass: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Divide each frequency channel by its bandpass value."""
    arr = np.asarray(dynspec, dtype=float)
    bp = bandpass_from_dynamic_spectrum(arr) if bandpass is None else np.asarray(bandpass, dtype=float)
    # PSRISM choice: eps is only a divide-by-near-zero safeguard.
    safe_bp = np.where(np.abs(bp) > eps, bp, 1.0)
    return arr / safe_bp[np.newaxis, :]
