"""Data-quality assessment and RFI masking for dynamic spectra."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import warnings

import numpy as np


@dataclass(frozen=True)
class DynamicSpectrumQuality:
    valid_mask: np.ndarray
    shape: tuple[int, int]
    sigma_threshold: float
    min_valid_fraction: float
    statistical_masking: bool
    input_invalid_cells: int
    bad_time_bins: tuple[int, ...]
    bad_frequency_channels: tuple[int, ...]
    isolated_flagged_cells: int

    @property
    def total_cells(self) -> int:
        return int(np.prod(self.shape))

    @property
    def valid_cells(self) -> int:
        return int(np.count_nonzero(self.valid_mask))

    @property
    def masked_cells(self) -> int:
        return self.total_cells - self.valid_cells

    @property
    def masked_fraction(self) -> float:
        if self.total_cells == 0:
            return 1.0
        return self.masked_cells / self.total_cells


def assess_dynamic_spectrum(
    dynspec: np.ndarray,
    sigma_threshold: float = 6.0,
    min_valid_fraction: float = 0.5,
    detect_rfi: bool = False,
) -> DynamicSpectrumQuality:
    """Build a valid-sample mask and optionally flag robust statistical outliers.

    Persistent narrow-band contamination is detected from channel level and
    spread, impulsive broad-band contamination from time-bin level and spread,
    and isolated contamination from residuals after subtracting robust time and
    frequency trends.
    """
    # Literature motivation: the Filothodoros thesis (ALEX.pdf), PDF p. 29,
    # Section 2.6.1, describes median/spike RFI excision plus manual review;
    # Cai et al. (scintillationJ0814.pdf), PDF p. 2, also use iterative cleaning
    # and visual inspection. This robust median/MAD detector is PSRISM's own
    # implementation; 6 sigma and 50% occupancy are project defaults.
    arr = np.asarray(dynspec, dtype=float)
    if arr.ndim != 2 or 0 in arr.shape:
        raise ValueError("dynspec must be a non-empty 2D array shaped as (time, frequency)")
    if not np.isfinite(sigma_threshold) or sigma_threshold <= 0:
        raise ValueError("sigma_threshold must be positive")
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in the interval (0, 1]")

    input_valid = np.isfinite(arr)
    time_occupancy = np.mean(input_valid, axis=1)
    frequency_occupancy = np.mean(input_valid, axis=0)
    bad_time = time_occupancy < min_valid_fraction
    bad_frequency = frequency_occupancy < min_valid_fraction

    isolated_bad = np.zeros(arr.shape, dtype=bool)
    if detect_rfi:
        analysis_valid = input_valid & ~bad_time[:, np.newaxis] & ~bad_frequency[np.newaxis, :]
        bad_time |= _axis_outliers(arr, analysis_valid, axis=1, threshold=sigma_threshold)
        bad_frequency |= _axis_outliers(arr, analysis_valid, axis=0, threshold=sigma_threshold)

        analysis_valid = input_valid & ~bad_time[:, np.newaxis] & ~bad_frequency[np.newaxis, :]
        isolated_bad = _residual_outliers(arr, analysis_valid, sigma_threshold)

    valid = (
        input_valid
        & ~bad_time[:, np.newaxis]
        & ~bad_frequency[np.newaxis, :]
        & ~isolated_bad
    )
    return DynamicSpectrumQuality(
        valid_mask=valid,
        shape=(int(arr.shape[0]), int(arr.shape[1])),
        sigma_threshold=float(sigma_threshold),
        min_valid_fraction=float(min_valid_fraction),
        statistical_masking=bool(detect_rfi),
        input_invalid_cells=int(np.count_nonzero(~input_valid)),
        bad_time_bins=tuple(int(value) for value in np.flatnonzero(bad_time)),
        bad_frequency_channels=tuple(int(value) for value in np.flatnonzero(bad_frequency)),
        isolated_flagged_cells=int(np.count_nonzero(isolated_bad & analysis_valid)) if detect_rfi else 0,
    )


def apply_quality_mask_to_archive(archive, valid_mask: np.ndarray) -> int:
    """Set PSRCHIVE weights to zero where ``valid_mask`` is false."""
    mask = np.asarray(valid_mask, dtype=bool)
    expected = (archive.get_nsubint(), archive.get_nchan())
    if mask.shape != expected:
        raise ValueError(f"valid_mask shape {mask.shape} does not match archive shape {expected}")

    newly_masked = 0
    for isub in range(expected[0]):
        integration = archive.get_Integration(isub)
        for ichan in np.flatnonzero(~mask[isub]):
            if integration.get_weight(int(ichan)) > 0:
                newly_masked += 1
            integration.set_weight(int(ichan), 0.0)
    return newly_masked


def write_quality_report(report: DynamicSpectrumQuality, output_path: str | Path) -> None:
    """Write compact, reproducible quality diagnostics as JSON."""
    payload = {
        "shape": list(report.shape),
        "statistical_masking": report.statistical_masking,
        "sigma_threshold": report.sigma_threshold,
        "min_valid_fraction": report.min_valid_fraction,
        "input_invalid_cells": report.input_invalid_cells,
        "bad_time_bins": list(report.bad_time_bins),
        "bad_frequency_channels": list(report.bad_frequency_channels),
        "isolated_flagged_cells": report.isolated_flagged_cells,
        "valid_cells": report.valid_cells,
        "masked_cells": report.masked_cells,
        "masked_fraction": report.masked_fraction,
    }
    with Path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _axis_outliers(
    arr: np.ndarray,
    valid: np.ndarray,
    axis: int,
    threshold: float,
) -> np.ndarray:
    masked = np.where(valid, arr, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        level = np.nanmedian(masked, axis=axis)
        if axis == 1:
            deviations = np.abs(masked - level[:, np.newaxis])
        else:
            deviations = np.abs(masked - level[np.newaxis, :])
        # PSRISM choice: 1.4826 makes MAD comparable to Gaussian sigma.
        spread = 1.4826 * np.nanmedian(deviations, axis=axis)
    return _robust_outliers(level, threshold) | _robust_outliers(spread, threshold)


def _residual_outliers(arr: np.ndarray, valid: np.ndarray, threshold: float) -> np.ndarray:
    if not np.any(valid):
        return np.zeros(arr.shape, dtype=bool)
    masked = np.where(valid, arr, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        global_level = float(np.nanmedian(masked))
        time_level = np.nanmedian(masked, axis=1)
        frequency_level = np.nanmedian(masked, axis=0)
    model = time_level[:, np.newaxis] + frequency_level[np.newaxis, :] - global_level
    residual = arr - model
    values = residual[valid]
    center = float(np.nanmedian(values))
    scale = 1.4826 * float(np.nanmedian(np.abs(values - center)))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        return np.zeros(arr.shape, dtype=bool)
    return valid & (np.abs(residual - center) > threshold * scale)


def _robust_outliers(values: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    result = ~finite
    if np.count_nonzero(finite) < 3:
        return result
    center = float(np.median(values[finite]))
    deviations = np.abs(values[finite] - center)
    scale = 1.4826 * float(np.median(deviations))
    if scale > np.finfo(float).eps:
        result[finite] = deviations > threshold * scale
    elif np.count_nonzero(deviations <= np.finfo(float).eps * max(abs(center), 1.0)) >= 3:
        result[finite] = deviations > np.finfo(float).eps * max(abs(center), 1.0)
    return result
