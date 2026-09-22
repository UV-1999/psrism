"""Parabolic-arc fitting for secondary spectra."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ParabolicArcFitResult:
    curvature: float
    curvature_error: float | None
    curvature_lower: float | None
    curvature_upper: float | None
    accepted_curvature: float | None
    accepted: bool
    rejection_reasons: tuple[str, ...]
    fringe_offset: float
    delay_offset: float
    score: float
    score_snr: float | None
    score_baseline: float | None
    score_noise: float | None
    half: str
    n_samples: int
    minimum_samples: int
    minimum_score_snr: float
    mask_bins: int
    peak_at_search_boundary: bool
    trial_curvatures: np.ndarray
    arc_strength: np.ndarray


def fit_parabolic_arc(
    spectrum,
    fringe_frequency,
    delay,
    curvature_min: float | None = None,
    curvature_max: float | None = None,
    n_trials: int = 200,
    half: str = "positive",
    mask_bins: int = 2,
    fringe_offset: float = 0.0,
    delay_offset: float = 0.0,
    minimum_samples: int = 20,
    minimum_score_snr: float = 5.0,
) -> ParabolicArcFitResult:
    """Fit ``delay - delay0 = eta * (fringe_frequency - fringe0)**2``.

    The input spectrum should be linear secondary-spectrum power, not dB power.
    Curvature has units of seconds cubed when fringe frequency is in Hz and delay
    is in seconds. ``positive`` searches positive delay, ``negative`` searches
    negative delay and returns negative curvature, and ``both`` searches both
    halves while reporting the positive curvature magnitude.
    """
    # Reference: Lorimer & Kramer (2005), psrhandbook.pdf, Section 7.4.4.3,
    # Eq. 7.45, gives the parabolic secondary-spectrum arc relation and its
    # dependence on screen location, wavelength and scintillation velocity.
    power = np.asarray(spectrum, dtype=float)
    fringe_frequency = np.asarray(fringe_frequency, dtype=float)
    delay = np.asarray(delay, dtype=float)
    half = half.lower()
    if half not in {"positive", "negative", "both"}:
        raise ValueError("half must be 'positive', 'negative', or 'both'")
    if power.shape != (len(fringe_frequency), len(delay)):
        raise ValueError("spectrum shape must match fringe_frequency and delay axes")
    if n_trials < 5:
        raise ValueError("n_trials must be at least 5")
    if mask_bins < 0:
        raise ValueError("mask_bins cannot be negative")
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be at least 1")
    if not np.isfinite(minimum_score_snr) or minimum_score_snr <= 0:
        raise ValueError("minimum_score_snr must be positive")
    if not np.isfinite(fringe_offset) or not np.isfinite(delay_offset):
        raise ValueError("arc apex offsets must be finite")
    for name, value in (
        ("curvature_min", curvature_min),
        ("curvature_max", curvature_max),
    ):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be positive")

    delay_order = np.argsort(delay)
    delay = delay[delay_order]
    power = power[:, delay_order]
    power = np.where(np.isfinite(power), power, 0.0)

    fringe_mask = mask_bins * _axis_step(fringe_frequency)
    delay_mask = mask_bins * _axis_step(delay)
    eta_abs = _trial_curvatures(
        fringe_frequency,
        delay,
        fringe_mask,
        delay_mask,
        fringe_offset,
        delay_offset,
        curvature_min,
        curvature_max,
        n_trials,
    )

    # PSRISM choice: score trial parabolas by interpolated mean linear power.
    # The bundled references motivate parabolic arcs but do not prescribe this
    # search statistic, grid, central mask or uncertainty estimator.
    scores = np.full_like(eta_abs, np.nan, dtype=float)
    sample_counts = np.zeros_like(eta_abs, dtype=int)
    for idx, eta in enumerate(eta_abs):
        values = []
        if half in {"positive", "both"}:
            branch, count = _sample_parabola(
                power,
                fringe_frequency,
                delay,
                eta,
                fringe_mask,
                delay_mask,
                fringe_offset,
                delay_offset,
            )
            if count:
                values.append(branch)
                sample_counts[idx] += count
        if half in {"negative", "both"}:
            branch, count = _sample_parabola(
                power,
                fringe_frequency,
                delay,
                -eta,
                fringe_mask,
                delay_mask,
                fringe_offset,
                delay_offset,
            )
            if count:
                values.append(branch)
                sample_counts[idx] += count
        if values:
            scores[idx] = float(np.mean(np.concatenate(values)))

    if not np.any(np.isfinite(scores)):
        raise ValueError("no valid parabolic arc samples were found")

    best_idx = int(np.nanargmax(scores))
    best_abs = float(eta_abs[best_idx])
    curvature = -best_abs if half == "negative" else best_abs
    lower_abs, upper_abs = _curvature_interval(eta_abs, scores, best_idx)
    if half == "negative":
        curvature_lower = None if upper_abs is None else -upper_abs
        curvature_upper = None if lower_abs is None else -lower_abs
    else:
        curvature_lower = lower_abs
        curvature_upper = upper_abs
    error = (
        None
        if curvature_lower is None or curvature_upper is None
        else 0.5 * (curvature_upper - curvature_lower)
    )

    score_baseline, score_noise, score_snr = _score_diagnostics(scores, best_idx)
    peak_at_boundary = best_idx in {0, len(eta_abs) - 1}
    rejection_reasons = []
    # PSRISM quality choices: a publishable curvature must have a closed
    # half-height search interval, avoid a trial-grid edge, sample at least 20
    # arc points, and exceed a robust score-curve S/N of five by default. The
    # bundled references establish arc physics, not these acceptance cuts.
    if peak_at_boundary:
        rejection_reasons.append("peak_at_search_boundary")
    if curvature_lower is None or curvature_upper is None:
        rejection_reasons.append("curvature_interval_not_closed")
    if sample_counts[best_idx] < minimum_samples:
        rejection_reasons.append("insufficient_arc_samples")
    if score_snr is None:
        rejection_reasons.append("score_snr_unavailable")
    elif score_snr < minimum_score_snr:
        rejection_reasons.append("score_snr_below_threshold")
    accepted = not rejection_reasons
    trial_curvatures = -eta_abs if half == "negative" else eta_abs
    return ParabolicArcFitResult(
        curvature=curvature,
        curvature_error=error,
        curvature_lower=curvature_lower,
        curvature_upper=curvature_upper,
        accepted_curvature=curvature if accepted else None,
        accepted=accepted,
        rejection_reasons=tuple(rejection_reasons),
        fringe_offset=float(fringe_offset),
        delay_offset=float(delay_offset),
        score=float(scores[best_idx]),
        score_snr=score_snr,
        score_baseline=score_baseline,
        score_noise=score_noise,
        half=half,
        n_samples=int(sample_counts[best_idx]),
        minimum_samples=int(minimum_samples),
        minimum_score_snr=float(minimum_score_snr),
        mask_bins=int(mask_bins),
        peak_at_search_boundary=peak_at_boundary,
        trial_curvatures=trial_curvatures,
        arc_strength=scores,
    )


def write_arc_fit_report(
    result: ParabolicArcFitResult,
    output_path: str | Path,
    secondary_metadata=None,
) -> None:
    """Write the raw and accepted arc fit, search curve, and FFT metadata."""
    payload = asdict(result)
    payload["trial_curvatures"] = result.trial_curvatures.tolist()
    payload["arc_strength"] = [
        None if not np.isfinite(value) else float(value) for value in result.arc_strength
    ]
    payload["secondary_spectrum"] = (
        None if secondary_metadata is None else asdict(secondary_metadata)
    )
    with Path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _trial_curvatures(
    fringe_frequency: np.ndarray,
    delay: np.ndarray,
    fringe_mask: float,
    delay_mask: float,
    fringe_offset: float,
    delay_offset: float,
    curvature_min: float | None,
    curvature_max: float | None,
    n_trials: int,
) -> np.ndarray:
    useful_fringe = np.abs((fringe_frequency - fringe_offset)[np.abs(fringe_frequency) > fringe_mask])
    useful_delay = np.abs((delay - delay_offset)[np.abs(delay) > delay_mask])
    if useful_fringe.size == 0 or useful_delay.size == 0:
        raise ValueError("not enough unmasked secondary-spectrum axis samples")

    max_fringe = float(np.nanmax(useful_fringe))
    min_fringe = float(np.nanmin(useful_fringe))
    min_delay = max(float(np.nanmin(useful_delay)), _axis_step(delay))
    max_delay = float(np.nanmax(useful_delay))
    eta_min = abs(curvature_min) if curvature_min is not None else min_delay / max_fringe**2
    # PSRISM search-grid choice: the shallowest resolvable arc reaches the
    # first delay bin at the largest fringe magnitude, while the steepest
    # reaches the largest delay at the first unmasked fringe bin.
    eta_max = abs(curvature_max) if curvature_max is not None else max_delay / min_fringe**2
    if eta_min <= 0 or eta_max <= 0:
        raise ValueError("curvature bounds must be positive in magnitude")
    if eta_min >= eta_max:
        eta_min, eta_max = eta_max, eta_min
    if eta_min == eta_max:
        eta_max = eta_min * 10.0
    return np.logspace(np.log10(eta_min), np.log10(eta_max), n_trials)


def _sample_parabola(
    power: np.ndarray,
    fringe_frequency: np.ndarray,
    delay: np.ndarray,
    curvature: float,
    fringe_mask: float,
    delay_mask: float,
    fringe_offset: float,
    delay_offset: float,
) -> tuple[np.ndarray, int]:
    target_delay = delay_offset + curvature * (fringe_frequency - fringe_offset) ** 2
    valid = (
        (np.abs(fringe_frequency) > fringe_mask)
        & (np.abs(target_delay) > delay_mask)
        & (target_delay >= delay[0])
        & (target_delay <= delay[-1])
    )
    if np.count_nonzero(valid) == 0:
        return np.asarray([], dtype=float), 0

    row_idx = np.where(valid)[0]
    y = target_delay[valid]
    delay_step = _axis_step(delay)
    pos = (y - delay[0]) / delay_step
    left = np.floor(pos).astype(int)
    right = np.clip(left + 1, 0, len(delay) - 1)
    left = np.clip(left, 0, len(delay) - 1)
    weight = np.clip(pos - left, 0.0, 1.0)
    samples = (1.0 - weight) * power[row_idx, left] + weight * power[row_idx, right]
    return samples[np.isfinite(samples)], int(np.count_nonzero(np.isfinite(samples)))


def _curvature_interval(
    curvatures: np.ndarray,
    scores: np.ndarray,
    best_idx: int,
) -> tuple[float | None, float | None]:
    finite = np.isfinite(scores)
    if np.count_nonzero(finite) < 5:
        return None, None
    baseline = float(np.nanmedian(scores))
    peak = float(scores[best_idx])
    if not np.isfinite(peak) or peak <= baseline:
        return None, None
    # PSRISM choice: use the half-height width above the median score as a
    # descriptive search-width uncertainty; it is not a confidence interval.
    threshold = baseline + 0.5 * (peak - baseline)
    left = _threshold_crossing(curvatures, scores, best_idx, threshold, direction=-1)
    right = _threshold_crossing(curvatures, scores, best_idx, threshold, direction=1)
    return left, right


def _threshold_crossing(
    curvatures: np.ndarray,
    scores: np.ndarray,
    best_idx: int,
    threshold: float,
    direction: int,
) -> float | None:
    idx = best_idx
    while 0 <= idx + direction < len(scores):
        nxt = idx + direction
        if np.isfinite(scores[nxt]) and scores[nxt] <= threshold:
            x0, x1 = curvatures[idx], curvatures[nxt]
            y0, y1 = scores[idx], scores[nxt]
            if y1 == y0:
                return float(x1)
            frac = (threshold - y0) / (y1 - y0)
            return float(x0 + frac * (x1 - x0))
        idx = nxt
    return None


def _score_diagnostics(
    scores: np.ndarray,
    best_idx: int,
) -> tuple[float | None, float | None, float | None]:
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size < 5:
        return None, None, None
    baseline = float(np.median(finite_scores))
    mad = float(np.median(np.abs(finite_scores - baseline)))
    # PSRISM choice: robustly standardize the correlated search curve with
    # the Gaussian-consistent MAD factor, falling back to its standard deviation.
    scale = 1.4826 * mad if mad > 0 else float(np.std(finite_scores))
    if scale <= 0:
        return baseline, None, None
    return baseline, scale, float((scores[best_idx] - baseline) / scale)


def _axis_step(axis: np.ndarray) -> float:
    diffs = np.diff(np.sort(np.unique(np.asarray(axis, dtype=float))))
    positive = diffs[diffs > 0]
    if len(positive) == 0:
        return 1.0
    return float(np.median(positive))
