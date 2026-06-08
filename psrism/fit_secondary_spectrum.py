"""Parabolic-arc fitting for secondary spectra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParabolicArcFitResult:
    curvature: float
    curvature_error: float | None
    fringe_offset: float
    delay_offset: float
    score: float
    score_snr: float | None
    half: str
    n_samples: int
    mask_bins: int
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
) -> ParabolicArcFitResult:
    """Fit ``delay - delay0 = eta * (fringe_frequency - fringe0)**2``.

    The input spectrum should be linear secondary-spectrum power, not dB power.
    Curvature has units of seconds cubed when fringe frequency is in Hz and delay
    is in seconds.
    """
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

    delay_order = np.argsort(delay)
    delay = delay[delay_order]
    power = power[:, delay_order]
    power = np.where(np.isfinite(power), power, 0.0)

    fringe_mask = max(mask_bins, 0) * _axis_step(fringe_frequency)
    delay_mask = max(mask_bins, 0) * _axis_step(delay)
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
    error = _curvature_error(eta_abs, scores, best_idx)
    if error is not None and half == "negative":
        error = abs(error)

    score_snr = _score_snr(scores, best_idx)
    trial_curvatures = -eta_abs if half == "negative" else eta_abs
    return ParabolicArcFitResult(
        curvature=curvature,
        curvature_error=error,
        fringe_offset=float(fringe_offset),
        delay_offset=float(delay_offset),
        score=float(scores[best_idx]),
        score_snr=score_snr,
        half=half,
        n_samples=int(sample_counts[best_idx]),
        mask_bins=int(mask_bins),
        trial_curvatures=trial_curvatures,
        arc_strength=scores,
    )


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
    min_delay = max(float(np.nanmin(useful_delay)), _axis_step(delay))
    max_delay = float(np.nanmax(useful_delay))
    eta_min = abs(curvature_min) if curvature_min is not None else min_delay / max_fringe**2
    eta_max = abs(curvature_max) if curvature_max is not None else max_delay / max_fringe**2
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


def _curvature_error(curvatures: np.ndarray, scores: np.ndarray, best_idx: int) -> float | None:
    finite = np.isfinite(scores)
    if np.count_nonzero(finite) < 5:
        return None
    baseline = float(np.nanmedian(scores))
    peak = float(scores[best_idx])
    if not np.isfinite(peak) or peak <= baseline:
        return None
    threshold = baseline + 0.5 * (peak - baseline)
    left = _threshold_crossing(curvatures, scores, best_idx, threshold, direction=-1)
    right = _threshold_crossing(curvatures, scores, best_idx, threshold, direction=1)
    if left is None or right is None:
        return None
    return float(0.5 * (right - left))


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


def _score_snr(scores: np.ndarray, best_idx: int) -> float | None:
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size < 5:
        return None
    baseline = float(np.median(finite_scores))
    mad = float(np.median(np.abs(finite_scores - baseline)))
    scale = 1.4826 * mad if mad > 0 else float(np.std(finite_scores))
    if scale <= 0:
        return None
    return float((scores[best_idx] - baseline) / scale)


def _axis_step(axis: np.ndarray) -> float:
    diffs = np.diff(np.sort(np.unique(np.asarray(axis, dtype=float))))
    positive = diffs[diffs > 0]
    if len(positive) == 0:
        return 1.0
    return float(np.median(positive))
