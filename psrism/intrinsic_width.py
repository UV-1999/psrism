"""Intrinsic Gaussian-width diagnostics across scattering-fit subbands."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


# Reference: Filothodoros et al. (Alex_and_Boe.pdf), PDF pp. 11-12,
# Appendix B, define the diagnostic as the fitted intrinsic Gaussian FWHM.
# The factor below is the standard Gaussian sigma-to-FWHM identity.
GAUSSIAN_FWHM_FACTOR = float(2.0 * np.sqrt(2.0 * np.log(2.0)))


@dataclass(frozen=True)
class IntrinsicWidthMeasurement:
    component: int
    frequency_mhz: float
    central_frequency_mhz: float
    bandwidth_mhz: float
    accepted: bool
    rejection_reasons: tuple[str, ...]
    sigma_bins: float
    sigma_error_bins: float | None
    fwhm_bins: float
    fwhm_error_bins: float | None
    fwhm_phase: float
    fwhm_phase_error: float | None
    fwhm_percent_period: float
    fwhm_percent_period_error: float | None
    fwhm_seconds: float | None
    fwhm_seconds_error: float | None


@dataclass(frozen=True)
class IntrinsicWidthTrend:
    component: int
    n_subbands: int
    reference_frequency_mhz: float
    reference_width_percent: float
    reference_width_percent_error: float | None
    frequency_exponent: float
    frequency_exponent_error: float | None
    exponent_p_value: float | None
    significant: bool
    chi_square_log_width: float
    reduced_chi_square_log_width: float | None
    dof: int
    pearson_r_log_frequency_width: float | None
    pearson_p_value: float | None
    uncertainty_method: str


@dataclass(frozen=True)
class IntrinsicWidthReport:
    scattering_model: str
    intrinsic_model: str
    width_available: bool
    measurement_definition: str
    component_matching: str
    n_fitted_subbands: int
    n_accepted_subbands: int
    n_components: int
    min_trend_subbands: int
    trend_p_threshold: float
    measurements: tuple[IntrinsicWidthMeasurement, ...]
    trends: tuple[IntrinsicWidthTrend, ...]
    warnings: tuple[str, ...]


def analyze_intrinsic_widths(
    scattering_result,
    scattering_model: str,
    min_trend_subbands: int = 3,
    p_threshold: float = 0.05,
) -> IntrinsicWidthReport:
    """Measure fitted intrinsic FWHM and its frequency trend per component."""
    if min_trend_subbands < 2:
        raise ValueError("min_trend_subbands must be at least 2")
    if not np.isfinite(p_threshold) or not 0.0 < p_threshold < 1.0:
        raise ValueError("p_threshold must be between zero and one")

    subbands = tuple(sorted(scattering_result.subbands, key=lambda item: item.frequency_mhz))
    intrinsic_models = {item.fit.intrinsic_model for item in subbands}
    intrinsic_model = (
        next(iter(intrinsic_models))
        if len(intrinsic_models) == 1
        else "mixed"
    )
    measurements: list[IntrinsicWidthMeasurement] = []

    for subband in subbands:
        fit = subband.fit
        nbin = fit.n_bins if fit.n_bins is not None else len(subband.profile)
        if nbin <= 0:
            continue
        for index, component in enumerate(fit.components):
            sigma_error = _component_error(fit.component_sigma_errors, index)
            fwhm_bins = GAUSSIAN_FWHM_FACTOR * float(component.sigma)
            fwhm_error_bins = _scale_optional(sigma_error, GAUSSIAN_FWHM_FACTOR)
            fwhm_phase = fwhm_bins / float(nbin)
            fwhm_phase_error = _scale_optional(fwhm_error_bins, 1.0 / float(nbin))
            period_s = fit.period_s

            # Reference: Filothodoros et al. (Alex_and_Boe.pdf), PDF pp. 11-12,
            # Appendix B and Fig. B1, report the fitted intrinsic Gaussian FWHM
            # as a percentage of pulse period versus observing frequency.
            measurements.append(
                IntrinsicWidthMeasurement(
                    component=index + 1,
                    frequency_mhz=float(subband.frequency_mhz),
                    central_frequency_mhz=float(subband.central_frequency_mhz),
                    bandwidth_mhz=float(subband.bandwidth_mhz),
                    accepted=bool(subband.accepted),
                    rejection_reasons=tuple(subband.rejection_reasons),
                    sigma_bins=float(component.sigma),
                    sigma_error_bins=sigma_error,
                    fwhm_bins=fwhm_bins,
                    fwhm_error_bins=fwhm_error_bins,
                    fwhm_phase=fwhm_phase,
                    fwhm_phase_error=fwhm_phase_error,
                    fwhm_percent_period=100.0 * fwhm_phase,
                    fwhm_percent_period_error=_scale_optional(fwhm_phase_error, 100.0),
                    fwhm_seconds=_scale_optional(period_s, fwhm_phase),
                    fwhm_seconds_error=_scale_optional(period_s, fwhm_phase_error),
                )
            )

    warnings: list[str] = []
    n_components = max((item.component for item in measurements), default=0)
    if not measurements:
        warnings.append(
            "Intrinsic Gaussian FWHM is unavailable because a template fit does not "
            "estimate Gaussian component widths."
        )
    elif n_components > 1:
        warnings.append(
            "Components are matched by fitted phase order independently in each subband; "
            "crossing, wrapping, or blended components can exchange identity."
        )

    reference_frequency = float(scattering_result.alpha_fit.reference_freq_mhz)
    trends: list[IntrinsicWidthTrend] = []
    for component in range(1, n_components + 1):
        selected = [
            item
            for item in measurements
            if item.component == component and item.accepted
        ]
        if len(selected) < min_trend_subbands:
            warnings.append(
                f"Component {component} has {len(selected)} accepted width measurements; "
                f"at least {min_trend_subbands} are required for a frequency trend."
            )
            continue
        trend = fit_intrinsic_width_trend(
            selected,
            reference_frequency_mhz=reference_frequency,
            p_threshold=p_threshold,
        )
        trends.append(trend)
        if trend.significant:
            direction = "narrows" if trend.frequency_exponent < 0 else "broadens"
            warnings.append(
                f"Component {component} {direction} significantly with increasing "
                "frequency in this fit. This may reflect intrinsic profile evolution/RFM "
                "or scattering-model degeneracy; it is not by itself proof of either."
            )

    return IntrinsicWidthReport(
        scattering_model=str(scattering_model),
        intrinsic_model=intrinsic_model,
        width_available=bool(measurements),
        measurement_definition="FWHM of each fitted intrinsic Gaussian component",
        component_matching="phase_order_per_subband",
        n_fitted_subbands=len(subbands),
        n_accepted_subbands=sum(bool(item.accepted) for item in subbands),
        n_components=n_components,
        min_trend_subbands=int(min_trend_subbands),
        trend_p_threshold=float(p_threshold),
        measurements=tuple(measurements),
        trends=tuple(trends),
        warnings=tuple(warnings),
    )


def fit_intrinsic_width_trend(
    measurements: list[IntrinsicWidthMeasurement],
    reference_frequency_mhz: float,
    p_threshold: float = 0.05,
) -> IntrinsicWidthTrend:
    """Fit W(nu) = W_ref (nu / nu_ref)^gamma in log space."""
    from scipy.stats import pearsonr, t as student_t

    if not np.isfinite(reference_frequency_mhz) or reference_frequency_mhz <= 0:
        raise ValueError("reference_frequency_mhz must be finite and positive")
    if not np.isfinite(p_threshold) or not 0.0 < p_threshold < 1.0:
        raise ValueError("p_threshold must be between zero and one")

    frequency = np.asarray([item.frequency_mhz for item in measurements], dtype=float)
    width = np.asarray([item.fwhm_percent_period for item in measurements], dtype=float)
    width_error = np.asarray(
        [
            np.nan
            if item.fwhm_percent_period_error is None
            else item.fwhm_percent_period_error
            for item in measurements
        ],
        dtype=float,
    )
    valid = np.isfinite(frequency) & np.isfinite(width) & (frequency > 0) & (width > 0)
    frequency = frequency[valid]
    width = width[valid]
    width_error = width_error[valid]
    if len(frequency) < 2:
        raise ValueError("at least two finite positive width measurements are required")
    if len(np.unique(frequency)) < 2:
        raise ValueError("width trend requires at least two distinct frequencies")

    x = np.log(frequency / float(reference_frequency_mhz))
    y = np.log(width)
    design = np.column_stack([np.ones_like(x), x])
    use_errors = np.all(np.isfinite(width_error) & (width_error > 0))
    if use_errors:
        log_error = width_error / width
        weights = 1.0 / log_error**2
        normal = design.T @ (weights[:, None] * design)
        covariance = np.linalg.inv(normal)
        coefficients = covariance @ (design.T @ (weights * y))
        residual = y - design @ coefficients
        chi_square = float(np.sum((residual / log_error) ** 2))
        uncertainty_method = "covariance_weighted_log_linear"
    else:
        coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coefficients
        chi_square = float(np.sum(residual**2))
        normal_inverse = np.linalg.inv(design.T @ design)
        dof_for_scale = len(y) - 2
        scale = chi_square / dof_for_scale if dof_for_scale > 0 else float("nan")
        covariance = normal_inverse * scale
        uncertainty_method = "unweighted_log_linear"

    dof = int(len(y) - 2)
    intercept = float(coefficients[0])
    exponent = float(coefficients[1])
    intercept_error = _finite_sqrt(covariance[0, 0])
    exponent_error = _finite_sqrt(covariance[1, 1])
    p_value = None
    if exponent_error is not None and exponent_error > 0 and dof > 0:
        p_value = float(2.0 * student_t.sf(abs(exponent / exponent_error), dof))
    if np.ptp(y) > 0:
        pearson = pearsonr(x, y)
        pearson_r = float(pearson.statistic)
        pearson_p = float(pearson.pvalue)
    else:
        pearson_r = None
        pearson_p = None
    reference_width = float(np.exp(intercept))

    # PSRISM diagnostic choice: Appendix B plots width against frequency but
    # does not prescribe a trend model. The power law and p-value threshold
    # make that visual check reproducible without assigning a physical cause.
    return IntrinsicWidthTrend(
        component=int(measurements[0].component),
        n_subbands=int(len(y)),
        reference_frequency_mhz=float(reference_frequency_mhz),
        reference_width_percent=reference_width,
        reference_width_percent_error=_scale_optional(reference_width, intercept_error),
        frequency_exponent=exponent,
        frequency_exponent_error=exponent_error,
        exponent_p_value=p_value,
        significant=bool(p_value is not None and p_value <= p_threshold),
        chi_square_log_width=chi_square,
        reduced_chi_square_log_width=(chi_square / dof if dof > 0 else None),
        dof=dof,
        pearson_r_log_frequency_width=pearson_r,
        pearson_p_value=pearson_p,
        uncertainty_method=uncertainty_method,
    )


def plot_intrinsic_widths(
    report: IntrinsicWidthReport,
    title: str | None = None,
    output_path: str | Path | None = None,
):
    """Plot intrinsic Gaussian FWHM as percentage of pulse period."""
    import matplotlib.pyplot as plt

    if not report.width_available:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colors = plt.get_cmap("tab10")
    for component in range(1, report.n_components + 1):
        values = [item for item in report.measurements if item.component == component]
        color = colors((component - 1) % 10)
        for accepted, marker, alpha in ((True, "o", 1.0), (False, "x", 0.65)):
            subset = [item for item in values if item.accepted is accepted]
            if not subset:
                continue
            errors = [item.fwhm_percent_period_error for item in subset]
            yerr = errors if all(value is not None for value in errors) else None
            ax.errorbar(
                [item.frequency_mhz for item in subset],
                [item.fwhm_percent_period for item in subset],
                yerr=yerr,
                fmt=marker,
                color=color,
                alpha=alpha,
                capsize=3,
                label=(
                    f"Component {component} ({'accepted' if accepted else 'rejected'})"
                ),
            )
        trend = next((item for item in report.trends if item.component == component), None)
        if trend is not None:
            frequencies = np.asarray([item.frequency_mhz for item in values], dtype=float)
            grid = np.geomspace(np.min(frequencies), np.max(frequencies), 200)
            model = trend.reference_width_percent * (
                grid / trend.reference_frequency_mhz
            ) ** trend.frequency_exponent
            ax.plot(grid, model, color=color, linewidth=1.6)

    # Reference: Filothodoros et al. (Alex_and_Boe.pdf), PDF p. 12, Fig. B1,
    # uses observing frequency and intrinsic FWHM as percentage of period.
    ax.set_xscale("log")
    ax.set_xlabel("Observing frequency (MHz)")
    ax.set_ylabel("Intrinsic Gaussian FWHM (% of pulse period)")
    if title:
        ax.set_title(Path(title).name)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150)
    return fig


def write_intrinsic_width_report(
    report: IntrinsicWidthReport,
    output_path: str | Path,
) -> None:
    """Write a strict-JSON intrinsic-width report."""
    payload = _json_safe(asdict(report))
    Path(output_path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _component_error(errors: tuple[float, ...], index: int) -> float | None:
    if index >= len(errors):
        return None
    value = float(errors[index])
    return value if np.isfinite(value) and value >= 0 else None


def _finite_sqrt(value: float) -> float | None:
    if not np.isfinite(value) or value < 0:
        return None
    return float(np.sqrt(value))


def _scale_optional(value: float | None, scale: float | None) -> float | None:
    if value is None or scale is None:
        return None
    result = float(value * scale)
    return result if np.isfinite(result) else None


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value
