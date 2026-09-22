"""Dispersion-measure variability and solar-separation diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


# Standards provenance: 365.25 days is the Julian-year definition, not a
# value drawn from the bundled pulsar references.
DAYS_PER_YEAR = 365.25
DM_SERIES = {"dm", "scattering-corrected"}


@dataclass(frozen=True)
class DmLinearFit:
    n_points: int
    start_mjd: float
    end_mjd: float
    reference_mjd: float
    intercept_pc_cm3: float
    slope_pc_cm3_per_year: float
    slope_error_pc_cm3_per_year: float | None
    weighted: bool


@dataclass(frozen=True)
class DmSlopeSegment:
    segment: int
    start_index: int
    end_index: int
    direction: str
    accepted: bool
    rejection_reasons: tuple[str, ...]
    fit: DmLinearFit | None


@dataclass(frozen=True)
class DmSolarVariabilityReport:
    dm_series: str
    n_epochs_total: int
    n_dm_finite: int
    n_solar_pairs: int
    p_value_threshold: float
    solar_warning_angle_deg: float
    minimum_sun_separation_deg: float | None
    epochs_below_warning_angle: int
    solar_pearson_r: float | None
    solar_pearson_p_value: float | None
    solar_correlation_significant: bool | None
    solar_ephemeris_methods: tuple[str, ...]
    global_fit: DmLinearFit | None
    dm_turning_sigma: float
    dm_segment_min_points: int
    turning_point_overlap: bool
    segments: tuple[DmSlopeSegment, ...]
    accepted_segments: int
    mean_absolute_slope_pc_cm3_per_year: float | None
    mean_absolute_slope_error_pc_cm3_per_year: float | None


# Reference defaults: Filothodoros thesis (ALEX.pdf), PDF p. 55, Section 4.2,
# uses p=0.05 for temporal decisions; PDF p. 85, Section 5.5.1, identifies
# approximately 10 degrees as the regime of stronger solar-wind influence.
def analyze_dm_solar_variability(
    measurements,
    dm_series: str = "dm",
    p_value_threshold: float = 0.05,
    solar_warning_angle_deg: float = 10.0,
    dm_turning_sigma: float = 1.0,
    dm_segment_min_points: int = 3,
) -> DmSolarVariabilityReport:
    """Analyze DM trends and their association with Sun-pulsar separation."""
    if dm_series not in DM_SERIES:
        raise ValueError(f"dm_series must be one of: {', '.join(sorted(DM_SERIES))}")
    if not np.isfinite(p_value_threshold) or not 0 < p_value_threshold < 1:
        raise ValueError("p_value_threshold must be between zero and one")
    if not np.isfinite(solar_warning_angle_deg) or solar_warning_angle_deg <= 0:
        raise ValueError("solar_warning_angle_deg must be positive")
    if not np.isfinite(dm_turning_sigma) or dm_turning_sigma < 0:
        raise ValueError("dm_turning_sigma cannot be negative")
    if dm_segment_min_points < 2:
        raise ValueError("dm_segment_min_points must be at least 2")

    mjd, dm, dm_error = dm_measurement_arrays(measurements, dm_series)
    sun_separation = _optional_array(
        getattr(item, "sun_separation_mid_deg", None) for item in measurements
    )
    finite_dm = np.isfinite(mjd) & np.isfinite(dm)
    order = np.argsort(mjd[finite_dm])
    fit_mjd = mjd[finite_dm][order]
    fit_dm = dm[finite_dm][order]
    fit_error = dm_error[finite_dm][order]

    solar_pairs = finite_dm & np.isfinite(sun_separation)
    pair_count = int(np.count_nonzero(solar_pairs))
    coefficient = None
    p_value = None
    if (
        pair_count >= 3
        and np.ptp(dm[solar_pairs]) > 0
        and np.ptp(sun_separation[solar_pairs]) > 0
    ):
        from scipy.stats import pearsonr

        # Reference: Filothodoros thesis (ALEX.pdf), PDF p. 70, Section 4.7,
        # and PDF p. 73, Section 4.8, report Pearson r and p-values when
        # comparing temporal DM with Sun-pulsar separation and other series.
        r_value, p_value_raw = pearsonr(dm[solar_pairs], sun_separation[solar_pairs])
        coefficient = _optional_float(r_value)
        p_value = _optional_float(p_value_raw)

    global_fit = _linear_fit(fit_mjd, fit_dm, fit_error)
    segments = _piecewise_segments(
        fit_mjd,
        fit_dm,
        fit_error,
        dm_turning_sigma=dm_turning_sigma,
        minimum_points=dm_segment_min_points,
    )
    accepted = [item for item in segments if item.accepted and item.fit is not None]
    mean_absolute_slope = None
    mean_absolute_slope_error = None
    if accepted:
        # Reference: Filothodoros thesis (ALEX.pdf), PDF pp. 71-72,
        # Section 4.7.2, divides a DM series into same-direction sections and
        # combines |Delta DM| slopes with the number of measurements as weight.
        weights = np.asarray([item.fit.n_points for item in accepted], dtype=float)
        slopes = np.asarray(
            [abs(item.fit.slope_pc_cm3_per_year) for item in accepted],
            dtype=float,
        )
        mean_absolute_slope = float(np.average(slopes, weights=weights))
        slope_errors = np.asarray(
            [
                np.nan
                if item.fit.slope_error_pc_cm3_per_year is None
                else item.fit.slope_error_pc_cm3_per_year
                for item in accepted
            ],
            dtype=float,
        )
        if np.all(np.isfinite(slope_errors)):
            # PSRISM choice: propagate independent segment-slope errors through
            # the measurement-count-weighted arithmetic mean.
            mean_absolute_slope_error = float(
                np.sqrt(np.sum((weights * slope_errors) ** 2)) / np.sum(weights)
            )

    finite_sun = sun_separation[np.isfinite(sun_separation)]
    methods = tuple(
        sorted(
            {
                str(item.solar_ephemeris_method)
                for item in measurements
                if getattr(item, "solar_ephemeris_method", None)
            }
        )
    )
    return DmSolarVariabilityReport(
        dm_series=dm_series,
        n_epochs_total=len(measurements),
        n_dm_finite=int(np.count_nonzero(finite_dm)),
        n_solar_pairs=pair_count,
        p_value_threshold=float(p_value_threshold),
        solar_warning_angle_deg=float(solar_warning_angle_deg),
        minimum_sun_separation_deg=(
            _optional_float(np.min(finite_sun)) if len(finite_sun) else None
        ),
        epochs_below_warning_angle=int(
            np.count_nonzero(finite_sun < solar_warning_angle_deg)
        ),
        solar_pearson_r=coefficient,
        solar_pearson_p_value=p_value,
        solar_correlation_significant=(
            None if p_value is None else bool(p_value < p_value_threshold)
        ),
        solar_ephemeris_methods=methods,
        global_fit=global_fit,
        dm_turning_sigma=float(dm_turning_sigma),
        dm_segment_min_points=int(dm_segment_min_points),
        turning_point_overlap=True,
        segments=segments,
        accepted_segments=len(accepted),
        mean_absolute_slope_pc_cm3_per_year=mean_absolute_slope,
        mean_absolute_slope_error_pc_cm3_per_year=mean_absolute_slope_error,
    )


def write_dm_solar_variability_report(
    report: DmSolarVariabilityReport,
    output_path: str | Path,
) -> None:
    """Write the DM and solar diagnostics as strict JSON."""
    with Path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(asdict(report), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def plot_dm_vs_sun_separation(
    measurements,
    report: DmSolarVariabilityReport,
    output_path: str | Path,
) -> bool:
    """Plot DM against the mid-observation Sun-pulsar separation."""
    _, dm, dm_error = dm_measurement_arrays(measurements, report.dm_series)
    separation = _optional_array(
        getattr(item, "sun_separation_mid_deg", None) for item in measurements
    )
    finite = np.isfinite(dm) & np.isfinite(separation)
    if not np.any(finite):
        return False

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    errors = dm_error[finite]
    valid_errors = np.isfinite(errors) & (errors > 0)
    if np.any(valid_errors):
        ax.errorbar(
            separation[finite][valid_errors],
            dm[finite][valid_errors],
            yerr=errors[valid_errors],
            fmt="o",
            color="black",
            capsize=2,
            label="DM with uncertainty",
        )
    if np.any(~valid_errors):
        ax.plot(
            separation[finite][~valid_errors],
            dm[finite][~valid_errors],
            "o",
            color="tab:blue",
            label="DM without uncertainty",
        )
    ax.axvline(
        report.solar_warning_angle_deg,
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label=f"warning angle ({report.solar_warning_angle_deg:g} deg)",
    )
    if report.solar_pearson_r is not None:
        ax.text(
            0.02,
            0.98,
            f"Pearson r={report.solar_pearson_r:.3g}\n"
            f"p={report.solar_pearson_p_value:.3g}\n"
            f"n={report.n_solar_pairs}",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
    ax.set_xlabel("Sun-pulsar separation (deg)")
    ax.set_ylabel(_dm_axis_label(report.dm_series))
    ax.set_title("Dispersion measure versus Sun-pulsar separation")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def plot_dm_piecewise_slopes(
    measurements,
    report: DmSolarVariabilityReport,
    output_path: str | Path,
) -> bool:
    """Plot the chronological DM series with global and piecewise fits."""
    mjd, dm, dm_error = dm_measurement_arrays(measurements, report.dm_series)
    finite = np.isfinite(mjd) & np.isfinite(dm)
    if not np.any(finite):
        return False
    order = np.argsort(mjd[finite])
    x = mjd[finite][order]
    y = dm[finite][order]
    error = dm_error[finite][order]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    valid_errors = np.isfinite(error) & (error > 0)
    if np.any(valid_errors):
        ax.errorbar(
            x[valid_errors],
            y[valid_errors],
            yerr=error[valid_errors],
            fmt="o",
            color="black",
            capsize=2,
            label="DM with uncertainty",
        )
    if np.any(~valid_errors):
        ax.plot(x[~valid_errors], y[~valid_errors], "o", color="black")

    if report.global_fit is not None:
        fit = report.global_fit
        fit_x = np.asarray([fit.start_mjd, fit.end_mjd])
        fit_y = _evaluate_fit(fit, fit_x)
        ax.plot(fit_x, fit_y, ":", color="0.35", linewidth=1.4, label="global fit")

    accepted_label_used = False
    rejected_label_used = False
    for segment in report.segments:
        if segment.fit is None:
            continue
        fit = segment.fit
        fit_x = np.asarray([fit.start_mjd, fit.end_mjd])
        fit_y = _evaluate_fit(fit, fit_x)
        if segment.accepted:
            label = "accepted same-direction fit" if not accepted_label_used else None
            accepted_label_used = True
            ax.plot(fit_x, fit_y, color="tab:blue", linewidth=2.0, label=label)
        else:
            label = "rejected short segment" if not rejected_label_used else None
            rejected_label_used = True
            ax.plot(fit_x, fit_y, color="0.65", linewidth=1.2, label=label)

    ax.set_xlabel("MJD")
    ax.set_ylabel(_dm_axis_label(report.dm_series))
    ax.set_title("Dispersion-measure global and same-direction slopes")
    ax.grid(alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def dm_measurement_arrays(measurements, dm_series: str):
    """Return MJD, selected DM, and selected DM uncertainty arrays."""
    if dm_series not in DM_SERIES:
        raise ValueError(f"dm_series must be one of: {', '.join(sorted(DM_SERIES))}")
    mjd = _optional_array(getattr(item, "mjd", None) for item in measurements)
    if dm_series == "dm":
        values = _optional_array(getattr(item, "dm", None) for item in measurements)
        errors = _optional_array(
            getattr(item, "dm_error", None) for item in measurements
        )
        return mjd, values, errors

    values = _optional_array(
        getattr(item, "scattering_corrected_dm", None) for item in measurements
    )
    error_values = []
    for item in measurements:
        dm_error = _positive_optional(getattr(item, "dm_error", None))
        correction_error = _positive_optional(
            getattr(item, "scattering_delta_dm_error", None)
        )
        if dm_error is not None and correction_error is not None:
            # PSRISM choice: treat the pdmp and residual-scattering DM errors as
            # independent and combine them in quadrature for this diagnostic.
            error_values.append(float(np.hypot(dm_error, correction_error)))
        elif dm_error is not None:
            error_values.append(dm_error)
        else:
            error_values.append(correction_error)
    return mjd, values, _optional_array(error_values)


def _piecewise_segments(
    mjd: np.ndarray,
    dm: np.ndarray,
    errors: np.ndarray,
    dm_turning_sigma: float,
    minimum_points: int,
) -> tuple[DmSlopeSegment, ...]:
    if len(dm) < 2:
        return ()
    differences = np.diff(dm)
    directions = np.sign(differences).astype(int)
    for index, difference in enumerate(differences):
        left_error = _positive_optional(errors[index])
        right_error = _positive_optional(errors[index + 1])
        if left_error is None or right_error is None:
            continue
        pair_error = float(np.hypot(left_error, right_error))
        # PSRISM choice: differences no larger than this configurable
        # uncertainty multiple do not introduce a turning point.
        if abs(difference) <= dm_turning_sigma * pair_error:
            directions[index] = 0

    # PSRISM choice: attach uncertainty-compatible flat steps to the preceding
    # trend (or the following trend at the beginning) before finding reversals.
    for index in range(1, len(directions)):
        if directions[index] == 0:
            directions[index] = directions[index - 1]
    for index in range(len(directions) - 2, -1, -1):
        if directions[index] == 0:
            directions[index] = directions[index + 1]

    change_points = [
        index + 1
        for index in range(len(directions) - 1)
        if directions[index] != directions[index + 1]
        and directions[index] != 0
        and directions[index + 1] != 0
    ]
    starts = [0, *change_points]
    ends = [*change_points, len(dm) - 1]
    result = []
    for segment_number, (start, end) in enumerate(zip(starts, ends), start=1):
        fit = _linear_fit(
            mjd[start : end + 1],
            dm[start : end + 1],
            errors[start : end + 1],
        )
        reasons = []
        if end - start + 1 < minimum_points:
            reasons.append("insufficient_points")
        if fit is None:
            reasons.append("slope_fit_failed")
        slope = None if fit is None else fit.slope_pc_cm3_per_year
        if slope is None:
            direction = "undetermined"
        elif slope > 0:
            direction = "increasing"
        elif slope < 0:
            direction = "decreasing"
        else:
            direction = "flat"
        result.append(
            DmSlopeSegment(
                segment=segment_number,
                start_index=start,
                end_index=end,
                direction=direction,
                accepted=not reasons,
                rejection_reasons=tuple(reasons),
                fit=fit,
            )
        )
    return tuple(result)


def _linear_fit(
    mjd: np.ndarray,
    dm: np.ndarray,
    errors: np.ndarray,
) -> DmLinearFit | None:
    finite = np.isfinite(mjd) & np.isfinite(dm)
    x_mjd = np.asarray(mjd[finite], dtype=float)
    y = np.asarray(dm[finite], dtype=float)
    error = np.asarray(errors[finite], dtype=float)
    if len(y) < 2 or np.ptp(x_mjd) <= 0:
        return None
    reference_mjd = float(np.mean(x_mjd))
    # PSRISM time-unit choice: fit in Julian years of 365.25 days and center
    # the independent variable to keep the regression numerically stable.
    x = (x_mjd - reference_mjd) / DAYS_PER_YEAR
    design = np.column_stack((np.ones(len(x)), x))
    weighted = bool(np.all(np.isfinite(error) & (error > 0)))
    try:
        if weighted:
            weights = 1.0 / error**2
            normal = design.T @ (weights[:, None] * design)
            covariance = np.linalg.inv(normal)
            coefficients = covariance @ (design.T @ (weights * y))
            slope_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
        else:
            coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
            slope_error = None
            if len(y) > 2:
                residual = y - design @ coefficients
                variance = float(np.sum(residual**2) / (len(y) - 2))
                covariance = variance * np.linalg.inv(design.T @ design)
                slope_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    except np.linalg.LinAlgError:
        return None
    return DmLinearFit(
        n_points=len(y),
        start_mjd=float(np.min(x_mjd)),
        end_mjd=float(np.max(x_mjd)),
        reference_mjd=reference_mjd,
        intercept_pc_cm3=float(coefficients[0]),
        slope_pc_cm3_per_year=float(coefficients[1]),
        slope_error_pc_cm3_per_year=_optional_float(slope_error),
        weighted=weighted,
    )


def _evaluate_fit(fit: DmLinearFit, mjd: np.ndarray) -> np.ndarray:
    return fit.intercept_pc_cm3 + fit.slope_pc_cm3_per_year * (
        np.asarray(mjd) - fit.reference_mjd
    ) / DAYS_PER_YEAR


def _dm_axis_label(dm_series: str) -> str:
    prefix = "Scattering-corrected DM" if dm_series == "scattering-corrected" else "DM"
    return f"{prefix} (pc cm$^{{-3}}$)"


def _positive_optional(value) -> float | None:
    converted = _optional_float(value)
    if converted is None or converted <= 0:
        return None
    return converted


def _optional_array(values) -> np.ndarray:
    return np.asarray(
        [np.nan if value is None else float(value) for value in values],
        dtype=float,
    )


def _optional_float(value) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)
