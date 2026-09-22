"""Non-ML variability statistics for PSRISM directory time series."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


PARAMETER_FIELDS = {
    "dm": ("dm", "dm_error", "DM", "pc cm^-3"),
    "tau": ("tau_s", "tau_error_s", "Scattering timescale tau", "s"),
    # Reference: Filothodoros thesis (ALEX.pdf), PDF pp. 34-35,
    # Section 2.7.4, for the reference-frequency tau series.
    "tau_ref": (
        "tau0_s",
        "tau0_error_s",
        "Reference-frequency scattering timescale tau_ref",
        "s",
    ),
    "alpha": ("alpha", "alpha_error", "Scattering spectral index alpha", ""),
    "dnu_d": (
        "decorrelation_bandwidth_mhz",
        "decorrelation_bandwidth_error_mhz",
        "Decorrelation bandwidth Delta_nu_d",
        "MHz",
    ),
    "dt_d": (
        "diffractive_timescale_s",
        "diffractive_timescale_error_s",
        "Diffractive timescale Delta_t_d",
        "s",
    ),
    "t_r": (
        "refractive_timescale_days",
        "refractive_timescale_error_days",
        "Refractive timescale T_r",
        "days",
    ),
}


@dataclass(frozen=True)
class ParameterVariabilityStatistics:
    parameter: str
    label: str
    unit: str
    n_epochs_total: int
    n_finite: int
    n_with_uncertainty: int
    start_mjd: float | None
    end_mjd: float | None
    time_span_days: float | None
    mean: float | None
    standard_deviation: float | None
    median: float | None
    minimum: float | None
    maximum: float | None
    weighted_mean: float | None
    weighted_mean_error: float | None
    constant_chi_square: float | None
    constant_degrees_of_freedom: int | None
    constant_p_value: float | None
    constant_rejected: bool | None
    runs_count: int | None
    runs_expected: float | None
    runs_variance: float | None
    runs_z: float | None
    runs_p_value: float | None
    runs_randomness_rejected: bool | None
    maximum_lag: int
    autocorrelation: tuple[float | None, ...]
    autocorrelation_confidence: tuple[float | None, ...]
    ljung_box_q: tuple[float | None, ...]
    ljung_box_p_values: tuple[float | None, ...]
    ljung_box_p_at_maximum_lag: float | None
    ljung_box_minimum_p_value: float | None
    ljung_box_maximum_p_value: float | None
    ljung_box_independence_rejected: bool | None
    ljung_box_all_lags_rejected: bool | None


@dataclass(frozen=True)
class TemporalCorrelation:
    parameter_x: str
    parameter_y: str
    n_paired: int
    pearson_r: float | None
    p_value: float | None
    significant: bool | None


@dataclass(frozen=True)
class TemporalStatisticsReport:
    p_value_threshold: float
    configured_maximum_lag: int | None
    lag_unit: str
    parameters: tuple[ParameterVariabilityStatistics, ...]
    correlations: tuple[TemporalCorrelation, ...]


def analyze_temporal_statistics(
    measurements,
    parameters,
    maximum_lag: int | None = None,
    p_value_threshold: float = 0.05,
) -> TemporalStatisticsReport:
    """Calculate non-ML variability diagnostics for selected measurements."""
    if maximum_lag is not None and maximum_lag < 1:
        raise ValueError("maximum_lag must be at least 1 when supplied")
    if not np.isfinite(p_value_threshold) or not 0 < p_value_threshold < 1:
        raise ValueError("p_value_threshold must be between zero and one")

    selected = [name for name in PARAMETER_FIELDS if name in set(parameters)]
    series = {name: _measurement_arrays(measurements, name) for name in selected}
    summaries = tuple(
        _parameter_statistics(
            name,
            *series[name],
            n_epochs_total=len(measurements),
            maximum_lag=maximum_lag,
            p_value_threshold=p_value_threshold,
        )
        for name in selected
    )
    correlations = tuple(
        _pearson_correlation(
            left,
            right,
            series[left][1],
            series[right][1],
            p_value_threshold,
        )
        for left_index, left in enumerate(selected)
        for right in selected[left_index + 1 :]
    )
    return TemporalStatisticsReport(
        p_value_threshold=float(p_value_threshold),
        configured_maximum_lag=maximum_lag,
        lag_unit="observation_index",
        parameters=summaries,
        correlations=correlations,
    )


def write_temporal_statistics_report(
    report: TemporalStatisticsReport,
    output_path: str | Path,
) -> None:
    """Write temporal statistics and pairwise correlations as JSON."""
    with Path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(asdict(report), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def plot_parameter_autocorrelation(
    result: ParameterVariabilityStatistics,
    output_path: str | Path,
) -> bool:
    """Plot observation-order ACF with lag-dependent Bartlett limits."""
    if result.maximum_lag < 1:
        return False
    import matplotlib.pyplot as plt

    lags = np.arange(result.maximum_lag + 1)
    acf = np.asarray(result.autocorrelation, dtype=float)
    confidence = np.asarray(result.autocorrelation_confidence, dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.vlines(lags, 0.0, acf, color="black", linewidth=1.0)
    ax.plot(lags, acf, "o", color="black", markersize=4)
    ax.fill_between(lags, -confidence, confidence, color="tab:blue", alpha=0.16)
    ax.set_xlabel("Lag (observation index)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(f"{result.label}: observation-order autocorrelation")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def plot_temporal_correlation_matrix(
    report: TemporalStatisticsReport,
    output_path: str | Path,
) -> bool:
    """Plot Pearson coefficients and p-values for available parameter pairs."""
    names = [item.parameter for item in report.parameters if item.n_finite >= 3]
    if len(names) < 2:
        return False
    import matplotlib.pyplot as plt

    size = len(names)
    matrix = np.eye(size, dtype=float)
    p_values = np.full((size, size), np.nan, dtype=float)
    counts = np.zeros((size, size), dtype=int)
    np.fill_diagonal(p_values, 0.0)
    for item in report.correlations:
        if item.parameter_x not in names or item.parameter_y not in names:
            continue
        row = names.index(item.parameter_x)
        column = names.index(item.parameter_y)
        if item.pearson_r is not None:
            matrix[row, column] = matrix[column, row] = item.pearson_r
        else:
            matrix[row, column] = matrix[column, row] = np.nan
        if item.p_value is not None:
            p_values[row, column] = p_values[column, row] = item.p_value
        counts[row, column] = counts[column, row] = item.n_paired

    fig, ax = plt.subplots(figsize=(max(5.2, 1.15 * size), max(4.6, 1.0 * size)))
    image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(np.arange(size), labels=names)
    ax.set_yticks(np.arange(size), labels=names)
    for row in range(size):
        for column in range(size):
            if row == column:
                text = "1.00"
            elif np.isfinite(matrix[row, column]):
                text = (
                    f"r={matrix[row, column]:.2f}\n"
                    f"p={p_values[row, column]:.2g}\n"
                    f"n={counts[row, column]}"
                )
            else:
                text = "insufficient"
            color = "white" if np.isfinite(matrix[row, column]) and abs(matrix[row, column]) > 0.55 else "black"
            ax.text(column, row, text, ha="center", va="center", color=color, fontsize=8)
    ax.set_title("Pairwise Pearson correlations")
    fig.colorbar(image, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def _parameter_statistics(
    parameter: str,
    mjd: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
    n_epochs_total: int,
    maximum_lag: int | None,
    p_value_threshold: float,
) -> ParameterVariabilityStatistics:
    from scipy.stats import chi2

    _, _, label, unit = PARAMETER_FIELDS[parameter]
    finite = np.isfinite(mjd) & np.isfinite(values)
    x = mjd[finite]
    y = values[finite]
    error = errors[finite]
    n_finite = int(len(y))
    error_valid = np.isfinite(error) & (error > 0)

    mean = _optional_float(np.mean(y)) if n_finite else None
    standard_deviation = _optional_float(np.std(y, ddof=1)) if n_finite >= 2 else None
    median = _optional_float(np.median(y)) if n_finite else None
    minimum = _optional_float(np.min(y)) if n_finite else None
    maximum = _optional_float(np.max(y)) if n_finite else None
    start_mjd = _optional_float(np.min(x)) if n_finite else None
    end_mjd = _optional_float(np.max(x)) if n_finite else None
    time_span = None if n_finite < 2 else float(end_mjd - start_mjd)

    weighted_mean = None
    weighted_mean_error = None
    constant_chi_square = None
    constant_degrees_of_freedom = None
    constant_p_value = None
    constant_rejected = None
    if np.count_nonzero(error_valid):
        weights = 1.0 / error[error_valid] ** 2
        # Reference: Filothodoros thesis (ALEX.pdf), PDF p. 55, Section 4.2,
        # compares temporal values with their weighted mean. Inverse-variance
        # weighting and the constant-model chi-square are PSRISM choices.
        weighted_mean = float(np.sum(weights * y[error_valid]) / np.sum(weights))
        weighted_mean_error = float(np.sqrt(1.0 / np.sum(weights)))
        if np.count_nonzero(error_valid) >= 2:
            constant_chi_square = float(
                np.sum(((y[error_valid] - weighted_mean) / error[error_valid]) ** 2)
            )
            constant_degrees_of_freedom = int(np.count_nonzero(error_valid) - 1)
            constant_p_value = float(
                chi2.sf(constant_chi_square, constant_degrees_of_freedom)
            )
            constant_rejected = bool(constant_p_value < p_value_threshold)

    if n_finite >= 2:
        selected_maximum_lag = min(
            n_finite - 1,
            n_finite - 1 if maximum_lag is None else maximum_lag,
        )
        acf = _autocorrelation(y, selected_maximum_lag)
        confidence = _bartlett_confidence(acf, n_finite, p_value_threshold)
        q_values, q_p_values = _ljung_box(acf, n_finite)
    else:
        selected_maximum_lag = 0
        acf = np.asarray([1.0] if n_finite == 1 else [], dtype=float)
        confidence = np.asarray([0.0] if n_finite == 1 else [], dtype=float)
        q_values = np.asarray([], dtype=float)
        q_p_values = np.asarray([], dtype=float)

    runs = _runs_test(y, p_value_threshold)
    p_at_maximum = _optional_float(q_p_values[-1]) if len(q_p_values) else None
    finite_q_p = q_p_values[np.isfinite(q_p_values)]
    minimum_q_p = _optional_float(np.min(finite_q_p)) if len(finite_q_p) else None
    maximum_q_p = _optional_float(np.max(finite_q_p)) if len(finite_q_p) else None
    return ParameterVariabilityStatistics(
        parameter=parameter,
        label=label,
        unit=unit,
        n_epochs_total=int(n_epochs_total),
        n_finite=n_finite,
        n_with_uncertainty=int(np.count_nonzero(error_valid)),
        start_mjd=start_mjd,
        end_mjd=end_mjd,
        time_span_days=time_span,
        mean=mean,
        standard_deviation=standard_deviation,
        median=median,
        minimum=minimum,
        maximum=maximum,
        weighted_mean=weighted_mean,
        weighted_mean_error=weighted_mean_error,
        constant_chi_square=constant_chi_square,
        constant_degrees_of_freedom=constant_degrees_of_freedom,
        constant_p_value=constant_p_value,
        constant_rejected=constant_rejected,
        runs_count=runs[0],
        runs_expected=runs[1],
        runs_variance=runs[2],
        runs_z=runs[3],
        runs_p_value=runs[4],
        runs_randomness_rejected=runs[5],
        maximum_lag=selected_maximum_lag,
        autocorrelation=_optional_tuple(acf),
        autocorrelation_confidence=_optional_tuple(confidence),
        ljung_box_q=_optional_tuple(q_values),
        ljung_box_p_values=_optional_tuple(q_p_values),
        ljung_box_p_at_maximum_lag=p_at_maximum,
        ljung_box_minimum_p_value=minimum_q_p,
        ljung_box_maximum_p_value=maximum_q_p,
        ljung_box_independence_rejected=(
            None if p_at_maximum is None else bool(p_at_maximum < p_value_threshold)
        ),
        # Reference: Filothodoros thesis (ALEX.pdf), PDF p. 55, Section 4.2,
        # Table 4.3 reports the largest p-value across tested lags. This field
        # reproduces that stricter all-lags decision separately from the
        # standard selected-maximum-lag decision above.
        ljung_box_all_lags_rejected=(
            None if maximum_q_p is None else bool(maximum_q_p < p_value_threshold)
        ),
    )


def _autocorrelation(values: np.ndarray, maximum_lag: int) -> np.ndarray:
    centered = np.asarray(values, dtype=float) - np.mean(values)
    denominator = float(np.sum(centered**2))
    result = np.full(maximum_lag + 1, np.nan, dtype=float)
    if denominator <= 0:
        result[0] = 1.0
        return result
    result[0] = 1.0
    # Reference: Filothodoros thesis (ALEX.pdf), PDF p. 52, Section 4.1.1,
    # Eq. 4.1. PSRISM evaluates the discrete sample analogue in epoch order.
    for lag in range(1, maximum_lag + 1):
        result[lag] = float(np.sum(centered[:-lag] * centered[lag:]) / denominator)
    return result


def _bartlett_confidence(
    autocorrelation: np.ndarray,
    sample_size: int,
    p_value_threshold: float,
) -> np.ndarray:
    from scipy.stats import norm

    critical = float(norm.ppf(1.0 - p_value_threshold / 2.0))
    confidence = np.zeros(len(autocorrelation), dtype=float)
    # Reference: Filothodoros thesis (ALEX.pdf), PDF pp. 52-53,
    # Section 4.1.1, Eqs. 4.2-4.3, motivates lag-dependent Bartlett limits.
    for lag in range(1, len(autocorrelation)):
        previous = autocorrelation[1:lag]
        previous = previous[np.isfinite(previous)]
        variance = (1.0 + 2.0 * np.sum(previous**2)) / sample_size
        confidence[lag] = critical * np.sqrt(variance)
    return confidence


def _runs_test(values: np.ndarray, p_value_threshold: float):
    from scipy.stats import norm

    if len(values) < 2:
        return None, None, None, None, None, None
    median = float(np.median(values))
    signs = np.sign(values - median)
    signs = signs[signs != 0]
    n_positive = int(np.count_nonzero(signs > 0))
    n_negative = int(np.count_nonzero(signs < 0))
    n = n_positive + n_negative
    if n < 2 or n_positive == 0 or n_negative == 0:
        return None, None, None, None, None, None
    runs = int(1 + np.count_nonzero(signs[1:] != signs[:-1]))
    # Reference: Filothodoros thesis (ALEX.pdf), PDF pp. 53-54,
    # Section 4.1.1, Eqs. 4.4-4.6. PSRISM uses the conventional median split,
    # omits ties, and reports a two-sided normal-approximation p-value.
    expected = 1.0 + 2.0 * n_positive * n_negative / n
    variance = (
        2.0
        * n_positive
        * n_negative
        * (2.0 * n_positive * n_negative - n)
        / (n**2 * (n - 1))
    )
    if variance <= 0:
        return runs, float(expected), float(variance), None, None, None
    z_value = float((runs - expected) / np.sqrt(variance))
    p_value = float(2.0 * norm.sf(abs(z_value)))
    return (
        runs,
        float(expected),
        float(variance),
        z_value,
        p_value,
        bool(p_value < p_value_threshold),
    )


def _ljung_box(autocorrelation: np.ndarray, sample_size: int):
    from scipy.stats import chi2

    if len(autocorrelation) <= 1:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    q_values = []
    p_values = []
    accumulated = 0.0
    # Reference: Filothodoros thesis (ALEX.pdf), PDF p. 54,
    # Section 4.1.1, Eq. 4.7, for the Ljung-Box Q statistic.
    for lag in range(1, len(autocorrelation)):
        rho = autocorrelation[lag]
        if not np.isfinite(rho):
            q_values.append(np.nan)
            p_values.append(np.nan)
            continue
        accumulated += rho**2 / (sample_size - lag)
        q_value = float(sample_size * (sample_size + 2) * accumulated)
        q_values.append(q_value)
        p_values.append(float(chi2.sf(q_value, lag)))
    return np.asarray(q_values, dtype=float), np.asarray(p_values, dtype=float)


def _pearson_correlation(
    parameter_x: str,
    parameter_y: str,
    values_x: np.ndarray,
    values_y: np.ndarray,
    p_value_threshold: float,
) -> TemporalCorrelation:
    from scipy.stats import pearsonr

    paired = np.isfinite(values_x) & np.isfinite(values_y)
    n_paired = int(np.count_nonzero(paired))
    coefficient = None
    p_value = None
    if (
        n_paired >= 3
        and np.ptp(values_x[paired]) > 0
        and np.ptp(values_y[paired]) > 0
    ):
        # Reference: Filothodoros thesis (ALEX.pdf), PDF pp. 73-74,
        # Section 4.8, reports Pearson r and p-values for temporal ISM pairs.
        coefficient_value, p_value_value = pearsonr(
            values_x[paired], values_y[paired]
        )
        coefficient = _optional_float(coefficient_value)
        p_value = _optional_float(p_value_value)
    return TemporalCorrelation(
        parameter_x=parameter_x,
        parameter_y=parameter_y,
        n_paired=n_paired,
        pearson_r=coefficient,
        p_value=p_value,
        significant=(
            None if p_value is None else bool(p_value < p_value_threshold)
        ),
    )


def _measurement_arrays(measurements, parameter: str):
    value_field, error_field, _, _ = PARAMETER_FIELDS[parameter]
    mjd = np.asarray([float(item.mjd) for item in measurements], dtype=float)
    values = _optional_array(getattr(item, value_field) for item in measurements)
    errors = _optional_array(getattr(item, error_field) for item in measurements)
    return mjd, values, errors


def _optional_array(values) -> np.ndarray:
    return np.asarray(
        [np.nan if value is None else float(value) for value in values],
        dtype=float,
    )


def _optional_float(value) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _optional_tuple(values) -> tuple[float | None, ...]:
    return tuple(_optional_float(value) for value in values)
