"""Mixed-source archive discovery and cross-pulsar catalogue exports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import re
from statistics import median
from typing import Callable

import numpy as np


_PULSAR_NAME = re.compile(r"(?<![A-Za-z0-9])([BJ]\d{3,4}[+-]\d{2,4})(?!\d)", re.IGNORECASE)


@dataclass(frozen=True)
class BatchDiscoveryFailure:
    archive_path: str
    error: str


@dataclass(frozen=True)
class BatchIgnoredSource:
    archive_path: str
    source: str


@dataclass(frozen=True)
class BatchDiscovery:
    input_files: int
    groups: dict[str, tuple[Path, ...]]
    failures: tuple[BatchDiscoveryFailure, ...]
    ignored_non_pulsars: tuple[BatchIgnoredSource, ...] = ()


@dataclass(frozen=True)
class BatchPulsarRun:
    pulsar: str
    archive_paths: tuple[Path, ...]
    exit_code: int
    output_directory: Path
    time_series_csv: Path
    terminal_log: Path


@dataclass(frozen=True)
class BatchExportPaths:
    observations_csv: Path
    pulsars_csv: Path
    failures_csv: Path
    manifest_json: Path
    scattering_comparison_plot: Path | None


def discover_archive_groups(
    archive_paths: list[Path],
    load_archive: Callable[[str], object],
    selected_sources: set[str] | None = None,
) -> BatchDiscovery:
    """Load archive headers and group paths by source name."""
    selected = (
        None
        if selected_sources is None
        else {_normalize_pulsar_name(item).casefold() for item in selected_sources}
    )
    groups: dict[str, list[Path]] = {}
    failures: list[BatchDiscoveryFailure] = []
    ignored: list[BatchIgnoredSource] = []
    for path in sorted(archive_paths):
        try:
            archive = load_archive(str(path))
            raw_source = _raw_archive_source(archive)
            source = _pulsar_name_from_text(raw_source)
            if source is None:
                source = _pulsar_name_from_text(path.name)
            if source is None and raw_source:
                ignored.append(BatchIgnoredSource(str(path), raw_source))
                continue
            if source is None:
                raise ValueError("archive has no source name and none could be inferred from its filename")
        except Exception as exc:
            failures.append(BatchDiscoveryFailure(str(path), str(exc)))
            continue
        if selected is not None and source.casefold() not in selected:
            continue
        groups.setdefault(source, []).append(path)

    return BatchDiscovery(
        input_files=len(archive_paths),
        groups={key: tuple(value) for key, value in sorted(groups.items())},
        failures=tuple(failures),
        ignored_non_pulsars=tuple(ignored),
    )


def write_batch_exports(
    runs: list[BatchPulsarRun],
    discovery: BatchDiscovery,
    output_prefix: Path,
    input_root: Path,
    pattern: str,
    recursive: bool,
    selected_sources: list[str],
    requested_parameters: str,
) -> BatchExportPaths:
    """Combine per-pulsar time-series tables into PSRDEX-friendly exports."""
    observations_path = output_prefix.parent / f"{output_prefix.name}_observations.csv"
    pulsars_path = output_prefix.parent / f"{output_prefix.name}_pulsars.csv"
    failures_path = output_prefix.parent / f"{output_prefix.name}_failures.csv"
    manifest_path = output_prefix.parent / f"{output_prefix.name}_manifest.json"
    comparison_path = (
        output_prefix.parent / f"{output_prefix.name}_alpha_vs_tau_ref_fraction.png"
    )

    observations: list[dict[str, str]] = []
    observation_fields: list[str] = []
    failures: list[dict[str, str]] = [
        {
            "archive_path": item.archive_path,
            "pulsar": "",
            "stage": "source_discovery",
            "error": item.error,
        }
        for item in discovery.failures
    ]
    summaries: list[dict[str, object]] = []
    manifest_runs: list[dict[str, object]] = []

    for run in runs:
        rows, fields = _read_time_series_rows(run.time_series_csv) if run.exit_code == 0 else ([], [])
        for field in fields:
            if field not in observation_fields:
                observation_fields.append(field)
        successful_paths = {
            str(Path(row["archive_path"]).resolve())
            for row in rows
            if row.get("archive_path")
        }
        for row in rows:
            observations.append({"pulsar": run.pulsar, **row})
        for path in run.archive_paths:
            if str(path.resolve()) not in successful_paths:
                failures.append(
                    {
                        "archive_path": str(path),
                        "pulsar": run.pulsar,
                        "stage": "epoch_processing",
                        "error": (
                            "pulsar run failed; see terminal log"
                            if run.exit_code != 0
                            else "epoch was skipped; see terminal log"
                        ),
                    }
                )

        summary = _summarize_pulsar_run(run, rows)
        summaries.append(summary)
        manifest_runs.append(
            {
                **summary,
                "archive_paths": [str(path) for path in run.archive_paths],
                "time_series_csv": str(run.time_series_csv),
                "terminal_log": str(run.terminal_log),
                "output_directory": str(run.output_directory),
            }
        )

    _write_csv(
        observations_path,
        observations,
        ["pulsar", *[field for field in observation_fields if field != "pulsar"]],
    )
    summary_fields = [
        "pulsar",
        "status",
        "files_discovered",
        "epochs_processed",
        "epochs_skipped",
        "first_mjd",
        "last_mjd",
        "time_span_days",
        "frequency_min_mhz",
        "frequency_max_mhz",
        "period_s_median",
        "dm_median",
        "tau_s_median",
        "tau_ref_s_median",
        "alpha_median",
        "alpha_weighted_mean",
        "alpha_weighted_mean_error",
        "tau_ref_s_weighted_mean",
        "tau_ref_s_weighted_mean_error",
        "tau_ref_fraction_period",
        "tau_ref_fraction_period_error",
        "anisotropic_alpha_weighted_mean",
        "anisotropic_alpha_weighted_mean_error",
        "anisotropic_tau_ref_s_weighted_mean",
        "anisotropic_tau_ref_s_weighted_mean_error",
        "anisotropic_tau_ref_fraction_period",
        "anisotropic_tau_ref_fraction_period_error",
        "decorrelation_bandwidth_mhz_median",
        "diffractive_timescale_s_median",
        "refractive_timescale_days_median",
        "time_series_csv",
        "output_directory",
    ]
    _write_csv(pulsars_path, summaries, summary_fields)
    _write_csv(
        failures_path,
        failures,
        ["archive_path", "pulsar", "stage", "error"],
    )
    comparison_plot = (
        comparison_path
        if plot_cross_pulsar_scattering_summary(summaries, comparison_path)
        else None
    )

    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "pattern": pattern,
        "recursive": bool(recursive),
        "selected_sources": list(selected_sources),
        "requested_parameters": requested_parameters,
        "input_files": discovery.input_files,
        "sources_discovered": len(discovery.groups),
        "source_discovery_failures": len(discovery.failures),
        "non_pulsar_archives_ignored": len(discovery.ignored_non_pulsars),
        "pulsar_runs_succeeded": sum(run.exit_code == 0 for run in runs),
        "pulsar_runs_failed": sum(run.exit_code != 0 for run in runs),
        "epochs_exported": len(observations),
        "products": {
            "observations_csv": str(observations_path),
            "pulsars_csv": str(pulsars_path),
            "failures_csv": str(failures_path),
            "scattering_comparison_plot": (
                None if comparison_plot is None else str(comparison_plot)
            ),
        },
        "discovery_failures": [asdict(item) for item in discovery.failures],
        "ignored_non_pulsars": [asdict(item) for item in discovery.ignored_non_pulsars],
        "runs": manifest_runs,
    }
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return BatchExportPaths(
        observations_csv=observations_path,
        pulsars_csv=pulsars_path,
        failures_csv=failures_path,
        manifest_json=manifest_path,
        scattering_comparison_plot=comparison_plot,
    )


def _raw_archive_source(archive) -> str:
    try:
        return str(archive.get_source()).strip()
    except (AttributeError, TypeError, ValueError):
        return ""


def _pulsar_name_from_text(value: str) -> str | None:
    normalized = re.sub(r"^\s*PSR\s*", "", str(value), flags=re.IGNORECASE)
    match = _PULSAR_NAME.search(normalized)
    return None if match is None else match.group(1).upper()


def _normalize_pulsar_name(value: str) -> str:
    return _pulsar_name_from_text(value) or str(value).strip()


def _read_time_series_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        return [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _summarize_pulsar_run(
    run: BatchPulsarRun,
    rows: list[dict[str, str]],
) -> dict[str, object]:
    mjd = _finite_values(rows, "mjd")
    frequency = _finite_values(rows, "observing_frequency_mhz")
    period = _finite_values(rows, "period_s")
    first_mjd = min(mjd) if mjd else None
    last_mjd = max(mjd) if mjd else None
    period_median = None if not period else float(median(period))
    alpha_weighted, alpha_weighted_error = _weighted_mean_field(
        rows, "alpha", "alpha_error"
    )
    tau_weighted, tau_weighted_error = _weighted_mean_field(
        rows, "tau0_s", "tau0_error_s"
    )
    anisotropic_alpha, anisotropic_alpha_error = _weighted_mean_field(
        rows, "anisotropic_alpha", "anisotropic_alpha_error"
    )
    anisotropic_tau, anisotropic_tau_error = _weighted_mean_field(
        rows, "anisotropic_tau0_s", "anisotropic_tau0_error_s"
    )
    return {
        "pulsar": run.pulsar,
        "status": "succeeded" if run.exit_code == 0 else "failed",
        "files_discovered": len(run.archive_paths),
        "epochs_processed": len(rows),
        "epochs_skipped": max(len(run.archive_paths) - len(rows), 0),
        "first_mjd": first_mjd,
        "last_mjd": last_mjd,
        "time_span_days": (
            None if first_mjd is None or last_mjd is None else last_mjd - first_mjd
        ),
        "frequency_min_mhz": min(frequency) if frequency else None,
        "frequency_max_mhz": max(frequency) if frequency else None,
        "period_s_median": period_median,
        "dm_median": _median_field(rows, "dm"),
        "tau_s_median": _median_field(rows, "tau_s"),
        "tau_ref_s_median": _median_field(rows, "tau0_s"),
        "alpha_median": _median_field(rows, "alpha"),
        "alpha_weighted_mean": alpha_weighted,
        "alpha_weighted_mean_error": alpha_weighted_error,
        "tau_ref_s_weighted_mean": tau_weighted,
        "tau_ref_s_weighted_mean_error": tau_weighted_error,
        "tau_ref_fraction_period": _ratio_optional(tau_weighted, period_median),
        "tau_ref_fraction_period_error": _ratio_optional(
            tau_weighted_error, period_median
        ),
        "anisotropic_alpha_weighted_mean": anisotropic_alpha,
        "anisotropic_alpha_weighted_mean_error": anisotropic_alpha_error,
        "anisotropic_tau_ref_s_weighted_mean": anisotropic_tau,
        "anisotropic_tau_ref_s_weighted_mean_error": anisotropic_tau_error,
        "anisotropic_tau_ref_fraction_period": _ratio_optional(
            anisotropic_tau, period_median
        ),
        "anisotropic_tau_ref_fraction_period_error": _ratio_optional(
            anisotropic_tau_error, period_median
        ),
        "decorrelation_bandwidth_mhz_median": _median_field(
            rows, "decorrelation_bandwidth_mhz"
        ),
        "diffractive_timescale_s_median": _median_field(
            rows, "diffractive_timescale_s"
        ),
        "refractive_timescale_days_median": _median_field(
            rows, "refractive_timescale_days"
        ),
        "time_series_csv": str(run.time_series_csv),
        "output_directory": str(run.output_directory),
    }


def _finite_values(rows: list[dict[str, str]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(field, ""))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def _median_field(rows: list[dict[str, str]], field: str) -> float | None:
    values = _finite_values(rows, field)
    # PSRISM catalogue choice: medians provide compact, outlier-resistant
    # summaries. They are exports for browsing, not population estimators.
    return None if not values else float(median(values))


def _weighted_mean_field(
    rows: list[dict[str, str]],
    value_field: str,
    error_field: str,
) -> tuple[float | None, float | None]:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        try:
            value = float(row.get(value_field, ""))
            error = float(row.get(error_field, ""))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and np.isfinite(error) and error > 0:
            pairs.append((value, error))
    if not pairs:
        return None, None
    values = np.asarray([item[0] for item in pairs], dtype=float)
    errors = np.asarray([item[1] for item in pairs], dtype=float)
    weights = 1.0 / errors**2
    # Reference: Filothodoros thesis (ALEX.pdf), PDF pp. 43-44,
    # Section 3.2 and Table 3.1, use inverse-square uncertainty weights for
    # the per-pulsar mean alpha and tau150 values.
    return (
        float(np.sum(weights * values) / np.sum(weights)),
        float(np.sqrt(1.0 / np.sum(weights))),
    )


def _ratio_optional(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return float(numerator / denominator)


def plot_cross_pulsar_scattering_summary(
    summaries: list[dict[str, object]],
    output_path: str | Path,
) -> bool:
    """Plot weighted alpha against weighted reference tau as a period fraction."""
    import matplotlib.pyplot as plt

    model_fields = (
        (
            "Isotropic",
            "tau_ref_fraction_period",
            "tau_ref_fraction_period_error",
            "alpha_weighted_mean",
            "alpha_weighted_mean_error",
            "o",
            "tab:blue",
        ),
        (
            "Anisotropic",
            "anisotropic_tau_ref_fraction_period",
            "anisotropic_tau_ref_fraction_period_error",
            "anisotropic_alpha_weighted_mean",
            "anisotropic_alpha_weighted_mean_error",
            "s",
            "tab:orange",
        ),
    )
    plotted: list[tuple[str, float, float, float | None, float | None, str, str]] = []
    for summary in summaries:
        for label, x_field, xerr_field, y_field, yerr_field, marker, color in model_fields:
            x = _as_optional_float(summary.get(x_field))
            y = _as_optional_float(summary.get(y_field))
            if x is None or y is None:
                continue
            plotted.append(
                (
                    str(summary["pulsar"]),
                    x,
                    y,
                    _as_optional_float(summary.get(xerr_field)),
                    _as_optional_float(summary.get(yerr_field)),
                    marker,
                    color,
                )
            )
    if not plotted:
        return False

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    used_labels: set[str] = set()
    for pulsar, x, y, xerr, yerr, marker, color in plotted:
        model_label = "Isotropic" if marker == "o" else "Anisotropic"
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt=marker,
            color=color,
            capsize=3,
            label=model_label if model_label not in used_labels else None,
        )
        used_labels.add(model_label)
        ax.annotate(pulsar, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)

    # Reference: Filothodoros thesis (ALEX.pdf), PDF pp. 43-44,
    # Section 3.2 and Fig. 3.1, compare weighted alpha with weighted tau150
    # expressed as a fraction of pulse period for both scattering models.
    ax.set_xlabel("Weighted reference scattering time / pulse period")
    ax.set_ylabel("Weighted scattering spectral index alpha")
    ax.set_title("Cross-pulsar scattering comparison")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def _as_optional_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return ""
    return value


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value
