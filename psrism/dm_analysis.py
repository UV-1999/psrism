"""Dispersion-measure estimation and scattering-aware diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np


# Reference: Lorimer & Kramer (2005), psrhandbook.pdf, Section 4.1.1,
# Eq. 4.6. Units are MHz^2 pc^-1 cm^3 s.
DISPERSION_CONSTANT_MHZ2_S = 4.148808e3

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_DM_RESULT = re.compile(
    rf"Best\s+DM\s*=\s*({_NUMBER})\s+Correction\s*=\s*({_NUMBER})"
    rf"\s+Error\s*=\s*({_NUMBER})",
    re.IGNORECASE,
)
_SNR_RESULT = re.compile(rf"Best\s+S/N\s*=\s*({_NUMBER})", re.IGNORECASE)
_PERIOD_RESULT = re.compile(
    rf"Best\s+BC\s+Period\s*\(ms\)\s*=\s*({_NUMBER}).*?Error\s*\(ms\)\s*=\s*({_NUMBER})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PdmpResult:
    archive_path: str
    archive_dm: float
    best_dm: float
    dm_correction: float
    dm_error: float
    best_snr: float | None
    best_barycentric_period_ms: float | None
    period_error_ms: float | None
    executable: str
    command: tuple[str, ...]
    stdout: str


@dataclass(frozen=True)
class ScatteringDMResult:
    input_dm: float
    delta_dm: float
    delta_dm_error: float
    corrected_dm: float
    reference_frequency_mhz: float
    intercept_s: float
    chi_square: float
    reduced_chi_square: float | None
    rms_residual_s: float
    n_points: int


def run_pdmp(
    archive_path: str | Path,
    archive_dm: float,
    executable: str = "pdmp",
    tempo_path: str | Path | None = None,
    dm_half_range: float | None = None,
    dm_step: float | None = None,
    max_channels: int | None = None,
    max_subints: int | None = None,
    max_bins: int | None = None,
    timeout_s: float = 300.0,
) -> PdmpResult:
    """Run PSRCHIVE ``pdmp`` and parse its best-DM result."""
    # Reference: Filothodoros et al. (Alex_and_Boe.pdf), PDF p. 3,
    # Section 2.1, and the Filothodoros thesis (ALEX.pdf), PDF p. 68,
    # Section 4.6, use pdmp's profile-S/N maximization for each observation.
    archive = Path(archive_path).expanduser().resolve()
    if not archive.is_file():
        raise ValueError(f"archive does not exist: {archive}")
    if not np.isfinite(archive_dm):
        raise ValueError("archive_dm must be finite")
    # PSRISM choice: the 300 s default is an operational guard, not a
    # literature-derived scientific constant.
    if not np.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be positive")

    resolved_executable = shutil.which(executable)
    if resolved_executable is None:
        candidate = Path(executable).expanduser()
        if not candidate.is_file():
            raise RuntimeError(
                f"pdmp executable not found: {executable!r}; activate the PSRCHIVE "
                "environment or pass --pdmp-executable"
            )
        resolved_executable = str(candidate.resolve())

    # PSRISM integration choice: force non-interactive execution and use the
    # PGPLOT null device; these flags do not change the scientific DM grid.
    command = [resolved_executable, "-f", "-g", "/null"]
    _append_positive_option(command, "-dr", dm_half_range)
    _append_positive_option(command, "-ds", dm_step)
    _append_positive_int_option(command, "-mc", max_channels)
    _append_positive_int_option(command, "-ms", max_subints)
    _append_positive_int_option(command, "-mb", max_bins)
    command.append(str(archive))

    environment = os.environ.copy()
    if tempo_path is not None:
        tempo = Path(tempo_path).expanduser().resolve()
        if not tempo.is_dir():
            raise ValueError(f"TEMPO directory does not exist: {tempo}")
        environment["TEMPO"] = str(tempo)

    # PSRISM choice: run in an isolated temporary directory because pdmp writes
    # pdmp.posn and pdmp.per in its working directory even for non-plot runs.
    try:
        with tempfile.TemporaryDirectory(prefix="psrism-pdmp-") as workdir:
            completed = subprocess.run(
                command,
                cwd=workdir,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=float(timeout_s),
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"pdmp timed out after {timeout_s:g} s") from exc
    except OSError as exc:
        raise RuntimeError(f"could not execute pdmp: {exc}") from exc

    output = completed.stdout or ""
    match = _DM_RESULT.search(output)
    if completed.returncode != 0:
        detail = _compact_failure_output(output)
        raise RuntimeError(
            f"pdmp failed with exit status {completed.returncode}"
            + (f": {detail}" if detail else "")
        )
    if match is None:
        detail = _compact_failure_output(output)
        raise RuntimeError(
            "pdmp did not produce a Best DM result"
            + (f": {detail}" if detail else f" (exit status {completed.returncode})")
        )

    best_dm, correction, error = (float(value) for value in match.groups())
    if (
        not all(np.isfinite(value) for value in (best_dm, correction, error))
        or best_dm < 0
        or error <= 0
    ):
        raise RuntimeError(
            "pdmp returned a non-finite/negative DM or non-positive DM error"
        )
    snr_match = _SNR_RESULT.search(output)
    period_match = _PERIOD_RESULT.search(output)
    best_snr = float(snr_match.group(1)) if snr_match else None
    best_period = float(period_match.group(1)) if period_match else None
    period_error = float(period_match.group(2)) if period_match else None

    return PdmpResult(
        archive_path=str(archive),
        archive_dm=float(archive_dm),
        best_dm=best_dm,
        dm_correction=correction,
        dm_error=error,
        best_snr=best_snr,
        best_barycentric_period_ms=best_period,
        period_error_ms=period_error,
        executable=resolved_executable,
        command=tuple(command),
        stdout=output,
    )


def scattering_dm_from_subbands(
    frequency_mhz,
    intrinsic_phase_bins,
    period_s: float,
    nbin: int,
    input_dm: float,
    phase_error_bins=None,
    reference_frequency_mhz: float | None = None,
) -> ScatteringDMResult:
    """Estimate a residual DM from deconvolved intrinsic-profile phases."""
    frequency = np.asarray(frequency_mhz, dtype=float)
    phase = np.asarray(intrinsic_phase_bins, dtype=float)
    phase_error = (
        None if phase_error_bins is None else np.asarray(phase_error_bins, dtype=float)
    )
    if frequency.shape != phase.shape:
        raise ValueError("frequency_mhz and intrinsic_phase_bins must have the same shape")
    if phase_error is not None and phase_error.shape != phase.shape:
        raise ValueError("phase_error_bins and intrinsic_phase_bins must have the same shape")
    if not np.isfinite(period_s) or period_s <= 0:
        raise ValueError("period_s must be positive")
    if nbin <= 0:
        raise ValueError("nbin must be positive")
    if not np.isfinite(input_dm):
        raise ValueError("input_dm must be finite")

    valid = np.isfinite(frequency) & np.isfinite(phase) & (frequency > 0)
    if phase_error is not None:
        valid &= np.isfinite(phase_error) & (phase_error > 0)
    frequency = frequency[valid]
    phase = phase[valid]
    if phase_error is not None:
        phase_error = phase_error[valid]
    # PSRISM choice: three points are the minimum that leaves a residual
    # degree of freedom after fitting a slope and intercept.
    if len(frequency) < 3:
        raise ValueError("at least three valid subbands are required for a scattering-DM check")

    order = np.argsort(frequency)
    frequency = frequency[order]
    phase = phase[order]
    if phase_error is not None:
        phase_error = phase_error[order]
    # PSRISM choice: unwrap periodic phase in ascending-frequency order.
    unwrapped_bins = np.unwrap(2.0 * np.pi * phase / nbin) * nbin / (2.0 * np.pi)
    arrival_s = unwrapped_bins * (period_s / nbin)
    # PSRISM choice for direct API calls: use the geometric-mean frequency
    # when the caller does not provide a reference frequency.
    reference = float(
        reference_frequency_mhz
        if reference_frequency_mhz is not None
        else np.exp(np.mean(np.log(frequency)))
    )
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("reference_frequency_mhz must be positive")

    # Reference: Lorimer & Kramer (2005), psrhandbook.pdf, Section 4.1.1,
    # Eqs. 4.4-4.7. Fitting deconvolved intrinsic-profile phases addresses the
    # scattering bias described by Geyer & Karastergiou (2018),
    # anomalous-pulsar-scattering-at-lofar-frequencies.pdf, PDF p. 3, Section 3.
    dispersion_axis = DISPERSION_CONSTANT_MHZ2_S * (
        np.power(frequency, -2.0) - reference**-2.0
    )
    design = np.column_stack([dispersion_axis, np.ones_like(dispersion_axis)])
    # PSRISM choice: use phase-covariance weighting when every phase error is
    # valid; otherwise direct callers receive an unweighted regression whose
    # covariance is scaled by the residual variance below.
    if phase_error is None:
        sigma_s = np.ones_like(arrival_s)
    else:
        sigma_s = phase_error * (period_s / nbin)
    weights = 1.0 / np.square(sigma_s)
    normal = design.T @ (weights[:, np.newaxis] * design)
    try:
        covariance = np.linalg.inv(normal)
    except np.linalg.LinAlgError as exc:
        raise ValueError("subband phases do not constrain a unique DM correction") from exc
    coefficients = covariance @ (design.T @ (weights * arrival_s))
    residual = arrival_s - design @ coefficients
    chi_square = float(np.sum(np.square(residual / sigma_s)))
    dof = len(arrival_s) - 2
    reduced = float(chi_square / dof) if dof > 0 else None
    if phase_error is None and dof > 0:
        covariance *= reduced

    delta_dm = float(coefficients[0])
    delta_dm_error = float(np.sqrt(max(covariance[0, 0], 0.0)))
    return ScatteringDMResult(
        input_dm=float(input_dm),
        delta_dm=delta_dm,
        delta_dm_error=delta_dm_error,
        corrected_dm=float(input_dm + delta_dm),
        reference_frequency_mhz=reference,
        intercept_s=float(coefficients[1]),
        chi_square=chi_square,
        reduced_chi_square=reduced,
        rms_residual_s=float(np.sqrt(np.mean(np.square(residual)))),
        n_points=int(len(arrival_s)),
    )


def write_dm_report(
    output_path: str | Path,
    pdmp_result: PdmpResult | None,
    scattering_result: ScatteringDMResult | None,
) -> None:
    payload = {
        "pdmp": None if pdmp_result is None else _pdmp_payload(pdmp_result),
        "scattering_dm": None if scattering_result is None else asdict(scattering_result),
    }
    with Path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _pdmp_payload(result: PdmpResult) -> dict:
    payload = asdict(result)
    payload["command"] = list(result.command)
    # PSRISM choice: parsed values and the exact command keep the JSON audit
    # record compact; pdmp's potentially long raw console output is omitted.
    payload.pop("stdout", None)
    return payload


def _append_positive_option(command: list[str], name: str, value: float | None) -> None:
    if value is None:
        return
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} value must be positive")
    command.extend([name, f"{value:g}"])


def _append_positive_int_option(command: list[str], name: str, value: int | None) -> None:
    if value is None:
        return
    if value <= 0:
        raise ValueError(f"{name} value must be positive")
    command.extend([name, str(int(value))])


def _compact_failure_output(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return " | ".join(lines[-8:])
