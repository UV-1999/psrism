"""Archive loading, preprocessing, and conversion to NumPy arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ArchiveMetadata:
    filename: str
    outname: str
    raw_shape: tuple[int, ...]
    processed_shape: tuple[int, ...]
    observation_time_s: float
    centre_frequency_mhz: float
    bandwidth_mhz: float
    frequency_low_mhz: float
    frequency_high_mhz: float
    dispersion_measure: float
    telescope: str


def load_archive(path: str):
    """Load a PSRCHIVE-supported archive."""
    import psrchive

    try:
        return psrchive.Archive_load(path)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not load archive '{path}' with PSRCHIVE. "
            "The file may be incomplete/corrupt, unsupported by this PSRCHIVE build, "
            "or missing external data blocks."
        ) from exc


def preprocess_archive(
    archive,
    dm: Optional[float] = None,
    nsub: Optional[int] = None,
    nchan: Optional[int] = None,
    nbin: Optional[int] = None,
):
    """Dedisperse, pscrunch, and optionally scrunch an archive in place."""
    if dm is not None and dm > 0.0:
        archive.set_dispersion_measure(dm)

    archive.dedisperse()
    archive.pscrunch()

    if nchan:
        _validate_scrunch_target("nchan", archive.get_nchan(), nchan)
        archive.fscrunch_to_nchan(nchan)
    if nsub:
        _validate_scrunch_target("nsub", archive.get_nsubint(), nsub)
        archive.tscrunch_to_nsub(nsub)
    if nbin:
        _validate_scrunch_target("nbin", archive.get_nbin(), nbin)
        archive.bscrunch_to_nbin(nbin)

    return archive


def _validate_scrunch_target(name: str, current: int, target: int) -> None:
    if target <= 0:
        raise ValueError(f"--{name} must be positive")
    if target > current:
        raise ValueError(f"--{name}={target} is larger than current {name}={current}")
    if current % target != 0:
        raise ValueError(
            f"--{name}={target} is not valid for current {name}={current}; "
            f"choose one of: {format_scrunch_targets(current)}"
        )


def valid_scrunch_targets(current: int) -> list[int]:
    """Return all target sizes that PSRCHIVE can scrunch to for a dimension."""
    if current <= 0:
        return []
    return [value for value in range(1, current + 1) if current % value == 0]


def suggested_scrunch_targets(current: int, max_items: int = 16) -> list[int]:
    """Return a compact list of useful smaller scrunch targets."""
    values = [value for value in valid_scrunch_targets(current) if value < current]
    if len(values) <= max_items:
        return values
    head_count = max_items // 2
    tail_count = max_items - head_count
    return values[:head_count] + values[-tail_count:]


def format_scrunch_targets(current: int) -> str:
    """Format valid scrunch targets for terminal messages."""
    targets = suggested_scrunch_targets(current)
    if not targets:
        return "none"
    all_targets = [value for value in valid_scrunch_targets(current) if value < current]
    text = ", ".join(str(value) for value in targets)
    if len(all_targets) > len(targets):
        text += f" ... ({len(all_targets)} total smaller divisors)"
    return text


def archive_metadata(archive, raw_shape: Optional[tuple[int, ...]] = None) -> ArchiveMetadata:
    """Collect common metadata used by the CLI and plotting functions."""
    filename = archive.get_filename()
    tobs = archive.integration_length()
    freq_c = archive.get_centre_frequency()
    bw = abs(archive.get_bandwidth())
    freq_lo = freq_c - bw / 2.0
    freq_hi = freq_c + bw / 2.0

    return ArchiveMetadata(
        filename=filename,
        outname=str(Path(filename).with_suffix("")),
        raw_shape=tuple(raw_shape or archive_shape(archive)),
        processed_shape=archive_shape(archive),
        observation_time_s=tobs,
        centre_frequency_mhz=freq_c,
        bandwidth_mhz=bw,
        frequency_low_mhz=freq_lo,
        frequency_high_mhz=freq_hi,
        dispersion_measure=archive.get_dispersion_measure(),
        telescope=archive.get_telescope(),
    )


def archive_shape(archive) -> tuple[int, int, int, int]:
    """Return archive shape as (nsub, npol, nchan, nbin) without loading data."""
    return (
        archive.get_nsubint(),
        archive.get_npol(),
        archive.get_nchan(),
        archive.get_nbin(),
    )


def archive_data_to_numpy(archive, remove_baseline: bool = True) -> np.ndarray:
    """Reduce PSRCHIVE data to a NumPy array shaped as (nsub, npol, nchan, nbin)."""
    temp = archive.clone()
    if remove_baseline:
        temp.remove_baseline()
    return np.asarray(temp.get_data())


def integrated_profile_array(archive, normalize: bool = True) -> np.ndarray:
    """Return a baseline-removed integrated pulse profile."""
    temp = archive.clone()
    temp.tscrunch()
    temp.fscrunch()
    temp.remove_baseline()

    profile = np.asarray(temp[0].get_Profile(0, 0).get_amps(), dtype=float)
    if normalize:
        profile = profile - np.min(profile)
        mx = np.max(profile)
        if mx != 0:
            profile = profile / mx
    return profile


def time_phase_array(archive) -> np.ndarray:
    """Return a time-vs-phase matrix after frequency scrunching."""
    temp = archive.clone()
    temp.fscrunch()
    temp.remove_baseline()

    nsub = temp.get_nsubint()
    nbin = temp.get_nbin()
    matrix = np.zeros((nsub, nbin))

    for isub in range(nsub):
        prof_obj = temp[isub].get_Profile(0, 0)
        prof = np.asarray(prof_obj.get_amps(), dtype=float)
        matrix[isub] = np.roll(prof, (nbin // 4) - prof_obj.find_max_bin())

    return matrix


def frequency_phase_array(archive) -> np.ndarray:
    """Return a frequency-vs-phase matrix after time scrunching."""
    temp = archive.clone()
    temp.tscrunch()
    temp.remove_baseline()

    nchan = temp.get_nchan()
    nbin = temp.get_nbin()
    matrix = np.zeros((nchan, nbin))

    for ichan in range(nchan):
        prof_obj = temp[0].get_Profile(0, ichan)
        prof = np.asarray(prof_obj.get_amps(), dtype=float)
        matrix[ichan] = np.roll(prof, (nbin // 4) - prof_obj.find_max_bin())

    return matrix
