"""Dynamic-spectrum flux extraction from archive profiles."""

from __future__ import annotations

import numpy as np


def single_pulse_window(archive, width_factor: int = 10) -> tuple[int, int]:
    """Find a centered on-pulse window for a single-peaked pulse."""
    temp = archive.clone()
    temp.tscrunch()
    temp.fscrunch()
    temp.remove_baseline()
    temp.centre_max_bin()

    prof = np.asarray(temp[0].get_Profile(0, 0).get_amps(), dtype=float)
    nbin = len(prof)
    prof = prof - np.min(prof)
    threshold = np.max(prof) / max(width_factor, 1)
    idx = np.where(prof > threshold)[0]

    if len(idx) == 0:
        return 0, 0

    return (nbin // 2) - (idx[0] - 1), (idx[-1] + 1) - (nbin // 2)


def interpulse_windows(archive, width_factor: int = 10) -> tuple[int, int, int, int]:
    """Find two on-pulse windows for a pulse with an interpulse."""
    temp = archive.clone()
    temp.tscrunch()
    temp.fscrunch()
    temp.remove_baseline()

    prof_obj = temp[0].get_Profile(0, 0)
    prof = np.asarray(prof_obj.get_amps(), dtype=float)
    nbin = len(prof)
    prof = np.roll(prof, (nbin // 4) - prof_obj.find_max_bin())
    prof = prof - np.min(prof)

    threshold = np.max(prof) / max(width_factor, 1)
    idx = np.where(prof > threshold)[0]

    if len(idx) == 0:
        return 0, 0, 0, 0

    groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    if len(groups) < 2:
        group = groups[0]
        return int(group[0]), int(group[-1]), int(group[0]), int(group[-1])

    main, inter = groups[0], groups[1]
    return int(main[0]), int(main[-1]), int(inter[0]), int(inter[-1])


def _safe_offpulse_indices(nbin: int, windows: list[tuple[int, int]]) -> np.ndarray:
    mask = np.ones(nbin, dtype=bool)
    for left, right in windows:
        left = max(0, left)
        right = min(nbin - 1, right)
        mask[left : right + 1] = False
    return np.where(mask)[0]


def fluxes_single_peak(archive, left_edge: int, right_edge: int) -> np.ndarray:
    """Calculate noise-normalized fluxes for a single-pulse dynamic spectrum."""
    temp = archive.clone()
    temp.remove_baseline()
    temp.centre_max_bin()

    nsub = temp.get_nsubint()
    nchan = temp.get_nchan()
    nbin = temp.get_nbin()

    w0 = max(0, (nbin // 2) - left_edge)
    w1 = min(nbin - 1, (nbin // 2) + right_edge)
    xon = np.arange(w0, w1 + 1)
    xoff = _safe_offpulse_indices(nbin, [(w0, w1)])

    dynspec = np.zeros((nsub, nchan))
    for isub in range(nsub):
        sub = temp[isub]
        for ichan in range(nchan):
            prof = np.asarray(sub.get_Profile(0, ichan).get_amps(), dtype=float)
            yoff = prof[xoff]
            std = np.std(yoff)
            if std == 0:
                continue
            cal = (prof[xon] - np.mean(yoff)) / std
            dynspec[isub, ichan] = max(np.trapezoid(cal) / len(cal), 0)

    return dynspec


def fluxes_two_peak(
    archive,
    left_main: int,
    right_main: int,
    left_inter: int,
    right_inter: int,
) -> np.ndarray:
    """Calculate noise-normalized fluxes for an interpulse dynamic spectrum."""
    temp = archive.clone()
    temp.remove_baseline()

    nsub = temp.get_nsubint()
    nchan = temp.get_nchan()
    nbin = temp.get_nbin()
    dynspec = np.zeros((nsub, nchan))

    xon1 = np.arange(max(0, left_main), min(nbin - 1, right_main) + 1)
    xon2 = np.arange(max(0, left_inter), min(nbin - 1, right_inter) + 1)
    xoff = _safe_offpulse_indices(nbin, [(left_main, right_main), (left_inter, right_inter)])

    for isub in range(nsub):
        sub = temp[isub]
        for ichan in range(nchan):
            prof_obj = sub.get_Profile(0, ichan)
            prof = np.asarray(prof_obj.get_amps(), dtype=float)
            prof = np.roll(prof, (nbin // 4) - prof_obj.find_max_bin())
            yoff = prof[xoff]
            std = np.std(yoff)
            if std == 0:
                continue
            mean = np.mean(yoff)
            cal1 = (prof[xon1] - mean) / std
            cal2 = (prof[xon2] - mean) / std
            dynspec[isub, ichan] = max(
                (np.trapezoid(cal1) + np.trapezoid(cal2)) / (len(cal1) + len(cal2)),
                0,
            )

    return dynspec
