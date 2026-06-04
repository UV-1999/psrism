#!/usr/bin/env python3

import os
import argparse

parser = argparse.ArgumentParser(
    usage='%(prog)s [ARCHIVE] [OPTION]...',
    description="""
psrism: Pulsar ISM analysis tool
Author: Piyush Marmat, PhD Student
Purpose: extract ISM related parameters from pulsar data.
""",
    formatter_class=argparse.RawDescriptionHelpFormatter
)

# positional argument
parser.add_argument(
    'archive',
    help='Pulsar data file (PSRCHIVE-supported format)'
)
# plotting control flags
group_plot = parser.add_argument_group('Plotting options')
group_plot.add_argument(
    '--dspec',
    action='store_true',
    help='plot dynamic spectrum'
)
group_plot.add_argument(
    '--sspec',
    action='store_true',
    help='plot secondary spectrum'
)
group_plot.add_argument(
    '--acspec',
    action='store_true',
    help='plot autocorrelation spectrum'
)
group_plot.add_argument(
    '--intpf',
    action='store_true',
    help='plot integrated pulse profile'
)
# metadata / processing options
group_meta = parser.add_argument_group('Processing options')
group_meta.add_argument(
    '--dm',
    type=float,
    help='dispersion measure to apply (pc cm^-3)'
)
group_meta.add_argument(
    '--nsub',
    type=int,
    help='scrunch to N subintegrations'
)
group_meta.add_argument(
    '--nchan',
    type=int,
    help='scrunch to N frequency channels'
)
group_meta.add_argument(
    '--nbin',
    type=int,
    help='scrunch to N phase bins'
)
group_meta.add_argument(
    '--interpulse',
    type=int,
    choices=[0, 1],
    default=0,
    help='0 = single pulse, 1 = interpulse present'
)
group_meta.add_argument(
    '--onw',
    type=int,
    default=10,
    help='on-pulse width factor'
)
args = parser.parse_args()

import psrchive

archive = psrchive.Archive_load(args.archive)

print(f"\nLoaded archive: {args.archive}")

outname = os.path.splitext(archive.get_filename())[0]
#outname = args.archive

# Original state
raw_shape = archive.get_data().shape
axes = ["Nsub (time)", "Npol", "Nchan (freq)", "Nbin (phase)"]

print("\nOriginal data shape:")
for i, (s, name) in enumerate(zip(raw_shape, axes)):
    print(f" Axis {i}: {name} = {s}")

tobs = archive.integration_length()
freq_c = archive.get_centre_frequency()
bw = abs(archive.get_bandwidth())
freq_lo = freq_c - bw / 2.0
freq_hi = freq_c + bw / 2.0
dm = archive.get_dispersion_measure()

print("\nObservation metadata:")
print(f" Observation time: {tobs:.3f} s ({tobs/60.0:.3f} min or {tobs/3600.0:.3f} hour(s))")
print(f" Centre frequency: {freq_c:.3f} MHz")
print(f" Bandwidth: {bw:.3f} MHz")
print(f" Frequency range: {freq_lo:.3f} MHz to {freq_hi:.3f} MHz")
print(f" DM: {dm:.6f} pc cm^-3")
print(f"Telescope used: {archive.get_telescope()}")

# DM handling
if args.dm is not None and args.dm > 0.0:
    archive.set_dispersion_measure(args.dm)
archive.dedisperse()
archive.pscrunch()

# On-pulse width factor
N = args.onw if args.onw > 0 else 10

# User-defined scrunching
if args.nchan:
    archive.fscrunch_to_nchan(args.nchan)
if args.nsub:
    archive.tscrunch_to_nsub(args.nsub)
if args.nbin:
    archive.bscrunch_to_nbin(args.nbin)

proc_shape = archive.get_data().shape
print("\nProcessed data shape:")
for i, (s, name) in enumerate(zip(proc_shape, axes)):
    print(f" Axis {i}: {name} = {s}")
nsub, npol, nchan, nbin = proc_shape

def plot_intpf(archive, N, outname, tobs, freq_lo, freq_hi):

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # -------- TIME-RESOLVED (fscrunch) --------
    temp_t = archive.clone()
    temp_t.fscrunch()
    temp_t.remove_baseline()

    nsub = temp_t.get_nsubint()
    nbin = temp_t.get_nbin()

    mat_time = np.zeros((nsub, nbin))

    for isub in range(nsub):
        prof = temp_t[isub].get_Profile(0, 0).get_amps()
        b0 = temp_t[isub].get_Profile(0, 0).find_max_bin()

        shift = (nbin // 4) - b0
        prof = np.roll(prof, shift)

        mat_time[isub] = prof

    # -------- FREQ-RESOLVED (tscrunch) --------
    temp_f = archive.clone()
    temp_f.tscrunch()
    temp_f.remove_baseline()

    nchan = temp_f.get_nchan()

    mat_freq = np.zeros((nchan, nbin))

    for ichan in range(nchan):
        prof = temp_f[0].get_Profile(0, ichan).get_amps()
        b0 = temp_f[0].get_Profile(0, ichan).find_max_bin()

        shift = (nbin // 4) - b0
        prof = np.roll(prof, shift)

        mat_freq[ichan] = prof

    # -------- AXES --------
    phase_axis = np.linspace(0, 1, nbin)

    dt = tobs / nsub
    time_axis = np.arange(nsub) * dt

    freq_axis = np.linspace(freq_lo, freq_hi, nchan)

    # -------- INTEGRATED PROFILE --------
    prof_int = np.mean(mat_time, axis=0)

    # -------- PLOTTING --------
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 2])

    ax_top  = fig.add_subplot(gs[0])
    ax_freq = fig.add_subplot(gs[1], sharex=ax_top)
    ax_time = fig.add_subplot(gs[2], sharex=ax_top)

    # Integrated profile
    ax_top.plot(phase_axis, prof_int, color='black')
    mx = np.max(prof_int)
    ax_top.axhline(mx, color="teal", label="max")
    ax_top.axhline(mx / N, color="green", label=f"1/{N} max")
    ax_top.axhline(mx / 10.0, color="red", label="1/10 max")
    ax_top.axhline(mx / 2.0, color="blue", label="1/2 max")
    ax_top.set_ylabel("Flux")
    ax_top.set_title("Integrated Pulse Profile")
    ax_top.legend(loc="best")

    # Frequency vs phase
    ax_freq.imshow(
        mat_freq,
        aspect="auto",
        origin="lower",
        cmap="afmhot",
        extent=(0, 1, freq_axis.min(), freq_axis.max())
    )
    ax_freq.set_ylabel("Frequency (MHz)")
    ax_freq.set_title("Frequency vs Phase")

    # Time vs phase
    ax_time.imshow(
        mat_time,
        aspect="auto",
        origin="lower",
        cmap="afmhot",
        extent=(0, 1, time_axis.min(), time_axis.max())
    )
    ax_time.set_ylabel("Time (s)")
    ax_time.set_xlabel("Pulse phase")
    ax_time.set_title("Time vs Phase")

    plt.tight_layout()
    plt.savefig(outname + "_intpf.png")

def plot_dspec(dynspec, archive, tobs, freq_lo, freq_hi, outname):
    import numpy as np
    import matplotlib.pyplot as plt

    nsub = archive.get_nsubint()
    nchan = archive.get_nchan()

    arr = dynspec
    # projections
    freq_proj = np.mean(arr, axis=0)   # time-averaged flux vs frequency
    time_proj = np.mean(arr, axis=1)   # frequency-averaged flux vs time

    fig, ax = plt.subplots(2, 2, figsize=(8, 6),
                           gridspec_kw={'height_ratios': [1, 4],
                                        'width_ratios': [4, 1]})

    # main dynamic spectrum
    im = ax[1, 0].imshow(
        arr,
        extent=(0, tobs / 60.0, freq_lo, freq_hi),
        aspect="auto",
        cmap="afmhot"
    )

    ax[1, 0].set_xlabel("Time (min)")
    ax[1, 0].set_ylabel("Frequency (MHz)")
    ax[1, 0].set_title("Dynamic Spectrum")

    # time projection (top)
    ax[0, 0].plot(np.linspace(0, tobs/60.0, nsub), time_proj)
    ax[0, 0].set_xlabel("Time (min)")
    ax[0, 0].set_ylabel("Flux")

    # frequency projection (right)
    ax[1, 1].plot(freq_proj, np.linspace(freq_lo, freq_hi, nchan))
    ax[1, 1].set_xlabel("Flux")
    ax[1, 1].set_ylabel("Frequency (MHz)")

    # remove empty panel
    ax[0, 1].axis("off")

    #plt.colorbar(im, ax=ax[1, 0])

    plt.tight_layout()
    plt.savefig(outname + "_dspec.png")

def plot_acspec(dynspec, archive, tobs, bw, outname):
    import numpy as np
    import matplotlib.pyplot as plt

    arr = dynspec - np.mean(dynspec, axis=None)

    ft = np.fft.fft2(arr)
    acf2d = np.fft.ifft2(np.abs(ft)**2)
    acf2d = np.fft.fftshift(np.real(acf2d))

    mx = np.max(acf2d)
    if mx != 0:
        acf2d /= mx

    nsub = archive.get_nsubint()
    nchan = archive.get_nchan()

    dt = tobs / nsub
    df = bw / nchan
    
    time_lag = (np.arange(nsub) - nsub//2) * dt
    freq_lag = (np.arange(nchan) - nchan//2) * df 

    plt.figure()

    plt.title("Autocorrelation Spectrum")
    plt.xlabel('time lag (s)')
    plt.ylabel('Frequency lag (MHz)')

    plt.imshow(
        acf2d,
        extent=(time_lag.min(), time_lag.max(), freq_lag.min(), freq_lag.max()),
        origin="lower",
        aspect="auto",
        cmap="afmhot"
    )

    plt.colorbar()
    plt.savefig(outname + "_acspec.png")

def plot_sspec(dynspec, archive, tobs, bw, outname):
    import numpy as np
    import matplotlib.pyplot as plt

    arr = dynspec - np.mean(dynspec, axis=None)

    nsub = archive.get_nsubint()
    nchan = archive.get_nchan()

    dt = tobs / nsub
    df = bw / nchan

    ft = np.fft.fft2(arr)
    ss = np.fft.fftshift(np.abs(ft)**2)
    ss = 10 * np.log10(ss + 1e-12)

    conjT = np.fft.fftshift(np.fft.fftfreq(nsub, d=dt))
    conjF = np.fft.fftshift(np.fft.fftfreq(nchan, d=df))

    ss[nsub // 2, :] = 0
    low  = np.median(ss)
    high = np.max(ss)
    plt.figure()

    plt.title("Secondary Spectrum")
    plt.xlabel("Fringe frequency (Hz)")
    plt.ylabel("Delay (s)")

    plt.imshow(
        ss,
        #ss[:int(ss.shape[0]/2), : int(ss.shape[1])],
        vmin = low, vmax = high,
        extent=(conjT.min(), conjT.max(), 0, conjF.max()),
        origin="lower",
        aspect="auto",
        cmap="afmhot"
    )

    plt.colorbar()
    plt.savefig(outname + "_sspec.png")

def integrated_profile(archive, N):
    import numpy as np
    temp = archive.clone()
    temp.tscrunch()
    temp.fscrunch()
    temp.remove_baseline()
    temp.centre_max_bin()

    prof = temp[0].get_Profile(0, 0).get_amps()

    nbin = len(prof)

    prof = prof - np.min(prof)
    threshold = np.max(prof) / N

    idx = np.where(prof > threshold)[0]

    if len(idx) == 0:
        return [0, 0]

    return [
        (nbin // 2) - (idx[0] - 1),
        (idx[-1] + 1) - (nbin // 2)
    ]

def interpulse_profile(archive, N):
    import numpy as np
    temp = archive.clone()
    temp.tscrunch()
    temp.fscrunch()
    temp.remove_baseline()

    prof = temp[0].get_Profile(0, 0).get_amps()

    nbin = len(prof)

    b0 = temp[0].get_Profile(0, 0).find_max_bin()

    shift = nbin // 4 - b0
    prof = np.roll(prof, shift)

    prof = prof - np.min(prof)

    threshold = np.max(prof) / N
    idx = np.where(prof > threshold)[0]

    if len(idx) == 0:
        return [0, 0, 0, 0]

    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)

    if len(groups) < 2:
        g = groups[0]
        return [g[0], g[-1], g[0], g[-1]]

    main = groups[0]
    inter = groups[1]

    return [main[0], main[-1], inter[0], inter[-1]]


def noise_cal_single_peak(left_edge, right_edge):
    import numpy as np
    """
    Noise-normalized dynamic spectrum for single-peak pulsar
    Returns: 2D array (nsub, nchan)
    """

    temp = archive.clone()

    temp.remove_baseline()
    temp.centre_max_bin()

    nsub = temp.get_nsubint()
    nchan = temp.get_nchan()
    nbin = temp.get_nbin()

    # On-pulse window (centered)
    w0 = (nbin // 2) - left_edge
    w1 = (nbin // 2) + right_edge

    xon = np.arange(w0, w1 + 1)
    xoff = np.concatenate([
        np.arange(0, w0),
        np.arange(w1, nbin)
    ])

    # Allocate dynamic spectrum
    dynspec = np.zeros((nsub, nchan))

    for isub in range(nsub):
        sub = temp[isub]

        for ichan in range(nchan):

            prof = sub.get_Profile(0, ichan).get_amps()

            # Off-pulse noise estimation
            yoff = prof[xoff]
            mean = np.mean(yoff)
            std = np.std(yoff)

            if std == 0:
                continue

            # S/N normalized on-pulse
            cal = (prof[xon] - mean) / std

            # Integrated normalized flux
            flux = np.trapezoid(cal) / len(cal)

            dynspec[isub, ichan] = max(flux, 0)

    print("Noise calibration done")

    return dynspec

def noise_cal_two_peak(left_main, right_main, left_inter, right_inter):
    import numpy as np
    """
    Noise-normalized dynamic spectrum for double-peak (interpulse) pulsar
    Returns: 2D array (nsub, nchan)
    """

    temp = archive.clone()
    temp.remove_baseline()

    nsub = temp.get_nsubint()
    nchan = temp.get_nchan()
    nbin = temp.get_nbin()

    dynspec = np.zeros((nsub, nchan))

    for isub in range(nsub):
        sub = temp[isub]

        for ichan in range(nchan):

            prof = sub.get_Profile(0, ichan).get_amps()

            # Align main pulse
            b0 = sub.get_Profile(0, ichan).find_max_bin()
            shift = (nbin // 4) - b0
            prof = np.roll(prof, shift)

            # Windows
            w0, w1 = left_main, right_main
            w2, w3 = left_inter, right_inter

            xon1 = np.arange(w0, w1 + 1)
            xon2 = np.arange(w2, w3 + 1)

            xoff = np.concatenate([
                np.arange(0, w0),
                np.arange(w1, w2),
                np.arange(w3, nbin)
            ])

            # Noise estimation
            yoff = prof[xoff]
            mean = np.mean(yoff)
            std = np.std(yoff)

            if std == 0:
                continue

            # S/N normalization
            cal1 = (prof[xon1] - mean) / std
            cal2 = (prof[xon2] - mean) / std

            # Integrated normalized flux
            flux = (
                np.trapezoid(cal1) +
                np.trapezoid(cal2)
            ) / (len(cal1) + len(cal2))

            dynspec[isub, ichan] = max(flux, 0)

    print("Noise calibration done")

    return dynspec

def build_dynspec(archive, N):

    if args.interpulse == 0:
        print("Single-pulse mode")
        w = integrated_profile(archive, N)   # [left, right]
        dynspec = noise_cal_single_peak(w[0], w[1])

    elif args.interpulse == 1:
        print("Double-pulse mode")
        w = interpulse_profile(archive, N)   # [w0,w1,w2,w3]
        dynspec = noise_cal_two_peak(w[0], w[1], w[2], w[3])

    else:
        raise ValueError("interpulse must be 0 or 1")

    return dynspec

if args.intpf:
    plot_intpf(archive, N, outname, tobs, freq_lo, freq_hi)

if args.dspec:
    dynspec = build_dynspec(archive, N)
    plot_dspec(dynspec, archive, tobs, freq_lo, freq_hi, outname)

if args.acspec:
    dynspec = build_dynspec(archive, N)
    plot_acspec(dynspec, archive, tobs, bw, outname)

if args.sspec:
    dynspec = build_dynspec(archive, N)
    plot_sspec(dynspec, archive, tobs, bw, outname)

del archive
