"""Command line interface for psrism."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [ARCHIVE] [OPTION]...",
        description="""
psrism: Pulsar ISM analysis tool
Author: Piyush Marmat, PhD Student
Purpose: extract ISM related parameters from pulsar data.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("archive", help="Pulsar data file (PSRCHIVE-supported format)")

    group_plot = parser.add_argument_group("Plotting options")
    group_plot.add_argument("--dspec", action="store_true", help="plot dynamic spectrum")
    group_plot.add_argument("--sspec", action="store_true", help="plot secondary spectrum")
    group_plot.add_argument("--acspec", action="store_true", help="plot autocorrelation spectrum")
    group_plot.add_argument("--intpf", action="store_true", help="plot integrated pulse profile")
    group_plot.add_argument("--fit-tau", action="store_true", help="fit tau from the integrated profile")
    group_plot.add_argument("--fit-alpha", action="store_true", help="fit tau per subband and alpha")
    group_plot.add_argument("--inspect", action="store_true", help="show archive dimensions and valid scrunch targets")

    group_meta = parser.add_argument_group("Processing options")
    group_meta.add_argument("--dm", type=float, help="dispersion measure to apply (pc cm^-3)")
    group_meta.add_argument("--nsub", type=int, help="scrunch to N subintegrations")
    group_meta.add_argument("--nchan", type=int, help="scrunch to N frequency channels")
    group_meta.add_argument("--nbin", type=int, help="scrunch to N phase bins")
    group_meta.add_argument(
        "--interpulse",
        type=int,
        choices=[0, 1],
        default=0,
        help="0 = single pulse, 1 = interpulse present",
    )
    group_meta.add_argument("--onw", type=int, default=10, help="on-pulse width factor")
    group_meta.add_argument("--normalize-dspec", action="store_true", help="normalize dynamic spectrum")
    group_meta.add_argument("--tau-subbands", type=int, default=4, help="frequency subbands for --fit-alpha")
    group_meta.add_argument(
        "--tau-reference-freq",
        type=float,
        help="reference frequency in MHz for tau-frequency alpha fit",
    )
    return parser


def _print_metadata(metadata):
    axes = ["Nsub (time)", "Npol", "Nchan (freq)", "Nbin (phase)"]
    print("\nOriginal data shape:")
    for i, (size, name) in enumerate(zip(metadata.raw_shape, axes)):
        print(f" Axis {i}: {name} = {size}")

    print("\nObservation metadata:")
    print(
        f" Observation time: {metadata.observation_time_s:.3f} s "
        f"({metadata.observation_time_s / 60.0:.3f} min or "
        f"{metadata.observation_time_s / 3600.0:.3f} hour(s))"
    )
    print(f" Centre frequency: {metadata.centre_frequency_mhz:.3f} MHz")
    print(f" Bandwidth: {metadata.bandwidth_mhz:.3f} MHz")
    print(
        f" Frequency range: {metadata.frequency_low_mhz:.3f} MHz "
        f"to {metadata.frequency_high_mhz:.3f} MHz"
    )
    print(f" DM: {metadata.dispersion_measure:.6f} pc cm^-3")
    print(f"Telescope used: {metadata.telescope}")

    print("\nProcessed data shape:")
    for i, (size, name) in enumerate(zip(metadata.processed_shape, axes)):
        print(f" Axis {i}: {name} = {size}")


def _print_scrunch_targets(shape, formatter, file=None) -> None:
    stream = file or sys.stdout
    nsub, _npol, nchan, nbin = shape
    print("\nValid scrunch targets smaller than current dimensions:", file=stream)
    print(f" --nsub  current {nsub}: {formatter(nsub)}", file=stream)
    print(f" --nchan current {nchan}: {formatter(nchan)}", file=stream)
    print(f" --nbin  current {nbin}: {formatter(nbin)}", file=stream)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from .archive_io import (
        archive_metadata,
        archive_shape,
        format_scrunch_targets,
        integrated_profile_array,
        load_archive,
        preprocess_archive,
    )
    from .autocorrelation_spectrum import (
        autocorrelation_axes,
        calculate_autocorrelation_spectrum,
        measure_acf_scales,
    )
    from .dynamic_spectrum import calculate_dynamic_spectrum
    from .fit_tau import (
        fit_tau_alpha_from_archive,
        fit_tau_from_archive,
        plot_subband_tau_fits,
        plot_tau_fit,
    )
    from .plot_autocorrelation_spectrum import plot_autocorrelation_spectrum
    from .plot_dynamic_spectrum import plot_dynamic_spectrum, plot_integrated_profile
    from .plot_scintillation_spectrum import plot_scintillation_spectrum
    from .plot_tau_vs_freq import plot_tau_vs_frequency
    from .scintillation_spectrum import calculate_scintillation_spectrum

    try:
        archive = load_archive(args.archive)
    except RuntimeError as exc:
        print(f"psrism: error: {exc}", file=sys.stderr)
        if exc.__cause__ is not None:
            print(f"PSRCHIVE detail: {exc.__cause__}", file=sys.stderr)
        return 2
    raw_shape = archive_shape(archive)

    print(f"\nLoaded archive: {args.archive}")
    if args.inspect:
        metadata = archive_metadata(archive, raw_shape=raw_shape)
        _print_metadata(metadata)
        _print_scrunch_targets(raw_shape, format_scrunch_targets)
        return 0

    try:
        preprocess_archive(
            archive,
            dm=args.dm,
            nsub=args.nsub,
            nchan=args.nchan,
            nbin=args.nbin,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"psrism: error: {exc}", file=sys.stderr)
        _print_scrunch_targets(raw_shape, format_scrunch_targets, file=sys.stderr)
        return 2
    metadata = archive_metadata(archive, raw_shape=raw_shape)
    _print_metadata(metadata)

    dynspec = None
    needs_dynspec = args.dspec or args.acspec or args.sspec
    if needs_dynspec:
        dynspec = calculate_dynamic_spectrum(
            archive,
            width_factor=max(args.onw, 1),
            interpulse=bool(args.interpulse),
            normalize=args.normalize_dspec,
        )

    integrated_tau_result = None
    if args.intpf or args.fit_tau:
        try:
            integrated_tau_result = fit_tau_from_archive(archive)
        except (RuntimeError, ValueError) as exc:
            print(f"psrism: error: integrated-profile tau fit failed: {exc}", file=sys.stderr)
            return 2

    if args.intpf:
        _print_tau_result(integrated_tau_result, label="Integrated-profile tau fit")
        plot_integrated_profile(
            archive,
            metadata,
            width_factor=max(args.onw, 1),
            tau_fit_result=integrated_tau_result,
            output_path=f"{metadata.outname}_intpf.png",
        )

    if args.dspec and dynspec is not None:
        plot_dynamic_spectrum(dynspec, metadata, output_path=f"{metadata.outname}_dspec.png")

    if args.acspec and dynspec is not None:
        acf2d = calculate_autocorrelation_spectrum(dynspec)
        time_lag, freq_lag = autocorrelation_axes(
            metadata.processed_shape[0],
            metadata.processed_shape[2],
            metadata.observation_time_s,
            metadata.bandwidth_mhz,
        )
        scales = measure_acf_scales(acf2d, time_lag, freq_lag)
        _print_acf_scales(scales)
        plot_autocorrelation_spectrum(acf2d, metadata, output_path=f"{metadata.outname}_acspec.png")

    if args.sspec and dynspec is not None:
        spectrum, fringe_frequency, delay = calculate_scintillation_spectrum(
            dynspec,
            metadata.observation_time_s,
            metadata.bandwidth_mhz,
        )
        plot_scintillation_spectrum(
            spectrum,
            fringe_frequency,
            delay,
            title=metadata.filename,
            output_path=f"{metadata.outname}_sspec.png",
        )

    if args.fit_tau:
        if not args.intpf:
            _print_tau_result(integrated_tau_result, label="Integrated-profile tau fit")
        profile = integrated_profile_array(archive)
        plot_tau_fit(profile, integrated_tau_result, output_path=f"{metadata.outname}_tau_fit.png")

    if args.fit_alpha:
        try:
            result = fit_tau_alpha_from_archive(
                archive,
                n_subbands=args.tau_subbands,
                reference_freq_mhz=args.tau_reference_freq,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"psrism: error: {exc}", file=sys.stderr)
            return 2
        alpha_fit = result.alpha_fit
        print("\nSubband tau fits:")
        for item in result.subbands:
            print(
                f" channels {item.channel_start}-{item.channel_stop}: "
                f"nu={item.frequency_mhz:.3f} MHz, "
                f"tau={item.tau:.6e} +/- {item.tau_error:.6e} s"
            )
        print(
            f"\nalpha = {alpha_fit.alpha:.6f} +/- {alpha_fit.alpha_error:.6f}"
        )
        print(
            f"tau0 = {alpha_fit.tau0:.6e} +/- {alpha_fit.tau0_error:.6e} s "
            f"at nu0={alpha_fit.reference_freq_mhz:.3f} MHz"
        )
        plot_tau_vs_frequency(
            [item.frequency_mhz for item in result.subbands],
            [item.tau for item in result.subbands],
            tau_error=[item.tau_error for item in result.subbands],
            alpha_result=alpha_fit,
            reference_freq_mhz=alpha_fit.reference_freq_mhz,
            title=metadata.filename,
            output_path=f"{metadata.outname}_tau_vs_freq.png",
        )
        plot_subband_tau_fits(
            result,
            title=metadata.filename,
            output_path=f"{metadata.outname}_subband_tau_fits.png",
        )

    return 0


def _print_tau_result(result, label: str) -> None:
    print(f"\n{label}:")
    print(f" tau (bins) = {result.tau_bins:.3f} +/- {result.tau_bins_error:.3f}")
    if result.tau_seconds is not None:
        print(
            f" tau = {result.tau_seconds:.6e} +/- "
            f"{result.tau_seconds_error:.6e} s"
        )
    print(
        f" goodness: unweighted reduced chi-square = {_format_optional_float(result.reduced_chi_square)}, "
        f"RMS residual = {_format_optional_float(result.rms_residual)}"
    )


def _print_acf_scales(scales) -> None:
    print("\nACF decorrelation measurements:")
    if scales.decorrelation_bandwidth_mhz is None:
        print(" decorrelation bandwidth: unavailable")
    else:
        print(f" decorrelation bandwidth = {scales.decorrelation_bandwidth_mhz:.6g} MHz")

    if scales.diffractive_timescale_s is None:
        print(" diffractive timescale: unavailable")
    else:
        print(f" diffractive timescale = {scales.diffractive_timescale_s:.6g} s")


def _format_optional_float(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


if __name__ == "__main__":
    raise SystemExit(main())
