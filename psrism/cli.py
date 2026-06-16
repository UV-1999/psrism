"""Command line interface for psrism."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys

if "MPLCONFIGDIR" not in os.environ:
    _mpl_config_dir = Path("/tmp/psrism-matplotlib")
    try:
        _mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(_mpl_config_dir)
    except OSError:
        pass


_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _style(text: str, *styles: str) -> str:
    if os.environ.get("PSRISM_NO_COLOR"):
        return text
    prefix = "".join(_ANSI[item] for item in styles)
    return f"{prefix}{text}{_ANSI['reset']}"


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _section(title: str) -> str:
    return "\n" + _style(title, "bold", "cyan")


def _ok(text: str) -> str:
    return _style(text, "green")


def _warn(text: str) -> str:
    return _style(text, "yellow")


def _err(text: str) -> str:
    return _style(text, "red")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [ARCHIVE_OR_DIRECTORY] [OPTION]...",
        description=f"""
{_style("PSRISM", "bold", "cyan")}  Pulsar ISM analysis tool
{_style("Author", "bold")}: Piyush Marmat, PhD Student
{_style("Purpose", "bold")}: extract scattering and scintillation parameters from pulsar archives.
""",
        epilog=f"""
{_style("Common workflows", "bold", "cyan")}
  Inspect dimensions and valid scrunch targets:
    {_style("psrism ARCHIVE --inspect", "green")}

  Dynamic spectrum:
    {_style("psrism ARCHIVE --nsub 71 --nchan 183 --nbin 512 --dspec", "green")}

  ACF with fitted slant and optional zoom:
    {_style("psrism ARCHIVE --acspec --zoom-acf", "green")}

  Tau and alpha from subbands:
    {_style("psrism ARCHIVE --fit-alpha --tau-subbands 8", "green")}

  Anisotropic scattering with a custom plot grid:
    {_style("psrism ARCHIVE --fit-anisotropy --tau-subbands 8 --subband-plot-rows 2 --subband-plot-cols 4", "green")}

  Parameter time series from a directory of archives:
    {_style("psrism ARCHIVE_DIR --time-params dm,tau,alpha,dnu_d,dt_d,t_r", "green")}

{_style("Outputs", "bold", "cyan")}
  Plots and terminal logs are saved inside a pulsar-named folder in the current directory.
  Set PSRISM_NO_COLOR=1 to disable ANSI colors.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("archive", help="Pulsar archive file or directory of archive files")

    group_plot = parser.add_argument_group(_style("Plotting and fitting options", "bold", "magenta"))
    group_plot.add_argument("--dspec", action="store_true", help="plot dynamic spectrum")
    group_plot.add_argument("--sspec", action="store_true", help="plot secondary spectrum")
    group_plot.add_argument("--acspec", action="store_true", help="plot autocorrelation spectrum")
    group_plot.add_argument("--intpf", action="store_true", help="plot integrated pulse profile")
    group_plot.add_argument("--fit-tau", action="store_true", help="fit tau from the integrated profile")
    group_plot.add_argument("--fit-alpha", action="store_true", help="fit tau per subband and alpha")
    group_plot.add_argument("--fit-anisotropy", action="store_true", help="fit anisotropic scattering in subbands")
    group_plot.add_argument("--estimate-refractive", action="store_true", help="estimate refractive scintillation parameters from fitted tau")
    group_plot.add_argument("--fit-arc", action="store_true", help="fit a parabolic arc in the secondary spectrum")
    group_plot.add_argument("--zoom-acf", action="store_true", help="crop the ACF plot around the fitted central ellipse")
    group_plot.add_argument("--inspect", action="store_true", help="show archive dimensions and valid scrunch targets")

    group_meta = parser.add_argument_group(_style("Processing and model options", "bold", "magenta"))
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
        "--time-params",
        default="all",
        help="comma-separated time-series parameters: dm,tau,alpha,dnu_d,dt_d,t_r or all",
    )
    group_meta.add_argument(
        "--time-axis",
        choices=["utc", "mjd"],
        default="utc",
        help="legacy option; directory plots now show UTC and MJD axes together",
    )
    group_meta.add_argument(
        "--time-pattern",
        default="*.nop",
        help="file pattern used when ARCHIVE_OR_DIRECTORY is a directory",
    )
    group_meta.add_argument("--subband-plot-rows", type=int, help="rows for subband fit plot grids")
    group_meta.add_argument("--subband-plot-cols", type=int, help="columns for subband fit plot grids")
    group_meta.add_argument(
        "--tau-reference-freq",
        type=float,
        help="reference frequency in MHz for tau-frequency alpha fit",
    )
    group_meta.add_argument("--distance-kpc", type=float, help="pulsar distance in kpc for scintillation estimates")
    group_meta.add_argument("--velocity-kms", type=float, help="effective transverse velocity in km/s for scintillation estimates")
    group_meta.add_argument("--c1", type=float, default=1.16, help="constant C1 for Delta_nu_d = C1 / (2 pi tau)")
    group_meta.add_argument("--eta-time", type=float, default=0.2, help="time filling factor for finite-scintle uncertainty")
    group_meta.add_argument("--eta-freq", type=float, default=0.2, help="frequency filling factor for finite-scintle uncertainty")
    group_meta.add_argument(
        "--arc-curvature-min",
        type=float,
        help="minimum parabolic-arc curvature magnitude in s^3 for --fit-arc",
    )
    group_meta.add_argument(
        "--arc-curvature-max",
        type=float,
        help="maximum parabolic-arc curvature magnitude in s^3 for --fit-arc",
    )
    group_meta.add_argument(
        "--arc-curvature-trials",
        type=int,
        default=200,
        help="number of trial parabolic-arc curvatures for --fit-arc",
    )
    group_meta.add_argument(
        "--arc-half",
        choices=["positive", "negative", "both"],
        default="positive",
        help="secondary-spectrum delay half to search for parabolic arcs",
    )
    group_meta.add_argument(
        "--arc-fringe-offset",
        type=float,
        default=0.0,
        help="parabolic-arc apex fringe-frequency offset in Hz",
    )
    group_meta.add_argument(
        "--arc-delay-offset",
        type=float,
        default=0.0,
        help="parabolic-arc apex delay offset in seconds",
    )
    group_meta.add_argument(
        "--arc-mask-bins",
        type=int,
        default=2,
        help="number of central axis bins to mask during --fit-arc",
    )
    return parser


def _print_metadata(metadata):
    axes = ["Nsub (time)", "Npol", "Nchan (freq)", "Nbin (phase)"]
    print(_section("Original Data Shape"))
    for i, (size, name) in enumerate(zip(metadata.raw_shape, axes)):
        print(f"  {_style(f'Axis {i}', 'bold')}: {name} = {_style(str(size), 'green')}")

    print(_section("Observation Metadata"))
    print(
        f"  Observation time: {_style(f'{metadata.observation_time_s:.3f} s', 'green')} "
        f"({metadata.observation_time_s / 60.0:.3f} min or "
        f"{metadata.observation_time_s / 3600.0:.3f} hour(s))"
    )
    print(f"  Centre frequency: {_style(f'{metadata.centre_frequency_mhz:.3f} MHz', 'green')}")
    print(f"  Bandwidth: {_style(f'{metadata.bandwidth_mhz:.3f} MHz', 'green')}")
    print(
        f"  Frequency range: {_style(f'{metadata.frequency_low_mhz:.3f} MHz', 'green')} "
        f"to {metadata.frequency_high_mhz:.3f} MHz"
    )
    print(f"  DM: {_style(f'{metadata.dispersion_measure:.6f} pc cm^-3', 'green')}")
    print(f"  Telescope used: {_style(metadata.telescope, 'green')}")

    print(_section("Processed Data Shape"))
    for i, (size, name) in enumerate(zip(metadata.processed_shape, axes)):
        print(f"  {_style(f'Axis {i}', 'bold')}: {name} = {_style(str(size), 'green')}")


def _print_scrunch_targets(shape, formatter, file=None) -> None:
    stream = file or sys.stdout
    nsub, _npol, nchan, nbin = shape
    print(_section("Valid Scrunch Targets Smaller Than Current Dimensions"), file=stream)
    print(f"  {_style('--nsub ', 'bold')} current {nsub}: {formatter(nsub)}", file=stream)
    print(f"  {_style('--nchan', 'bold')} current {nchan}: {formatter(nchan)}", file=stream)
    print(f"  {_style('--nbin ', 'bold')} current {nbin}: {formatter(nbin)}", file=stream)


class _Tee:
    """Write terminal output to both the original stream and a log file."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, text):
        self.stream.write(text)
        self.log_file.write(_strip_ansi(text))

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


def _safe_path_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.+-]+", "_", value.strip())
    return name.strip("._") or "unknown_pulsar"


def _archive_stem(path: str) -> str:
    return Path(path).with_suffix("").name


def _pulsar_name(archive, archive_path: str) -> str:
    try:
        source = archive.get_source()
    except AttributeError:
        source = ""
    if source:
        return _safe_path_name(source)
    return _safe_path_name(_archive_stem(archive_path).split("_")[0])


def _plot_path(output_prefix: Path, suffix: str) -> str:
    return str(output_prefix.parent / f"{output_prefix.name}_{suffix}.png")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from .archive_io import archive_shape, load_archive

    archive_arg = Path(args.archive)
    if archive_arg.is_dir():
        output_dir = Path.cwd() / _safe_path_name(archive_arg.name)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_prefix = output_dir / f"{_safe_path_name(archive_arg.name)}_time_series"
        log_path = output_dir / f"{output_prefix.name}_terminal_output.txt"
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        with log_path.open("w", encoding="utf-8") as log_file:
            sys.stdout = _Tee(original_stdout, log_file)
            sys.stderr = _Tee(original_stderr, log_file)
            try:
                return _run_time_series_directory(args, archive_arg, output_dir, output_prefix, log_path)
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

    try:
        archive = load_archive(args.archive)
    except RuntimeError as exc:
        print(_err(f"psrism: error: {exc}"), file=sys.stderr)
        if exc.__cause__ is not None:
            print(_err(f"PSRCHIVE detail: {exc.__cause__}"), file=sys.stderr)
        return 2
    raw_shape = archive_shape(archive)

    output_dir = Path.cwd() / _pulsar_name(archive, args.archive)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = output_dir / _archive_stem(args.archive)
    log_path = output_dir / f"{output_prefix.name}_terminal_output.txt"

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = _Tee(original_stdout, log_file)
        sys.stderr = _Tee(original_stderr, log_file)
        try:
            return _run_loaded_archive(args, archive, raw_shape, output_dir, output_prefix, log_path)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def _run_time_series_directory(
    args,
    archive_dir: Path,
    output_dir: Path,
    output_prefix: Path,
    log_path: Path,
) -> int:
    from .archive_io import archive_metadata, archive_shape, load_archive, preprocess_archive
    from .autocorrelation_spectrum import (
        autocorrelation_axes,
        calculate_autocorrelation_spectrum,
        measure_acf_scales,
    )
    from .dynamic_spectrum import calculate_dynamic_spectrum
    from .fit_tau import fit_tau_alpha_from_archive, fit_tau_from_archive
    from .time_series_analysis import (
        EpochMeasurement,
        archive_epoch_datetime,
        parse_time_params,
        plot_parameter_vs_time,
        write_time_series_csv,
    )

    try:
        params = parse_time_params(args.time_params)
    except ValueError as exc:
        print(_err(f"psrism: error: {exc}"), file=sys.stderr)
        return 2

    archive_paths = sorted(archive_dir.glob(args.time_pattern))
    if not archive_paths:
        print(_err(f"psrism: error: no files matching {args.time_pattern!r} in {archive_dir}"), file=sys.stderr)
        return 2

    print(_section("Time-Series Run Setup"))
    print(f"  {_style('Archive directory', 'bold')}: {_ok(str(archive_dir))}")
    print(f"  {_style('Files found', 'bold')}: {_ok(str(len(archive_paths)))}")
    print(f"  {_style('Requested parameters', 'bold')}: {_ok(', '.join(sorted(params)))}")
    print(f"  {_style('Time axes', 'bold')}: {_ok('UTC bottom, MJD top')}")
    print(f"  {_style('Output directory', 'bold')}: {_ok(str(output_dir))}")
    print(f"  {_style('Terminal output log', 'bold')}: {_ok(str(log_path))}")

    measurements: list[EpochMeasurement] = []
    band_lows: list[float] = []
    band_highs: list[float] = []
    needs_tau = "tau" in params
    needs_alpha = "alpha" in params
    needs_scintillation = bool(params & {"dnu_d", "dt_d", "t_r"})

    print(_section("Epoch Processing"))
    for idx, path in enumerate(archive_paths, start=1):
        print(f"  [{idx}/{len(archive_paths)}] {_style(path.name, 'bold')}")
        try:
            archive = load_archive(str(path))
            utc, mjd = archive_epoch_datetime(archive, str(path))
            dm_value = float(archive.get_dispersion_measure())
            centre_mhz = float(archive.get_centre_frequency())
            bandwidth_mhz = abs(float(archive.get_bandwidth()))
            band_lows.append(centre_mhz - bandwidth_mhz / 2.0)
            band_highs.append(centre_mhz + bandwidth_mhz / 2.0)
            metadata = None
            if needs_tau or needs_alpha or needs_scintillation:
                raw_shape = archive_shape(archive)
                preprocess_archive(
                    archive,
                    dm=args.dm,
                    nsub=args.nsub,
                    nchan=args.nchan,
                    nbin=args.nbin,
                )
                metadata = archive_metadata(archive, raw_shape=raw_shape)

            tau_result = None
            tau_value = None
            tau_error = None
            if needs_tau:
                tau_result = fit_tau_from_archive(archive, center_peak=True)
                tau_value = tau_result.tau_seconds
                tau_error = tau_result.tau_seconds_error

            alpha = None
            alpha_error = None
            tau0 = None
            tau0_error = None
            if needs_alpha:
                alpha_result = fit_tau_alpha_from_archive(
                    archive,
                    n_subbands=args.tau_subbands,
                    reference_freq_mhz=args.tau_reference_freq,
                ).alpha_fit
                alpha = alpha_result.alpha
                alpha_error = alpha_result.alpha_error
                tau0 = alpha_result.tau0
                tau0_error = alpha_result.tau0_error

            dnu_d = None
            dt_d = None
            tr_days = None
            if needs_scintillation and metadata is not None:
                dynspec = calculate_dynamic_spectrum(
                    archive,
                    width_factor=max(args.onw, 1),
                    interpulse=bool(args.interpulse),
                    normalize=args.normalize_dspec,
                )
                acf2d = calculate_autocorrelation_spectrum(dynspec)
                time_lag, freq_lag = autocorrelation_axes(
                    metadata.processed_shape[0],
                    metadata.processed_shape[2],
                    metadata.observation_time_s,
                    metadata.bandwidth_mhz,
                )
                scales = measure_acf_scales(acf2d, time_lag, freq_lag)
                dnu_d = scales.decorrelation_bandwidth_mhz
                dt_d = scales.diffractive_timescale_s
                if dnu_d is not None and dt_d is not None and dnu_d > 0 and dt_d > 0:
                    tr_days = (
                        (4.0 / 3.141592653589793)
                        * (metadata.centre_frequency_mhz / dnu_d)
                        * dt_d
                        / 86400.0
                    )

            measurements.append(
                EpochMeasurement(
                    archive_path=str(path),
                    archive_name=path.name,
                    utc=utc,
                    mjd=mjd,
                    dm=dm_value if "dm" in params else None,
                    tau_s=tau_value if "tau" in params else None,
                    tau_error_s=tau_error if "tau" in params else None,
                    alpha=alpha if "alpha" in params else None,
                    alpha_error=alpha_error if "alpha" in params else None,
                    tau0_s=tau0 if "alpha" in params else None,
                    tau0_error_s=tau0_error if "alpha" in params else None,
                    decorrelation_bandwidth_mhz=dnu_d if "dnu_d" in params else None,
                    diffractive_timescale_s=dt_d if "dt_d" in params else None,
                    refractive_timescale_days=tr_days if "t_r" in params else None,
                )
            )
            reported = []
            if "dm" in params:
                reported.append(f"DM={dm_value:.6g}")
            if "tau" in params:
                reported.append(
                    f"tau={_format_optional_float(tau_value)} +/- {_format_optional_float(tau_error)} s"
                )
            if "alpha" in params:
                reported.append(
                    f"alpha={_format_optional_float(alpha)} +/- {_format_optional_float(alpha_error)}"
                )
            if "dnu_d" in params:
                reported.append(f"dnu_d={_format_optional_float(dnu_d)} MHz")
            if "dt_d" in params:
                reported.append(f"dt_d={_format_optional_float(dt_d)} s")
            if "t_r" in params:
                reported.append(f"T_r={_format_optional_float(tr_days)} days")
            print(f"     {_ok('ok')}  UTC={utc.isoformat()}  MJD={mjd:.6f}")
            if reported:
                print(f"     {'; '.join(reported)}")
        except Exception as exc:
            print(_warn(f"     skipped: {exc}"))

    if not measurements:
        print(_err("psrism: error: no epochs were successfully processed"), file=sys.stderr)
        return 2

    measurements.sort(key=lambda item: item.mjd)
    csv_path = output_prefix.parent / f"{output_prefix.name}.csv"
    write_time_series_csv(measurements, csv_path)

    print(_section("Saved Time-Series Products"))
    print(f"  {_style('CSV table', 'bold')}: {_ok(str(csv_path))}")
    band_title = ""
    if band_lows and band_highs:
        band_title = f" ({min(band_lows):.3f}-{max(band_highs):.3f} MHz)"
    plotted = 0
    param_titles = {
        "dm": "DM",
        "tau": "scattering timescale τ",
        "alpha": "scattering spectral index α",
        "dnu_d": "decorrelation bandwidth Δν_d",
        "dt_d": "diffractive timescale Δt_d",
        "t_r": "refractive timescale T_r",
    }
    for param in ["dm", "tau", "alpha", "dnu_d", "dt_d", "t_r"]:
        if param not in params:
            continue
        output_path = output_prefix.parent / f"{output_prefix.name}_{param}_vs_time.png"
        for legacy_axis in ("utc", "mjd"):
            legacy_path = output_prefix.parent / f"{output_prefix.name}_{param}_vs_time_{legacy_axis}.png"
            try:
                legacy_path.unlink(missing_ok=True)
            except OSError:
                pass
        title = f"{archive_dir.name}{band_title}: {param_titles[param]} vs time"
        if plot_parameter_vs_time(measurements, param, args.time_axis, title, output_path):
            print(f"  {_style(param, 'bold')}: {_ok(str(output_path))}")
            plotted += 1
        else:
            print(f"  {_style(param, 'bold')}: {_warn('no finite values to plot')}")

    if plotted == 0:
        print(_warn("No time-series plots were produced because no requested parameter had finite values."))
    return 0


def _run_loaded_archive(args, archive, raw_shape, output_dir: Path, output_prefix: Path, log_path: Path) -> int:
    from .archive_io import (
        archive_metadata,
        format_scrunch_targets,
        integrated_profile_array,
        preprocess_archive,
    )
    from .autocorrelation_spectrum import (
        autocorrelation_axes,
        calculate_autocorrelation_spectrum,
        measure_acf_scales,
    )
    from .anisotropic_scattering import (
        fit_anisotropic_scattering_from_archive,
        plot_anisotropic_subband_fits,
    )
    from .dynamic_spectrum import calculate_dynamic_spectrum
    from .fit_tau import (
        fit_tau_alpha_from_archive,
        fit_tau_from_archive,
        plot_subband_tau_fits,
        plot_tau_fit,
    )
    from .fit_autocorrelation_spectrum import fit_autocorrelation_spectrum
    from .fit_secondary_spectrum import fit_parabolic_arc
    from .plot_autocorrelation_spectrum import plot_autocorrelation_spectrum
    from .plot_dynamic_spectrum import plot_dynamic_spectrum, plot_integrated_profile
    from .plot_scintillation_spectrum import plot_scintillation_spectrum
    from .plot_tau_vs_freq import plot_tau_vs_frequency
    from .refractive_scintillation import estimate_scintillation_from_tau
    from .scintillation_spectrum import calculate_scintillation_spectrum

    print(_section("Run Setup"))
    print(f"  {_style('Loaded archive', 'bold')}: {_ok(args.archive)}")
    print(f"  {_style('Output directory', 'bold')}: {_ok(str(output_dir))}")
    print(f"  {_style('Terminal output log', 'bold')}: {_ok(str(log_path))}")
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
        print(_err(f"psrism: error: {exc}"), file=sys.stderr)
        _print_scrunch_targets(raw_shape, format_scrunch_targets, file=sys.stderr)
        return 2
    metadata = archive_metadata(archive, raw_shape=raw_shape)
    _print_metadata(metadata)

    dynspec = None
    needs_dynspec = args.dspec or args.acspec or args.sspec or args.fit_arc
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
            integrated_tau_result = fit_tau_from_archive(archive, center_peak=True)
        except (RuntimeError, ValueError) as exc:
            print(_err(f"psrism: error: integrated-profile tau fit failed: {exc}"), file=sys.stderr)
            return 2

    if args.intpf:
        _print_tau_result(integrated_tau_result, label="Integrated-profile tau fit")
        plot_integrated_profile(
            archive,
            metadata,
            width_factor=max(args.onw, 1),
            tau_fit_result=integrated_tau_result,
            output_path=_plot_path(output_prefix, "intpf"),
        )

    if args.dspec and dynspec is not None:
        plot_dynamic_spectrum(dynspec, metadata, output_path=_plot_path(output_prefix, "dspec"))

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
        acf_fit_result = None
        try:
            acf_fit_result = fit_autocorrelation_spectrum(acf2d, time_lag, freq_lag)
            _print_acf_tilt_fit(acf_fit_result)
        except (RuntimeError, ValueError) as exc:
            print(_warn(f"psrism: warning: tilted ACF fit failed: {exc}"), file=sys.stderr)
        plot_autocorrelation_spectrum(
            acf2d,
            metadata,
            output_path=_plot_path(output_prefix, f"acspec{'_zoom' if args.zoom_acf else ''}"),
            acf_fit_result=acf_fit_result,
            zoom_fit=args.zoom_acf,
        )

    arc_fit_result = None
    if args.fit_arc and dynspec is not None:
        spectrum_linear, fringe_frequency, delay = calculate_scintillation_spectrum(
            dynspec,
            metadata.observation_time_s,
            metadata.bandwidth_mhz,
            log_scale=False,
        )
        try:
            arc_fit_result = fit_parabolic_arc(
                spectrum_linear,
                fringe_frequency,
                delay,
                curvature_min=args.arc_curvature_min,
                curvature_max=args.arc_curvature_max,
                n_trials=args.arc_curvature_trials,
                half=args.arc_half,
                mask_bins=args.arc_mask_bins,
                fringe_offset=args.arc_fringe_offset,
                delay_offset=args.arc_delay_offset,
            )
            _print_arc_fit(arc_fit_result)
        except (RuntimeError, ValueError) as exc:
            print(_err(f"psrism: error: parabolic arc fit failed: {exc}"), file=sys.stderr)
            return 2

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
            output_path=_plot_path(output_prefix, "sspec"),
            arc_fit_result=arc_fit_result,
        )

    if args.fit_tau:
        if not args.intpf:
            _print_tau_result(integrated_tau_result, label="Integrated-profile tau fit")
        profile = integrated_profile_array(archive, center_peak=True)
        plot_tau_fit(profile, integrated_tau_result, output_path=_plot_path(output_prefix, "tau_fit"))

    alpha_result_container = None
    if args.fit_alpha:
        try:
            result = fit_tau_alpha_from_archive(
                archive,
                n_subbands=args.tau_subbands,
                reference_freq_mhz=args.tau_reference_freq,
            )
        except (RuntimeError, ValueError) as exc:
            print(_err(f"psrism: error: {exc}"), file=sys.stderr)
            return 2
        alpha_result_container = result
        alpha_fit = result.alpha_fit
        print(_section("Subband Tau Fits"))
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
            output_path=_plot_path(output_prefix, "tau_vs_freq"),
        )
        plot_subband_tau_fits(
            result,
            title=metadata.filename,
            output_path=_plot_path(output_prefix, "subband_tau_fits"),
            nrows=args.subband_plot_rows,
            ncols=args.subband_plot_cols,
        )

    anisotropic_result = None
    if args.fit_anisotropy:
        try:
            anisotropic_result = fit_anisotropic_scattering_from_archive(
                archive,
                n_subbands=args.tau_subbands,
                reference_freq_mhz=args.tau_reference_freq,
            )
        except (RuntimeError, ValueError) as exc:
            print(_err(f"psrism: error: anisotropic scattering fit failed: {exc}"), file=sys.stderr)
            return 2
        _print_anisotropic_result(anisotropic_result)
        plot_tau_vs_frequency(
            [item.frequency_mhz for item in anisotropic_result.subbands],
            [item.tau_eff for item in anisotropic_result.subbands],
            tau_error=[item.tau_eff_error for item in anisotropic_result.subbands],
            alpha_result=anisotropic_result.alpha_fit,
            reference_freq_mhz=anisotropic_result.alpha_fit.reference_freq_mhz,
            title=metadata.filename,
            output_path=_plot_path(output_prefix, "anisotropic_tau_eff_vs_freq"),
        )
        plot_anisotropic_subband_fits(
            anisotropic_result,
            title=metadata.filename,
            output_path=_plot_path(output_prefix, "anisotropic_subband_fits"),
            nrows=args.subband_plot_rows,
            ncols=args.subband_plot_cols,
        )

    if args.estimate_refractive:
        try:
            if anisotropic_result is not None:
                freq = [item.frequency_mhz for item in anisotropic_result.subbands]
                tau = [item.tau_eff for item in anisotropic_result.subbands]
                tau_error = [item.tau_eff_error for item in anisotropic_result.subbands]
                source_label = "anisotropic τ_eff"
            else:
                if alpha_result_container is None:
                    alpha_result_container = fit_tau_alpha_from_archive(
                        archive,
                        n_subbands=args.tau_subbands,
                        reference_freq_mhz=args.tau_reference_freq,
                    )
                freq = [item.frequency_mhz for item in alpha_result_container.subbands]
                tau = [item.tau for item in alpha_result_container.subbands]
                tau_error = [item.tau_error for item in alpha_result_container.subbands]
                source_label = "isotropic τ"
            estimates = estimate_scintillation_from_tau(
                freq,
                tau,
                tau_error,
                observing_bandwidth_mhz=metadata.bandwidth_mhz,
                observing_duration_s=metadata.observation_time_s,
                distance_kpc=args.distance_kpc,
                velocity_kms=args.velocity_kms,
                c1=args.c1,
                eta_time=args.eta_time,
                eta_freq=args.eta_freq,
            )
        except (RuntimeError, ValueError) as exc:
            print(_err(f"psrism: error: refractive estimate failed: {exc}"), file=sys.stderr)
            return 2
        _print_refractive_estimates(
            estimates,
            source_label=source_label,
            distance_kpc=args.distance_kpc,
            velocity_kms=args.velocity_kms,
        )

    return 0


def _print_tau_result(result, label: str) -> None:
    print(_section(label))
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
    print(_section("ACF Decorrelation Measurements"))
    if scales.decorrelation_bandwidth_mhz is None:
        print(f" decorrelation bandwidth: {_warn('unavailable')}")
    else:
        print(f" decorrelation bandwidth = {scales.decorrelation_bandwidth_mhz:.6g} MHz")

    if scales.diffractive_timescale_s is None:
        print(f" diffractive timescale: {_warn('unavailable')}")
    else:
        print(f" diffractive timescale = {scales.diffractive_timescale_s:.6g} s")


def _print_acf_tilt_fit(result) -> None:
    print(_section("Tilted ACF Gaussian Fit"))
    print(f" fit decorrelation bandwidth = {result.delta_f_diss:.6g} MHz")
    print(f" fit diffractive timescale = {result.delta_t_diss:.6g} s")
    print(f" correlation coefficient = {result.correlation:.6g}")
    print(f" ellipse rotation angle = {result.rotation_angle_deg:.6g} deg")
    slope = result.drift_slope_s_per_mhz
    rate = result.drift_rate_mhz_per_s
    if slope is None:
        print(f" scintle drift slope d(time lag)/d(freq lag): {_warn('unavailable')}")
    else:
        print(f" scintle drift slope d(time lag)/d(freq lag) = {slope:.6g} s/MHz")
    if rate is not None:
        print(f" inverse drift rate d(freq lag)/d(time lag) = {rate:.6g} MHz/s")
    print(
        " quadratic coefficients: "
        f"a={result.a:.6g}, b={result.b:.6g}, c={result.c:.6g}"
    )
    print(
        f" goodness: unweighted reduced chi-square = {_format_optional_float(result.reduced_chi_square)}, "
        f"RMS residual = {_format_optional_float(result.rms_residual)}, "
        f"fit points = {result.n_fit_points}"
    )


def _print_arc_fit(result) -> None:
    print(_section("Parabolic Arc Fit"))
    print(f" delay half searched = {result.half}")
    print(f" apex fringe-frequency offset = {result.fringe_offset:.6g} Hz")
    print(f" apex delay offset = {result.delay_offset:.6g} s")
    print(
        f" curvature eta = {result.curvature:.6e} "
        f"+/- {_format_optional_float(result.curvature_error)} s^3"
    )
    print(f" arc strength = {result.score:.6g}")
    print(f" arc strength S/N = {_format_optional_float(result.score_snr)}")
    print(f" samples on best arc = {result.n_samples}")


def _print_anisotropic_result(result) -> None:
    print(_section("Anisotropic Scattering Fits"))
    for item in result.subbands:
        print(
            f" channels {item.channel_start}-{item.channel_stop}: "
            f"nu={item.frequency_mhz:.3f} MHz, "
            f"tau_x={item.tau_x:.6e} +/- {item.tau_x_error:.6e} s, "
            f"tau_y={item.tau_y:.6e} +/- {item.tau_y_error:.6e} s, "
            f"tau_eff={item.tau_eff:.6e} +/- {item.tau_eff_error:.6e} s, "
            f"tau_ratio={item.anisotropy_ratio:.6g} +/- {item.anisotropy_ratio_error:.6g}"
        )
        print(
            f"  goodness: anisotropic red. chi-square = "
            f"{_format_optional_float(item.fit.reduced_chi_square)}, "
            f"isotropic red. chi-square = {_format_optional_float(item.isotropic_reduced_chi_square)}, "
            f"RMS = {_format_optional_float(item.fit.rms_residual)}"
        )
    alpha_fit = result.alpha_fit
    print(
        f"\nanisotropic tau_eff alpha = {alpha_fit.alpha:.6f} "
        f"+/- {alpha_fit.alpha_error:.6f}"
    )
    print(
        f"tau_eff0 = {alpha_fit.tau0:.6e} +/- {alpha_fit.tau0_error:.6e} s "
        f"at nu0={alpha_fit.reference_freq_mhz:.3f} MHz"
    )


def _print_refractive_estimates(
    estimates,
    source_label: str,
    distance_kpc: float | None,
    velocity_kms: float | None,
) -> None:
    print(_section(f"Scintillation and Refractive Estimates From {source_label}"))
    if distance_kpc is None or velocity_kms is None:
        print(f" {_warn('distance/velocity not supplied; reporting Delta_nu_d only')}")
        print(f" {_warn('provide --distance-kpc and --velocity-kms for Delta_t_d, N_scintles, and T_r')}")
    for item in estimates:
        print(
            f" nu={item.frequency_mhz:.3f} MHz, "
            f"tau={item.tau_s:.6e} s, "
            f"Delta_nu_d={item.decorrelation_bandwidth_mhz:.6e} MHz"
        )
        if item.diffractive_timescale_s is not None:
            print(
                f"  Delta_t_d={item.diffractive_timescale_s:.6e} s, "
                f"T_r={item.refractive_timescale_days:.6e} days, "
                f"N_scintles={item.n_scintles:.6g}, "
                f"sigma_fse={item.finite_scintle_error_s:.6e} s, "
                f"sigma_total={item.tau_total_error_s:.6e} s"
            )


def _format_optional_float(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


if __name__ == "__main__":
    raise SystemExit(main())
