"""Command line interface for psrism."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys

import numpy as np

from .conventions import SCATTERING_INDEX_CONVENTION, TAU_REFERENCE_FREQUENCY_MHZ

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

  Dynamic spectrum with auditable RFI masking:
    {_style("psrism ARCHIVE --dspec --rfi-mask", "green")}

  ACF with fitted slant and optional zoom:
    {_style("psrism ARCHIVE --acspec --zoom-acf", "green")}

  Tau and alpha from subbands:
    {_style("psrism ARCHIVE --fit-alpha --tau-subbands 8", "green")}

  Anisotropic scattering with a custom plot grid:
    {_style("psrism ARCHIVE --fit-anisotropy --tau-subbands 8 --subband-plot-rows 2 --subband-plot-cols 4", "green")}

  Parameter time series from a directory of archives:
    {_style("psrism ARCHIVE_DIR --time-params dm,tau,tau_ref,alpha,dnu_d,dt_d,t_r", "green")}

  Multiple pulsars from a common archive tree:
    {_style("psrism ARCHIVE_ROOT --batch-pulsars --batch-recursive --time-pattern '*.nop' --time-params dm,tau,alpha", "green")}

  Annual scintillation anisotropy from a directory:
    {_style("psrism ARCHIVE_DIR --fit-annual-anisotropy --annual-distance-kpc D --annual-pm-ra-cosdec-mas-yr PMRA --annual-pm-dec-mas-yr PMDEC", "green")}

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
    group_plot.add_argument(
        "--measure-dm",
        action="store_true",
        help="measure each archive DM with PSRCHIVE pdmp before other analyses",
    )
    group_plot.add_argument(
        "--check-scattering-dm",
        action="store_true",
        help="estimate residual DM from deconvolved subband phases (single archives require --fit-alpha)",
    )
    group_plot.add_argument("--fit-anisotropy", action="store_true", help="fit anisotropic scattering in subbands")
    group_plot.add_argument(
        "--fit-annual-anisotropy",
        action="store_true",
        help="fit the multi-epoch annual anisotropic scintillation model",
    )
    group_plot.add_argument("--estimate-refractive", action="store_true", help="estimate refractive scintillation parameters from fitted tau")
    group_plot.add_argument("--fit-arc", action="store_true", help="fit a parabolic arc in the secondary spectrum")
    group_plot.add_argument("--zoom-acf", action="store_true", help="crop the ACF plot around the fitted central ellipse")
    group_plot.add_argument("--inspect", action="store_true", help="show archive dimensions and valid scrunch targets")

    group_meta = parser.add_argument_group(_style("Processing and model options", "bold", "magenta"))
    group_meta.add_argument(
        "--batch-pulsars",
        action="store_true",
        help="group a mixed-source archive directory and run each pulsar independently",
    )
    group_meta.add_argument(
        "--batch-recursive",
        action="store_true",
        help="search below the input directory recursively in --batch-pulsars mode",
    )
    group_meta.add_argument(
        "--batch-source",
        action="append",
        default=[],
        help="source name to include in batch mode; repeat to select multiple sources",
    )
    group_meta.add_argument("--dm", type=float, help="dispersion measure to apply (pc cm^-3)")
    group_meta.add_argument(
        "--pdmp-executable",
        default="pdmp",
        help="pdmp executable name or path (default: pdmp)",
    )
    group_meta.add_argument(
        "--pdmp-tempo-path",
        help="TEMPO installation directory supplied to pdmp via the TEMPO environment variable",
    )
    group_meta.add_argument(
        "--pdmp-dm-range",
        type=float,
        help="positive half-range searched around the archive DM by pdmp",
    )
    group_meta.add_argument(
        "--pdmp-dm-step",
        type=float,
        help="positive DM trial step supplied to pdmp",
    )
    group_meta.add_argument(
        "--pdmp-max-channels",
        type=int,
        help="maximum frequency channels retained by pdmp",
    )
    group_meta.add_argument(
        "--pdmp-max-subints",
        type=int,
        help="maximum subintegrations retained by pdmp",
    )
    group_meta.add_argument(
        "--pdmp-max-bins",
        type=int,
        help="maximum phase bins retained by pdmp",
    )
    group_meta.add_argument(
        "--pdmp-timeout",
        type=float,
        default=300.0,
        help="pdmp timeout in seconds (PSRISM default: 300)",
    )
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
    group_meta.add_argument(
        "--rfi-mask",
        action="store_true",
        help="apply conservative median/MAD RFI masking before requested analyses",
    )
    group_meta.add_argument(
        "--rfi-sigma",
        type=float,
        default=6.0,
        help="robust sigma threshold for --rfi-mask (default: 6)",
    )
    group_meta.add_argument(
        "--rfi-min-valid-fraction",
        type=float,
        default=0.5,
        help="minimum valid fraction in each time/frequency bin (default: 0.5)",
    )
    group_meta.add_argument(
        "--scintillation-min-resolution-bins",
        type=float,
        default=2.0,
        help="minimum samples across an accepted ACF width (PSRISM default: 2)",
    )
    group_meta.add_argument(
        "--secondary-window",
        choices=["none", "hann"],
        default="hann",
        help="window applied before the secondary-spectrum FFT (default: hann)",
    )
    group_meta.add_argument("--tau-subbands", type=int, default=4, help="frequency subbands for --fit-alpha")
    group_meta.add_argument(
        "--profile-components",
        type=int,
        default=1,
        help="Gaussian components in isotropic/anisotropic profile fits (default: 1)",
    )
    group_meta.add_argument(
        "--intrinsic-template",
        help="intrinsic profile template: NPY, text/CSV, or PSRCHIVE-readable archive",
    )
    # PSRISM choices: Appendix B of reference [2] plots the fitted widths but
    # does not prescribe a minimum sample count or significance threshold.
    group_meta.add_argument(
        "--intrinsic-width-min-subbands",
        type=int,
        default=3,
        help="accepted subbands required for an intrinsic-width trend (default: 3)",
    )
    group_meta.add_argument(
        "--intrinsic-width-p-threshold",
        type=float,
        default=0.05,
        help="p-value threshold for intrinsic-width evolution (default: 0.05)",
    )
    group_meta.add_argument(
        "--time-params",
        default="all",
        help="comma-separated time-series parameters: dm,tau,tau_ref,alpha,dnu_d,dt_d,t_r or all",
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
    group_meta.add_argument(
        "--variability-max-lag",
        type=int,
        default=0,
        help="maximum observation-order lag for temporal tests; 0 uses all available",
    )
    # Reference: Filothodoros thesis (ALEX.pdf), PDF p. 55, Section 4.2,
    # uses p=0.05 as the temporal-test decision threshold.
    group_meta.add_argument(
        "--variability-p-threshold",
        type=float,
        default=0.05,
        help="p-value threshold for temporal diagnostics (default: 0.05)",
    )
    group_meta.add_argument(
        "--dm-analysis-series",
        choices=["dm", "scattering-corrected"],
        default="dm",
        help="DM series used for solar and slope diagnostics (default: dm)",
    )
    # Reference: Filothodoros thesis (ALEX.pdf), PDF p. 85, Section 5.5.1,
    # notes that solar-wind effects become important below about 10 degrees.
    group_meta.add_argument(
        "--solar-warning-angle",
        type=float,
        default=10.0,
        help="Sun-separation warning angle in degrees (default: 10)",
    )
    # PSRISM choices: one combined adjacent-error sigma suppresses noisy
    # turning points, and three points is the minimum accepted segment fit.
    # Neither value is prescribed by the bundled references.
    group_meta.add_argument(
        "--dm-turning-sigma",
        type=float,
        default=1.0,
        help="uncertainty deadband for automatic DM turning points (default: 1)",
    )
    group_meta.add_argument(
        "--dm-segment-min-points",
        type=int,
        default=3,
        help="minimum points in an accepted same-direction DM fit (default: 3)",
    )
    group_meta.add_argument(
        "--annual-distance-kpc",
        type=float,
        help="pulsar distance in kpc for --fit-annual-anisotropy",
    )
    group_meta.add_argument(
        "--annual-pm-ra-cosdec-mas-yr",
        type=float,
        help="proper motion mu_RA*cos(dec) in mas/yr for the annual model",
    )
    group_meta.add_argument(
        "--annual-pm-dec-mas-yr",
        type=float,
        help="declination proper motion in mas/yr for the annual model",
    )
    # PSRISM fit-quality and prior choices; the bundled annual-variation paper
    # does not prescribe these generic bounds or a minimum sample size.
    group_meta.add_argument(
        "--annual-min-epochs",
        type=int,
        default=8,
        help="minimum usable epochs for the five-parameter annual fit (default: 8)",
    )
    group_meta.add_argument(
        "--annual-max-screen-speed",
        type=float,
        default=500.0,
        help="symmetric screen-velocity prior bound in km/s (default: 500)",
    )
    group_meta.add_argument(
        "--annual-max-axial-ratio",
        type=float,
        default=20.0,
        help="upper uniform-prior bound for annual axial ratio (default: 20)",
    )
    group_meta.add_argument(
        "--annual-mcmc-walkers",
        type=int,
        default=32,
        help="emcee walkers for the annual fit (default: 32)",
    )
    group_meta.add_argument(
        "--annual-mcmc-steps",
        type=int,
        default=5000,
        help="emcee steps per walker for the annual fit (default: 5000)",
    )
    group_meta.add_argument(
        "--annual-mcmc-burn",
        type=int,
        default=1000,
        help="discarded burn-in steps per annual-fit walker (default: 1000)",
    )
    group_meta.add_argument(
        "--annual-mcmc-seed",
        type=int,
        default=0,
        help="random seed for reproducible annual-fit initialization (default: 0)",
    )
    group_meta.add_argument("--subband-plot-rows", type=int, help="rows for subband fit plot grids")
    group_meta.add_argument("--subband-plot-cols", type=int, help="columns for subband fit plot grids")
    group_meta.add_argument(
        "--tau-reference-freq",
        type=float,
        default=TAU_REFERENCE_FREQUENCY_MHZ,
        help=f"reference frequency in MHz for tau-frequency alpha fit (default: {TAU_REFERENCE_FREQUENCY_MHZ:g})",
    )
    group_meta.add_argument(
        "--alpha-min-subbands",
        type=int,
        default=3,
        help="minimum accepted tau subbands required for alpha (default: 3)",
    )
    group_meta.add_argument(
        "--alpha-min-profile-snr",
        type=float,
        default=5.0,
        help="minimum residual-based profile-fit S/N per alpha subband (default: 5)",
    )
    group_meta.add_argument(
        "--alpha-max-relative-tau-error",
        type=float,
        default=1.0,
        help="maximum tau uncertainty divided by tau per alpha subband (default: 1)",
    )
    group_meta.add_argument(
        "--alpha-min-valid-channel-fraction",
        type=float,
        default=0.5,
        help="minimum positive-weight channel fraction per alpha subband (default: 0.5)",
    )
    group_meta.add_argument(
        "--alpha-mc-samples",
        type=int,
        default=10000,
        help="Monte Carlo draws for asymmetric alpha/tau_ref errors; 0 disables (default: 10000)",
    )
    group_meta.add_argument(
        "--alpha-mc-seed",
        type=int,
        default=0,
        help="random seed for reproducible alpha Monte Carlo errors (default: 0)",
    )
    group_meta.add_argument("--distance-kpc", type=float, help="pulsar distance in kpc for scintillation estimates")
    group_meta.add_argument("--velocity-kms", type=float, help="effective transverse velocity in km/s for scintillation estimates")
    # References for these scientific defaults: Lorimer & Kramer (2005),
    # psrhandbook.pdf, Sections 4.2.3 and 4.2.5.2; Geiger & Lam
    # (NANOGrav-Memo-008.pdf), PDF pp. 5-6, Eqs. 6-8.
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
    group_meta.add_argument(
        "--arc-min-samples",
        type=int,
        default=20,
        help="minimum sampled points on an accepted arc (PSRISM default: 20)",
    )
    group_meta.add_argument(
        "--arc-min-score-snr",
        type=float,
        default=5.0,
        help="minimum robust arc-strength S/N for acceptance (PSRISM default: 5)",
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


def _print_observation_geometry(geometry) -> None:
    print(_section("Observation Geometry"))
    print(f"  Telescope: {_style(geometry.telescope, 'green')}")
    if geometry.telescope_latitude_deg is None or geometry.telescope_longitude_deg is None:
        print(f"  Telescope coordinates: {_warn('unknown')}")
    else:
        height = "n/a" if geometry.telescope_height_m is None else f"{geometry.telescope_height_m:.1f} m"
        print(
            f"  Telescope coordinates: "
            f"lat={_style(f'{geometry.telescope_latitude_deg:.5f} deg', 'green')}, "
            f"lon={geometry.telescope_longitude_deg:.5f} deg, height={height}"
        )
        print(f"  Coordinate source: {geometry.telescope_location_source}")
    print(
        f"  Source coordinates: "
        f"RA={_style(f'{geometry.source_ra_deg:.6f} deg', 'green')}, "
        f"Dec={geometry.source_dec_deg:.6f} deg"
    )
    print(f"  Start UTC: {_style(geometry.start_utc.isoformat(), 'green')} (MJD {geometry.start_mjd:.9f})")
    print(f"  End UTC: {_style(geometry.end_utc.isoformat(), 'green')} (MJD {geometry.end_mjd:.9f})")
    print(
        f"  Sun separation: start={_style(f'{geometry.sun_separation_start_deg:.3f} deg', 'green')}, "
        f"mid={geometry.sun_separation_mid_deg:.3f} deg, "
        f"end={geometry.sun_separation_end_deg:.3f} deg"
    )
    print(f"  Solar ephemeris: {geometry.solar_ephemeris_method}")


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


def _frequency_scaling_options(args) -> dict:
    return {
        "min_subbands": args.alpha_min_subbands,
        "min_profile_snr": args.alpha_min_profile_snr,
        "max_relative_tau_error": args.alpha_max_relative_tau_error,
        "min_valid_channel_fraction": args.alpha_min_valid_channel_fraction,
        "monte_carlo_samples": args.alpha_mc_samples,
        "random_seed": args.alpha_mc_seed,
    }


def _intrinsic_width_options(args) -> dict:
    return {
        "min_trend_subbands": args.intrinsic_width_min_subbands,
        "p_threshold": args.intrinsic_width_p_threshold,
    }


def _pdmp_options(args) -> dict:
    return {
        "executable": args.pdmp_executable,
        "tempo_path": args.pdmp_tempo_path,
        "dm_half_range": args.pdmp_dm_range,
        "dm_step": args.pdmp_dm_step,
        "max_channels": args.pdmp_max_channels,
        "max_subints": args.pdmp_max_subints,
        "max_bins": args.pdmp_max_bins,
        "timeout_s": args.pdmp_timeout,
    }


def _scattering_dm_result(alpha_container, archive, input_dm: float):
    from .dm_analysis import scattering_dm_from_subbands

    accepted = alpha_container.accepted_subbands
    phase_errors = [item.fit.intrinsic_phase_bins_error for item in accepted]
    use_phase_errors = all(
        value is not None and np.isfinite(value) and value > 0 for value in phase_errors
    )
    return scattering_dm_from_subbands(
        [item.frequency_mhz for item in accepted],
        [item.fit.intrinsic_phase_bins for item in accepted],
        period_s=float(archive.get_Integration(0).get_folding_period()),
        nbin=int(archive.get_nbin()),
        input_dm=float(input_dm),
        phase_error_bins=phase_errors if use_phase_errors else None,
        reference_frequency_mhz=alpha_container.alpha_fit.reference_freq_mhz,
    )


def _refractive_timescale_error_days(
    refractive_timescale_days: float,
    decorrelation_bandwidth_mhz: float,
    decorrelation_bandwidth_error_mhz: float | None,
    diffractive_timescale_s: float,
    diffractive_timescale_error_s: float | None,
    width_covariance_mhz_s: float | None,
) -> float | None:
    values = (
        decorrelation_bandwidth_error_mhz,
        diffractive_timescale_error_s,
        width_covariance_mhz_s,
    )
    if any(value is None or not np.isfinite(value) for value in values):
        return None
    # PSRISM choice: first-order covariance propagation through
    # T_r proportional to Delta_t_DISS / Delta_nu_DISS. The physical relation
    # is from Geiger & Lam (NANOGrav-Memo-008.pdf), PDF p. 9, Eq. 14.
    derivative_dnu = -refractive_timescale_days / decorrelation_bandwidth_mhz
    derivative_dt = refractive_timescale_days / diffractive_timescale_s
    variance = (
        derivative_dnu**2 * decorrelation_bandwidth_error_mhz**2
        + derivative_dt**2 * diffractive_timescale_error_s**2
        + 2.0 * derivative_dnu * derivative_dt * width_covariance_mhz_s
    )
    if not np.isfinite(variance) or variance < 0:
        return None
    return float(np.sqrt(variance))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.profile_components < 1:
        parser.error("--profile-components must be at least 1")
    if args.intrinsic_template and args.profile_components != 1:
        parser.error("--intrinsic-template cannot be combined with --profile-components other than 1")
    if args.alpha_min_subbands < 2:
        parser.error("--alpha-min-subbands must be at least 2")
    if args.alpha_min_subbands > args.tau_subbands:
        parser.error("--alpha-min-subbands cannot exceed --tau-subbands")
    if args.alpha_mc_samples < 0:
        parser.error("--alpha-mc-samples cannot be negative")
    if args.intrinsic_width_min_subbands < 2:
        parser.error("--intrinsic-width-min-subbands must be at least 2")
    if (
        not np.isfinite(args.intrinsic_width_p_threshold)
        or not 0 < args.intrinsic_width_p_threshold < 1
    ):
        parser.error("--intrinsic-width-p-threshold must be between zero and one")
    if args.variability_max_lag < 0:
        parser.error("--variability-max-lag cannot be negative")
    if (
        not np.isfinite(args.variability_p_threshold)
        or not 0 < args.variability_p_threshold < 1
    ):
        parser.error("--variability-p-threshold must be between zero and one")
    if not np.isfinite(args.solar_warning_angle) or args.solar_warning_angle <= 0:
        parser.error("--solar-warning-angle must be positive")
    if not np.isfinite(args.dm_turning_sigma) or args.dm_turning_sigma < 0:
        parser.error("--dm-turning-sigma cannot be negative")
    if args.dm_segment_min_points < 2:
        parser.error("--dm-segment-min-points must be at least 2")
    if args.annual_min_epochs < 6:
        parser.error("--annual-min-epochs must be at least 6")
    if (
        not np.isfinite(args.annual_max_screen_speed)
        or args.annual_max_screen_speed <= 0
    ):
        parser.error("--annual-max-screen-speed must be positive")
    if (
        not np.isfinite(args.annual_max_axial_ratio)
        or args.annual_max_axial_ratio <= 1
    ):
        parser.error("--annual-max-axial-ratio must exceed 1")
    if args.annual_mcmc_walkers < 10 or args.annual_mcmc_walkers % 2:
        parser.error("--annual-mcmc-walkers must be an even integer of at least 10")
    if args.annual_mcmc_steps < 2:
        parser.error("--annual-mcmc-steps must be at least 2")
    if not 0 <= args.annual_mcmc_burn < args.annual_mcmc_steps:
        parser.error("--annual-mcmc-burn must be smaller than --annual-mcmc-steps")
    if args.annual_mcmc_seed < 0:
        parser.error("--annual-mcmc-seed cannot be negative")
    if args.fit_annual_anisotropy:
        required_annual = (
            ("--annual-distance-kpc", args.annual_distance_kpc),
            (
                "--annual-pm-ra-cosdec-mas-yr",
                args.annual_pm_ra_cosdec_mas_yr,
            ),
            ("--annual-pm-dec-mas-yr", args.annual_pm_dec_mas_yr),
        )
        missing_annual = [name for name, value in required_annual if value is None]
        if missing_annual:
            parser.error(
                "--fit-annual-anisotropy requires " + ", ".join(missing_annual)
            )
        if not np.isfinite(args.annual_distance_kpc) or args.annual_distance_kpc <= 0:
            parser.error("--annual-distance-kpc must be positive")
        for option, value in required_annual[1:]:
            if not np.isfinite(value):
                parser.error(f"{option} must be finite")
    if (
        not np.isfinite(args.scintillation_min_resolution_bins)
        or args.scintillation_min_resolution_bins <= 0
    ):
        parser.error("--scintillation-min-resolution-bins must be positive")
    if args.zoom_acf and not args.acspec:
        parser.error("--zoom-acf requires --acspec")
    if args.arc_curvature_trials < 5:
        parser.error("--arc-curvature-trials must be at least 5")
    if args.arc_mask_bins < 0:
        parser.error("--arc-mask-bins cannot be negative")
    if args.arc_min_samples < 1:
        parser.error("--arc-min-samples must be at least 1")
    if not np.isfinite(args.arc_min_score_snr) or args.arc_min_score_snr <= 0:
        parser.error("--arc-min-score-snr must be positive")
    for option, value in (
        ("--arc-curvature-min", args.arc_curvature_min),
        ("--arc-curvature-max", args.arc_curvature_max),
    ):
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error(f"{option} must be positive")
    for option, value in (
        ("--arc-fringe-offset", args.arc_fringe_offset),
        ("--arc-delay-offset", args.arc_delay_offset),
    ):
        if not np.isfinite(value):
            parser.error(f"{option} must be finite")
    for option, value in (
        ("--c1", args.c1),
        ("--eta-time", args.eta_time),
        ("--eta-freq", args.eta_freq),
    ):
        if not np.isfinite(value) or value <= 0:
            parser.error(f"{option} must be positive")
    for option, value in (
        ("--distance-kpc", args.distance_kpc),
        ("--velocity-kms", args.velocity_kms),
    ):
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error(f"{option} must be positive")
    if (args.distance_kpc is None) != (args.velocity_kms is None):
        parser.error("--distance-kpc and --velocity-kms must be supplied together")
    if args.measure_dm and args.dm is not None:
        parser.error("--measure-dm and --dm cannot be used together")
    if args.dm is not None and (not np.isfinite(args.dm) or args.dm < 0):
        parser.error("--dm must be finite and nonnegative")
    if args.check_scattering_dm and not args.fit_alpha and not Path(args.archive).is_dir():
        parser.error("--check-scattering-dm requires --fit-alpha for a single archive")
    if not np.isfinite(args.pdmp_timeout) or args.pdmp_timeout <= 0:
        parser.error("--pdmp-timeout must be positive")
    for option, value in (
        ("--pdmp-dm-range", args.pdmp_dm_range),
        ("--pdmp-dm-step", args.pdmp_dm_step),
        ("--pdmp-max-channels", args.pdmp_max_channels),
        ("--pdmp-max-subints", args.pdmp_max_subints),
        ("--pdmp-max-bins", args.pdmp_max_bins),
    ):
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error(f"{option} must be positive")

    from .archive_io import archive_shape, load_archive

    archive_arg = Path(args.archive)
    if args.fit_annual_anisotropy and not archive_arg.is_dir():
        parser.error("--fit-annual-anisotropy requires an archive directory")
    if args.batch_pulsars and not archive_arg.is_dir():
        parser.error("--batch-pulsars requires an archive directory")
    if (args.batch_recursive or args.batch_source) and not args.batch_pulsars:
        parser.error("--batch-recursive and --batch-source require --batch-pulsars")
    if args.batch_pulsars and args.fit_annual_anisotropy:
        parser.error(
            "--fit-annual-anisotropy is source-specific and cannot be used with "
            "--batch-pulsars"
        )
    if args.batch_pulsars and args.dm is not None:
        parser.error(
            "--dm cannot be shared across pulsars; use archive DMs or --measure-dm"
        )
    if args.batch_pulsars and args.intrinsic_template:
        parser.error(
            "--intrinsic-template is source-specific and cannot be shared in batch mode"
        )
    if archive_arg.is_dir():
        directory_name = (
            f"{_safe_path_name(archive_arg.name)}_psrism_batch"
            if args.batch_pulsars
            else _safe_path_name(archive_arg.name)
        )
        output_dir = Path.cwd() / directory_name
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = "batch" if args.batch_pulsars else "time_series"
        output_prefix = output_dir / f"{_safe_path_name(archive_arg.name)}_{suffix}"
        log_path = output_dir / f"{output_prefix.name}_terminal_output.txt"
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        with log_path.open("w", encoding="utf-8") as log_file:
            sys.stdout = _Tee(original_stdout, log_file)
            sys.stderr = _Tee(original_stderr, log_file)
            try:
                if args.batch_pulsars:
                    return _run_batch_directory(
                        args,
                        archive_arg,
                        output_dir,
                        output_prefix,
                        log_path,
                    )
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


def _run_batch_directory(
    args,
    archive_root: Path,
    output_dir: Path,
    output_prefix: Path,
    log_path: Path,
) -> int:
    from .archive_io import load_archive
    from .batch_analysis import (
        BatchPulsarRun,
        discover_archive_groups,
        write_batch_exports,
    )

    candidates = sorted(
        path
        for path in (
            archive_root.rglob(args.time_pattern)
            if args.batch_recursive
            else archive_root.glob(args.time_pattern)
        )
        if path.is_file()
    )
    if not candidates:
        scope = "recursively" if args.batch_recursive else "in the top-level directory"
        print(
            _err(
                f"psrism: error: no files matching {args.time_pattern!r} "
                f"{scope} under {archive_root}"
            ),
            file=sys.stderr,
        )
        return 2

    discovery = discover_archive_groups(
        candidates,
        load_archive,
        selected_sources=set(args.batch_source) if args.batch_source else None,
    )
    if not discovery.groups:
        selection = (
            f" selected by {', '.join(args.batch_source)}"
            if args.batch_source
            else ""
        )
        print(
            _err(f"psrism: error: no readable archive sources were found{selection}"),
            file=sys.stderr,
        )
        return 2

    print(_section("Cross-Pulsar Batch Setup"))
    print(f"  {_style('Archive root', 'bold')}: {_ok(str(archive_root))}")
    print(f"  {_style('Pattern', 'bold')}: {_ok(args.time_pattern)}")
    print(f"  {_style('Recursive', 'bold')}: {args.batch_recursive}")
    print(f"  {_style('Candidate files', 'bold')}: {_ok(str(len(candidates)))}")
    print(f"  {_style('Pulsars selected', 'bold')}: {_ok(str(len(discovery.groups)))}")
    print(
        f"  {_style('Header failures', 'bold')}: "
        f"{_warn(str(len(discovery.failures))) if discovery.failures else _ok('0')}"
    )
    print(
        f"  {_style('Non-pulsar archives ignored', 'bold')}: "
        f"{len(discovery.ignored_non_pulsars)}"
    )
    print(f"  {_style('Batch output', 'bold')}: {_ok(str(output_dir))}")
    print(f"  {_style('Batch log', 'bold')}: {_ok(str(log_path))}")

    runs: list[BatchPulsarRun] = []
    used_directory_names: set[str] = set()
    parent_stdout = sys.stdout
    parent_stderr = sys.stderr
    for index, (source, paths) in enumerate(discovery.groups.items(), start=1):
        base_name = _safe_path_name(source)
        directory_name = base_name
        collision_index = 2
        while directory_name in used_directory_names:
            directory_name = f"{base_name}_{collision_index}"
            collision_index += 1
        used_directory_names.add(directory_name)

        source_dir = output_dir / directory_name
        source_dir.mkdir(parents=True, exist_ok=True)
        source_prefix = source_dir / f"{directory_name}_time_series"
        source_log = source_dir / f"{source_prefix.name}_terminal_output.txt"
        print(
            _section(
                f"Batch Pulsar {index}/{len(discovery.groups)}: {source} "
                f"({len(paths)} file(s))"
            )
        )
        with source_log.open("w", encoding="utf-8") as source_log_file:
            sys.stdout = _Tee(parent_stdout, source_log_file)
            sys.stderr = _Tee(parent_stderr, source_log_file)
            try:
                try:
                    exit_code = _run_time_series_directory(
                        args,
                        archive_root,
                        source_dir,
                        source_prefix,
                        source_log,
                        archive_paths=list(paths),
                        series_name=source,
                    )
                except Exception as exc:
                    print(
                        _err(f"psrism: pulsar batch failed unexpectedly: {exc}"),
                        file=sys.stderr,
                    )
                    exit_code = 2
            finally:
                sys.stdout = parent_stdout
                sys.stderr = parent_stderr
        runs.append(
            BatchPulsarRun(
                pulsar=source,
                archive_paths=paths,
                exit_code=exit_code,
                output_directory=source_dir,
                time_series_csv=source_dir / f"{source_prefix.name}.csv",
                terminal_log=source_log,
            )
        )

    exports = write_batch_exports(
        runs,
        discovery,
        output_prefix,
        input_root=archive_root,
        pattern=args.time_pattern,
        recursive=args.batch_recursive,
        selected_sources=args.batch_source,
        requested_parameters=args.time_params,
    )
    print(_section("Saved Cross-Pulsar Products"))
    print(f"  {_style('Observations', 'bold')}: {_ok(str(exports.observations_csv))}")
    print(f"  {_style('Pulsar summary', 'bold')}: {_ok(str(exports.pulsars_csv))}")
    print(f"  {_style('Failures', 'bold')}: {_ok(str(exports.failures_csv))}")
    print(f"  {_style('Manifest', 'bold')}: {_ok(str(exports.manifest_json))}")
    if exports.scattering_comparison_plot is not None:
        print(
            f"  {_style('Scattering comparison', 'bold')}: "
            f"{_ok(str(exports.scattering_comparison_plot))}"
        )

    successful = sum(run.exit_code == 0 for run in runs)
    failed = len(runs) - successful
    print(
        f"  Completed pulsars: {_ok(str(successful))}; "
        f"failed pulsars: {_warn(str(failed)) if failed else _ok('0')}"
    )
    if successful == 0:
        return 2
    if failed or discovery.failures:
        return 1
    return 0


def _run_time_series_directory(
    args,
    archive_dir: Path,
    output_dir: Path,
    output_prefix: Path,
    log_path: Path,
    archive_paths: list[Path] | None = None,
    series_name: str | None = None,
) -> int:
    from .annual_anisotropy import (
        fit_annual_anisotropy,
        plot_annual_anisotropy,
        plot_annual_anisotropy_posterior,
        write_annual_anisotropy_report,
    )
    from .anisotropic_scattering import fit_anisotropic_scattering_from_archive
    from .archive_io import archive_metadata, archive_shape, load_archive, preprocess_archive
    from .autocorrelation_spectrum import (
        autocorrelation_axes,
        calculate_autocorrelation_spectrum,
        measure_acf_scales,
        summarize_scintillation_measurement,
    )
    from .data_quality import apply_quality_mask_to_archive, assess_dynamic_spectrum
    from .dm_analysis import run_pdmp
    from .dm_variability import (
        analyze_dm_solar_variability,
        plot_dm_piecewise_slopes,
        plot_dm_vs_sun_separation,
        write_dm_solar_variability_report,
    )
    from .dynamic_spectrum import calculate_dynamic_spectrum, normalize_dynamic_spectrum
    from .fit_tau import fit_tau_alpha_from_archive, fit_tau_from_archive
    from .fit_autocorrelation_spectrum import fit_autocorrelation_spectrum
    from .solar_geometry import observation_geometry
    from .time_series_analysis import (
        EpochMeasurement,
        archive_epoch_datetime,
        parse_time_params,
        plot_parameter_vs_time,
        write_time_series_csv,
    )
    from .temporal_statistics import (
        analyze_temporal_statistics,
        plot_parameter_autocorrelation,
        plot_temporal_correlation_matrix,
        write_temporal_statistics_report,
    )

    try:
        params = parse_time_params(args.time_params)
    except ValueError as exc:
        print(_err(f"psrism: error: {exc}"), file=sys.stderr)
        return 2
    if args.fit_annual_anisotropy:
        params.update({"dnu_d", "dt_d"})
    if (
        "dm" in params
        and args.dm_analysis_series == "scattering-corrected"
        and not args.check_scattering_dm
    ):
        print(
            _err(
                "psrism: error: --dm-analysis-series scattering-corrected "
                "requires --check-scattering-dm"
            ),
            file=sys.stderr,
        )
        return 2

    archive_paths = (
        sorted(archive_dir.glob(args.time_pattern))
        if archive_paths is None
        else sorted(archive_paths)
    )
    if not archive_paths:
        print(_err(f"psrism: error: no files matching {args.time_pattern!r} in {archive_dir}"), file=sys.stderr)
        return 2

    print(_section("Time-Series Run Setup"))
    print(f"  {_style('Archive directory', 'bold')}: {_ok(str(archive_dir))}")
    if series_name is not None:
        print(f"  {_style('Pulsar group', 'bold')}: {_ok(series_name)}")
    print(f"  {_style('Files found', 'bold')}: {_ok(str(len(archive_paths)))}")
    print(f"  {_style('Requested parameters', 'bold')}: {_ok(', '.join(sorted(params)))}")
    print(f"  {_style('Time axes', 'bold')}: {_ok('UTC bottom, MJD top')}")
    if "dm" in params:
        print(
            f"  {_style('DM diagnostic series', 'bold')}: "
            f"{_ok(args.dm_analysis_series)}"
        )
    if args.fit_annual_anisotropy:
        print(
            f"  {_style('Annual anisotropy', 'bold')}: "
            f"{_ok(f'{args.annual_mcmc_walkers} walkers x {args.annual_mcmc_steps} steps')}"
        )
    print(
        f"  {_style('RFI policy', 'bold')}: "
        f"{'robust statistical masking' if args.rfi_mask else 'archive weights/non-finite samples only'}"
    )
    profile_model = (
        f"template ({args.intrinsic_template})"
        if args.intrinsic_template
        else f"{args.profile_components} Gaussian component(s)"
    )
    print(f"  {_style('Profile model', 'bold')}: {_ok(profile_model)}")
    if params & {"alpha", "tau_ref"}:
        print(f"  {_style('Alpha convention', 'bold')}: {SCATTERING_INDEX_CONVENTION}")
        print(f"  {_style('Tau reference', 'bold')}: {_ok(f'{args.tau_reference_freq:g} MHz')}")
    print(f"  {_style('Output directory', 'bold')}: {_ok(str(output_dir))}")
    print(f"  {_style('Terminal output log', 'bold')}: {_ok(str(log_path))}")

    measurements: list[EpochMeasurement] = []
    band_lows: list[float] = []
    band_highs: list[float] = []
    needs_tau = "tau" in params
    needs_alpha = bool(params & {"alpha", "tau_ref"}) or args.check_scattering_dm
    needs_anisotropic = bool(args.fit_anisotropy)
    needs_scintillation = bool(params & {"dnu_d", "dt_d", "t_r"})

    print(_section("Epoch Processing"))
    for idx, path in enumerate(archive_paths, start=1):
        print(f"  [{idx}/{len(archive_paths)}] {_style(path.name, 'bold')}")
        try:
            archive = load_archive(str(path))
            utc, mjd = archive_epoch_datetime(archive, str(path))
            geometry = observation_geometry(archive)
            archive_dm = float(archive.get_dispersion_measure())
            pdmp_result = None
            processing_dm = args.dm
            if args.measure_dm:
                pdmp_result = run_pdmp(path, archive_dm, **_pdmp_options(args))
                processing_dm = pdmp_result.best_dm
            if pdmp_result is not None:
                dm_value = pdmp_result.best_dm
                dm_error = pdmp_result.dm_error
                dm_method = "pdmp_snr"
            elif args.dm is not None:
                dm_value = float(args.dm)
                dm_error = None
                dm_method = "manual_override"
            else:
                dm_value = archive_dm
                dm_error = None
                dm_method = "archive_metadata"
            centre_mhz = float(archive.get_centre_frequency())
            bandwidth_mhz = abs(float(archive.get_bandwidth()))
            period_s = float(archive.get_Integration(0).get_folding_period())
            band_lows.append(centre_mhz - bandwidth_mhz / 2.0)
            band_highs.append(centre_mhz + bandwidth_mhz / 2.0)
            metadata = None
            if (
                needs_tau
                or needs_alpha
                or needs_anisotropic
                or needs_scintillation
                or args.rfi_mask
            ):
                raw_shape = archive_shape(archive)
                preprocess_archive(
                    archive,
                    dm=processing_dm,
                    nsub=args.nsub,
                    nchan=args.nchan,
                    nbin=args.nbin,
                )
                metadata = archive_metadata(archive, raw_shape=raw_shape)

            dynspec = None
            quality = None
            if needs_scintillation or args.rfi_mask:
                dynspec = calculate_dynamic_spectrum(
                    archive,
                    width_factor=max(args.onw, 1),
                    interpulse=bool(args.interpulse),
                    normalize=False,
                )
                quality = assess_dynamic_spectrum(
                    dynspec,
                    sigma_threshold=args.rfi_sigma,
                    min_valid_fraction=args.rfi_min_valid_fraction,
                    detect_rfi=args.rfi_mask,
                )
                if args.rfi_mask:
                    apply_quality_mask_to_archive(archive, quality.valid_mask)
                dynspec = np.where(quality.valid_mask, dynspec, np.nan)
                if args.normalize_dspec:
                    dynspec = normalize_dynamic_spectrum(dynspec)

            tau_result = None
            tau_value = None
            tau_error = None
            if needs_tau:
                tau_result = fit_tau_from_archive(
                    archive,
                    center_peak=True,
                    n_components=args.profile_components,
                    intrinsic_template=args.intrinsic_template,
                )
                tau_value = tau_result.tau_seconds
                tau_error = tau_result.tau_seconds_error

            alpha = None
            alpha_error = None
            alpha_error_lower = None
            alpha_error_upper = None
            alpha_covariance_error = None
            alpha_monte_carlo_samples = None
            alpha_subbands_accepted = None
            alpha_subbands_rejected = None
            alpha_subbands_failed = None
            tau0 = None
            tau0_error = None
            tau0_error_lower = None
            tau0_error_upper = None
            tau_reference_frequency_mhz = None
            alpha_container = None
            if needs_alpha:
                alpha_container = fit_tau_alpha_from_archive(
                    archive,
                    n_subbands=args.tau_subbands,
                    reference_freq_mhz=args.tau_reference_freq,
                    n_components=args.profile_components,
                    intrinsic_template=args.intrinsic_template,
                    **_frequency_scaling_options(args),
                )
                alpha_result = alpha_container.alpha_fit
                alpha = alpha_result.alpha
                alpha_error = alpha_result.alpha_error
                alpha_error_lower = alpha_result.alpha_error_lower
                alpha_error_upper = alpha_result.alpha_error_upper
                alpha_covariance_error = alpha_result.covariance_alpha_error
                alpha_monte_carlo_samples = alpha_result.monte_carlo_samples
                alpha_subbands_accepted = len(alpha_container.accepted_subbands)
                alpha_subbands_rejected = len(alpha_container.rejected_subbands)
                alpha_subbands_failed = len(alpha_container.failed_subbands)
                tau0 = alpha_result.tau0
                tau0_error = alpha_result.tau0_error
                tau0_error_lower = alpha_result.tau0_error_lower
                tau0_error_upper = alpha_result.tau0_error_upper
                tau_reference_frequency_mhz = alpha_result.reference_freq_mhz

            anisotropic_alpha = None
            anisotropic_alpha_error = None
            anisotropic_alpha_error_lower = None
            anisotropic_alpha_error_upper = None
            anisotropic_tau0 = None
            anisotropic_tau0_error = None
            anisotropic_tau0_error_lower = None
            anisotropic_tau0_error_upper = None
            anisotropic_reference_frequency_mhz = None
            anisotropic_subbands_accepted = None
            anisotropic_subbands_rejected = None
            anisotropic_subbands_failed = None
            if needs_anisotropic:
                anisotropic_container = fit_anisotropic_scattering_from_archive(
                    archive,
                    n_subbands=args.tau_subbands,
                    reference_freq_mhz=args.tau_reference_freq,
                    n_components=args.profile_components,
                    intrinsic_template=args.intrinsic_template,
                    **_frequency_scaling_options(args),
                )
                anisotropic_fit = anisotropic_container.alpha_fit
                anisotropic_alpha = anisotropic_fit.alpha
                anisotropic_alpha_error = anisotropic_fit.alpha_error
                anisotropic_alpha_error_lower = anisotropic_fit.alpha_error_lower
                anisotropic_alpha_error_upper = anisotropic_fit.alpha_error_upper
                anisotropic_tau0 = anisotropic_fit.tau0
                anisotropic_tau0_error = anisotropic_fit.tau0_error
                anisotropic_tau0_error_lower = anisotropic_fit.tau0_error_lower
                anisotropic_tau0_error_upper = anisotropic_fit.tau0_error_upper
                anisotropic_reference_frequency_mhz = (
                    anisotropic_fit.reference_freq_mhz
                )
                anisotropic_subbands_accepted = len(
                    anisotropic_container.accepted_subbands
                )
                anisotropic_subbands_rejected = len(
                    anisotropic_container.rejected_subbands
                )
                anisotropic_subbands_failed = len(
                    anisotropic_container.failed_subbands
                )

            scattering_dm = None
            if args.check_scattering_dm:
                scattering_dm = _scattering_dm_result(
                    alpha_container,
                    archive,
                    processing_dm if processing_dm is not None else archive_dm,
                )

            dnu_d = None
            dnu_d_error = None
            dt_d = None
            dt_d_error = None
            tr_days = None
            tr_error_days = None
            scintillation_measurement = None
            if needs_scintillation and metadata is not None and dynspec is not None:
                acf2d = calculate_autocorrelation_spectrum(
                    dynspec,
                    valid_mask=quality.valid_mask,
                )
                time_lag, freq_lag = autocorrelation_axes(
                    metadata.processed_shape[0],
                    metadata.processed_shape[2],
                    metadata.observation_time_s,
                    metadata.bandwidth_mhz,
                )
                scales = measure_acf_scales(acf2d, time_lag, freq_lag)
                try:
                    acf_fit = fit_autocorrelation_spectrum(acf2d, time_lag, freq_lag)
                except (RuntimeError, ValueError):
                    acf_fit = None
                scintillation_measurement = summarize_scintillation_measurement(
                    scales,
                    time_lag,
                    freq_lag,
                    fit_result=acf_fit,
                    minimum_resolution_bins=args.scintillation_min_resolution_bins,
                    observing_duration_s=metadata.observation_time_s,
                    observing_bandwidth_mhz=metadata.bandwidth_mhz,
                    eta_time=args.eta_time,
                    eta_freq=args.eta_freq,
                )
                dnu_d = scintillation_measurement.decorrelation_bandwidth_mhz
                dnu_d_error = (
                    scintillation_measurement.decorrelation_bandwidth_error_mhz
                )
                dt_d = scintillation_measurement.diffractive_timescale_s
                dt_d_error = scintillation_measurement.diffractive_timescale_error_s
                if dnu_d is not None and dt_d is not None and dnu_d > 0 and dt_d > 0:
                    # Reference: Geiger & Lam (NANOGrav-Memo-008.pdf),
                    # PDF p. 9, Eq. 14.
                    tr_days = (
                        (4.0 / 3.141592653589793)
                        * (metadata.centre_frequency_mhz / dnu_d)
                        * dt_d
                        / 86400.0
                    )
                    tr_error_days = _refractive_timescale_error_days(
                        tr_days,
                        dnu_d,
                        dnu_d_error,
                        dt_d,
                        dt_d_error,
                        scintillation_measurement.width_covariance_mhz_s,
                    )

            measurements.append(
                EpochMeasurement(
                    archive_path=str(path),
                    archive_name=path.name,
                    utc=utc,
                    mjd=mjd,
                    start_utc=geometry.start_utc,
                    end_utc=geometry.end_utc,
                    start_mjd=geometry.start_mjd,
                    end_mjd=geometry.end_mjd,
                    telescope=geometry.telescope,
                    telescope_latitude_deg=geometry.telescope_latitude_deg,
                    telescope_longitude_deg=geometry.telescope_longitude_deg,
                    telescope_height_m=geometry.telescope_height_m,
                    source_ra_deg=geometry.source_ra_deg,
                    source_dec_deg=geometry.source_dec_deg,
                    observing_frequency_mhz=centre_mhz,
                    period_s=period_s,
                    sun_separation_start_deg=geometry.sun_separation_start_deg,
                    sun_separation_mid_deg=geometry.sun_separation_mid_deg,
                    sun_separation_end_deg=geometry.sun_separation_end_deg,
                    solar_ephemeris_method=geometry.solar_ephemeris_method,
                    dm=dm_value if "dm" in params else None,
                    archive_dm=archive_dm,
                    dm_error=dm_error if "dm" in params else None,
                    dm_method=dm_method if "dm" in params else None,
                    dm_correction=(
                        pdmp_result.dm_correction if pdmp_result is not None else None
                    ),
                    pdmp_snr=(pdmp_result.best_snr if pdmp_result is not None else None),
                    scattering_delta_dm=(
                        scattering_dm.delta_dm if scattering_dm is not None else None
                    ),
                    scattering_delta_dm_error=(
                        scattering_dm.delta_dm_error if scattering_dm is not None else None
                    ),
                    scattering_corrected_dm=(
                        scattering_dm.corrected_dm if scattering_dm is not None else None
                    ),
                    scattering_dm_reduced_chi_square=(
                        scattering_dm.reduced_chi_square
                        if scattering_dm is not None
                        else None
                    ),
                    tau_s=tau_value if "tau" in params else None,
                    tau_error_s=tau_error if "tau" in params else None,
                    alpha=alpha if "alpha" in params else None,
                    alpha_error=alpha_error if "alpha" in params else None,
                    alpha_error_lower=alpha_error_lower if "alpha" in params else None,
                    alpha_error_upper=alpha_error_upper if "alpha" in params else None,
                    alpha_covariance_error=(
                        alpha_covariance_error if "alpha" in params else None
                    ),
                    alpha_monte_carlo_samples=(
                        alpha_monte_carlo_samples if "alpha" in params else None
                    ),
                    alpha_subbands_accepted=(
                        alpha_subbands_accepted
                        if params & {"alpha", "tau_ref"}
                        else None
                    ),
                    alpha_subbands_rejected=(
                        alpha_subbands_rejected
                        if params & {"alpha", "tau_ref"}
                        else None
                    ),
                    alpha_subbands_failed=(
                        alpha_subbands_failed
                        if params & {"alpha", "tau_ref"}
                        else None
                    ),
                    tau0_s=tau0 if params & {"alpha", "tau_ref"} else None,
                    tau0_error_s=(
                        tau0_error if params & {"alpha", "tau_ref"} else None
                    ),
                    tau0_error_lower_s=(
                        tau0_error_lower
                        if params & {"alpha", "tau_ref"}
                        else None
                    ),
                    tau0_error_upper_s=(
                        tau0_error_upper
                        if params & {"alpha", "tau_ref"}
                        else None
                    ),
                    tau_reference_frequency_mhz=(
                        tau_reference_frequency_mhz
                        if params & {"alpha", "tau_ref"}
                        else None
                    ),
                    decorrelation_bandwidth_mhz=dnu_d if "dnu_d" in params else None,
                    decorrelation_bandwidth_error_mhz=(
                        dnu_d_error if "dnu_d" in params else None
                    ),
                    diffractive_timescale_s=dt_d if "dt_d" in params else None,
                    diffractive_timescale_error_s=(
                        dt_d_error if "dt_d" in params else None
                    ),
                    scintillation_measurement_method=(
                        scintillation_measurement.measurement_method
                        if scintillation_measurement is not None
                        else None
                    ),
                    acf_fit_decorrelation_bandwidth_mhz=(
                        scintillation_measurement.fit_decorrelation_bandwidth_mhz
                        if scintillation_measurement is not None
                        else None
                    ),
                    acf_fit_decorrelation_bandwidth_error_mhz=(
                        scintillation_measurement.fit_decorrelation_bandwidth_error_mhz
                        if scintillation_measurement is not None
                        else None
                    ),
                    acf_fit_diffractive_timescale_s=(
                        scintillation_measurement.fit_diffractive_timescale_s
                        if scintillation_measurement is not None
                        else None
                    ),
                    acf_fit_diffractive_timescale_error_s=(
                        scintillation_measurement.fit_diffractive_timescale_error_s
                        if scintillation_measurement is not None
                        else None
                    ),
                    acf_fit_width_covariance_mhz_s=(
                        scintillation_measurement.fit_width_covariance_mhz_s
                        if scintillation_measurement is not None
                        else None
                    ),
                    acf_width_covariance_mhz_s=(
                        scintillation_measurement.width_covariance_mhz_s
                        if scintillation_measurement is not None
                        else None
                    ),
                    acf_slice_decorrelation_bandwidth_mhz=(
                        scintillation_measurement.slice_decorrelation_bandwidth_mhz
                        if scintillation_measurement is not None
                        else None
                    ),
                    acf_slice_diffractive_timescale_s=(
                        scintillation_measurement.slice_diffractive_timescale_s
                        if scintillation_measurement is not None
                        else None
                    ),
                    acf_drift_slope_s_per_mhz=(
                        scintillation_measurement.drift_slope_s_per_mhz
                        if scintillation_measurement is not None
                        else None
                    ),
                    acf_drift_slope_error_s_per_mhz=(
                        scintillation_measurement.drift_slope_error_s_per_mhz
                        if scintillation_measurement is not None
                        else None
                    ),
                    acf_correlation=(
                        scintillation_measurement.correlation
                        if scintillation_measurement is not None
                        else None
                    ),
                    acf_correlation_error=(
                        scintillation_measurement.correlation_error
                        if scintillation_measurement is not None
                        else None
                    ),
                    scintillation_frequency_resolution_mhz=(
                        scintillation_measurement.frequency_resolution_mhz
                        if scintillation_measurement is not None
                        else None
                    ),
                    scintillation_time_resolution_s=(
                        scintillation_measurement.time_resolution_s
                        if scintillation_measurement is not None
                        else None
                    ),
                    decorrelation_bandwidth_resolution_bins=(
                        scintillation_measurement.decorrelation_bandwidth_resolution_bins
                        if scintillation_measurement is not None
                        else None
                    ),
                    diffractive_timescale_resolution_bins=(
                        scintillation_measurement.diffractive_timescale_resolution_bins
                        if scintillation_measurement is not None
                        else None
                    ),
                    decorrelation_bandwidth_resolved=(
                        scintillation_measurement.decorrelation_bandwidth_resolved
                        if scintillation_measurement is not None
                        else None
                    ),
                    diffractive_timescale_resolved=(
                        scintillation_measurement.diffractive_timescale_resolved
                        if scintillation_measurement is not None
                        else None
                    ),
                    scintillation_n_scintles=(
                        scintillation_measurement.n_scintles
                        if scintillation_measurement is not None
                        else None
                    ),
                    scintillation_finite_scintle_fraction=(
                        scintillation_measurement.finite_scintle_fraction
                        if scintillation_measurement is not None
                        else None
                    ),
                    scintillation_eta_time=(
                        scintillation_measurement.eta_time
                        if scintillation_measurement is not None
                        else None
                    ),
                    scintillation_eta_frequency=(
                        scintillation_measurement.eta_frequency
                        if scintillation_measurement is not None
                        else None
                    ),
                    refractive_timescale_days=tr_days if "t_r" in params else None,
                    refractive_timescale_error_days=(
                        tr_error_days if "t_r" in params else None
                    ),
                    quality_masked_fraction=(
                        quality.masked_fraction if quality is not None else None
                    ),
                    quality_bad_time_bins=(
                        len(quality.bad_time_bins) if quality is not None else None
                    ),
                    quality_bad_frequency_channels=(
                        len(quality.bad_frequency_channels) if quality is not None else None
                    ),
                    quality_isolated_flagged_cells=(
                        quality.isolated_flagged_cells if quality is not None else None
                    ),
                    profile_intrinsic_model=(
                        "template"
                        if args.intrinsic_template
                        else ("gaussian" if args.profile_components == 1 else "multi_gaussian")
                    )
                    if needs_tau or needs_alpha or needs_anisotropic
                    else None,
                    profile_components=(
                        None if args.intrinsic_template else args.profile_components
                    )
                    if needs_tau or needs_alpha or needs_anisotropic
                    else None,
                    intrinsic_template=(
                        args.intrinsic_template
                        if needs_tau or needs_alpha or needs_anisotropic
                        else None
                    ),
                    anisotropic_alpha=anisotropic_alpha,
                    anisotropic_alpha_error=anisotropic_alpha_error,
                    anisotropic_alpha_error_lower=anisotropic_alpha_error_lower,
                    anisotropic_alpha_error_upper=anisotropic_alpha_error_upper,
                    anisotropic_tau0_s=anisotropic_tau0,
                    anisotropic_tau0_error_s=anisotropic_tau0_error,
                    anisotropic_tau0_error_lower_s=anisotropic_tau0_error_lower,
                    anisotropic_tau0_error_upper_s=anisotropic_tau0_error_upper,
                    anisotropic_tau_reference_frequency_mhz=(
                        anisotropic_reference_frequency_mhz
                    ),
                    anisotropic_subbands_accepted=anisotropic_subbands_accepted,
                    anisotropic_subbands_rejected=anisotropic_subbands_rejected,
                    anisotropic_subbands_failed=anisotropic_subbands_failed,
                )
            )
            reported = []
            if "dm" in params:
                dm_uncertainty = (
                    "" if dm_error is None else f" +/- {dm_error:.3g}"
                )
                reported.append(f"DM={dm_value:.6g}{dm_uncertainty} ({dm_method})")
            if scattering_dm is not None:
                reported.append(
                    f"scattering delta_DM={scattering_dm.delta_dm:+.6g} "
                    f"+/- {scattering_dm.delta_dm_error:.3g}"
                )
            if "tau" in params:
                reported.append(
                    f"tau={_format_optional_float(tau_value)} +/- {_format_optional_float(tau_error)} s"
                )
            if "tau_ref" in params:
                reported.append(
                    f"tau_ref={_format_optional_float(tau0)} "
                    f"+{_format_optional_float(tau0_error_upper)}"
                    f"/-{_format_optional_float(tau0_error_lower)} s "
                    f"at {_format_optional_float(tau_reference_frequency_mhz)} MHz"
                )
            if "alpha" in params:
                reported.append(
                    f"alpha={_format_optional_float(alpha)} "
                    f"+{_format_optional_float(alpha_error_upper)}"
                    f"/-{_format_optional_float(alpha_error_lower)} "
                    f"(accepted/rejected/failed subbands="
                    f"{alpha_subbands_accepted}/{alpha_subbands_rejected}/{alpha_subbands_failed})"
                )
            if needs_anisotropic:
                reported.append(
                    f"anisotropic alpha={_format_optional_float(anisotropic_alpha)} "
                    f"+{_format_optional_float(anisotropic_alpha_error_upper)}"
                    f"/-{_format_optional_float(anisotropic_alpha_error_lower)}, "
                    f"tau_ref={_format_optional_float(anisotropic_tau0)} "
                    f"+{_format_optional_float(anisotropic_tau0_error_upper)}"
                    f"/-{_format_optional_float(anisotropic_tau0_error_lower)} s"
                )
            if "dnu_d" in params:
                reported.append(
                    f"dnu_d={_format_optional_float(dnu_d)} +/- "
                    f"{_format_optional_float(dnu_d_error)} MHz"
                )
            if "dt_d" in params:
                reported.append(
                    f"dt_d={_format_optional_float(dt_d)} +/- "
                    f"{_format_optional_float(dt_d_error)} s"
                )
            if (
                scintillation_measurement is not None
                and scintillation_measurement.n_scintles is not None
            ):
                reported.append(
                    f"N_scintles={scintillation_measurement.n_scintles:.6g}"
                )
            if scintillation_measurement is not None:
                reported.append(
                    f"ACF method={scintillation_measurement.measurement_method}, "
                    f"resolved(dnu/dt)="
                    f"{scintillation_measurement.decorrelation_bandwidth_resolved}/"
                    f"{scintillation_measurement.diffractive_timescale_resolved}"
                )
            if "t_r" in params:
                reported.append(
                    f"T_r={_format_optional_float(tr_days)} +/- "
                    f"{_format_optional_float(tr_error_days)} days"
                )
            print(f"     {_ok('ok')}  UTC={utc.isoformat()}  MJD={mjd:.6f}")
            print(
                "     "
                f"telescope={geometry.telescope}, "
                f"start={geometry.start_utc.isoformat()}, "
                f"end={geometry.end_utc.isoformat()}, "
                f"Sun sep.={geometry.sun_separation_mid_deg:.3f} deg"
            )
            if quality is not None:
                print(
                    "     "
                    f"quality masked={100.0 * quality.masked_fraction:.3f}%, "
                    f"bad time bins={len(quality.bad_time_bins)}, "
                    f"bad channels={len(quality.bad_frequency_channels)}, "
                    f"isolated cells={quality.isolated_flagged_cells}"
                )
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
    variability_report = analyze_temporal_statistics(
        measurements,
        params,
        maximum_lag=(
            None if args.variability_max_lag == 0 else args.variability_max_lag
        ),
        p_value_threshold=args.variability_p_threshold,
    )
    variability_path = (
        output_prefix.parent / f"{output_prefix.name}_variability.json"
    )
    write_temporal_statistics_report(variability_report, variability_path)
    dm_solar_report = None
    dm_solar_path = None
    if "dm" in params:
        dm_solar_report = analyze_dm_solar_variability(
            measurements,
            dm_series=args.dm_analysis_series,
            p_value_threshold=args.variability_p_threshold,
            solar_warning_angle_deg=args.solar_warning_angle,
            dm_turning_sigma=args.dm_turning_sigma,
            dm_segment_min_points=args.dm_segment_min_points,
        )
        dm_solar_path = (
            output_prefix.parent / f"{output_prefix.name}_dm_solar.json"
        )
        write_dm_solar_variability_report(dm_solar_report, dm_solar_path)
    annual_report = None
    annual_path = None
    annual_chain_path = None
    if args.fit_annual_anisotropy:
        annual_path = (
            output_prefix.parent / f"{output_prefix.name}_annual_anisotropy.json"
        )
        annual_chain_path = (
            output_prefix.parent
            / f"{output_prefix.name}_annual_anisotropy_chain.npz"
        )
        try:
            annual_report = fit_annual_anisotropy(
                measurements,
                distance_kpc=args.annual_distance_kpc,
                proper_motion_ra_cosdec_mas_per_year=(
                    args.annual_pm_ra_cosdec_mas_yr
                ),
                proper_motion_dec_mas_per_year=args.annual_pm_dec_mas_yr,
                minimum_epochs=args.annual_min_epochs,
                maximum_screen_speed_kms=args.annual_max_screen_speed,
                maximum_axial_ratio=args.annual_max_axial_ratio,
                mcmc_walkers=args.annual_mcmc_walkers,
                mcmc_steps=args.annual_mcmc_steps,
                mcmc_burn=args.annual_mcmc_burn,
                mcmc_seed=args.annual_mcmc_seed,
                posterior_samples_path=annual_chain_path,
            )
        except (RuntimeError, ValueError) as exc:
            print(_err(f"psrism: annual anisotropy fit failed: {exc}"), file=sys.stderr)
            return 2
        write_annual_anisotropy_report(annual_report, annual_path)

    print(_section("Saved Time-Series Products"))
    print(f"  {_style('CSV table', 'bold')}: {_ok(str(csv_path))}")
    print(f"  {_style('Variability report', 'bold')}: {_ok(str(variability_path))}")
    if dm_solar_path is not None:
        print(f"  {_style('DM/solar report', 'bold')}: {_ok(str(dm_solar_path))}")
    if annual_path is not None:
        print(f"  {_style('Annual fit report', 'bold')}: {_ok(str(annual_path))}")
        print(f"  {_style('Annual MCMC chain', 'bold')}: {_ok(str(annual_chain_path))}")
    _print_temporal_statistics(variability_report)
    if dm_solar_report is not None:
        _print_dm_solar_variability(dm_solar_report)
    if annual_report is not None:
        _print_annual_anisotropy(annual_report)
    band_title = ""
    if band_lows and band_highs:
        band_title = f" ({min(band_lows):.3f}-{max(band_highs):.3f} MHz)"
    plotted = 0
    param_titles = {
        "dm": "DM",
        "tau": "scattering timescale τ",
        "tau_ref": "reference-frequency scattering timescale τ_ref",
        "alpha": "scattering spectral index α",
        "dnu_d": "decorrelation bandwidth Δν_d",
        "dt_d": "diffractive timescale Δt_d",
        "t_r": "refractive timescale T_r",
    }
    for param in ["dm", "tau", "tau_ref", "alpha", "dnu_d", "dt_d", "t_r"]:
        if param not in params:
            continue
        output_path = output_prefix.parent / f"{output_prefix.name}_{param}_vs_time.png"
        for legacy_axis in ("utc", "mjd"):
            legacy_path = output_prefix.parent / f"{output_prefix.name}_{param}_vs_time_{legacy_axis}.png"
            try:
                legacy_path.unlink(missing_ok=True)
            except OSError:
                pass
        title_name = series_name if series_name is not None else archive_dir.name
        title = f"{title_name}{band_title}: {param_titles[param]} vs time"
        if plot_parameter_vs_time(measurements, param, args.time_axis, title, output_path):
            print(f"  {_style(param, 'bold')}: {_ok(str(output_path))}")
            plotted += 1
        else:
            print(f"  {_style(param, 'bold')}: {_warn('no finite values to plot')}")

    for result in variability_report.parameters:
        acf_path = (
            output_prefix.parent
            / f"{output_prefix.name}_{result.parameter}_autocorrelation.png"
        )
        if plot_parameter_autocorrelation(result, acf_path):
            print(
                f"  {_style(result.parameter + ' ACF', 'bold')}: "
                f"{_ok(str(acf_path))}"
            )

    correlation_path = (
        output_prefix.parent / f"{output_prefix.name}_correlations.png"
    )
    if plot_temporal_correlation_matrix(variability_report, correlation_path):
        print(f"  {_style('Correlations', 'bold')}: {_ok(str(correlation_path))}")

    if dm_solar_report is not None:
        solar_plot_path = (
            output_prefix.parent / f"{output_prefix.name}_dm_vs_sun_separation.png"
        )
        if plot_dm_vs_sun_separation(
            measurements, dm_solar_report, solar_plot_path
        ):
            print(f"  {_style('DM/Sun', 'bold')}: {_ok(str(solar_plot_path))}")
        slope_plot_path = (
            output_prefix.parent / f"{output_prefix.name}_dm_piecewise_slopes.png"
        )
        if plot_dm_piecewise_slopes(
            measurements, dm_solar_report, slope_plot_path
        ):
            print(f"  {_style('DM slopes', 'bold')}: {_ok(str(slope_plot_path))}")

    if annual_report is not None:
        annual_plot_path = (
            output_prefix.parent / f"{output_prefix.name}_annual_anisotropy.png"
        )
        if plot_annual_anisotropy(annual_report, annual_plot_path):
            print(
                f"  {_style('Annual anisotropy', 'bold')}: "
                f"{_ok(str(annual_plot_path))}"
            )
        posterior_plot_path = (
            output_prefix.parent
            / f"{output_prefix.name}_annual_anisotropy_posterior.png"
        )
        if plot_annual_anisotropy_posterior(
            annual_report, posterior_plot_path
        ):
            print(
                f"  {_style('Annual posterior', 'bold')}: "
                f"{_ok(str(posterior_plot_path))}"
            )

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
        summarize_scintillation_measurement,
        write_scintillation_report,
    )
    from .anisotropic_scattering import (
        fit_anisotropic_scattering_from_archive,
        plot_anisotropic_subband_fits,
    )
    from .data_quality import (
        apply_quality_mask_to_archive,
        assess_dynamic_spectrum,
        write_quality_report,
    )
    from .dynamic_spectrum import calculate_dynamic_spectrum, normalize_dynamic_spectrum
    from .dm_analysis import run_pdmp, write_dm_report
    from .fit_tau import (
        fit_tau_alpha_from_archive,
        fit_tau_from_archive,
        plot_subband_tau_fits,
        plot_tau_fit,
    )
    from .fit_autocorrelation_spectrum import fit_autocorrelation_spectrum
    from .intrinsic_width import (
        analyze_intrinsic_widths,
        plot_intrinsic_widths,
        write_intrinsic_width_report,
    )
    from .fit_secondary_spectrum import fit_parabolic_arc, write_arc_fit_report
    from .plot_autocorrelation_spectrum import plot_autocorrelation_spectrum
    from .plot_dynamic_spectrum import plot_dynamic_spectrum, plot_integrated_profile
    from .plot_scintillation_spectrum import plot_arc_search, plot_scintillation_spectrum
    from .plot_tau_vs_freq import plot_tau_vs_frequency
    from .refractive_scintillation import estimate_scintillation_from_tau
    from .scintillation_spectrum import (
        calculate_scintillation_spectrum,
        secondary_spectrum_metadata,
    )
    from .solar_geometry import observation_geometry

    print(_section("Run Setup"))
    print(f"  {_style('Loaded archive', 'bold')}: {_ok(args.archive)}")
    print(f"  {_style('Output directory', 'bold')}: {_ok(str(output_dir))}")
    print(f"  {_style('Terminal output log', 'bold')}: {_ok(str(log_path))}")
    profile_model = (
        f"template ({args.intrinsic_template})"
        if args.intrinsic_template
        else f"{args.profile_components} Gaussian component(s)"
    )
    print(f"  {_style('Profile model', 'bold')}: {_ok(profile_model)}")
    archive_dm = float(archive.get_dispersion_measure())
    pdmp_result = None
    scattering_dm_result = None
    dm_report_path = output_prefix.parent / f"{output_prefix.name}_dm_measurement.json"
    processing_dm = args.dm
    if args.measure_dm:
        try:
            pdmp_result = run_pdmp(args.archive, archive_dm, **_pdmp_options(args))
        except (RuntimeError, ValueError) as exc:
            print(_err(f"psrism: error: pdmp DM measurement failed: {exc}"), file=sys.stderr)
            return 2
        processing_dm = pdmp_result.best_dm
        _print_pdmp_result(pdmp_result)
        write_dm_report(dm_report_path, pdmp_result, None)
        print(f" DM report = {_ok(str(dm_report_path))}")
    if args.inspect:
        metadata = archive_metadata(archive, raw_shape=raw_shape)
        _print_metadata(metadata)
        _print_observation_geometry(observation_geometry(archive))
        _print_scrunch_targets(raw_shape, format_scrunch_targets)
        return 0

    try:
        preprocess_archive(
            archive,
            dm=processing_dm,
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
    _print_observation_geometry(observation_geometry(archive))

    dynspec = None
    quality = None
    needs_dynspec = args.dspec or args.acspec or args.sspec or args.fit_arc or args.rfi_mask
    if needs_dynspec:
        try:
            dynspec = calculate_dynamic_spectrum(
                archive,
                width_factor=max(args.onw, 1),
                interpulse=bool(args.interpulse),
                normalize=False,
            )
            quality = assess_dynamic_spectrum(
                dynspec,
                sigma_threshold=args.rfi_sigma,
                min_valid_fraction=args.rfi_min_valid_fraction,
                detect_rfi=args.rfi_mask,
            )
            if args.rfi_mask:
                apply_quality_mask_to_archive(archive, quality.valid_mask)
            dynspec = np.where(quality.valid_mask, dynspec, np.nan)
            if args.normalize_dspec:
                dynspec = normalize_dynamic_spectrum(dynspec)
        except (RuntimeError, ValueError) as exc:
            print(_err(f"psrism: error: data-quality assessment failed: {exc}"), file=sys.stderr)
            return 2
        _print_data_quality(quality)
        quality_path = output_prefix.parent / f"{output_prefix.name}_data_quality.json"
        write_quality_report(quality, quality_path)
        print(f" quality report = {_ok(str(quality_path))}")

    integrated_tau_result = None
    if args.intpf or args.fit_tau:
        try:
            integrated_tau_result = fit_tau_from_archive(
                archive,
                center_peak=True,
                n_components=args.profile_components,
                intrinsic_template=args.intrinsic_template,
            )
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
        plot_dynamic_spectrum(
            dynspec,
            metadata,
            output_path=_plot_path(output_prefix, "dspec"),
            valid_mask=quality.valid_mask,
        )

    if args.acspec and dynspec is not None:
        acf2d = calculate_autocorrelation_spectrum(dynspec, valid_mask=quality.valid_mask)
        time_lag, freq_lag = autocorrelation_axes(
            metadata.processed_shape[0],
            metadata.processed_shape[2],
            metadata.observation_time_s,
            metadata.bandwidth_mhz,
        )
        scales = measure_acf_scales(acf2d, time_lag, freq_lag)
        acf_fit_result = None
        try:
            acf_fit_result = fit_autocorrelation_spectrum(acf2d, time_lag, freq_lag)
        except (RuntimeError, ValueError) as exc:
            print(_warn(f"psrism: warning: tilted ACF fit failed: {exc}"), file=sys.stderr)
        scintillation_measurement = summarize_scintillation_measurement(
            scales,
            time_lag,
            freq_lag,
            fit_result=acf_fit_result,
            minimum_resolution_bins=args.scintillation_min_resolution_bins,
            observing_duration_s=metadata.observation_time_s,
            observing_bandwidth_mhz=metadata.bandwidth_mhz,
            eta_time=args.eta_time,
            eta_freq=args.eta_freq,
        )
        _print_scintillation_measurement(scintillation_measurement)
        if acf_fit_result is not None:
            _print_acf_tilt_fit(acf_fit_result)
        scintillation_path = (
            output_prefix.parent / f"{output_prefix.name}_scintillation.json"
        )
        write_scintillation_report(scintillation_measurement, scintillation_path)
        print(f" scintillation report = {_ok(str(scintillation_path))}")
        plot_autocorrelation_spectrum(
            acf2d,
            metadata,
            output_path=_plot_path(output_prefix, f"acspec{'_zoom' if args.zoom_acf else ''}"),
            acf_fit_result=acf_fit_result,
            zoom_fit=args.zoom_acf,
        )

    arc_fit_result = None
    secondary_metadata = None
    if (args.fit_arc or args.sspec) and dynspec is not None:
        secondary_metadata = secondary_spectrum_metadata(
            dynspec,
            metadata.observation_time_s,
            metadata.bandwidth_mhz,
            valid_mask=quality.valid_mask,
            window=args.secondary_window,
            centre_frequency_mhz=metadata.centre_frequency_mhz,
        )
        _print_secondary_spectrum_metadata(secondary_metadata)

    if args.fit_arc and dynspec is not None:
        spectrum_linear, fringe_frequency, delay = calculate_scintillation_spectrum(
            dynspec,
            metadata.observation_time_s,
            metadata.bandwidth_mhz,
            log_scale=False,
            valid_mask=quality.valid_mask,
            window=args.secondary_window,
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
                minimum_samples=args.arc_min_samples,
                minimum_score_snr=args.arc_min_score_snr,
            )
            _print_arc_fit(arc_fit_result)
            arc_report_path = output_prefix.parent / f"{output_prefix.name}_arc_fit.json"
            write_arc_fit_report(
                arc_fit_result,
                arc_report_path,
                secondary_metadata=secondary_metadata,
            )
            plot_arc_search(
                arc_fit_result,
                output_path=_plot_path(output_prefix, "arc_search"),
            )
            print(f" arc fit report = {_ok(str(arc_report_path))}")
        except (RuntimeError, ValueError) as exc:
            print(_err(f"psrism: error: parabolic arc fit failed: {exc}"), file=sys.stderr)
            return 2

    if args.sspec and dynspec is not None:
        spectrum, fringe_frequency, delay = calculate_scintillation_spectrum(
            dynspec,
            metadata.observation_time_s,
            metadata.bandwidth_mhz,
            valid_mask=quality.valid_mask,
            window=args.secondary_window,
        )
        plot_scintillation_spectrum(
            spectrum,
            fringe_frequency,
            delay,
            title=metadata.filename,
            output_path=_plot_path(output_prefix, "sspec"),
            arc_fit_result=arc_fit_result,
            central_mask_bins=args.arc_mask_bins,
        )

    if args.fit_tau:
        if not args.intpf:
            _print_tau_result(integrated_tau_result, label="Integrated-profile tau fit")
        profile = integrated_profile_array(
            archive,
            normalize=False,
            center_peak=True,
            remove_baseline=False,
        )
        plot_tau_fit(profile, integrated_tau_result, output_path=_plot_path(output_prefix, "tau_fit"))

    alpha_result_container = None
    if args.fit_alpha:
        try:
            result = fit_tau_alpha_from_archive(
                archive,
                n_subbands=args.tau_subbands,
                reference_freq_mhz=args.tau_reference_freq,
                n_components=args.profile_components,
                intrinsic_template=args.intrinsic_template,
                **_frequency_scaling_options(args),
            )
        except (RuntimeError, ValueError) as exc:
            print(_err(f"psrism: error: {exc}"), file=sys.stderr)
            return 2
        alpha_result_container = result
        alpha_fit = result.alpha_fit
        print(_section("Subband Tau Fits"))
        for item in result.subbands:
            status = _ok("accepted") if item.accepted else _warn("rejected")
            print(
                f" channels {item.channel_start}-{item.channel_stop}: "
                f"fc={item.central_frequency_mhz:.3f} MHz, "
                f"bandwidth={item.bandwidth_mhz:.3f} MHz, "
                f"fm={item.frequency_mhz:.3f} MHz, "
                f"tau={item.tau:.6e} +/- {item.tau_error:.6e} s"
            )
            print(
                f"  intrinsic model = {item.fit.intrinsic_model}, "
                f"DC baseline = {item.fit.baseline:.6g}, "
                f"fit S/N = {_format_optional_float(item.profile_snr)}, "
                f"valid channels = {item.valid_channel_fraction:.3f}, "
                f"red. chi-square = {_format_optional_float(item.fit.reduced_chi_square)}, "
                f"RMS = {_format_optional_float(item.fit.rms_residual)}, "
                f"status = {status}"
            )
            if item.rejection_reasons:
                print(f"  rejection reason(s): {'; '.join(item.rejection_reasons)}")
        for item in result.failed_subbands:
            print(
                f" channels {item.channel_start}-{item.channel_stop}: "
                f"fm={item.frequency_mhz:.3f} MHz, status = {_warn('fit failed')}; "
                f"reason: {item.rejection_reason}"
            )
        _print_alpha_summary(alpha_fit)
        plot_tau_vs_frequency(
            [item.frequency_mhz for item in result.subbands],
            [item.tau for item in result.subbands],
            tau_error=[item.tau_error for item in result.subbands],
            alpha_result=alpha_fit,
            accepted_mask=[item.accepted for item in result.subbands],
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
        intrinsic_width_report = analyze_intrinsic_widths(
            result,
            scattering_model="isotropic",
            **_intrinsic_width_options(args),
        )
        intrinsic_width_path = (
            output_prefix.parent / f"{output_prefix.name}_intrinsic_width.json"
        )
        write_intrinsic_width_report(intrinsic_width_report, intrinsic_width_path)
        plot_intrinsic_widths(
            intrinsic_width_report,
            title=metadata.filename,
            output_path=_plot_path(output_prefix, "intrinsic_width_vs_freq"),
        )
        _print_intrinsic_width_report(intrinsic_width_report, intrinsic_width_path)

    if args.check_scattering_dm:
        try:
            input_dm = processing_dm if processing_dm is not None else archive_dm
            scattering_dm_result = _scattering_dm_result(
                alpha_result_container,
                archive,
                input_dm,
            )
        except (RuntimeError, ValueError) as exc:
            print(_err(f"psrism: error: scattering-aware DM check failed: {exc}"), file=sys.stderr)
            return 2
        _print_scattering_dm_result(scattering_dm_result)

    anisotropic_result = None
    if args.fit_anisotropy:
        try:
            anisotropic_result = fit_anisotropic_scattering_from_archive(
                archive,
                n_subbands=args.tau_subbands,
                reference_freq_mhz=args.tau_reference_freq,
                n_components=args.profile_components,
                intrinsic_template=args.intrinsic_template,
                **_frequency_scaling_options(args),
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
            accepted_mask=[item.accepted for item in anisotropic_result.subbands],
            measurement_label="Subband τ_eff",
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
        anisotropic_width_report = analyze_intrinsic_widths(
            anisotropic_result,
            scattering_model="anisotropic",
            **_intrinsic_width_options(args),
        )
        anisotropic_width_path = (
            output_prefix.parent
            / f"{output_prefix.name}_anisotropic_intrinsic_width.json"
        )
        write_intrinsic_width_report(anisotropic_width_report, anisotropic_width_path)
        plot_intrinsic_widths(
            anisotropic_width_report,
            title=metadata.filename,
            output_path=_plot_path(output_prefix, "anisotropic_intrinsic_width_vs_freq"),
        )
        _print_intrinsic_width_report(
            anisotropic_width_report,
            anisotropic_width_path,
        )

    if args.estimate_refractive:
        try:
            if anisotropic_result is not None:
                accepted = anisotropic_result.accepted_subbands
                freq = [item.frequency_mhz for item in accepted]
                tau = [item.tau_eff for item in accepted]
                tau_error = [item.tau_eff_error for item in accepted]
                source_label = "anisotropic τ_eff"
            else:
                if alpha_result_container is None:
                    alpha_result_container = fit_tau_alpha_from_archive(
                        archive,
                        n_subbands=args.tau_subbands,
                        reference_freq_mhz=args.tau_reference_freq,
                        n_components=args.profile_components,
                        intrinsic_template=args.intrinsic_template,
                        **_frequency_scaling_options(args),
                    )
                accepted = alpha_result_container.accepted_subbands
                freq = [item.frequency_mhz for item in accepted]
                tau = [item.tau for item in accepted]
                tau_error = [item.tau_error for item in accepted]
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

    if scattering_dm_result is not None:
        write_dm_report(dm_report_path, pdmp_result, scattering_dm_result)
        label = "updated DM report" if pdmp_result is not None else "DM report"
        print(f" {label} = {_ok(str(dm_report_path))}")

    return 0


def _print_tau_result(result, label: str) -> None:
    print(_section(label))
    print(f" intrinsic model = {result.intrinsic_model}")
    if result.components:
        print(f" Gaussian components = {len(result.components)}")
    if result.phase_shift_bins is not None:
        print(f" template phase shift = {result.phase_shift_bins:.6g} bins")
    print(f" fitted DC baseline = {result.baseline:.6g}")
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


def _print_pdmp_result(result) -> None:
    print(_section("PDMP Dispersion-Measure Fit"))
    print(f" archive DM = {result.archive_dm:.12g} pc cm^-3")
    print(
        f" measured DM = {result.best_dm:.12g} +/- "
        f"{result.dm_error:.6g} pc cm^-3"
    )
    print(f" correction = {result.dm_correction:+.12g} pc cm^-3")
    print(f" best profile S/N = {_format_optional_float(result.best_snr)}")


def _print_scattering_dm_result(result) -> None:
    print(_section("Scattering-Aware DM Check"))
    print(f" accepted intrinsic phases = {result.n_points}")
    print(f" phase reference frequency = {result.reference_frequency_mhz:.6g} MHz")
    print(
        f" residual delta DM = {result.delta_dm:+.12g} +/- "
        f"{result.delta_dm_error:.6g} pc cm^-3"
    )
    print(f" corrected DM = {result.corrected_dm:.12g} pc cm^-3")
    print(
        f" goodness: reduced chi-square = "
        f"{_format_optional_float(result.reduced_chi_square)}, "
        f"RMS phase residual = {result.rms_residual_s:.6e} s"
    )


def _print_data_quality(report) -> None:
    print(_section("Dynamic Spectrum Data Quality"))
    mode = "archive weights + robust statistical RFI mask" if report.statistical_masking else "archive weights/non-finite samples only"
    print(f" masking mode = {mode}")
    print(f" input invalid cells = {report.input_invalid_cells}")
    print(f" flagged time bins = {len(report.bad_time_bins)}")
    print(f" flagged frequency channels = {len(report.bad_frequency_channels)}")
    print(f" isolated flagged cells = {report.isolated_flagged_cells}")
    fraction_text = f"{100.0 * report.masked_fraction:.3f}%"
    if report.masked_fraction >= 0.5:
        print(f" total masked = {_warn(fraction_text)}")
        print(_warn(" warning: at least half of the dynamic spectrum is masked"))
    else:
        print(f" total masked = {fraction_text}")


def _print_scintillation_measurement(result) -> None:
    print(_section("ACF Scintillation Measurement"))
    print(f" primary method = {result.measurement_method}")
    print(
        f" frequency resolution = {result.frequency_resolution_mhz:.6g} MHz; "
        f"minimum accepted width = {result.minimum_resolution_bins:g} samples"
    )
    if result.decorrelation_bandwidth_mhz is None:
        print(
            f" decorrelation bandwidth: {_warn('unresolved')} "
            f"(fit={_format_optional_float(result.fit_decorrelation_bandwidth_mhz)} MHz, "
            f"width samples="
            f"{_format_optional_float(result.decorrelation_bandwidth_resolution_bins)}, "
            f"half-power crossing={result.frequency_crossing_found})"
        )
    else:
        print(
            f" decorrelation bandwidth = {result.decorrelation_bandwidth_mhz:.6g} +/- "
            f"{_format_optional_float(result.decorrelation_bandwidth_error_mhz)} MHz "
            f"({result.decorrelation_bandwidth_resolution_bins:.3g} samples)"
        )

    print(f" time resolution = {result.time_resolution_s:.6g} s")
    if result.diffractive_timescale_s is None:
        print(
            f" diffractive timescale: {_warn('unresolved')} "
            f"(fit={_format_optional_float(result.fit_diffractive_timescale_s)} s, "
            f"width samples="
            f"{_format_optional_float(result.diffractive_timescale_resolution_bins)}, "
            f"1/e crossing={result.time_crossing_found})"
        )
    else:
        print(
            f" diffractive timescale = {result.diffractive_timescale_s:.6g} +/- "
            f"{_format_optional_float(result.diffractive_timescale_error_s)} s "
            f"({result.diffractive_timescale_resolution_bins:.3g} samples)"
        )
    print(
        " slice cross-checks: "
        f"dnu_d={_format_optional_float(result.slice_decorrelation_bandwidth_mhz)} MHz, "
        f"dt_d={_format_optional_float(result.slice_diffractive_timescale_s)} s"
    )
    if result.n_scintles is not None:
        print(
            f" finite coverage: N_scintles={result.n_scintles:.6g}, "
            f"1/sqrt(N_scintles)={result.finite_scintle_fraction:.6g}, "
            f"eta_t={result.eta_time:g}, eta_nu={result.eta_frequency:g}"
        )


def _print_acf_tilt_fit(result) -> None:
    print(_section("Tilted ACF Gaussian Fit"))
    print(
        f" fit decorrelation bandwidth = {result.delta_f_diss:.6g} +/- "
        f"{_format_optional_float(result.delta_f_diss_error)} MHz"
    )
    print(
        f" fit diffractive timescale = {result.delta_t_diss:.6g} +/- "
        f"{_format_optional_float(result.delta_t_diss_error)} s"
    )
    print(
        f" correlation coefficient = {result.correlation:.6g} +/- "
        f"{_format_optional_float(result.correlation_error)}"
    )
    print(f" ellipse rotation angle = {result.rotation_angle_deg:.6g} deg")
    slope = result.drift_slope_s_per_mhz
    rate = result.drift_rate_mhz_per_s
    if slope is None:
        print(f" scintle drift slope d(time lag)/d(freq lag): {_warn('unavailable')}")
    else:
        print(
            f" scintle drift slope d(time lag)/d(freq lag) = {slope:.6g} +/- "
            f"{_format_optional_float(result.drift_slope_error_s_per_mhz)} s/MHz"
        )
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
    print(
        f" status = "
        f"{_ok('accepted') if result.accepted else _warn('rejected')}"
    )
    print(f" delay half searched = {result.half}")
    print(f" apex fringe-frequency offset = {result.fringe_offset:.6g} Hz")
    print(f" apex delay offset = {result.delay_offset:.6g} s")
    print(
        f" curvature eta = {result.curvature:.6e} "
        f"+/- {_format_optional_float(result.curvature_error)} s^3"
    )
    print(
        " half-height interval = "
        f"[{_format_optional_float(result.curvature_lower)}, "
        f"{_format_optional_float(result.curvature_upper)}] s^3"
    )
    print(f" arc strength = {result.score:.6g}")
    print(f" arc strength S/N = {_format_optional_float(result.score_snr)}")
    print(
        f" samples on best arc = {result.n_samples} "
        f"(minimum {result.minimum_samples})"
    )
    print(f" peak at search boundary = {result.peak_at_search_boundary}")
    if result.rejection_reasons:
        print(f" rejection reasons = {', '.join(result.rejection_reasons)}")


def _print_secondary_spectrum_metadata(result) -> None:
    print(_section("Secondary Spectrum Sampling"))
    print(f" FFT window = {result.window}")
    print(f" masked dynamic-spectrum fraction = {result.masked_fraction:.6g}")
    print(
        f" dynamic-spectrum sampling = {result.time_resolution_s:.6g} s x "
        f"{result.frequency_resolution_mhz:.6g} MHz"
    )
    print(
        f" Fourier resolution = {result.fringe_frequency_resolution_hz:.6g} Hz x "
        f"{result.delay_resolution_s:.6e} s"
    )
    print(
        f" Nyquist limits = +/-{result.fringe_frequency_nyquist_hz:.6g} Hz, "
        f"+/-{result.delay_nyquist_s:.6e} s"
    )


def _print_intrinsic_width_report(report, report_path: Path) -> None:
    print(_section(f"Intrinsic Profile Widths ({report.scattering_model})"))
    if not report.width_available:
        print(f"  Width measurements: {_warn('unavailable for intrinsic templates')}")
    else:
        for measurement in report.measurements:
            status = _ok("accepted") if measurement.accepted else _warn("rejected")
            uncertainty = (
                ""
                if measurement.fwhm_percent_period_error is None
                else f" +/- {measurement.fwhm_percent_period_error:.4g}"
            )
            print(
                f"  component {measurement.component}, "
                f"nu={measurement.frequency_mhz:.3f} MHz: "
                f"FWHM={measurement.fwhm_percent_period:.6g}{uncertainty}% "
                f"of period ({status})"
            )
        for trend in report.trends:
            exponent_error = (
                "n/a"
                if trend.frequency_exponent_error is None
                else f"{trend.frequency_exponent_error:.4g}"
            )
            p_value = (
                "n/a"
                if trend.exponent_p_value is None
                else f"{trend.exponent_p_value:.4g}"
            )
            print(
                f"  component {trend.component} trend: "
                f"W proportional to nu^{trend.frequency_exponent:.5g} "
                f"+/- {exponent_error}, p={p_value}, "
                f"significant={'yes' if trend.significant else 'no'}"
            )
    for warning in report.warnings:
        print(f"  {_warn('Caution')}: {warning}")
    print(f"  {_style('JSON report', 'bold')}: {_ok(str(report_path))}")


def _print_temporal_statistics(report) -> None:
    print(_section("Temporal Variability Diagnostics"))
    print(f" p-value threshold = {report.p_value_threshold:g}")
    print(f" lag convention = {report.lag_unit}")
    for item in report.parameters:
        print(
            f" {item.parameter}: n={item.n_finite}/{item.n_epochs_total}, "
            f"span={_format_optional_float(item.time_span_days)} days, "
            f"mean={_format_optional_float(item.mean)}, "
            f"weighted mean={_format_optional_float(item.weighted_mean)} "
            f"+/- {_format_optional_float(item.weighted_mean_error)}"
        )
        print(
            "  constant/runs/Ljung-Box p = "
            f"{_format_optional_float(item.constant_p_value)} / "
            f"{_format_optional_float(item.runs_p_value)} / "
            f"{_format_optional_float(item.ljung_box_p_at_maximum_lag)} "
            f"(maximum lag {item.maximum_lag})"
        )
    for item in report.correlations:
        if item.pearson_r is None:
            continue
        print(
            f" Pearson {item.parameter_x}/{item.parameter_y}: "
            f"r={item.pearson_r:.6g}, p={item.p_value:.6g}, n={item.n_paired}"
        )


def _print_dm_solar_variability(report) -> None:
    print(_section("DM Variability and Solar Diagnostics"))
    print(
        f" series={report.dm_series}, "
        f"finite epochs={report.n_dm_finite}/{report.n_epochs_total}"
    )
    methods = ", ".join(report.solar_ephemeris_methods) or "unknown"
    print(
        f" solar ephemeris={methods}, "
        f"minimum separation={_format_optional_float(report.minimum_sun_separation_deg)} deg, "
        f"epochs below {report.solar_warning_angle_deg:g} deg="
        f"{report.epochs_below_warning_angle}"
    )
    print(
        " DM/Sun Pearson r/p/n = "
        f"{_format_optional_float(report.solar_pearson_r)} / "
        f"{_format_optional_float(report.solar_pearson_p_value)} / "
        f"{report.n_solar_pairs}"
    )
    if report.global_fit is not None:
        print(
            " global DM slope = "
            f"{report.global_fit.slope_pc_cm3_per_year:.6g} +/- "
            f"{_format_optional_float(report.global_fit.slope_error_pc_cm3_per_year)} "
            "pc cm^-3 yr^-1"
        )
    print(
        f" accepted same-direction segments={report.accepted_segments}/"
        f"{len(report.segments)}, weighted mean |slope|="
        f"{_format_optional_float(report.mean_absolute_slope_pc_cm3_per_year)} +/- "
        f"{_format_optional_float(report.mean_absolute_slope_error_pc_cm3_per_year)} "
        "pc cm^-3 yr^-1"
    )


def _print_annual_anisotropy(report) -> None:
    print(_section("Annual Anisotropic Scintillation Fit"))
    print(
        f" fitted epochs={report.n_epochs_fitted}/{report.n_epochs_total}, "
        f"span={report.time_span_days:.6g} days, "
        f"mean Delta_nu_d={report.mean_decorrelation_bandwidth_mhz:.6g} +/- "
        f"{report.mean_decorrelation_bandwidth_error_mhz:.6g} MHz"
    )
    print(
        f" chi-square/dof={report.chi_square:.6g}/{report.degrees_of_freedom}, "
        f"reduced chi-square={report.reduced_chi_square:.6g}, "
        f"mean acceptance={report.mean_acceptance_fraction:.4f}"
    )
    for parameter in report.parameters:
        unit = f" {parameter.unit}" if parameter.unit else ""
        print(
            f" {parameter.name}={parameter.value:.6g} "
            f"-{parameter.error_lower:.3g}/+{parameter.error_upper:.3g}{unit}"
        )


def _print_anisotropic_result(result) -> None:
    print(_section("Anisotropic Scattering Fits"))
    for item in result.subbands:
        status = _ok("accepted") if item.accepted else _warn("rejected")
        print(
            f" channels {item.channel_start}-{item.channel_stop}: "
            f"fc={item.central_frequency_mhz:.3f} MHz, "
            f"bandwidth={item.bandwidth_mhz:.3f} MHz, "
            f"fm={item.frequency_mhz:.3f} MHz, "
            f"tau_x={item.tau_x:.6e} +/- {item.tau_x_error:.6e} s, "
            f"tau_y={item.tau_y:.6e} +/- {item.tau_y_error:.6e} s, "
            f"tau_eff={item.tau_eff:.6e} +/- {item.tau_eff_error:.6e} s, "
            f"tau_ratio={item.anisotropy_ratio:.6g} +/- {item.anisotropy_ratio_error:.6g}"
        )
        print(
            f"  intrinsic model = {item.fit.intrinsic_model}, "
            f"DC baseline = {item.fit.baseline:.6g}"
        )
        print(
            f"  goodness: anisotropic red. chi-square = "
            f"{_format_optional_float(item.fit.reduced_chi_square)}, "
            f"isotropic red. chi-square = {_format_optional_float(item.isotropic_reduced_chi_square)}, "
            f"RMS = {_format_optional_float(item.fit.rms_residual)}, "
            f"fit S/N = {_format_optional_float(item.profile_snr)}, "
            f"valid channels = {item.valid_channel_fraction:.3f}, status = {status}"
        )
        if item.rejection_reasons:
            print(f"  rejection reason(s): {'; '.join(item.rejection_reasons)}")
    for item in result.failed_subbands:
        print(
            f" channels {item.channel_start}-{item.channel_stop}: "
            f"fm={item.frequency_mhz:.3f} MHz, status = {_warn('fit failed')}; "
            f"reason: {item.rejection_reason}"
        )
    _print_alpha_summary(result.alpha_fit, prefix="anisotropic tau_eff ", tau_prefix="tau_eff,")


def _print_alpha_summary(alpha_fit, prefix: str = "", tau_prefix: str = "tau") -> None:
    print(
        f"\n{prefix}alpha = {alpha_fit.alpha:.6f} "
        f"+{alpha_fit.alpha_error_upper:.6f}/-{alpha_fit.alpha_error_lower:.6f} "
        f"(adopted +/- {alpha_fit.alpha_error:.6f})"
    )
    print(f"alpha convention: {SCATTERING_INDEX_CONVENTION}")
    tau_name = (
        f"{tau_prefix}150"
        if abs(alpha_fit.reference_freq_mhz - TAU_REFERENCE_FREQUENCY_MHZ) < 1e-9
        else f"{tau_prefix}ref"
    )
    print(
        f"{tau_name} = {alpha_fit.tau0:.6e} "
        f"+{alpha_fit.tau0_error_upper:.6e}/-{alpha_fit.tau0_error_lower:.6e} s "
        f"(adopted +/- {alpha_fit.tau0_error:.6e} s) "
        f"at nu_ref={alpha_fit.reference_freq_mhz:.3f} MHz"
    )
    print(
        f"uncertainty = {alpha_fit.uncertainty_method}; "
        f"Monte Carlo samples = {alpha_fit.monte_carlo_samples}; "
        f"weighted-fit alpha error = {alpha_fit.covariance_alpha_error:.6f}"
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
