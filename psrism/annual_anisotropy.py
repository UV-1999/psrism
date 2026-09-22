"""Annual anisotropy equations and multi-epoch scintillation fitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np


# Standards provenance: the MJD epoch is a time-coordinate definition, not a
# value drawn from the bundled pulsar references.
MJD0 = datetime(1858, 11, 17, tzinfo=timezone.utc)


@dataclass(frozen=True)
class AnnualAnisotropyCoefficients:
    axial_ratio: float
    psi_deg: float
    r: float
    a: float
    b: float
    c: float


@dataclass(frozen=True)
class AnnualParameterEstimate:
    name: str
    unit: str
    value: float
    error_lower: float
    error_upper: float
    percentile_16: float
    percentile_50: float
    percentile_84: float


@dataclass(frozen=True)
class AnnualEpochResult:
    archive_name: str
    mjd: float
    day_of_year: float
    observing_frequency_mhz: float
    diffractive_timescale_s: float
    diffractive_timescale_error_s: float
    earth_velocity_alpha_kms: float
    earth_velocity_delta_kms: float
    observed_q: float
    observed_q_error: float
    model_q: float
    observed_scintillation_velocity_kms: float
    observed_scintillation_velocity_error_kms: float
    model_scintillation_velocity_kms: float
    normalized_residual: float


@dataclass(frozen=True)
class AnnualAnisotropyReport:
    model: str
    solitary_pulsar_only: bool
    n_epochs_total: int
    n_epochs_fitted: int
    minimum_epochs: int
    start_mjd: float
    end_mjd: float
    time_span_days: float
    source_ra_deg: float
    source_dec_deg: float
    pulsar_distance_kpc: float
    proper_motion_ra_cosdec_mas_per_year: float
    proper_motion_dec_mas_per_year: float
    pulsar_velocity_alpha_kms: float
    pulsar_velocity_delta_kms: float
    mean_decorrelation_bandwidth_mhz: float
    mean_decorrelation_bandwidth_error_mhz: float
    bandwidth_average_method: str
    a_iss_model: str
    maximum_screen_speed_kms: float
    maximum_axial_ratio: float
    chi_square: float
    degrees_of_freedom: int
    reduced_chi_square: float
    log_likelihood: float
    mcmc_walkers: int
    mcmc_steps: int
    mcmc_burn: int
    mcmc_seed: int
    mcmc_samples: int
    mean_acceptance_fraction: float
    autocorrelation_time_steps: tuple[float | None, ...]
    point_estimate: str
    posterior_samples_file: str | None
    posterior_sample_columns: tuple[str, ...]
    parameters: tuple[AnnualParameterEstimate, ...]
    epochs: tuple[AnnualEpochResult, ...]


def anisotropy_coefficients(axial_ratio: float, psi_deg: float) -> AnnualAnisotropyCoefficients:
    """Return coefficients for the anisotropic effective-velocity quadratic."""
    # Reference: Cai et al. (scintillationJ0814.pdf), PDF p. 3, Eqs. 5-6.
    axial_ratio = max(float(axial_ratio), 1.0)
    psi_rad = np.deg2rad(float(psi_deg))
    r = (axial_ratio**2 - 1.0) / (axial_ratio**2 + 1.0)
    denom = np.sqrt(max(1.0 - r**2, 1e-15))
    a = (1.0 - r * np.cos(2.0 * psi_rad)) / denom
    b = (1.0 + r * np.cos(2.0 * psi_rad)) / denom
    c = -2.0 * r * np.sin(2.0 * psi_rad) / denom
    return AnnualAnisotropyCoefficients(
        axial_ratio=axial_ratio,
        psi_deg=float(psi_deg),
        r=float(r),
        a=float(a),
        b=float(b),
        c=float(c),
    )


def effective_velocity_components(
    distance_kpc: float,
    screen_distance_kpc: float,
    earth_velocity_alpha_kms,
    earth_velocity_delta_kms,
    pulsar_velocity_alpha_kms,
    pulsar_velocity_delta_kms,
    screen_velocity_alpha_kms=0.0,
    screen_velocity_delta_kms=0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute thin-screen effective velocity components."""
    distance = float(distance_kpc)
    screen_distance = float(screen_distance_kpc)
    if distance <= 0:
        raise ValueError("distance_kpc must be positive")
    # Reference: Cai et al. (scintillationJ0814.pdf), PDF p. 3, Eqs. 3 and 5.
    weight_earth = (distance - screen_distance) / distance
    weight_pulsar = screen_distance / distance
    v_alpha = (
        weight_earth * np.asarray(earth_velocity_alpha_kms, dtype=float)
        + weight_pulsar * np.asarray(pulsar_velocity_alpha_kms, dtype=float)
        - np.asarray(screen_velocity_alpha_kms, dtype=float)
    )
    v_delta = (
        weight_earth * np.asarray(earth_velocity_delta_kms, dtype=float)
        + weight_pulsar * np.asarray(pulsar_velocity_delta_kms, dtype=float)
        - np.asarray(screen_velocity_delta_kms, dtype=float)
    )
    return v_alpha, v_delta


def anisotropic_effective_speed(
    v_eff_alpha_kms,
    v_eff_delta_kms,
    axial_ratio: float,
    psi_deg: float,
) -> np.ndarray:
    """Evaluate the anisotropic effective speed magnitude."""
    # Reference: Cai et al. (scintillationJ0814.pdf), PDF p. 3, Eq. 5.
    coeff = anisotropy_coefficients(axial_ratio, psi_deg)
    v_alpha = np.asarray(v_eff_alpha_kms, dtype=float)
    v_delta = np.asarray(v_eff_delta_kms, dtype=float)
    speed2 = coeff.a * v_alpha**2 + coeff.b * v_delta**2 + coeff.c * v_alpha * v_delta
    return np.sqrt(np.clip(speed2, 0.0, np.inf))


def scintillation_velocity(
    a_iss: float,
    distance_kpc,
    decorrelation_bandwidth_mhz,
    frequency_ghz,
    diffractive_timescale_s,
) -> np.ndarray:
    """Compute V_ISS = A_ISS sqrt(D Delta_nu_d) / (f Delta_t_d)."""
    # Reference: Cai et al. (scintillationJ0814.pdf), PDF p. 3, Eq. 1.
    return (
        float(a_iss)
        * np.sqrt(np.asarray(distance_kpc, dtype=float) * np.asarray(decorrelation_bandwidth_mhz, dtype=float))
        / (np.asarray(frequency_ghz, dtype=float) * np.asarray(diffractive_timescale_s, dtype=float))
    )


def thin_screen_a_iss(
    distance_kpc: float,
    screen_distance_kpc: float,
    axial_ratio: float,
) -> float:
    """Return anisotropy-dependent A_ISS for a thin screen."""
    distance = float(distance_kpc)
    screen_distance = float(screen_distance_kpc)
    axial_ratio = float(axial_ratio)
    if distance <= 0:
        raise ValueError("distance_kpc must be positive")
    if not 0 < screen_distance < distance:
        raise ValueError("screen_distance_kpc must lie between Earth and pulsar")
    if axial_ratio < 1:
        raise ValueError("axial_ratio must be at least 1")
    # Reference: Cai et al. (scintillationJ0814.pdf), PDF p. 3,
    # Section 3.1, gives the anisotropy-dependent thin-screen A_ISS.
    return float(
        2.78e4
        * np.sqrt((axial_ratio + 1.0 / axial_ratio) / 2.0)
        * np.sqrt(2.0 * screen_distance / (distance - screen_distance))
    )


def annual_q_model(
    earth_velocity_alpha_kms,
    earth_velocity_delta_kms,
    distance_kpc: float,
    pulsar_velocity_alpha_kms: float,
    pulsar_velocity_delta_kms: float,
    screen_velocity_alpha_kms: float,
    screen_velocity_delta_kms: float,
    axial_ratio: float,
    psi_deg: float,
    screen_distance_kpc: float,
) -> np.ndarray:
    """Evaluate the annual thin-screen model for Q."""
    v_alpha, v_delta = effective_velocity_components(
        distance_kpc,
        screen_distance_kpc,
        earth_velocity_alpha_kms,
        earth_velocity_delta_kms,
        pulsar_velocity_alpha_kms,
        pulsar_velocity_delta_kms,
        screen_velocity_alpha_kms,
        screen_velocity_delta_kms,
    )
    speed = anisotropic_effective_speed(v_alpha, v_delta, axial_ratio, psi_deg)
    a_iss = thin_screen_a_iss(distance_kpc, screen_distance_kpc, axial_ratio)
    # Reference: Cai et al. (scintillationJ0814.pdf), PDF p. 3,
    # Section 3.1, Eq. 4, defines Q from the thin-screen velocity model.
    return (
        speed
        * np.sqrt(float(distance_kpc))
        / (a_iss * (float(distance_kpc) - float(screen_distance_kpc)))
    )


def earth_velocity_equatorial_kms(
    mjd,
    source_ra_deg: float,
    source_dec_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project Earth's barycentric velocity onto the source tangent plane."""
    try:
        from astropy.coordinates import get_body_barycentric_posvel
        from astropy.time import Time
        import astropy.units as u
    except ImportError as exc:
        raise RuntimeError(
            "annual anisotropy fitting requires astropy; reinstall PSRISM dependencies"
        ) from exc

    epochs = Time(np.asarray(mjd, dtype=float), format="mjd", scale="tdb")
    _position, velocity = get_body_barycentric_posvel("earth", epochs)
    vectors = np.moveaxis(velocity.xyz.to_value(u.km / u.s), 0, -1)
    ra = np.deg2rad(float(source_ra_deg))
    dec = np.deg2rad(float(source_dec_deg))
    alpha_hat = np.asarray([-np.sin(ra), np.cos(ra), 0.0])
    delta_hat = np.asarray(
        [-np.cos(ra) * np.sin(dec), -np.sin(ra) * np.sin(dec), np.cos(dec)]
    )
    # Reference: Cai et al. (scintillationJ0814.pdf), PDF pp. 3-4,
    # Sections 3.1-3.2, uses projected Earth orbital velocity in the annual
    # model. Astropy barycentric velocity and this ICRS tangent projection are
    # PSRISM implementation choices; topocentric station rotation is omitted.
    return vectors @ alpha_hat, vectors @ delta_hat


def proper_motion_velocity_kms(
    proper_motion_ra_cosdec_mas_per_year: float,
    proper_motion_dec_mas_per_year: float,
    distance_kpc: float,
) -> tuple[float, float]:
    """Convert equatorial proper motion and distance to transverse velocity."""
    # Standards provenance: 4.74047 converts arcsec/yr at pc to km/s. The
    # input RA component follows the common mu_alpha*cos(delta) convention.
    conversion = 4.74047
    return (
        float(conversion * proper_motion_ra_cosdec_mas_per_year * distance_kpc),
        float(conversion * proper_motion_dec_mas_per_year * distance_kpc),
    )


# PSRISM generic defaults: the minimum epoch count, broad prior bounds, chain
# length, burn-in, and reproducibility seed are not prescribed by the bundled
# annual-variation reference and remain user-configurable through the CLI.
def fit_annual_anisotropy(
    measurements,
    distance_kpc: float,
    proper_motion_ra_cosdec_mas_per_year: float,
    proper_motion_dec_mas_per_year: float,
    minimum_epochs: int = 8,
    maximum_screen_speed_kms: float = 500.0,
    maximum_axial_ratio: float = 20.0,
    mcmc_walkers: int = 32,
    mcmc_steps: int = 5000,
    mcmc_burn: int = 1000,
    mcmc_seed: int = 0,
    posterior_samples_path: str | Path | None = None,
) -> AnnualAnisotropyReport:
    """Fit the five-parameter annual anisotropic thin-screen model."""
    from scipy.optimize import least_squares

    try:
        import emcee
    except ImportError as exc:
        raise RuntimeError(
            "annual anisotropy fitting requires emcee; reinstall PSRISM dependencies"
        ) from exc

    distance = float(distance_kpc)
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("distance_kpc must be positive")
    if minimum_epochs < 6:
        raise ValueError("minimum_epochs must be at least 6 for a five-parameter fit")
    if not np.isfinite(maximum_screen_speed_kms) or maximum_screen_speed_kms <= 0:
        raise ValueError("maximum_screen_speed_kms must be positive")
    if not np.isfinite(maximum_axial_ratio) or maximum_axial_ratio <= 1:
        raise ValueError("maximum_axial_ratio must exceed 1")
    if mcmc_walkers < 10 or mcmc_walkers % 2:
        raise ValueError("mcmc_walkers must be an even integer of at least 10")
    if mcmc_steps < 2:
        raise ValueError("mcmc_steps must be at least 2")
    if not 0 <= mcmc_burn < mcmc_steps:
        raise ValueError("mcmc_burn must be nonnegative and smaller than mcmc_steps")
    if mcmc_seed < 0:
        raise ValueError("mcmc_seed cannot be negative")

    arrays = _annual_measurement_arrays(measurements)
    usable = (
        np.isfinite(arrays["mjd"])
        & np.isfinite(arrays["frequency_mhz"])
        & (arrays["frequency_mhz"] > 0)
        & np.isfinite(arrays["bandwidth_mhz"])
        & (arrays["bandwidth_mhz"] > 0)
        & np.isfinite(arrays["timescale_s"])
        & (arrays["timescale_s"] > 0)
        & np.isfinite(arrays["timescale_error_s"])
        & (arrays["timescale_error_s"] > 0)
        & np.isfinite(arrays["ra_deg"])
        & np.isfinite(arrays["dec_deg"])
    )
    indices = np.flatnonzero(usable)
    if len(indices) < minimum_epochs:
        raise ValueError(
            f"annual anisotropy requires at least {minimum_epochs} epochs with "
            "resolved bandwidth, timescale error, frequency, and source coordinates; "
            f"found {len(indices)}"
        )
    order = np.argsort(arrays["mjd"][indices])
    indices = indices[order]
    mjd = arrays["mjd"][indices]
    frequency_mhz = arrays["frequency_mhz"][indices]
    bandwidth_mhz = arrays["bandwidth_mhz"][indices]
    bandwidth_error_mhz = arrays["bandwidth_error_mhz"][indices]
    timescale_s = arrays["timescale_s"][indices]
    timescale_error_s = arrays["timescale_error_s"][indices]
    source_ra = float(np.median(arrays["ra_deg"][indices]))
    source_dec = float(np.median(arrays["dec_deg"][indices]))

    # Reference: Cai et al. (scintillationJ0814.pdf), PDF p. 3,
    # Section 3.2, uses the average of all measured Delta_nu_d values.
    mean_bandwidth = float(np.mean(bandwidth_mhz))
    valid_bandwidth_errors = np.isfinite(bandwidth_error_mhz) & (
        bandwidth_error_mhz > 0
    )
    if np.all(valid_bandwidth_errors):
        # PSRISM choice: propagate independent epoch errors through the
        # arithmetic mean used by the reference analysis.
        mean_bandwidth_error = float(
            np.sqrt(np.sum(bandwidth_error_mhz**2)) / len(bandwidth_mhz)
        )
        bandwidth_average_method = "arithmetic_mean_propagated_errors"
    else:
        # PSRISM fallback when one or more reported bandwidth errors are absent.
        mean_bandwidth_error = float(
            np.std(bandwidth_mhz, ddof=1) / np.sqrt(len(bandwidth_mhz))
        )
        bandwidth_average_method = "arithmetic_mean_empirical_standard_error"

    frequency_ghz = frequency_mhz / 1000.0
    observed_q = np.sqrt(mean_bandwidth) / (frequency_ghz * timescale_s)
    # PSRISM choice: first-order independent-error propagation through Eq. 4.
    observed_q_error = observed_q * np.sqrt(
        (0.5 * mean_bandwidth_error / mean_bandwidth) ** 2
        + (timescale_error_s / timescale_s) ** 2
    )
    if not np.all(np.isfinite(observed_q_error) & (observed_q_error > 0)):
        raise ValueError("annual anisotropy produced invalid Q uncertainties")

    earth_alpha, earth_delta = earth_velocity_equatorial_kms(
        mjd, source_ra, source_dec
    )
    pulsar_alpha, pulsar_delta = proper_motion_velocity_kms(
        proper_motion_ra_cosdec_mas_per_year,
        proper_motion_dec_mas_per_year,
        distance,
    )

    # PSRISM uniform-prior choices: each screen-velocity component is bounded
    # symmetrically, Ar is at least one, psi has 180-degree ellipse symmetry,
    # and the screen stays just inside the Earth-pulsar line of sight.
    lower = np.asarray(
        [
            -maximum_screen_speed_kms,
            -maximum_screen_speed_kms,
            1.0,
            0.0,
            distance * 1e-4,
        ],
        dtype=float,
    )
    upper = np.asarray(
        [
            maximum_screen_speed_kms,
            maximum_screen_speed_kms,
            maximum_axial_ratio,
            180.0,
            distance * (1.0 - 1e-4),
        ],
        dtype=float,
    )

    def model(parameters):
        return annual_q_model(
            earth_alpha,
            earth_delta,
            distance,
            pulsar_alpha,
            pulsar_delta,
            parameters[0],
            parameters[1],
            parameters[2],
            parameters[3],
            parameters[4],
        )

    def residuals(parameters):
        return (observed_q - model(parameters)) / observed_q_error

    # PSRISM choice: deterministic multi-start bounded least squares supplies
    # a stable walker center; scientific intervals come from MCMC below.
    best_optimizer = None
    axial_ratio_starts = tuple(
        1.0 + fraction * (maximum_axial_ratio - 1.0)
        for fraction in (0.05, 0.25, 0.7)
    )
    for axial_ratio in axial_ratio_starts:
        for psi_deg in (15.0, 60.0, 105.0, 150.0):
            for screen_fraction in (0.2, 0.5, 0.8):
                initial = np.asarray(
                    [0.0, 0.0, axial_ratio, psi_deg, distance * screen_fraction]
                )
                result = least_squares(
                    residuals,
                    initial,
                    bounds=(lower, upper),
                    x_scale=np.asarray(
                        [100.0, 100.0, 2.0, 90.0, max(distance / 2.0, 1e-6)]
                    ),
                    max_nfev=5000,
                )
                if best_optimizer is None or np.sum(result.fun**2) < np.sum(
                    best_optimizer.fun**2
                ):
                    best_optimizer = result

    rng = np.random.default_rng(mcmc_seed)
    span = upper - lower
    walker_center = np.clip(
        best_optimizer.x,
        lower + 0.01 * span,
        upper - 0.01 * span,
    )
    initial_walkers = walker_center + rng.normal(
        0.0,
        0.005 * span,
        size=(mcmc_walkers, len(walker_center)),
    )
    margin = span * 1e-8
    initial_walkers = np.clip(initial_walkers, lower + margin, upper - margin)

    normalization = float(np.sum(np.log(2.0 * np.pi * observed_q_error**2)))

    def log_probability(parameters):
        if np.any(parameters <= lower) or np.any(parameters >= upper):
            return -np.inf
        values = residuals(parameters)
        if not np.all(np.isfinite(values)):
            return -np.inf
        # Reference: Cai et al. (scintillationJ0814.pdf), PDF p. 4,
        # Section 3.2, fits five annual-model parameters with emcee and uniform
        # priors. The bounded intervals above are declared PSRISM priors.
        return float(-0.5 * (np.sum(values**2) + normalization))

    sampler = emcee.EnsembleSampler(
        mcmc_walkers,
        len(best_optimizer.x),
        log_probability,
    )
    # PSRISM choice: seed emcee's legacy RandomState separately from the
    # Generator used for walker initialization so the full chain is repeatable.
    sampler.random_state = np.random.RandomState(mcmc_seed).get_state()
    sampler.run_mcmc(initial_walkers, mcmc_steps, progress=False)
    samples = sampler.get_chain(discard=mcmc_burn, flat=True)
    sample_log_probability = sampler.get_log_prob(discard=mcmc_burn, flat=True)
    finite_samples = np.all(np.isfinite(samples), axis=1) & np.isfinite(
        sample_log_probability
    )
    samples = samples[finite_samples]
    sample_log_probability = sample_log_probability[finite_samples]
    if not len(samples):
        raise RuntimeError("annual anisotropy MCMC produced no finite samples")

    percentiles = np.percentile(samples, [16.0, 50.0, 84.0], axis=0)
    point = percentiles[1]
    parameter_names = (
        ("screen_velocity_alpha_kms", "km s^-1"),
        ("screen_velocity_delta_kms", "km s^-1"),
        ("axial_ratio", ""),
        ("psi_deg", "deg"),
        ("screen_distance_kpc", "kpc"),
    )
    if posterior_samples_path is not None:
        posterior_samples_path = Path(posterior_samples_path)
        # PSRISM provenance choice: retain the complete finite post-burn chain
        # and log probabilities so convergence and alternative summaries can
        # be checked without rerunning archive processing.
        np.savez_compressed(
            posterior_samples_path,
            samples=samples,
            log_probability=sample_log_probability,
            columns=np.asarray([name for name, _unit in parameter_names]),
        )
    estimates = tuple(
        AnnualParameterEstimate(
            name=name,
            unit=unit,
            value=float(point[index]),
            error_lower=float(percentiles[1, index] - percentiles[0, index]),
            error_upper=float(percentiles[2, index] - percentiles[1, index]),
            percentile_16=float(percentiles[0, index]),
            percentile_50=float(percentiles[1, index]),
            percentile_84=float(percentiles[2, index]),
        )
        for index, (name, unit) in enumerate(parameter_names)
    )
    fitted_q = model(point)
    normalized_residual = (observed_q - fitted_q) / observed_q_error
    chi_square = float(np.sum(normalized_residual**2))
    degrees_of_freedom = len(observed_q) - len(point)
    log_likelihood = float(-0.5 * (chi_square + normalization))
    fitted_a_iss = thin_screen_a_iss(distance, point[4], point[2])
    observed_viss = scintillation_velocity(
        fitted_a_iss,
        distance,
        mean_bandwidth,
        frequency_ghz,
        timescale_s,
    )
    observed_viss_error = observed_viss * observed_q_error / observed_q
    model_viss = fitted_q * fitted_a_iss * np.sqrt(distance)

    try:
        autocorrelation = sampler.get_autocorr_time(tol=0)
    except Exception:
        autocorrelation = np.full(len(point), np.nan)

    epoch_results = tuple(
        AnnualEpochResult(
            archive_name=str(arrays["archive_name"][source_index]),
            mjd=float(mjd[index]),
            day_of_year=_day_of_year(
                arrays["utc"][source_index], mjd=float(mjd[index])
            ),
            observing_frequency_mhz=float(frequency_mhz[index]),
            diffractive_timescale_s=float(timescale_s[index]),
            diffractive_timescale_error_s=float(timescale_error_s[index]),
            earth_velocity_alpha_kms=float(earth_alpha[index]),
            earth_velocity_delta_kms=float(earth_delta[index]),
            observed_q=float(observed_q[index]),
            observed_q_error=float(observed_q_error[index]),
            model_q=float(fitted_q[index]),
            observed_scintillation_velocity_kms=float(observed_viss[index]),
            observed_scintillation_velocity_error_kms=float(
                observed_viss_error[index]
            ),
            model_scintillation_velocity_kms=float(model_viss[index]),
            normalized_residual=float(normalized_residual[index]),
        )
        for index, source_index in enumerate(indices)
    )
    return AnnualAnisotropyReport(
        model="anisotropic_thin_screen_five_parameter",
        solitary_pulsar_only=True,
        n_epochs_total=len(measurements),
        n_epochs_fitted=len(indices),
        minimum_epochs=int(minimum_epochs),
        start_mjd=float(np.min(mjd)),
        end_mjd=float(np.max(mjd)),
        time_span_days=float(np.ptp(mjd)),
        source_ra_deg=source_ra,
        source_dec_deg=source_dec,
        pulsar_distance_kpc=distance,
        proper_motion_ra_cosdec_mas_per_year=float(
            proper_motion_ra_cosdec_mas_per_year
        ),
        proper_motion_dec_mas_per_year=float(proper_motion_dec_mas_per_year),
        pulsar_velocity_alpha_kms=pulsar_alpha,
        pulsar_velocity_delta_kms=pulsar_delta,
        mean_decorrelation_bandwidth_mhz=mean_bandwidth,
        mean_decorrelation_bandwidth_error_mhz=mean_bandwidth_error,
        bandwidth_average_method=bandwidth_average_method,
        a_iss_model="anisotropic_thin_screen_Cai2026",
        maximum_screen_speed_kms=float(maximum_screen_speed_kms),
        maximum_axial_ratio=float(maximum_axial_ratio),
        chi_square=chi_square,
        degrees_of_freedom=degrees_of_freedom,
        reduced_chi_square=float(chi_square / degrees_of_freedom),
        log_likelihood=log_likelihood,
        mcmc_walkers=int(mcmc_walkers),
        mcmc_steps=int(mcmc_steps),
        mcmc_burn=int(mcmc_burn),
        mcmc_seed=int(mcmc_seed),
        mcmc_samples=len(samples),
        mean_acceptance_fraction=float(np.mean(sampler.acceptance_fraction)),
        autocorrelation_time_steps=tuple(
            None if not np.isfinite(value) else float(value)
            for value in autocorrelation
        ),
        point_estimate="posterior_median",
        posterior_samples_file=(
            None if posterior_samples_path is None else str(posterior_samples_path)
        ),
        posterior_sample_columns=tuple(name for name, _unit in parameter_names),
        parameters=estimates,
        epochs=epoch_results,
    )


def write_annual_anisotropy_report(
    report: AnnualAnisotropyReport,
    output_path: str | Path,
) -> None:
    """Write an annual-anisotropy report as strict JSON."""
    with Path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(asdict(report), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def plot_annual_anisotropy(
    report: AnnualAnisotropyReport,
    output_path: str | Path,
) -> bool:
    """Plot annual scintillation velocity and normalized residuals."""
    if not report.epochs:
        return False
    import matplotlib.pyplot as plt

    day = np.asarray([item.day_of_year for item in report.epochs])
    observed = np.asarray(
        [item.observed_scintillation_velocity_kms for item in report.epochs]
    )
    observed_error = np.asarray(
        [item.observed_scintillation_velocity_error_kms for item in report.epochs]
    )
    model = np.asarray(
        [item.model_scintillation_velocity_kms for item in report.epochs]
    )
    residual = np.asarray([item.normalized_residual for item in report.epochs])
    order = np.argsort(day)

    fig, (ax, residual_ax) = plt.subplots(
        2,
        1,
        figsize=(8.2, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax.errorbar(
        day,
        observed,
        yerr=observed_error,
        fmt="o",
        color="0.35",
        markersize=4,
        capsize=2,
        label="observed",
    )
    ax.plot(day[order], model[order], color="black", linewidth=1.5, label="model")
    ax.set_ylabel("Scintillation velocity (km s$^{-1}$)")
    ax.set_title("Annual anisotropic thin-screen fit")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    residual_ax.axhline(0.0, color="black", linewidth=0.8)
    residual_ax.plot(day, residual, "o", color="tab:blue", markersize=4)
    residual_ax.set_xlabel("Day of year")
    residual_ax.set_ylabel("Residual / sigma")
    residual_ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def plot_annual_anisotropy_posterior(
    report: AnnualAnisotropyReport,
    output_path: str | Path,
) -> bool:
    """Plot one- and two-dimensional views of the saved MCMC posterior."""
    if report.posterior_samples_file is None:
        return False
    sample_path = Path(report.posterior_samples_file)
    if not sample_path.exists():
        return False
    with np.load(sample_path, allow_pickle=False) as payload:
        samples = np.asarray(payload["samples"], dtype=float)
    if samples.ndim != 2 or not len(samples):
        return False

    import matplotlib.pyplot as plt

    labels = (
        r"$V_{IISM,\alpha}$",
        r"$V_{IISM,\delta}$",
        r"$A_r$",
        r"$\psi$",
        r"$D_s$",
    )
    dimensions = samples.shape[1]
    if dimensions != len(labels):
        return False
    maximum_points = 5000
    # PSRISM plotting choice: preserve every sample in the NPZ while limiting
    # only the rendered pairwise scatter density and using 40 histogram bins.
    if len(samples) > maximum_points:
        selected = np.linspace(0, len(samples) - 1, maximum_points, dtype=int)
        plotted_samples = samples[selected]
    else:
        plotted_samples = samples

    fig, axes = plt.subplots(
        dimensions,
        dimensions,
        figsize=(11.0, 11.0),
        squeeze=False,
    )
    medians = np.asarray([item.percentile_50 for item in report.parameters])
    for row in range(dimensions):
        for column in range(dimensions):
            ax = axes[row, column]
            if row < column:
                ax.axis("off")
                continue
            if row == column:
                ax.hist(samples[:, column], bins=40, color="0.3", histtype="stepfilled")
                ax.axvline(medians[column], color="tab:red", linewidth=1.0)
            else:
                ax.plot(
                    plotted_samples[:, column],
                    plotted_samples[:, row],
                    ".",
                    color="0.25",
                    alpha=0.12,
                    markersize=1.0,
                )
                ax.axvline(medians[column], color="tab:red", linewidth=0.6)
                ax.axhline(medians[row], color="tab:red", linewidth=0.6)
            if row == dimensions - 1:
                ax.set_xlabel(labels[column])
            else:
                ax.set_xticklabels([])
            if column == 0 and row > 0:
                ax.set_ylabel(labels[row])
            elif column > 0:
                ax.set_yticklabels([])
            ax.grid(alpha=0.12)
    fig.suptitle("Annual-anisotropy posterior", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def _annual_measurement_arrays(measurements) -> dict[str, np.ndarray]:
    return {
        "archive_name": np.asarray(
            [getattr(item, "archive_name", "") for item in measurements],
            dtype=object,
        ),
        "utc": np.asarray(
            [getattr(item, "utc", None) for item in measurements], dtype=object
        ),
        "mjd": _optional_array(getattr(item, "mjd", None) for item in measurements),
        "frequency_mhz": _optional_array(
            getattr(item, "observing_frequency_mhz", None) for item in measurements
        ),
        "bandwidth_mhz": _optional_array(
            getattr(item, "decorrelation_bandwidth_mhz", None)
            for item in measurements
        ),
        "bandwidth_error_mhz": _optional_array(
            getattr(item, "decorrelation_bandwidth_error_mhz", None)
            for item in measurements
        ),
        "timescale_s": _optional_array(
            getattr(item, "diffractive_timescale_s", None) for item in measurements
        ),
        "timescale_error_s": _optional_array(
            getattr(item, "diffractive_timescale_error_s", None)
            for item in measurements
        ),
        "ra_deg": _optional_array(
            getattr(item, "source_ra_deg", None) for item in measurements
        ),
        "dec_deg": _optional_array(
            getattr(item, "source_dec_deg", None) for item in measurements
        ),
    }


def _day_of_year(value, mjd: float | None = None) -> float:
    if not isinstance(value, datetime):
        if mjd is None or not np.isfinite(mjd):
            raise ValueError("day of year requires a datetime or finite MJD")
        value = MJD0 + timedelta(days=float(mjd))
    start = value.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    return float((value - start).total_seconds() / 86400.0 + 1.0)


def _optional_array(values) -> np.ndarray:
    return np.asarray(
        [np.nan if value is None else float(value) for value in values],
        dtype=float,
    )
