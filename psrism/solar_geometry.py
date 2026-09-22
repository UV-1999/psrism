"""Observation geometry and Sun-separation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import re


# Standards provenance: the MJD epoch is a time-coordinate definition, not a
# value drawn from the bundled pulsar references.
MJD0 = datetime(1858, 11, 17, tzinfo=timezone.utc)


@dataclass(frozen=True)
class TelescopeLocation:
    name: str
    latitude_deg: float | None
    longitude_deg: float | None
    height_m: float | None
    source: str


@dataclass(frozen=True)
class ObservationGeometry:
    telescope: str
    telescope_latitude_deg: float | None
    telescope_longitude_deg: float | None
    telescope_height_m: float | None
    telescope_location_source: str
    source_ra_deg: float
    source_dec_deg: float
    start_utc: datetime
    end_utc: datetime
    start_mjd: float
    end_mjd: float
    sun_separation_start_deg: float
    sun_separation_mid_deg: float
    sun_separation_end_deg: float
    solar_ephemeris_method: str


TELESCOPE_LOCATIONS = {
    # Approximate geodetic positions for common LOFAR stations used in this project.
    # The Sun-pulsar angular separation is geocentric here, so these coordinates are
    # recorded for provenance and future topocentric extensions.
    "DE601": TelescopeLocation("DE601", 50.5248, 6.8836, 366.0, "PSRISM approximate value; not from bundled references"),
    "DE601HBA": TelescopeLocation("DE601HBA", 50.5248, 6.8836, 366.0, "PSRISM approximate value; not from bundled references"),
    "EFFELSBERG": TelescopeLocation("EFFELSBERG", 50.5248, 6.8836, 366.0, "PSRISM approximate value; not from bundled references"),
    "PL611": TelescopeLocation("PL611", 50.0890, 19.6090, 350.0, "PSRISM approximate value; not from bundled references"),
    "PL611HBA": TelescopeLocation("PL611HBA", 50.0890, 19.6090, 350.0, "PSRISM approximate value; not from bundled references"),
}


def observation_geometry(archive) -> ObservationGeometry:
    """Extract archive observing geometry and Sun-pulsar angular separations."""
    telescope = archive.get_telescope()
    location = telescope_location(telescope)
    source_ra_deg, source_dec_deg = source_coordinates_deg(archive)
    start_mjd, end_mjd = observation_start_end_mjd(archive)
    start_utc = mjd_to_datetime(start_mjd)
    end_utc = mjd_to_datetime(end_mjd)
    mid_mjd = 0.5 * (start_mjd + end_mjd)
    ephemeris_method = solar_ephemeris_method()

    return ObservationGeometry(
        telescope=telescope,
        telescope_latitude_deg=location.latitude_deg,
        telescope_longitude_deg=location.longitude_deg,
        telescope_height_m=location.height_m,
        telescope_location_source=location.source,
        source_ra_deg=source_ra_deg,
        source_dec_deg=source_dec_deg,
        start_utc=start_utc,
        end_utc=end_utc,
        start_mjd=start_mjd,
        end_mjd=end_mjd,
        sun_separation_start_deg=sun_pulsar_separation_deg(
            source_ra_deg, source_dec_deg, start_mjd, method=ephemeris_method
        ),
        sun_separation_mid_deg=sun_pulsar_separation_deg(
            source_ra_deg, source_dec_deg, mid_mjd, method=ephemeris_method
        ),
        sun_separation_end_deg=sun_pulsar_separation_deg(
            source_ra_deg, source_dec_deg, end_mjd, method=ephemeris_method
        ),
        solar_ephemeris_method=ephemeris_method,
    )


def telescope_location(telescope: str) -> TelescopeLocation:
    key = telescope.upper().strip()
    if key in TELESCOPE_LOCATIONS:
        return TELESCOPE_LOCATIONS[key]
    for suffix in ("HBA", "LBA"):
        if key.endswith(suffix) and key[: -len(suffix)] in TELESCOPE_LOCATIONS:
            base = TELESCOPE_LOCATIONS[key[: -len(suffix)]]
            return TelescopeLocation(telescope, base.latitude_deg, base.longitude_deg, base.height_m, base.source)
    return TelescopeLocation(telescope, None, None, None, "unknown")


def source_coordinates_deg(archive) -> tuple[float, float]:
    coord = archive.get_coordinates()
    values = coord.getDegrees()
    if isinstance(values, str):
        numbers = re.findall(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", values)
        if len(numbers) < 2:
            raise ValueError(f"could not parse source coordinates from {values!r}")
        return float(numbers[0]), float(numbers[1])
    return float(values[0]), float(values[1])


def observation_start_end_mjd(archive) -> tuple[float, float]:
    nsub = archive.get_nsubint()
    if nsub <= 0:
        epoch = float(archive.get_Integration(0).get_epoch().in_days())
        half = float(archive.integration_length()) / 86400.0 / 2.0
        return epoch - half, epoch + half
    start = float(archive.get_Integration(0).get_start_time().in_days())
    end = float(archive.get_Integration(nsub - 1).get_end_time().in_days())
    return start, end


def mjd_to_datetime(mjd: float) -> datetime:
    return MJD0 + timedelta(days=float(mjd))


def datetime_to_mjd(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt.astimezone(timezone.utc) - MJD0).total_seconds() / 86400.0


def solar_ephemeris_method() -> str:
    """Return the available Sun-position implementation."""
    try:
        import astropy.coordinates  # noqa: F401
        import astropy.time  # noqa: F401
        import astropy.units  # noqa: F401
    except ImportError:
        return "psrism_approximate_fallback"
    return "astropy_builtin"


def sun_pulsar_separation_deg(
    source_ra_deg: float,
    source_dec_deg: float,
    mjd: float,
    method: str | None = None,
) -> float:
    method = solar_ephemeris_method() if method is None else method
    if method == "astropy_builtin":
        from astropy.coordinates import SkyCoord, get_sun
        from astropy.time import Time
        import astropy.units as u

        # Reference: Filothodoros thesis (ALEX.pdf), PDF p. 70, Section 4.7,
        # uses Astropy to calculate the Sun-pulsar separation per observation.
        epoch = Time(float(mjd), format="mjd", scale="utc")
        sun = get_sun(epoch)
        source = SkyCoord(
            ra=float(source_ra_deg) * u.deg,
            dec=float(source_dec_deg) * u.deg,
            frame="icrs",
        ).transform_to(sun.frame)
        return float(source.separation(sun).deg)
    if method != "psrism_approximate_fallback":
        raise ValueError(f"unknown solar ephemeris method: {method}")
    sun_ra_deg, sun_dec_deg = approximate_sun_ra_dec_deg(mjd)
    return angular_separation_deg(source_ra_deg, source_dec_deg, sun_ra_deg, sun_dec_deg)


def approximate_sun_ra_dec_deg(mjd: float) -> tuple[float, float]:
    """Approximate apparent geocentric Sun RA/Dec in degrees.

    This is a PSRISM low-precision utility for qualitative time-series
    overlays, not a precision astrometry calculation.
    """
    # Method motivation: the Filothodoros thesis (ALEX.pdf), PDF p. 35,
    # Section 2.8, proposes comparing temporal scattering variation with the
    # Sun-pulsar angular distance. The ephemeris approximation and station
    # table are PSRISM utility choices, not formulas or values taken from the
    # bundled references. Use a precision astrometry package when required.
    jd = float(mjd) + 2400000.5
    n = jd - 2451545.0
    mean_long = (280.460 + 0.9856474 * n) % 360.0
    mean_anomaly = math.radians((357.528 + 0.9856003 * n) % 360.0)
    ecliptic_long = math.radians(
        (mean_long + 1.915 * math.sin(mean_anomaly) + 0.020 * math.sin(2.0 * mean_anomaly)) % 360.0
    )
    obliquity = math.radians(23.439 - 0.0000004 * n)
    ra = math.atan2(math.cos(obliquity) * math.sin(ecliptic_long), math.cos(ecliptic_long))
    dec = math.asin(math.sin(obliquity) * math.sin(ecliptic_long))
    return math.degrees(ra) % 360.0, math.degrees(dec)


def angular_separation_deg(ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
    # PSRISM utility implementation of spherical angular separation; this
    # equation is not taken from the bundled pulsar references.
    ra1 = math.radians(ra1_deg)
    dec1 = math.radians(dec1_deg)
    ra2 = math.radians(ra2_deg)
    dec2 = math.radians(dec2_deg)
    cos_sep = (
        math.sin(dec1) * math.sin(dec2)
        + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    )
    cos_sep = min(1.0, max(-1.0, cos_sep))
    return math.degrees(math.acos(cos_sep))
