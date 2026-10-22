"""Field stars around a target from the offline Gaia DR3 catalog.

The NIRDA trace contamination model (``trace.py``) needs every star near a
target that is bright enough to matter: its offset on the sky and its J-band
flux relative to the target. Both come from a local copy of Gaia DR3 managed
by ``gaiaoffline``, so no network access is needed at run time. The copy must
be downloaded once; see the gaiaoffline README.

J magnitudes are estimated from Gaia photometry alone (G and BP - RP), so a
star missing from Gaia, or fainter than the database's G limit (G < 16 by
default), is not in the field.

``query_visible_field`` serves the wider VISDA (visible camera) field drawn by
``visda.py`` from the same catalog, with Gaia photometry instead of J.
"""

import os
import warnings

import gaiaoffline
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from gaiaoffline import Gaia

__all__ = [
    "query_field",
    "query_stars",
    "query_visible_field",
    "estimate_jmag",
    "gnomonic_offsets",
]

# =====================================================================================
# Field selection
# =====================================================================================
# Comfortably beyond the farthest a neighbor's trace can reach the target's
# extraction aperture (~4.1 arcmin for the trace extent in trace.py).
FIELD_RADIUS_ARCMIN = 6.0
# Neighbors fainter than this J are ignored.
FIELD_J_LIMIT = 14.0
# The target is the brightest Gaia star whose proper-motion track passes within
# this radius of the given coordinate at some epoch between
# TARGET_TRACK_START_JYEAR and the observation. Name resolvers such as SIMBAD
# give J2000 positions while Gaia's are J2016, so matching on the track rather
# than the J2016 position keeps fast-moving targets matched (GJ 436 moves
# ~1.2 arcsec/yr, ~19 arcsec between the two epochs).
TARGET_MATCH_RADIUS_ARCSEC = 10.0
TARGET_TRACK_START_JYEAR = 2000.0
GAIA_EPOCH_JYEAR = 2016.0

# =====================================================================================
# Gaia to J-band photometry
# =====================================================================================
# G - J as a cubic in BP - RP (highest power first, for np.polyval). Fit on
# 2026-10-20 to the 2MASS crossmatch in the gaiaoffline database: 24 fields
# spread uniformly on the sky, 0.3 deg radius, 2MASS J < 14, 0 < BP - RP < 5.5,
# 4-sigma clipped (10,480 of 11,223 stars kept). Robust scatter 0.05 mag; bin
# medians within 0.04 mag for BP - RP < 3 and within 0.1 mag redder than that.
G_MINUS_J_COEFFS = (-0.01765, 0.04155, 1.20464, 0.08049)
G_MINUS_J_COLOR_RANGE = (0.0, 5.5)
# Stars without BP - RP: the 95th percentile of G - J in the same sample, so
# their J errs bright (toward more contamination) rather than faint.
G_MINUS_J_NO_COLOR = 2.89


def _check_database():
    """Raise FileNotFoundError when the gaiaoffline database is not installed.

    Checked before connecting because sqlite would otherwise create an empty
    file at the database path and fail later with a less helpful error.
    """
    path = gaiaoffline.DATABASEPATH
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(
            f"The gaiaoffline Gaia DR3 database was not found at {path}. The "
            "NIRDA trace check (Visibility use_trace_sweep=True) needs it: "
            "download it as described in the gaiaoffline README, or construct "
            "Visibility with use_trace_sweep=False."
        )


def estimate_jmag(g_mag, bp_rp):
    """J magnitude estimated from Gaia G and BP - RP.

    Parameters
    ----------
    g_mag : array_like
        Gaia G magnitude.
    bp_rp : array_like
        Gaia BP - RP color; NaN where unknown. Colors outside
        ``G_MINUS_J_COLOR_RANGE`` are clipped to it.

    Returns
    -------
    np.ndarray
        Estimated J. Stars without a color use ``G_MINUS_J_NO_COLOR``.
    """
    g_mag = np.asarray(g_mag, dtype=float)
    bp_rp = np.asarray(bp_rp, dtype=float)
    color = np.clip(np.nan_to_num(bp_rp), *G_MINUS_J_COLOR_RANGE)
    return np.where(
        np.isfinite(bp_rp),
        g_mag - np.polyval(G_MINUS_J_COEFFS, color),
        g_mag - G_MINUS_J_NO_COLOR,
    )


def gnomonic_offsets(ra, dec, ra0, dec0):
    """Standard coordinates of (ra, dec) about the tangent point (ra0, dec0).

    The same projection as a FITS TAN WCS.

    Parameters
    ----------
    ra, dec : array_like
        Positions in degrees.
    ra0, dec0 : float
        Tangent point in degrees.

    Returns
    -------
    xi, eta : np.ndarray
        Offsets in arcsec; xi points east and eta points north.
    """
    ra, dec, ra0, dec0 = (
        np.deg2rad(np.asarray(angle, dtype=float)) for angle in (ra, dec, ra0, dec0)
    )
    denominator = np.sin(dec) * np.sin(dec0) + np.cos(dec) * np.cos(dec0) * np.cos(
        ra - ra0
    )
    xi = np.cos(dec) * np.sin(ra - ra0) / denominator
    eta = (
        np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(ra - ra0)
    ) / denominator
    return np.rad2deg(xi) * 3600.0, np.rad2deg(eta) * 3600.0


def query_field(target_coord: SkyCoord, time: Time):
    """Neighbor stars of a target, positioned at the observation epoch.

    Every star is moved along its Gaia proper motion (linear, adequate for
    arcseconds over decades against 1.19 arcsec pixels) from J2016 to the mean
    of ``time``, and its offset is measured from the target at that epoch.

    Parameters
    ----------
    target_coord : SkyCoord
        Target position (scalar), at any epoch from
        ``TARGET_TRACK_START_JYEAR`` to the observation: the target is the
        brightest Gaia star whose proper-motion track passes within
        ``TARGET_MATCH_RADIUS_ARCSEC`` of it over that span.
    time : astropy.time.Time
        Observation time(s); their mean sets the epoch.

    Returns
    -------
    dict or None
        None, with a warning, when no Gaia star lies near ``target_coord``
        (e.g. a bare coordinate pair). Otherwise:

        target_source_id : int
            Gaia DR3 source id of the target.
        target_jmag : float
            Estimated J of the target.
        xi, eta : np.ndarray
            (n,) offsets of the neighbors from the target in arcsec, east
            and north. The target itself is excluded.
        jmag : np.ndarray
            (n,) estimated J of the neighbors, all below ``FIELD_J_LIMIT``.
        flux_ratio : np.ndarray
            (n,) J-band flux of each neighbor relative to the target.

    Raises
    ------
    FileNotFoundError
        If the gaiaoffline database is not installed.
    """
    ra0 = float(target_coord.icrs.ra.deg)
    dec0 = float(target_coord.icrs.dec.deg)
    epoch = float(np.mean(np.atleast_1d(time.jyear)))
    stars = _read_cone(
        ra0,
        dec0,
        FIELD_RADIUS_ARCMIN / 60.0 + TARGET_MATCH_RADIUS_ARCSEC / 3600.0,
        epoch,
    )
    target = _match_target(stars, ra0, dec0, epoch)
    if target is None:
        warnings.warn(
            f"No Gaia star passes within {TARGET_MATCH_RADIUS_ARCSEC:g} arcsec "
            f"of RA {ra0:.5f}, Dec {dec0:.5f} between "
            f"J{TARGET_TRACK_START_JYEAR:g} and the observation; the NIRDA "
            "trace check is skipped for this target.",
            stacklevel=2,
        )
        return None

    # Offsets from the target, both at the observation epoch
    ra_now, dec_now, jmag = stars["ra_now"], stars["dec_now"], stars["jmag"]
    xi, eta = gnomonic_offsets(ra_now, dec_now, ra_now[target], dec_now[target])
    keep = (
        (np.arange(len(jmag)) != target)
        & (jmag < FIELD_J_LIMIT)
        & (np.hypot(xi, eta) <= FIELD_RADIUS_ARCMIN * 60.0)
    )
    return {
        "target_source_id": int(stars["source_id"][target]),
        "target_jmag": float(jmag[target]),
        "xi": xi[keep],
        "eta": eta[keep],
        "jmag": jmag[keep],
        "flux_ratio": 10.0 ** (-0.4 * (jmag[keep] - jmag[target])),
    }


def query_stars(center: SkyCoord, time: Time):
    """Gaia stars around any sky position, positioned at the observation epoch.

    For targets that are not themselves Gaia stars, such as planets and their
    moons: no target is matched, and offsets are measured from ``center``.
    Selection, proper motion and J estimates are as in ``query_field``.

    Parameters
    ----------
    center : SkyCoord
        Position (scalar) the offsets are measured from.
    time : astropy.time.Time
        Observation time(s); their mean sets the epoch.

    Returns
    -------
    dict
        xi, eta : np.ndarray
            (n,) offsets from ``center`` in arcsec, east and north, of the
            stars within ``FIELD_RADIUS_ARCMIN`` and below ``FIELD_J_LIMIT``.
        jmag : np.ndarray
            (n,) their estimated J.

    Raises
    ------
    FileNotFoundError
        If the gaiaoffline database is not installed.
    """
    ra0 = float(center.icrs.ra.deg)
    dec0 = float(center.icrs.dec.deg)
    epoch = float(np.mean(np.atleast_1d(time.jyear)))
    stars = _read_cone(ra0, dec0, FIELD_RADIUS_ARCMIN / 60.0, epoch)
    xi, eta = gnomonic_offsets(stars["ra_now"], stars["dec_now"], ra0, dec0)
    keep = (stars["jmag"] < FIELD_J_LIMIT) & (
        np.hypot(xi, eta) <= FIELD_RADIUS_ARCMIN * 60.0
    )
    return {"xi": xi[keep], "eta": eta[keep], "jmag": stars["jmag"][keep]}


def query_visible_field(
    target_coord: SkyCoord, time: Time, radius_arcmin: float, match_target=True
):
    """Gaia stars for a VISDA (visible camera) field, at the observation epoch.

    Unlike ``query_field``, every star in the cone is returned whatever its
    brightness (down to the database's G limit) and the target is included.
    The target is matched as in ``query_field``; when no Gaia star matches (a
    bare coordinate), or ``match_target`` is False (a solar system body, which
    must not be swapped for a star near it), offsets are measured from
    ``target_coord`` instead, without a warning.

    Parameters
    ----------
    target_coord : SkyCoord
        Target position (scalar); see ``query_field``.
    time : astropy.time.Time
        Observation time(s); their mean sets the epoch.
    radius_arcmin : float
        Cone radius about the target, in arcmin.
    match_target : bool, optional
        Look for the target among the Gaia stars (default True).

    Returns
    -------
    dict
        xi, eta : np.ndarray
            (n,) offsets from the target (or ``target_coord``) in arcsec,
            east and north.
        g_mag, bp_mag, bp_rp : np.ndarray
            (n,) Gaia G, BP and BP - RP; NaN where Gaia has none.
        is_target : np.ndarray
            (n,) bool, True for the matched target (all False if none).

    Raises
    ------
    FileNotFoundError
        If the gaiaoffline database is not installed.
    """
    ra0 = float(target_coord.icrs.ra.deg)
    dec0 = float(target_coord.icrs.dec.deg)
    epoch = float(np.mean(np.atleast_1d(time.jyear)))
    stars = _read_cone(
        ra0, dec0, radius_arcmin / 60.0 + TARGET_MATCH_RADIUS_ARCSEC / 3600.0, epoch
    )
    target = _match_target(stars, ra0, dec0, epoch) if match_target else None
    ra_now, dec_now = stars["ra_now"], stars["dec_now"]
    if target is None:
        xi, eta = gnomonic_offsets(ra_now, dec_now, ra0, dec0)
    else:
        xi, eta = gnomonic_offsets(ra_now, dec_now, ra_now[target], dec_now[target])
    is_target = np.arange(len(xi)) == (-1 if target is None else target)
    keep = np.hypot(xi, eta) <= radius_arcmin * 60.0
    return {
        "xi": xi[keep],
        "eta": eta[keep],
        "g_mag": stars["g_mag"][keep],
        "bp_mag": stars["bp_mag"][keep],
        "bp_rp": stars["bp_rp"][keep],
        "is_target": is_target[keep],
    }


def _match_target(stars, ra0, dec0, epoch):
    """Index in ``stars`` of the target at (ra0, dec0), or None.

    The target is the brightest star whose proper-motion track passes within
    ``TARGET_MATCH_RADIUS_ARCSEC`` of the coordinate between
    ``TARGET_TRACK_START_JYEAR`` and ``epoch``.
    """
    # Offsets move linearly in time, p(t) = p_2016 + v (t - 2016), so the
    # closest approach is at t* = 2016 - (p_2016 . v) / |v|^2, clamped to the
    # span searched.
    xi_gaia, eta_gaia = gnomonic_offsets(stars["ra"], stars["dec"], ra0, dec0)
    speed_xi, speed_eta = stars["pmra"] / 1000.0, stars["pmdec"] / 1000.0  # "/yr
    speed_sq = speed_xi**2 + speed_eta**2
    with np.errstate(invalid="ignore", divide="ignore"):
        closest = -(xi_gaia * speed_xi + eta_gaia * speed_eta) / speed_sq
    closest = np.clip(
        np.nan_to_num(closest),
        TARGET_TRACK_START_JYEAR - GAIA_EPOCH_JYEAR,
        max(epoch, GAIA_EPOCH_JYEAR) - GAIA_EPOCH_JYEAR,
    )
    miss = np.hypot(xi_gaia + speed_xi * closest, eta_gaia + speed_eta * closest)
    near = np.flatnonzero(miss < TARGET_MATCH_RADIUS_ARCSEC)
    if near.size == 0:
        return None
    return near[np.argmin(stars["g_mag"][near])]


def _read_cone(ra0, dec0, radius_deg, epoch):
    """Gaia stars in a cone, with J estimates and positions at ``epoch``.

    Every star is moved along its Gaia proper motion (linear, adequate for
    arcseconds over decades against 1.19 arcsec pixels) from J2016 to
    ``epoch`` (Julian years).

    Returns
    -------
    dict of np.ndarray
        source_id, ra, dec (J2016, deg), pmra, pmdec (mas/yr, NaN as 0),
        g_mag, bp_mag, bp_rp, jmag, and ra_now, dec_now (deg at ``epoch``).

    Raises
    ------
    FileNotFoundError
        If the gaiaoffline database is not installed.
    """
    _check_database()
    with Gaia(photometry_output="mag") as gaia:
        stars = gaia.conesearch(ra=ra0, dec=dec0, radius=radius_deg)
    ra = stars["ra"].to_numpy(dtype=float)
    dec = stars["dec"].to_numpy(dtype=float)
    pmra = np.nan_to_num(stars["pmra"].to_numpy(dtype=float))  # includes cos(dec)
    pmdec = np.nan_to_num(stars["pmdec"].to_numpy(dtype=float))
    g_mag = stars["phot_g_mean_mag"].to_numpy(dtype=float)
    bp_mag = stars["phot_bp_mean_mag"].to_numpy(dtype=float)
    bp_rp = bp_mag - stars["phot_rp_mean_mag"].to_numpy(dtype=float)
    years = epoch - GAIA_EPOCH_JYEAR
    return {
        "source_id": stars["source_id"].to_numpy(),
        "ra": ra,
        "dec": dec,
        "pmra": pmra,
        "pmdec": pmdec,
        "g_mag": g_mag,
        "bp_mag": bp_mag,
        "bp_rp": bp_rp,
        "jmag": estimate_jmag(g_mag, bp_rp),
        "ra_now": ra + pmra * years / 3.6e6 / np.cos(np.deg2rad(dec)),
        "dec_now": dec + pmdec * years / 3.6e6,
    }
