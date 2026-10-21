"""Field stars around a target from the offline Gaia DR3 catalog.

The NIRDA trace contamination model (``trace.py``) needs every star near a
target that is bright enough to matter: its offset on the sky and its J-band
flux relative to the target. Both come from a local copy of Gaia DR3 managed
by ``gaiaoffline``, so no network access is needed at run time. The copy must
be downloaded once; see the gaiaoffline README.

J magnitudes are estimated from Gaia photometry alone (G and BP - RP), so a
star missing from Gaia, or fainter than the database's G limit (G < 16 by
default), is not in the field.
"""

import os
import warnings

import gaiaoffline
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from gaiaoffline import Gaia

__all__ = ["query_field", "estimate_jmag", "gnomonic_offsets"]

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
    _check_database()
    ra0 = float(target_coord.icrs.ra.deg)
    dec0 = float(target_coord.icrs.dec.deg)
    search_radius_deg = FIELD_RADIUS_ARCMIN / 60.0 + TARGET_MATCH_RADIUS_ARCSEC / 3600.0
    with Gaia(photometry_output="mag") as gaia:
        stars = gaia.conesearch(ra=ra0, dec=dec0, radius=search_radius_deg)

    ra = stars["ra"].to_numpy(dtype=float)
    dec = stars["dec"].to_numpy(dtype=float)
    g_mag = stars["phot_g_mean_mag"].to_numpy(dtype=float)
    bp_rp = (stars["phot_bp_mean_mag"] - stars["phot_rp_mean_mag"]).to_numpy(
        dtype=float
    )
    pmra = np.nan_to_num(stars["pmra"].to_numpy(dtype=float))  # includes cos(dec)
    pmdec = np.nan_to_num(stars["pmdec"].to_numpy(dtype=float))
    epoch = float(np.mean(np.atleast_1d(time.jyear)))

    # Identify the target as the brightest star whose track passes near the
    # coordinate. Offsets move linearly in time, p(t) = p_2016 + v (t - 2016),
    # so the closest approach is at t* = 2016 - (p_2016 . v) / |v|^2, clamped
    # to the span searched.
    xi_gaia, eta_gaia = gnomonic_offsets(ra, dec, ra0, dec0)
    speed_xi, speed_eta = pmra / 1000.0, pmdec / 1000.0  # arcsec per year
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
        warnings.warn(
            f"No Gaia star passes within {TARGET_MATCH_RADIUS_ARCSEC:g} arcsec "
            f"of RA {ra0:.5f}, Dec {dec0:.5f} between "
            f"J{TARGET_TRACK_START_JYEAR:g} and the observation; the NIRDA "
            "trace check is skipped for this target.",
            stacklevel=2,
        )
        return None
    target = near[np.argmin(g_mag[near])]

    # Carry every star along its proper motion to the observation epoch
    years = epoch - GAIA_EPOCH_JYEAR
    dec_now = dec + pmdec * years / 3.6e6
    ra_now = ra + pmra * years / 3.6e6 / np.cos(np.deg2rad(dec))
    xi, eta = gnomonic_offsets(ra_now, dec_now, ra_now[target], dec_now[target])

    jmag = estimate_jmag(g_mag, bp_rp)
    keep = (
        (np.arange(len(stars)) != target)
        & (jmag < FIELD_J_LIMIT)
        & (np.hypot(xi, eta) <= FIELD_RADIUS_ARCMIN * 60.0)
    )
    return {
        "target_source_id": int(stars["source_id"].iloc[target]),
        "target_jmag": float(jmag[target]),
        "xi": xi[keep],
        "eta": eta[keep],
        "jmag": jmag[keep],
        "flux_ratio": 10.0 ** (-0.4 * (jmag[keep] - jmag[target])),
    }
