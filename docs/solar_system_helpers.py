"""
Helper functions for ``solar_system_visibility.ipynb``.

Four groups of functions live here:

* resolving planet and moon names to JPL Horizons ids, and each body's system
  (its primary and the primary's major moons);
* JPL Horizons observer ephemerides, fetched over the Horizons API with the
  standard library, including Horizons' moon visibility codes (occulted,
  eclipsed, transiting);
* sampling extended bodies (an oblate planet disk and Saturn's main rings) as
  point sources, so ``pandoravisibility.trace`` can draw and score them;
* the NIRDA field around a target at one time step: the other bodies of its
  system plus Gaia field stars;
* the same for the VISDA (visible camera) field, for ``pandoravisibility.visda``.

The ephemerides are geocentric. Pandora's ~7,000 km offset from the Earth's
center moves every body of a planetary system by the same ~1 arcsec, so the
relative geometry (which decides occultations and trace overlaps) is unchanged.
"""

import json
import urllib.parse
import urllib.request

import numpy as np
from astropy import units as u
from astropy.coordinates import GCRS, SkyCoord

from pandoravisibility import star_field, trace

__all__ = [
    "resolve_body",
    "system_ids",
    "horizons_ephemeris",
    "system_ephemerides",
    "mean_coord",
    "is_clear",
    "pandora_centric",
    "body_samples",
    "system_field",
    "PLANETS",
    "MAJOR_MOONS",
]

#: JPL Horizons API endpoint.
HORIZONS_API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"

#: Planets by Horizons id of the body itself (x99), not the system barycenter.
PLANETS = {"Mars": 499, "Jupiter": 599, "Saturn": 699, "Uranus": 799, "Neptune": 899}

#: Major moons per planet, by Horizons id.
MAJOR_MOONS = {
    499: {"Phobos": 401, "Deimos": 402},
    599: {"Io": 501, "Europa": 502, "Ganymede": 503, "Callisto": 504},
    699: {
        "Mimas": 601,
        "Enceladus": 602,
        "Tethys": 603,
        "Dione": 604,
        "Rhea": 605,
        "Titan": 606,
        "Hyperion": 607,
        "Iapetus": 608,
    },
    799: {"Ariel": 701, "Umbriel": 702, "Titania": 703, "Oberon": 704, "Miranda": 705},
    899: {"Triton": 801},
}

#: Polar over equatorial radius, for the shape of a planet's disk.
POLAR_RATIO = {499: 0.995, 599: 0.935, 699: 0.902, 799: 0.977, 899: 0.983}

#: Main rings drawn, as (inner, outer) radius in planet equatorial radii:
#: Saturn's B ring inner edge (92,000 km) to A ring outer edge (136,775 km),
#: Cassini division not resolved. Equatorial radius 60,268 km.
RINGS = {699: (92000.0 / 60268.0, 136775.0 / 60268.0)}

#: Ring surface brightness relative to the planet's disk, per unit projected
#: area. A placeholder (Saturn's B ring and disk have similar albedos); the
#: planet's total flux is fixed by its Horizons magnitude either way.
RING_RELATIVE_BRIGHTNESS = 1.0

#: The Sun's V - J. Planets and moons reflect sunlight, so their J is taken as
#: V - SOLAR_V_MINUS_J. Methane bands make Saturn, Titan, Uranus and Neptune
#: fainter in the near infrared than this suggests.
SOLAR_V_MINUS_J = 1.15

#: The Sun's Gaia BP - RP and G - V (the Riello et al. 2021 G - V relation at
#: that color), for drawing sunlit bodies on VISDA.
SOLAR_BP_RP = 0.82
SOLAR_G_MINUS_V = -0.15

#: Spacing of the point samples of an extended body, in arcsec (half a pixel).
SAMPLE_SPACING = 0.6

#: Bodies smaller than this (arcsec) are drawn as single points.
EXTENDED_MIN_DIAMETER = 2.0 * trace.NIRDA_PIXEL_SCALE

#: Horizons satellite visibility codes. "Clear" is the only state counted as
#: observable; "-" marks a primary body.
VISIBILITY_CODES = {
    "*": "free and clear",
    "t": "transiting the planet's disk",
    "O": "occulted by the planet",
    "p": "partial umbral eclipse",
    "P": "occulted, partial umbral eclipse",
    "u": "total umbral eclipse",
    "U": "occulted, total umbral eclipse",
    "-": "primary body",
}

#: Codes whose light is hidden (behind the planet or in its full shadow).
HIDDEN_CODES = ("O", "P", "u", "U")

# Horizons observer quantities: 1 astrometric RA/Dec, 9 visual magnitude,
# 12 satellite separation and visibility code, 13 angular diameter,
# 14 sub-observer longitude/latitude, 17 north pole position angle,
# 20 observer range.
_QUANTITIES = "1,9,12,13,14,17,20"
_COLUMNS = {
    "R.A._(ICRF)": "ra",
    "DEC_(ICRF)": "dec",
    "APmag": "vmag",
    "vis.": "code",
    "Ang-diam": "diameter",
    "ObsSub-LAT": "sub_lat",
    "NP.ang": "pole_pa",
    "delta": "delta",
}

#: Kilometers per astronomical unit (IAU 2012).
_AU_KM = 149597870.7


def resolve_body(name):
    """
    Horizons id and display name of a planet or major moon.

    Parameters
    ----------
    name : str or int
        A name from ``PLANETS`` or ``MAJOR_MOONS`` (any case), or a Horizons
        id such as 606.

    Returns
    -------
    body_id : int
    label : str
    """
    names = dict(PLANETS)
    for moons in MAJOR_MOONS.values():
        names.update(moons)
    if isinstance(name, (int, np.integer)):
        for label, body_id in names.items():
            if body_id == name:
                return int(body_id), label
        raise ValueError(f"Horizons id {name} is not a planet or major moon here")
    for label, body_id in names.items():
        if label.lower() == str(name).strip().lower():
            return body_id, label
    raise ValueError(
        f"{name!r} is not a known planet or major moon; choose from " + ", ".join(names)
    )


def system_ids(body_id):
    """Ids of a body's system: its primary first, then the primary's moons."""
    primary = body_id if body_id % 100 == 99 else (body_id // 100) * 100 + 99
    return [primary] + list(MAJOR_MOONS.get(primary, {}).values())


def horizons_ephemeris(body_id, start_utc, stop_utc, step_minutes):
    """
    Geocentric observer ephemeris of one body from JPL Horizons.

    Parameters
    ----------
    body_id : int
        Horizons id.
    start_utc, stop_utc : str
        ISO UTC limits, e.g. ``"2026-10-22T00:00:00"``.
    step_minutes : float
        Step between rows, in whole minutes.

    Returns
    -------
    dict of np.ndarray
        jd (UT), ra and dec (astrometric ICRF, deg), vmag, code (Horizons
        visibility code, see ``VISIBILITY_CODES``), diameter (arcsec),
        sub_lat (sub-observer latitude, deg; the ring opening angle for a
        ringed planet), pole_pa (north pole position angle, deg east of
        north) and delta (distance from the Earth's center, au). Values
        Horizons does not provide are NaN.
    """
    params = {
        "format": "json",
        "COMMAND": f"'{body_id}'",
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "OBSERVER",
        "CENTER": "'500@399'",
        "START_TIME": f"'{start_utc.replace('T', ' ')}'",
        "STOP_TIME": f"'{stop_utc.replace('T', ' ')}'",
        "STEP_SIZE": f"'{int(round(step_minutes))} m'",
        "QUANTITIES": f"'{_QUANTITIES}'",
        "ANG_FORMAT": "DEG",
        "CSV_FORMAT": "YES",
        "CAL_FORMAT": "JD",
    }
    url = f"{HORIZONS_API_URL}?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as response:
        text = json.load(response)["result"]
    if "$$SOE" not in text:
        raise RuntimeError(f"Horizons returned no ephemeris for {body_id}:\n{text}")

    header_text, table = text.split("$$SOE")
    table = table.split("$$EOE")[0]
    header = [name.strip() for name in header_text.strip().splitlines()[-2].split(",")]
    rows = [
        [value.strip() for value in line.split(",")]
        for line in table.strip().splitlines()
    ]
    columns = {name: index for index, name in enumerate(header) if name}
    result = {"jd": np.array([float(row[0]) for row in rows])}
    for name, key in _COLUMNS.items():
        values = [row[columns[name]] for row in rows]
        if key == "code":
            result[key] = np.array(values)
        else:
            result[key] = np.array(
                [float(v) if v not in ("", "n.a.") else np.nan for v in values]
            )
    return result


def system_ephemerides(target_ids, start_utc, stop_utc, step_minutes, times):
    """
    Ephemerides of every body in the targets' systems, on the notebook grid.

    Each body is fetched once from Horizons and sampled at ``times``:
    numbers are interpolated, visibility codes taken from the nearest row.

    Returns
    -------
    dict
        Horizons id -> ephemeris dict (see ``horizons_ephemeris``), every
        array with one entry per time in ``times``.
    """
    ephemerides = {}
    for target_id in target_ids:
        for body_id in system_ids(target_id):
            if body_id in ephemerides:
                continue
            raw = horizons_ephemeris(body_id, start_utc, stop_utc, step_minutes)
            after = np.clip(np.searchsorted(raw["jd"], times.jd), 1, len(raw["jd"]) - 1)
            before_closer = (times.jd - raw["jd"][after - 1]) < (
                raw["jd"][after] - times.jd
            )
            nearest = np.where(before_closer, after - 1, after)
            sampled = {"code": raw["code"][nearest]}
            # RA through its unit vector, so 0/360 deg wraps interpolate cleanly
            for key in ("dec", "vmag", "diameter", "sub_lat", "pole_pa", "delta"):
                sampled[key] = np.interp(times.jd, raw["jd"], raw[key])
            ra = np.deg2rad(raw["ra"])
            sampled["ra"] = (
                np.rad2deg(
                    np.arctan2(
                        np.interp(times.jd, raw["jd"], np.sin(ra)),
                        np.interp(times.jd, raw["jd"], np.cos(ra)),
                    )
                )
                % 360.0
            )
            ephemerides[body_id] = sampled
    return ephemerides


def mean_coord(ephemeris, indices):
    """Mean sky position of a body over some time steps, as a SkyCoord."""
    ra = np.deg2rad(ephemeris["ra"][indices])
    dec = np.deg2rad(ephemeris["dec"][indices])
    # Average the unit vectors, which handles the 0/360 deg RA wrap
    x = np.mean(np.cos(dec) * np.cos(ra))
    y = np.mean(np.cos(dec) * np.sin(ra))
    z = np.mean(np.sin(dec))
    return SkyCoord(
        np.rad2deg(np.arctan2(y, x)) % 360.0,
        np.rad2deg(np.arctan2(z, np.hypot(x, y))),
        unit="deg",
    )


def pandora_centric(ephemeris, visibility, times):
    """
    A body's position as seen from Pandora rather than the Earth's center.

    Horizons positions here are geocentric; Pandora sits ~7,000 km from the
    Earth's center, which moves a body by ~7,000 km / its distance: about
    1 arcsec for Saturn, up to ~20 arcsec for Mars near opposition. The
    spacecraft position comes from the TLE through ``Visibility.get_state``.
    Like Horizons' positions these are astrometric ICRF (light time
    included, no aberration), the frame star catalog positions are given in.

    Parameters
    ----------
    ephemeris : dict
        The body's sampled ephemeris (see ``system_ephemerides``).
    visibility : pandoravisibility.Visibility
        Supplies the spacecraft orbit.
    times : astropy.time.Time
        The notebook time grid.

    Returns
    -------
    ra, dec : np.ndarray
        Pandora-centric position in degrees, one per time.
    """
    spacecraft = (
        visibility.get_state(times)
        .transform_to(GCRS(obstime=times))
        .cartesian.xyz.to_value(u.km)
    )
    ra = np.deg2rad(ephemeris["ra"])
    dec = np.deg2rad(ephemeris["dec"])
    distance = ephemeris["delta"] * _AU_KM
    body = distance * np.array(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
    )
    x, y, z = body - spacecraft
    return np.rad2deg(np.arctan2(y, x)) % 360.0, np.rad2deg(
        np.arctan2(z, np.hypot(x, y))
    )


def is_clear(body_id, ephemeris):
    """Per-step flag: True where a moon is free and clear (planets always)."""
    if body_id % 100 == 99:
        return np.ones(len(ephemeris["code"]), dtype=bool)
    return ephemeris["code"] == "*"


def body_samples(body_id, ephemeris, index):
    """
    Point samples of a body's light on the sky, for ``pandoravisibility.trace``.

    A body smaller than ``EXTENDED_MIN_DIAMETER`` is one point. A larger one
    is a uniform oblate disk (no limb darkening) and, for a planet in
    ``RINGS``, a ring annulus seen at the sub-observer latitude, sampled every
    ``SAMPLE_SPACING``. Ring shadows and the overlap of disk and rings are
    ignored.

    Parameters
    ----------
    body_id : int
        Horizons id.
    ephemeris : dict
        The body's sampled ephemeris (see ``system_ephemerides``).
    index : int
        Time step.

    Returns
    -------
    dxi, deta, weight : np.ndarray
        Offsets from the body's center in arcsec (east, north) and the share
        of its light at each sample, summing to 1. The first sample is the
        one nearest the center.
    """
    diameter = ephemeris["diameter"][index]
    if not np.isfinite(diameter) or diameter < EXTENDED_MIN_DIAMETER:
        return np.zeros(1), np.zeros(1), np.ones(1)

    radius = diameter / 2.0
    opening = np.deg2rad(np.nan_to_num(ephemeris["sub_lat"][index]))
    pole_pa = np.deg2rad(np.nan_to_num(ephemeris["pole_pa"][index]))
    # Apparent flattening of the disk depends on how the pole is tilted to us
    polar = POLAR_RATIO.get(body_id, 1.0)
    apparent_polar = np.sqrt((polar * np.cos(opening)) ** 2 + np.sin(opening) ** 2)

    # Disk: grid in (equator, pole) coordinates
    grid = np.arange(-radius, radius + SAMPLE_SPACING / 2, SAMPLE_SPACING)
    along_equator, along_pole = (axis.ravel() for axis in np.meshgrid(grid, grid))
    inside = (along_equator / radius) ** 2 + (
        along_pole / (radius * apparent_polar)
    ) ** 2 <= 1.0
    along_equator, along_pole = along_equator[inside], along_pole[inside]
    weight = np.ones(along_equator.size)

    if body_id in RINGS:
        # Rings: polar grid in the ring plane, foreshortened along the pole
        inner, outer = (radius * edge for edge in RINGS[body_id])
        ring_radii = np.arange(inner, outer, SAMPLE_SPACING) + SAMPLE_SPACING / 2
        ring_equator, ring_pole, ring_weight = [], [], []
        for ring_radius in ring_radii:
            n_azimuth = max(int(2 * np.pi * ring_radius / SAMPLE_SPACING), 8)
            azimuth = np.arange(n_azimuth) * 2 * np.pi / n_azimuth
            ring_equator.append(ring_radius * np.cos(azimuth))
            ring_pole.append(ring_radius * np.sin(azimuth) * np.sin(opening))
            # Projected area of each sample relative to a disk sample
            ring_weight.append(
                np.full(
                    n_azimuth,
                    RING_RELATIVE_BRIGHTNESS
                    * 2
                    * np.pi
                    * ring_radius
                    * SAMPLE_SPACING
                    / n_azimuth
                    * abs(np.sin(opening))
                    / SAMPLE_SPACING**2,
                )
            )
        along_equator = np.concatenate([along_equator, *ring_equator])
        along_pole = np.concatenate([along_pole, *ring_pole])
        weight = np.concatenate([weight, *ring_weight])

    # To the sky: the pole points to position angle pole_pa (east of north)
    dxi = along_equator * np.cos(pole_pa) + along_pole * np.sin(pole_pa)
    deta = -along_equator * np.sin(pole_pa) + along_pole * np.cos(pole_pa)
    first = np.argmin(np.hypot(dxi, deta))
    order = np.r_[first, np.delete(np.arange(dxi.size), first)]
    return dxi[order], deta[order], weight[order] / weight.sum()


def system_field(target_id, ephemerides, index, time, labels, include_stars=True):
    """
    NIRDA field around a target at one time step, for ``trace.plot_trace``.

    The target sits at the pivot, extended if it is (``target_shape``). The
    other bodies of its system are neighbors, dropped while hidden (occulted
    or fully eclipsed, see ``HIDDEN_CODES``), and Gaia field stars from
    ``star_field.query_stars`` are added. Brightness ratios use J estimated
    as V - ``SOLAR_V_MINUS_J`` for the bodies.

    Parameters
    ----------
    target_id : int
        Horizons id of the body at the pivot.
    ephemerides : dict
        From ``system_ephemerides``.
    index : int
        Time step.
    time : astropy.time.Time
        That time step, for the stars' proper motion.
    labels : dict
        Horizons id -> display name.
    include_stars : bool
        Add Gaia field stars (needs the gaiaoffline database).

    Returns
    -------
    dict
        ``xi``, ``eta``, ``flux_ratio``, ``jmag`` and ``label`` per entry,
        plus ``target_shape``, in the format ``pandoravisibility.trace``
        takes.
    """
    target = ephemerides[target_id]
    ra0, dec0 = target["ra"][index], target["dec"][index]
    target_jmag = target["vmag"][index] - SOLAR_V_MINUS_J

    xi, eta, flux, jmag, label = [], [], [], [], []
    for body_id in system_ids(target_id):
        body = ephemerides[body_id]
        if body_id == target_id or body["code"][index] in HIDDEN_CODES:
            continue
        center_xi, center_eta = star_field.gnomonic_offsets(
            body["ra"][index], body["dec"][index], ra0, dec0
        )
        dxi, deta, weight = body_samples(body_id, body, index)
        body_jmag = body["vmag"][index] - SOLAR_V_MINUS_J
        xi.append(center_xi + dxi)
        eta.append(center_eta + deta)
        flux.append(10.0 ** (-0.4 * (body_jmag - target_jmag)) * weight)
        jmag.append(np.full(weight.size, body_jmag))
        label.extend([labels[body_id]] + [""] * (weight.size - 1))

    if include_stars:
        stars = star_field.query_stars(SkyCoord(ra0, dec0, unit="deg"), time)
        xi.append(stars["xi"])
        eta.append(stars["eta"])
        flux.append(10.0 ** (-0.4 * (stars["jmag"] - target_jmag)))
        jmag.append(stars["jmag"])
        label.extend(f"J={value:.1f}" for value in stars["jmag"])

    shape_xi, shape_eta, shape_weight = body_samples(target_id, target, index)
    return {
        "xi": np.concatenate(xi) if xi else np.array([]),
        "eta": np.concatenate(eta) if eta else np.array([]),
        "flux_ratio": np.concatenate(flux) if flux else np.array([]),
        "jmag": np.concatenate(jmag) if jmag else np.array([]),
        "label": label,
        "target_shape": {"xi": shape_xi, "eta": shape_eta, "weight": shape_weight},
    }


def system_visible_sources(
    target_id, ephemerides, index, time, labels, radius_arcmin, include_stars=True
):
    """
    VISDA sources around a target at one time step, for ``visda.plot_visible_field``.

    Every body of the target's system that is not hidden (occulted or fully
    eclipsed, see ``HIDDEN_CODES``), the target included, sampled as in
    ``body_samples``, plus Gaia field stars from
    ``star_field.query_visible_field``. Bodies take G = V +
    ``SOLAR_G_MINUS_V`` and BP - RP = ``SOLAR_BP_RP``.

    Parameters
    ----------
    target_id : int
        Horizons id of the body the field is centered on.
    ephemerides : dict
        From ``system_ephemerides``.
    index : int
        Time step.
    time : astropy.time.Time
        That time step, for the stars' proper motion.
    labels : dict
        Horizons id -> display name.
    radius_arcmin : float
        Star cone radius, e.g. from ``visda.field_radius_arcmin``.
    include_stars : bool
        Add Gaia field stars (needs the gaiaoffline database).

    Returns
    -------
    dict
        ``xi``, ``eta``, ``g_mag``, ``bp_mag``, ``bp_rp``, ``weight`` and
        ``label`` per source, in the format ``visda.render_frame`` takes.
        Stars are labeled with their BP when brighter than 12.
    """
    target = ephemerides[target_id]
    ra0, dec0 = target["ra"][index], target["dec"][index]

    xi, eta, g_mag, bp_mag, bp_rp, weight, label = [], [], [], [], [], [], []
    for body_id in system_ids(target_id):
        body = ephemerides[body_id]
        if body_id != target_id and body["code"][index] in HIDDEN_CODES:
            continue
        center_xi, center_eta = star_field.gnomonic_offsets(
            body["ra"][index], body["dec"][index], ra0, dec0
        )
        dxi, deta, body_weight = body_samples(body_id, body, index)
        xi.append(center_xi + dxi)
        eta.append(center_eta + deta)
        g_mag.append(np.full(body_weight.size, body["vmag"][index] + SOLAR_G_MINUS_V))
        bp_mag.append(np.full(body_weight.size, np.nan))
        bp_rp.append(np.full(body_weight.size, SOLAR_BP_RP))
        weight.append(body_weight)
        label.extend([labels[body_id]] + [""] * (body_weight.size - 1))

    if include_stars:
        stars = star_field.query_visible_field(
            SkyCoord(ra0, dec0, unit="deg"), time, radius_arcmin, match_target=False
        )
        xi.append(stars["xi"])
        eta.append(stars["eta"])
        g_mag.append(stars["g_mag"])
        bp_mag.append(stars["bp_mag"])
        bp_rp.append(stars["bp_rp"])
        weight.append(np.ones(stars["xi"].size))
        label.extend(
            f"BP={value:.1f}" if value < 12.0 else "" for value in stars["bp_mag"]
        )

    return {
        "xi": np.concatenate(xi),
        "eta": np.concatenate(eta),
        "g_mag": np.concatenate(g_mag),
        "bp_mag": np.concatenate(bp_mag),
        "bp_rp": np.concatenate(bp_rp),
        "weight": np.concatenate(weight),
        "label": label,
    }
