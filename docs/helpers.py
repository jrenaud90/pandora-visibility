"""
Helper functions for ``target_visibility.ipynb``.

Three groups of functions live here:

* target handling — turning a ``targets_list`` entry into a name, a sky
  coordinate and (when a planet letter was given) a transit ephemeris
  fetched from the NASA Exoplanet Archive;
* time grids and run statistics;
* the small matplotlib helpers used to draw the timeline plots.
"""

import io
import re
import urllib.parse
import urllib.request
import warnings

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.coordinates.name_resolve import sesame_database
from astropy.time import Time, TimeDelta

from pandoravisibility import find_continuous_periods

__all__ = [
    "resolve_target",
    "target_label",
    "split_planet_name",
    "query_exoplanet_archive",
    "planet_ephemeris",
    "prepare_target",
    "prepare_targets",
    "predict_events",
    "event_segments",
    "event_visibility_stats",
    "build_time_grid",
    "run_stats",
    "mask_segments",
    "draw_row",
    "draw_split_row",
    "draw_event_box",
    "format_date_axis",
    "label_rows",
    "BAR_HEIGHT",
    "EVENT_HEIGHT",
    "TRANSIT_COLOR",
    "ECLIPSE_COLOR",
]

# Resolve target names against SIMBAD specifically, not the default mirror list.
sesame_database.set("simbad")

#: NASA Exoplanet Archive TAP endpoint.
ARCHIVE_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

#: Planet parameters pulled from the archive's composite-parameter table.
ARCHIVE_COLUMNS = (
    "pl_name",
    "hostname",
    "pl_letter",
    "ra",
    "dec",
    "tran_flag",
    "pl_tranmid",
    "pl_orbper",
    "pl_trandur",
    "pl_orbeccen",
    "pl_orblper",
    "pl_imppar",
    "pl_ratror",
    "pl_ratdor",
)

#: Colours for the event overlays drawn behind the visibility bars.
TRANSIT_COLOR = "tab:blue"
ECLIPSE_COLOR = "tab:purple"

_resolved_names = {}
_archive_cache = {}

# A trailing lowercase letter is a planet designation ("TRAPPIST-1 b").  It
# must follow a space or a digit so that star names ending in a lowercase
# letter ("Sirius", "alf Cen") are left alone.  Uppercase trailing letters
# are stellar components of a binary and stay part of the host name, which
# is why "WASP-160 B" is a star but "WASP-160 B b" is a planet orbiting it.
_PLANET_NAME_RE = re.compile(r"^(?P<host>.*(?:\d|\s))\s?(?P<letter>[b-z])$")


def split_planet_name(name):
    """
    Split a target name into a host star and a planet letter.

    Parameters
    ----------
    name : str
        A target name, e.g. ``"WASP-107"``, ``"WASP-107 c"`` or
        ``"WASP-160 B b"``.

    Returns
    -------
    host : str
        The host star name, with any binary component letter kept.
    letter : str or None
        The lowercase planet letter, or None when *name* is just a star.

    Examples
    --------
    >>> split_planet_name("TRAPPIST-1 b")
    ('TRAPPIST-1', 'b')
    >>> split_planet_name("WASP-160 B b")
    ('WASP-160 B', 'b')
    >>> split_planet_name("WASP-160 B")
    ('WASP-160 B', None)
    >>> split_planet_name("Sirius")
    ('Sirius', None)
    """
    match = _PLANET_NAME_RE.match(name.strip())
    if match is None:
        return name.strip(), None
    return match.group("host").strip(), match.group("letter")


def _query_archive(where_clause):
    """
    Run one ADQL query against the archive's composite-parameter table.

    Parameters
    ----------
    where_clause : str
        The body of the ADQL ``where`` clause.

    Returns
    -------
    pandas.DataFrame
        Matching rows, with :data:`ARCHIVE_COLUMNS` as columns.
    """
    query = (
        f"select {', '.join(ARCHIVE_COLUMNS)} from pscomppars " f"where {where_clause}"
    )
    url = f"{ARCHIVE_TAP_URL}?" + urllib.parse.urlencode(
        {"query": query, "format": "csv"}
    )
    with urllib.request.urlopen(url, timeout=120) as response:
        return pd.read_csv(io.StringIO(response.read().decode("utf-8")))


def _archive_value(row, column, default=np.nan):
    """Return one archive column as a float, mapping missing entries to *default*."""
    value = row.get(column, np.nan)
    if value is None or pd.isna(value):
        return default
    return float(value)


def query_exoplanet_archive(planet_name):
    """
    Look one planet up in the NASA Exoplanet Archive.

    Matching ignores spaces and case, so ``"HD209458 b"`` finds the archive's
    ``"HD 209458 b"``.

    Parameters
    ----------
    planet_name : str
        Full planet name, host plus letter.

    Returns
    -------
    dict or None
        Archive parameters, or None when the planet is not in the archive.
        The dict holds ``pl_name``, ``hostname``, ``transits`` and the
        ephemeris entries used by :func:`planet_ephemeris`.
    """
    key = planet_name.upper().replace(" ", "")
    if key in _archive_cache:
        return _archive_cache[key]

    quoted = key.replace("'", "''")
    table = _query_archive(f"upper(replace(pl_name,' ','')) = '{quoted}'")
    if len(table) == 0:
        _archive_cache[key] = None
        return None

    row = table.iloc[0]
    record = {
        "pl_name": row["pl_name"],
        "hostname": row["hostname"],
        "letter": row["pl_letter"],
        "transits": bool(row["tran_flag"]),
        "t0_bjd": _archive_value(row, "pl_tranmid"),
        "period_days": _archive_value(row, "pl_orbper"),
        "duration_hours": _archive_value(row, "pl_trandur"),
        "eccentricity": _archive_value(row, "pl_orbeccen", default=0.0),
        "omega_deg": _archive_value(row, "pl_orblper", default=90.0),
        "impact": _archive_value(row, "pl_imppar"),
        "radius_ratio": _archive_value(row, "pl_ratror"),
        "a_over_rstar": _archive_value(row, "pl_ratdor"),
    }
    _archive_cache[key] = record
    return record


def _mean_anomaly(true_anomaly, eccentricity):
    """Mean anomaly at a given true anomaly, via the eccentric anomaly."""
    eccentric_anomaly = 2.0 * np.arctan2(
        np.sqrt(1.0 - eccentricity) * np.sin(0.5 * true_anomaly),
        np.sqrt(1.0 + eccentricity) * np.cos(0.5 * true_anomaly),
    )
    return eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly)


def _conjunction_offset(period, eccentricity, omega):
    """
    Time from mid-transit to mid-eclipse.

    Inferior conjunction happens at true anomaly ``pi/2 - w`` and superior
    conjunction at ``-pi/2 - w`` (Winn 2010).  Converting both to mean
    anomaly gives the separation exactly, which matters for eccentric
    orbits where the usual first-order expansion is well off — HAT-P-2 b
    lands about two hours from its true eclipse time under that expansion.

    Parameters
    ----------
    period : float
        Orbital period, in days.
    eccentricity : float
        Orbital eccentricity.
    omega : float
        Argument of periastron, in radians.

    Returns
    -------
    float
        Offset in days, always within one period of the transit.
    """
    if eccentricity <= 0.0:
        return 0.5 * period
    separation = _mean_anomaly(-0.5 * np.pi - omega, eccentricity) - _mean_anomaly(
        0.5 * np.pi - omega, eccentricity
    )
    return period * (separation % (2.0 * np.pi)) / (2.0 * np.pi)


def planet_ephemeris(record):
    """
    Turn archive parameters into transit and secondary-eclipse ephemerides.

    The eclipse time comes from :func:`_conjunction_offset`, and its
    duration and impact parameter are scaled by the usual factor
    ``(1 + e sin w) / (1 - e sin w)`` (Winn 2010).  For a circular orbit
    this is simply half a period later with the same duration.

    Parameters
    ----------
    record : dict
        Output of :func:`query_exoplanet_archive`.

    Returns
    -------
    dict or None
        ``transit`` and ``eclipse`` sub-dicts, each with ``t0_bjd``,
        ``period_days`` and ``duration_hours``; ``eclipse`` is None when
        the planet's secondary eclipse does not occur.  Returns None when
        the archive is missing the ephemeris this needs.
    """
    period = record["period_days"]
    t0 = record["t0_bjd"]
    if not np.isfinite(period) or not np.isfinite(t0):
        return None

    duration = record["duration_hours"]
    impact = record["impact"]
    if not np.isfinite(duration):
        # Fall back on a central-transit estimate, T = P / (pi a/R*).
        a_over_rstar = record["a_over_rstar"]
        if not np.isfinite(a_over_rstar) or a_over_rstar <= 0:
            return None
        shape = np.sqrt(max(1.0 - (impact if np.isfinite(impact) else 0.0) ** 2, 0.0))
        duration = 24.0 * period * shape / (np.pi * a_over_rstar)

    eccentricity = record["eccentricity"]
    omega = np.deg2rad(record["omega_deg"])
    if not np.isfinite(eccentricity):
        eccentricity = 0.0

    # Ratio of eclipse to transit duration, and of the two impact parameters.
    stretch = (1.0 + eccentricity * np.sin(omega)) / (
        1.0 - eccentricity * np.sin(omega)
    )
    offset = _conjunction_offset(period, eccentricity, omega)

    # The secondary eclipse happens unless eccentricity swings the planet
    # far enough out of the way at superior conjunction.
    radius_ratio = record["radius_ratio"]
    grazing_limit = 1.0 + (radius_ratio if np.isfinite(radius_ratio) else 0.0)
    eclipse_occurs = not np.isfinite(impact) or abs(impact * stretch) < grazing_limit

    ephemeris = {
        "transit": {
            "t0_bjd": t0,
            "period_days": period,
            "duration_hours": duration,
        },
        "eclipse": None,
    }
    if eclipse_occurs:
        ephemeris["eclipse"] = {
            "t0_bjd": t0 + offset,
            "period_days": period,
            "duration_hours": duration * stretch,
        }
    return ephemeris


def _bjd_tdb_to_utc(jd_tdb, coord):
    """
    Convert barycentric BJD_TDB event times to geocentric UTC.

    Ephemerides are published in BJD_TDB, which leads geocentric time by up
    to about 8 minutes depending on where Earth sits in its orbit.  That is
    a noticeable fraction of a short transit, so it is worth removing.  The
    spacecraft's own offset from the geocentre is under 25 ms and ignored.

    Parameters
    ----------
    jd_tdb : float or ndarray
        Barycentric Julian date(s), TDB.
    coord : astropy.coordinates.SkyCoord
        Target coordinate, setting the light-travel direction.

    Returns
    -------
    astropy.time.Time
        The same instants as geocentric UTC.
    """
    geocentre = EarthLocation.from_geocentric(0 * u.m, 0 * u.m, 0 * u.m)
    barycentric = Time(jd_tdb, format="jd", scale="tdb", location=geocentre)
    travel = barycentric.light_travel_time(coord, kind="barycentric")
    return (barycentric - travel).utc


def predict_events(ephemeris, coord, times):
    """
    Predict the events of one ephemeris that overlap a time grid.

    Parameters
    ----------
    ephemeris : dict
        One of the sub-dicts from :func:`planet_ephemeris`, holding
        ``t0_bjd``, ``period_days`` and ``duration_hours``.
    coord : astropy.coordinates.SkyCoord
        Target coordinate, used for the barycentric time correction.
    times : astropy.time.Time
        The observation time grid.

    Returns
    -------
    list of dict
        One entry per event, each with ``mid``, ``start`` and ``stop``
        times, a boolean ``mask`` over *times*, and ``clipped`` marking
        events that run past the edge of the grid.
    """
    period = ephemeris["period_days"]
    half_duration = 0.5 * ephemeris["duration_hours"] / 24.0

    # Epoch numbers whose events can touch the grid, with one spare on each end.
    first = (
        int(np.floor((times[0].jd - ephemeris["t0_bjd"] - half_duration) / period)) - 1
    )
    last = (
        int(np.ceil((times[-1].jd - ephemeris["t0_bjd"] + half_duration) / period)) + 1
    )
    epochs = np.arange(first, last + 1)
    if epochs.size == 0:
        return []

    mid_times = _bjd_tdb_to_utc(ephemeris["t0_bjd"] + epochs * period, coord)
    grid_jd = times.jd

    events = []
    for epoch, mid in zip(epochs, mid_times):
        start = mid - half_duration * u.day
        stop = mid + half_duration * u.day
        if stop.jd < grid_jd[0] or start.jd > grid_jd[-1]:
            continue
        events.append(
            {
                "epoch": int(epoch),
                "mid": mid,
                "start": start,
                "stop": stop,
                "mask": (grid_jd >= start.jd) & (grid_jd <= stop.jd),
                "clipped": start.jd < grid_jd[0] or stop.jd > grid_jd[-1],
            }
        )
    return events


def event_visibility_stats(events, visible):
    """
    Fraction of sampled event time for which the target is visible.

    Parameters
    ----------
    events : list of dict
        Events from :func:`predict_events`.
    visible : array_like of bool
        Visibility flags on the same time grid.

    Returns
    -------
    dict
        ``n_events``, ``n_full`` (events fully inside the grid),
        ``visible_fraction`` over all sampled event steps, and
        ``fractions``, the per-event fractions in event order.
    """
    visible = np.asarray(visible, dtype=bool)
    fractions = []
    covered = 0
    seen = 0
    for event in events:
        steps = int(event["mask"].sum())
        if steps == 0:
            fractions.append(np.nan)
            continue
        hits = int((event["mask"] & visible).sum())
        fractions.append(hits / steps)
        covered += hits
        seen += steps
    return {
        "n_events": len(events),
        "n_full": sum(1 for event in events if not event["clipped"]),
        "visible_fraction": covered / seen if seen else np.nan,
        "fractions": fractions,
    }


def resolve_target(target):
    """
    Turn one ``targets_list`` entry into a name and a sky coordinate.

    Parameters
    ----------
    target : str or tuple
        One of: a name to look up in SIMBAD (``"WASP-50"``), a planet name
        ending in a lowercase letter (``"WASP-107 b"``, whose host star is
        looked up), a pair of decimal degrees (``(258.831, 4.96068)``), or
        a pair of sexagesimal strings (``("21h44m10.6s", "-05d05m41s")``).

    Returns
    -------
    name : str or None
        The name as supplied, or None when coordinates were given directly.
    coord : astropy.coordinates.SkyCoord
        ICRS coordinate of the target.
    """
    if isinstance(target, str):
        host, _ = split_planet_name(target)
        if host not in _resolved_names:
            _resolved_names[host] = SkyCoord.from_name(host)
        return target, _resolved_names[host]

    right_ascension, declination = target
    if isinstance(right_ascension, str):
        coord = SkyCoord(
            right_ascension,
            declination,
            frame="icrs",
            unit=(u.hourangle, u.deg),
        )
    else:
        coord = SkyCoord(right_ascension, declination, frame="icrs", unit=u.deg)
    return None, coord


def target_label(name, coord):
    """
    Short plot label: the name if there is one, else RA/Dec in degrees.

    Parameters
    ----------
    name : str or None
        Target name, or None.
    coord : astropy.coordinates.SkyCoord
        Target coordinate, used when *name* is None.

    Returns
    -------
    str
        Either the name, or ``"RA, Dec"`` to three decimal places.
    """
    if name is not None:
        return name
    return f"{coord.ra.deg:.3f}, {coord.dec.deg:+.3f}"


def prepare_target(target):
    """
    Resolve one target and, for a planet, fetch its transit ephemeris.

    Anything that cannot be looked up raises a warning rather than an
    error: an unresolvable star is dropped, while a planet the archive
    does not know — or knows not to transit — keeps its star and loses
    only the event overlay.

    Parameters
    ----------
    target : str or tuple
        One ``targets_list`` entry.

    Returns
    -------
    dict or None
        ``label``, ``name``, ``coord``, ``planet`` (the archive record or
        None) and ``ephemeris`` (from :func:`planet_ephemeris`, or None).
        None is returned when the star could not be resolved.
    """
    try:
        name, coord = resolve_target(target)
    except Exception as error:
        warnings.warn(
            f"{target!r}: could not resolve the star ({error}); skipping it.",
            stacklevel=2,
        )
        return None

    prepared = {
        "label": target_label(name, coord),
        "name": name,
        "coord": coord,
        "planet": None,
        "ephemeris": None,
    }

    if name is None or split_planet_name(name)[1] is None:
        return prepared

    try:
        record = query_exoplanet_archive(name)
    except Exception as error:
        warnings.warn(
            f"{name}: the NASA Exoplanet Archive query failed ({error}); "
            f"keeping the star without transit predictions.",
            stacklevel=2,
        )
        return prepared

    if record is None:
        warnings.warn(
            f"{name}: no such planet in the NASA Exoplanet Archive; "
            f"keeping the star without transit predictions.",
            stacklevel=2,
        )
        return prepared

    if not record["transits"]:
        warnings.warn(
            f"{record['pl_name']}: not known to transit; "
            f"skipping the transit and eclipse predictions.",
            stacklevel=2,
        )
        return prepared

    ephemeris = planet_ephemeris(record)
    if ephemeris is None:
        warnings.warn(
            f"{record['pl_name']}: transits, but the archive has no usable "
            f"ephemeris (period, mid-transit and duration); "
            f"skipping the transit and eclipse predictions.",
            stacklevel=2,
        )
        return prepared

    prepared["planet"] = record
    prepared["ephemeris"] = ephemeris
    return prepared


def prepare_targets(targets_list):
    """
    Run :func:`prepare_target` over a list, dropping unresolvable stars.

    Parameters
    ----------
    targets_list : sequence
        Entries accepted by :func:`resolve_target`.

    Returns
    -------
    list of dict
        The targets that resolved, in the order given.
    """
    prepared = [prepare_target(target) for target in targets_list]
    return [target for target in prepared if target is not None]


def build_time_grid(start_utc, stop_utc, step_minutes):
    """
    Regular UTC time grid covering [start_utc, stop_utc).

    Parameters
    ----------
    start_utc, stop_utc : str
        ISO UTC timestamps, e.g. ``"2026-05-05T00:00:00"``.
    step_minutes : float
        Spacing between time steps, in minutes.

    Returns
    -------
    astropy.time.Time
        Array of observation times.
    """
    start = Time(start_utc, scale="utc")
    stop = Time(stop_utc, scale="utc")
    if stop <= start:
        raise ValueError("stop_utc must be later than start_utc")
    span_minutes = (stop - start).to_value(u.min)
    return start + TimeDelta(np.arange(0.0, span_minutes, step_minutes) * u.min)


def run_stats(times, mask, step_minutes):
    """
    Count contiguous True runs in *mask* and measure the longest.

    Parameters
    ----------
    times : astropy.time.Time
        Time grid matching *mask*.
    mask : array_like of bool
        Per-timestep flags, typically a visibility result.
    step_minutes : float
        Grid spacing, used to give the final step of a run its dwell time.

    Returns
    -------
    n_runs : int
        Number of contiguous runs.
    longest_minutes : float
        Duration of the longest run, in minutes.
    """
    periods = find_continuous_periods(times, np.asarray(mask, dtype=bool), 0.0)
    if not periods:
        return 0, 0.0
    longest = max(period["duration_hours"] for period in periods) * 60.0
    return len(periods), longest + step_minutes


def mask_segments(times, mask, step_minutes):
    """
    Contiguous True runs of *mask* as (start, width) pairs for ``broken_barh``.

    Parameters
    ----------
    times : astropy.time.Time
        Time grid matching *mask*.
    mask : array_like of bool
        Per-timestep flags to draw.
    step_minutes : float
        Grid spacing.  Each run is padded by one step so that a single
        isolated timestep still draws a visible bar.

    Returns
    -------
    list of tuple
        ``(start, width)`` pairs in matplotlib date units.
    """
    time_numbers = mdates.date2num(times.to_datetime())
    step_width = step_minutes / (24.0 * 60.0)
    segments = []
    for period in find_continuous_periods(times, np.asarray(mask, dtype=bool), 0.0):
        start = time_numbers[period["start_index"]]
        stop = time_numbers[period["end_index"]]
        segments.append((start, stop - start + step_width))
    return segments


def event_segments(events, min_width_minutes=0.0):
    """
    Event start/stop times as (start, width) pairs for ``broken_barh``.

    Uses the predicted event times rather than the time grid, so a box
    keeps its true width even on a coarse grid.

    Parameters
    ----------
    events : list of dict
        Events from :func:`predict_events`.
    min_width_minutes : float
        Floor on the drawn width, centred on the event.  Pass the grid
        spacing to stop a short transit from collapsing to a hairline on
        a long axis; the visibility bars cannot resolve anything finer
        than one step either.

    Returns
    -------
    list of tuple
        ``(start, width)`` pairs in matplotlib date units.
    """
    min_width = min_width_minutes / (24.0 * 60.0)
    segments = []
    for event in events:
        start = mdates.date2num(event["start"].to_datetime())
        stop = mdates.date2num(event["stop"].to_datetime())
        width = stop - start
        if width < min_width:
            start -= 0.5 * (min_width - width)
            width = min_width
        segments.append((start, width))
    return segments


#: Thickness of one timeline row, in y-axis units.
BAR_HEIGHT = 0.18

#: Thickness of an event box, drawn behind the visibility bars.
EVENT_HEIGHT = 0.62


def draw_row(ax, row, segments, color):
    """
    Draw one row of horizontal bars centred on integer position *row*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    row : int
        Row index, matching the order of the y tick labels.
    segments : list of tuple
        ``(start, width)`` pairs from :func:`mask_segments`.
    color : str
        Any matplotlib colour specification.
    """
    ax.broken_barh(
        segments, (row - BAR_HEIGHT / 2, BAR_HEIGHT), facecolors=color, zorder=3
    )


def draw_split_row(ax, row, times, visible, roll_deg, step_minutes, cmap):
    """
    Draw a two-part row: visibility on top, the roll held underneath.

    The upper half is green wherever *visible*; the lower half colors
    every timestep with a finite *roll_deg* by its roll on the cyclic
    *cmap* (0 to 360 deg), one bar per run of constant roll.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on (rows run top to bottom, as set by
        :func:`label_rows`).
    row : int
        Row index, matching the order of the y tick labels.
    times : astropy.time.Time
        Time grid matching *visible* and *roll_deg*.
    visible : array_like of bool
        Per-timestep visibility.
    roll_deg : array_like of float
        Roll held at each timestep, in degrees; NaN leaves a gap.
    step_minutes : float
        Grid spacing, see :func:`mask_segments`.
    cmap : matplotlib.colors.Colormap
        Cyclic colormap, e.g. ``plt.get_cmap("twilight")``.
    """
    ax.broken_barh(
        mask_segments(times, visible, step_minutes),
        (row - BAR_HEIGHT, BAR_HEIGHT),
        facecolors="tab:green",
        zorder=3,
    )
    wrapped = np.mod(np.asarray(roll_deg, dtype=float), 360.0)
    for roll in np.unique(wrapped[np.isfinite(wrapped)]):
        ax.broken_barh(
            mask_segments(times, wrapped == roll, step_minutes),
            (row, BAR_HEIGHT),
            facecolors=cmap(roll / 360.0),
            zorder=3,
        )


def draw_event_box(ax, row, segments, color, alpha=0.30):
    """
    Draw transparent event boxes behind the bars of one row.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    row : int
        Row index, matching the order of the y tick labels.
    segments : list of tuple
        ``(start, width)`` pairs from :func:`event_segments`.
    color : str
        Any matplotlib colour specification.
    alpha : float
        Box transparency.  The outline is drawn at twice this, which is
        what keeps a short event readable on a long axis.
    """
    ax.broken_barh(
        segments,
        (row - EVENT_HEIGHT / 2, EVENT_HEIGHT),
        facecolors=to_rgba(color, alpha),
        edgecolors=to_rgba(color, min(1.0, 2.0 * alpha)),
        linewidth=0.7,
        zorder=1,
    )


def format_date_axis(ax, times):
    """
    Put a concise UTC scale on the x-axis and clip it to the time grid.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to format.
    times : astropy.time.Time
        Time grid defining the axis limits.
    """
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    time_numbers = mdates.date2num(times.to_datetime())
    ax.set_xlim(time_numbers[0], time_numbers[-1])
    ax.set_xlabel("UTC")


def label_rows(ax, labels):
    """
    Label one horizontal bar row per entry in *labels*, top to bottom.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes holding the bar rows.
    labels : sequence of str
        Row labels, in the order the rows were drawn.
    """
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.grid(axis="x", alpha=0.3)
