#!/usr/bin/env python
"""
Plot the keep-outs a ``Visibility`` instance enforces.

Unlike the target-visibility tools, nothing here depends on a target, a
time or an orbit: it draws the *rules* the instance is configured with, so
you can see what a set of keep-out arguments actually means before running
anything against it.

Two things get drawn:

* **Body keep-outs** — the minimum separation each body demands, as a
  function of the angle between the pointing direction and that body.
  Below the limit is forbidden, above it is allowed.
* **Earth limb keep-out** — the minimum angle above the limb, as a function
  of the Earth illumination angle at the limb point being grazed.  A fixed
  limit is a flat line; a day/night pair is a step; ``use_dynamic_earthlimb``
  and ``use_dynamic_earthlimb_st`` are the DPC wedges.

Every number is read back out of the instance — the wedge curves come from
``Visibility._dynamic_earthlimb_min_deg`` and
``Visibility._dynamic_st_earthlimb_min_deg``, the active checks from
``_st_checks_for`` — so this plot cannot drift from the constraint the
engine applies.

Usage (CLI)
-----------
    python scripts/plot_keepouts.py

Usage (notebook / import)
-------------------------
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path("scripts").resolve()))
    from plot_keepouts import plot_keepouts

    vis = Visibility(line1, line2, sun_min=91*u.deg, moon_min=20*u.deg,
                     st_sun_min=50*u.deg, use_dynamic_earthlimb=True,
                     use_dynamic_earthlimb_st=True)
    fig = plot_keepouts(vis)
"""

import pathlib

import numpy as np
from astropy import units as u

from pandoravisibility import Visibility
from pandoravisibility.visibility import _DYN_EARTH_ANGULAR_RADIUS_DEG

__all__ = [
    "body_keepouts",
    "earthlimb_keepouts",
    "keepout_summary",
    "plot_keepouts",
]

#: Forbidden-zone fill for the body keep-out bars.
FORBIDDEN_COLOR = "tab:red"
#: One colour per optic, shared by both panels.
OPTIC_COLORS = {
    "boresight": "tab:blue",
    "ST1": "tab:orange",
    "ST2": "tab:green",
}


def _deg(value):
    """Angle Quantity (or plain number) as a float in degrees."""
    if isinstance(value, u.Quantity):
        return float(value.to(u.deg).value)
    return float(value)


def _tracker_offset_deg(tracker):
    """Angle between a star tracker boresight and the science boresight."""
    vec = np.array(Visibility._get_star_tracker_body_xyz(tracker))
    return float(np.degrees(np.arccos(np.clip(vec[2], -1.0, 1.0))))


# ═══════════════════════════════════════════════════════════════════
# Reading the rules off the instance
# ═══════════════════════════════════════════════════════════════════

def body_keepouts(visibility):
    """Active body keep-outs, as a list of dicts.

    Only constraints that actually bite are returned: a limit of zero is
    switched off, and the star tracker entries are dropped entirely when
    ``st_required`` is 0 or no tracker limit is set.

    Parameters
    ----------
    visibility : Visibility
        The configured instance to read.

    Returns
    -------
    list of dict
        ``optic`` ("boresight", "ST1", "ST2"), ``body``, ``limit_deg``.
    """
    rows = []
    for body, limit in (
        ("Sun", visibility.sun_min),
        ("Moon", visibility.moon_min),
        ("Mars", visibility.mars_min),
        ("Jupiter", visibility.jupiter_min),
    ):
        if _deg(limit) > 0.0:
            rows.append(
                {"optic": "boresight", "body": body, "limit_deg": _deg(limit)}
            )

    if visibility._st_constraint_active:
        for tracker in (1, 2):
            # _st_checks_for is the engine's own list of what is switched
            # on for this tracker, including the per-tracker overrides.
            for name, limit, key in visibility._st_checks_for(tracker):
                if key == "earthlimb_angle":
                    continue  # drawn against illumination instead
                rows.append({
                    "optic": f"ST{tracker}",
                    "body": name.capitalize(),
                    "limit_deg": _deg(limit),
                })
    return rows


def earthlimb_keepouts(visibility, illumination_deg):
    """Earth limb keep-out curves against Earth illumination angle.

    Parameters
    ----------
    visibility : Visibility
        The configured instance to read.
    illumination_deg : ndarray
        Earth illumination angles to evaluate, in degrees.  0 is the
        sub-solar point (brightest limb), 90 the terminator, 180 the
        anti-solar point (fully dark limb).

    Returns
    -------
    list of dict
        One entry per curve: ``optic``, ``label``, ``limit_deg`` (an array
        matching *illumination_deg*), and ``illumination_dependent`` —
        False when the curve is drawn flat only because this x-axis does
        not control it, which happens for a day/night pair in
        ``daynight_mode="subsatellite"``.
    """
    illumination_deg = np.asarray(illumination_deg, dtype=float)
    ones = np.ones_like(illumination_deg)
    curves = []

    # ── Boresight ───────────────────────────────────────────────────
    if visibility.use_dynamic_earthlimb:
        curves.append({
            "optic": "boresight",
            "label": "boresight (DPC wedge)",
            "limit_deg": np.asarray(
                visibility._dynamic_earthlimb_min_deg(illumination_deg),
                dtype=float,
            ),
            "illumination_dependent": True,
        })
    elif (visibility.earthlimb_day_min is not None
          or visibility.earthlimb_night_min is not None):
        day = _deg(visibility.earthlimb_day_min
                   if visibility.earthlimb_day_min is not None
                   else visibility.earthlimb_min)
        night = _deg(visibility.earthlimb_night_min
                     if visibility.earthlimb_night_min is not None
                     else visibility.earthlimb_min)
        if visibility.daynight_mode == "limb":
            # The limb point is sunlit while dot(n, sun) > -sin(twilight),
            # i.e. while the illumination angle is below 90 + margin, so
            # the day/night pair really is a step on this axis.
            switch = 90.0 + _deg(visibility.twilight_margin)
            curves.append({
                "optic": "boresight",
                "label": f"boresight (day/night, step at {switch:.0f}°)",
                "limit_deg": np.where(illumination_deg < switch, day, night),
                "illumination_dependent": True,
            })
        else:
            # Set by the subsatellite point, which this axis says nothing
            # about — draw both levels and say so.
            for level, side in ((day, "day"), (night, "night")):
                curves.append({
                    "optic": "boresight",
                    "label": f"boresight ({side}, subsatellite)",
                    "limit_deg": level * ones,
                    "illumination_dependent": False,
                })
    else:
        curves.append({
            "optic": "boresight",
            "label": "boresight (fixed)",
            "limit_deg": _deg(visibility.earthlimb_min) * ones,
            "illumination_dependent": True,
        })

    # ── Star trackers ───────────────────────────────────────────────
    if visibility._st_constraint_active:
        for tracker in (1, 2):
            active = any(key == "earthlimb_angle"
                         for _, _, key in visibility._st_checks_for(tracker))
            if not active:
                continue
            if visibility.use_dynamic_earthlimb_st:
                curves.append({
                    "optic": f"ST{tracker}",
                    "label": f"ST{tracker} (DPC wedge)",
                    "limit_deg": np.asarray(
                        visibility._dynamic_st_earthlimb_min_deg(
                            illumination_deg),
                        dtype=float,
                    ),
                    "illumination_dependent": True,
                })
            else:
                curves.append({
                    "optic": f"ST{tracker}",
                    "label": f"ST{tracker} (fixed)",
                    "limit_deg": _deg(
                        visibility._st_earthlimb_min_for(tracker)) * ones,
                    "illumination_dependent": True,
                })
    return curves


def keepout_summary(visibility):
    """Human-readable summary of every keep-out the instance enforces."""
    lines = [repr(visibility), ""]

    rows = body_keepouts(visibility)
    if rows:
        lines.append("Body keep-outs (minimum separation):")
        for row in rows:
            lines.append(
                f"  {row['optic']:<10} {row['body']:<8} "
                f">= {row['limit_deg']:6.1f} deg"
            )
    else:
        lines.append("Body keep-outs: none active")

    illumination = np.array([0.0, 90.0, 180.0])
    curves = earthlimb_keepouts(visibility, illumination)
    lines.append("")
    lines.append("Earth limb keep-outs (minimum angle above the limb):")
    lines.append(f"  {'curve':<38} {'bright':>8}{'term.':>8}{'dark':>8}")
    for curve in curves:
        values = np.atleast_1d(curve["limit_deg"])
        if values.size == 1:
            values = np.repeat(values, illumination.size)
        lines.append(
            f"  {curve['label']:<38} "
            + "".join(f"{v:8.1f}" for v in values)
        )
        if not curve["illumination_dependent"]:
            lines.append(
                "      (which level applies is set by the subsatellite "
                "point, not by this angle)"
            )

    if visibility._st_constraint_active:
        need = "both trackers" if visibility.st_required == 2 else "either tracker"
        lines.append("")
        lines.append(f"Star trackers: {need} must pass.")
        for tracker in (1, 2):
            lines.append(
                f"  ST{tracker} boresight sits "
                f"{_tracker_offset_deg(tracker):.1f} deg off the science "
                f"boresight, so its angles are its own, not the target's."
            )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_keepouts(visibility, illumination_step=0.25, figsize=(12.5, 8.0),
                  savepath=None, show=False):
    """
    Draw every keep-out the instance enforces.

    Parameters
    ----------
    visibility : Visibility
        A configured instance.  Nothing is evaluated against a target or a
        time; only the keep-out settings are read.
    illumination_step : float
        Sampling of the Earth illumination axis, in degrees.  Fine enough
        to render the wedge corners sharply.
    figsize : tuple
        Figure size.
    savepath : str, optional
        If given, save the figure here.
    show : bool
        Call ``plt.show()``.  Leave False in notebooks that display
        figures inline.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plot_keepouts(). "
            "Install it with:  pip install pandoravisibility[plotting]"
        ) from None

    bodies = body_keepouts(visibility)
    illumination = np.arange(0.0, 180.0 + illumination_step,
                             illumination_step)
    curves = earthlimb_keepouts(visibility, illumination)

    fig, (ax_body, ax_limb) = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw={"height_ratios": [max(len(bodies), 3), 6], "hspace": 0.38},
    )

    # ── Panel 1: body keep-outs ─────────────────────────────────────
    if bodies:
        labels = [f"{row['optic']} – {row['body']}" for row in bodies]
        for row, y in zip(bodies, range(len(bodies))):
            limit = row["limit_deg"]
            ax_body.barh(y, limit, height=0.55, color=FORBIDDEN_COLOR,
                         alpha=0.55, edgecolor=FORBIDDEN_COLOR)
            ax_body.plot([limit, limit], [y - 0.34, y + 0.34],
                         color=OPTIC_COLORS.get(row["optic"], "0.3"), lw=2.0)
            ax_body.annotate(
                f"{limit:.0f}°", xy=(limit, y), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=8,
            )
        ax_body.set_yticks(range(len(bodies)))
        ax_body.set_yticklabels(labels, fontsize=9)
        ax_body.set_ylim(len(bodies) - 0.5, -0.5)
    else:
        ax_body.text(0.5, 0.5, "no body keep-outs active", ha="center",
                     va="center", transform=ax_body.transAxes, fontsize=11)
        ax_body.set_yticks([])
    ax_body.set_xlim(0, 180)
    ax_body.set_xticks(np.arange(0, 181, 30))
    ax_body.set_xlabel("angle between the optic and the body (deg)")
    ax_body.set_title(
        "Body keep-outs — shaded is forbidden, the bar end is the limit",
        fontsize=11,
    )
    ax_body.grid(axis="x", alpha=0.3)

    # ── Panel 2: Earth limb keep-out vs illumination ────────────────
    approximate = False
    for curve in curves:
        values = np.broadcast_to(
            np.asarray(curve["limit_deg"], dtype=float), illumination.shape)
        style = "-" if curve["illumination_dependent"] else "--"
        if not curve["illumination_dependent"]:
            approximate = True
        # The two trackers usually share a curve exactly, so draw ST2
        # thinner and last: a coincident pair then reads as a thin green
        # line centred on a thick orange one rather than as one line.
        width = 1.5 if curve["optic"] == "ST2" else 2.8
        ax_limb.plot(illumination, values, style, lw=width,
                     color=OPTIC_COLORS.get(curve["optic"], "0.3"),
                     label=curve["label"],
                     zorder=3 if curve["optic"] == "ST2" else 2)

    ax_limb.axvline(90.0, color="0.55", lw=1.0, ls=":")
    ax_limb.annotate("terminator", xy=(90.0, 1.0), xycoords=("data", "axes fraction"),
                     xytext=(4, -12), textcoords="offset points",
                     fontsize=8, color="0.4")
    ax_limb.set_xlim(0, 180)
    ax_limb.set_xticks(np.arange(0, 181, 30))
    ax_limb.set_xlabel(
        "Earth illumination angle at the grazed limb point (deg)   "
        "—   0 sub-solar, 90 terminator, 180 anti-solar"
    )
    ax_limb.set_ylabel("minimum angle above the limb (deg)")
    ax_limb.grid(alpha=0.3)
    ax_limb.legend(fontsize=8, loc="upper right", frameon=False)
    title = "Earth limb keep-out vs Earth illumination"
    if approximate:
        title += "  (dashed: level set by the subsatellite point, not this axis)"
    ax_limb.set_title(title, fontsize=11)

    # The wedges are specified from the Earth centre; show that scale too
    # so the plotted values line up with the anchor numbers in the source.
    ax_centre = ax_limb.secondary_yaxis(
        "right",
        functions=(lambda v: v + _DYN_EARTH_ANGULAR_RADIUS_DEG,
                   lambda v: v - _DYN_EARTH_ANGULAR_RADIUS_DEG),
    )
    ax_centre.set_ylabel(
        f"angle from the Earth centre (deg)\n"
        f"(limb + {_DYN_EARTH_ANGULAR_RADIUS_DEG:.0f}° nominal Earth radius)",
        fontsize=8,
    )

    fig.suptitle(f"Keep-outs — {visibility!r}", fontsize=10, y=0.985)
    # subplots_adjust rather than tight_layout: the secondary axis on the
    # limb panel is not a layout-managed artist, and tight_layout warns
    # and mis-sizes the figure when it meets one.
    fig.subplots_adjust(left=0.13, right=0.86, top=0.90, bottom=0.09,
                        hspace=0.42)

    if savepath:
        # Create the directory rather than failing at write time, so a
        # caller can pass any path without pre-making the folder.
        savepath = pathlib.Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"Saved: {savepath}")
    if show:
        plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Draft TLE for Pandora — the keep-outs do not depend on it, but
    # Visibility needs a valid orbit to construct.
    LINE1 = "1 67395U 80229J   26212.76861111  .00000000  00000-0  37770-3 0    05"
    LINE2 = "2 67395  97.8073 210.7389 0004787   3.5990  91.5851 14.88172851    02"

    example = Visibility(
        LINE1, LINE2,
        sun_min=91 * u.deg,
        moon_min=20 * u.deg,
        earthlimb_day_min=44 * u.deg,
        earthlimb_night_min=13 * u.deg,
        st_sun_min=45 * u.deg,
        st_moon_min=5 * u.deg,
        st_earthlimb_min=30 * u.deg,
        st_required=1,
        use_dynamic_earthlimb=True,
        use_dynamic_earthlimb_st=True,
    )
    print(keepout_summary(example))
    # Relative to this file, not the working directory, so the script runs
    # the same from the repo root and from inside scripts/.
    plot_keepouts(
        example,
        savepath=pathlib.Path(__file__).resolve().parent / "keepouts.png",
    )
