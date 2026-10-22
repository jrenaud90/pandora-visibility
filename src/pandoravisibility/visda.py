"""Simulated VISDA (visible camera) star field at a spacecraft roll.

VISDA images the sky around the target directly, so a frame is just the Gaia
stars in the field, each spread by the point-spread function (PSF) and scaled
to the counts it gives in one uncoadded 0.2 s science exposure. Fields are
drawn in the detector frame of the VisSci FITS files (the frame of their
ROI_TABLE and of the onboard astrometric solution, ``*_AstSoln.txt``): x along
image columns, y along rows, origin at the lower left.

Every constant below was measured on 2026-10-22 from on-orbit VisSci data
(single 0.2 s frames, or coadds divided by their frame count), mostly
WASP-52 (2026-09-10), TOI-181 (2026-09-06) and HD 131627 (2026-06-27), with
the roll and photometry fits extended to every AstSoln file then on hand (14
visits, 533 stars crossmatched to Gaia DR3).
"""

import numpy as np
from astropy import units as u

__all__ = [
    "detector_offsets",
    "frame_counts",
    "field_radius_arcmin",
    "render_frame",
    "plot_visible_field",
]

# =====================================================================================
# Detector and roll convention
# =====================================================================================
# Onboard astrometric solutions give 0.7879 to 0.7901 arcsec per pixel.
VISDA_PIXEL_SCALE = 0.788  # arcsec per pixel
# Position angle (deg east of north) of the detector +y axis is
# VISDA_ROLL_OFFSET - roll, with the roll in the Visibility (and science
# calendar) convention; +x points 90 deg further east. The AstSoln rotation
# (AROT) plus the calendar roll is 263.65 +- 0.04 deg over 13 visits of six
# targets; NIRDA's Level 2 ROLL header follows the same rule (263.69).
VISDA_ROLL_OFFSET = 263.65  # deg

# =====================================================================================
# Photometry of one 0.2 s frame
# =====================================================================================
# -2.5 log10(counts) = G + VISDA_COLOR_TERM (BP - RP) - VISDA_ZERO_POINT, with
# counts the background-subtracted total of one frame. The color term is fit
# to the AstSoln fluxes of 533 stars (G 7.9 to 16.5, BP - RP 0.6 to 1.7,
# scatter 0.17 mag). Those fluxes miss the PSF wings, so the zero point is set
# by the total counts of 45 bright star ROIs instead (0.22 mag brighter than
# the AstSoln fit, scatter 0.06 mag).
VISDA_ZERO_POINT = 22.59
VISDA_COLOR_TERM = 0.216
# BP - RP for stars without one: the median of the calibration stars.
VISDA_DEFAULT_COLOR = 1.05
# Single-frame bias and noise (AstSoln AIMMEAN and AIMSTD in sparse fields).
VISDA_BIAS = 112.0
VISDA_READ_NOISE = 3.7
# Raw counts clip at 2**15: HD 131627 (G 7.9) peaks at 21,000 to 32,845.
VISDA_SATURATION = 32767.0
# Largest share of a star's counts in its brightest pixel (WASP-52); the PSF
# is a lumpy ring, so its peak is set by hot spots the model below smooths over.
VISDA_PEAK_FRACTION = 0.05

# =====================================================================================
# Point-spread function
# =====================================================================================
# Two concentric Gaussians: the halo matches the measured ring's extent (85 to
# 91% of the light within r = 6 px, 90 to 97% within 8 px) and the core its
# peak (2.7 to 5% of the light in one pixel; this model gives 2.8%).
PSF_CORE_SIGMA = 1.5  # px
PSF_HALO_SIGMA = 4.0  # px
PSF_CORE_FRACTION = 0.3

# =====================================================================================
# Field defaults
# =====================================================================================
FIELD_SIZE = (1056, 1056)  # (x, y) pixels
# Box around the target: the VisSci star ROI size (STARDIMS).
TARGET_BOX = 50
CENTER_BOX = 256
# Stars fainter than this Gaia BP are left out: too faint to see in one frame.
BP_LIMIT = 16.0
# Sources rendered per matrix product in render_frame
_RENDER_CHUNK = 2048


def detector_offsets(xi, eta, roll_deg):
    """VISDA pixel offsets of stars from the target at a roll.

    Parameters
    ----------
    xi, eta : array_like
        Sky offsets from the target in arcsec, east and north.
    roll_deg : float
        Spacecraft roll in degrees (Visibility convention).

    Returns
    -------
    x, y : np.ndarray
        Offsets along detector columns (x) and rows (y), in pixels.
    """
    xi = np.asarray(xi, dtype=float)
    eta = np.asarray(eta, dtype=float)
    pa_y = np.deg2rad(VISDA_ROLL_OFFSET - roll_deg)
    pa_x = pa_y + np.pi / 2
    x = (xi * np.sin(pa_x) + eta * np.cos(pa_x)) / VISDA_PIXEL_SCALE
    y = (xi * np.sin(pa_y) + eta * np.cos(pa_y)) / VISDA_PIXEL_SCALE
    return x, y


def frame_counts(g_mag, bp_rp):
    """Background-subtracted counts of a star in one 0.2 s VISDA frame.

    Parameters
    ----------
    g_mag : array_like
        Gaia G magnitude.
    bp_rp : array_like
        Gaia BP - RP; NaN uses ``VISDA_DEFAULT_COLOR``.

    Returns
    -------
    np.ndarray
        Total counts (ADU) summed over the star's PSF.
    """
    color = np.asarray(bp_rp, dtype=float)
    color = np.where(np.isfinite(color), color, VISDA_DEFAULT_COLOR)
    magnitude = np.asarray(g_mag, dtype=float) + VISDA_COLOR_TERM * color
    return 10.0 ** (-0.4 * (magnitude - VISDA_ZERO_POINT))


def field_radius_arcmin(field_size=FIELD_SIZE, target_pixel=None):
    """Cone radius (arcmin) about the target that covers the whole field.

    Parameters
    ----------
    field_size : tuple of int
        Field (x, y) size in pixels.
    target_pixel : tuple of float, optional
        Target (x, y) pixel; the field center when omitted.

    Returns
    -------
    float
        Distance from the target to the farthest field corner, plus a PSF
        halo's worth of margin.
    """
    target_x, target_y = _target_pixel(field_size, target_pixel)
    corners_x = np.array([-0.5, field_size[0] - 0.5]) - target_x
    corners_y = np.array([-0.5, field_size[1] - 0.5]) - target_y
    farthest = np.hypot(np.abs(corners_x).max(), np.abs(corners_y).max())
    return (farthest + 5 * PSF_HALO_SIGMA) * VISDA_PIXEL_SCALE / 60.0


def _target_pixel(field_size, target_pixel):
    if target_pixel is None:
        return (field_size[0] - 1) / 2.0, (field_size[1] - 1) / 2.0
    return float(target_pixel[0]), float(target_pixel[1])


def _source_arrays(sources, roll_deg, field_size, target_pixel, bp_limit):
    """Pixel positions, counts and kept mask of a source dict (see render_frame)."""
    xi = np.asarray(sources["xi"], dtype=float)
    g_mag = np.asarray(sources["g_mag"], dtype=float)
    bp_rp = np.asarray(sources.get("bp_rp", np.full(xi.size, np.nan)), dtype=float)
    bp_mag = np.asarray(sources.get("bp_mag", np.full(xi.size, np.nan)), dtype=float)
    weight = np.asarray(sources.get("weight", np.ones(xi.size)), dtype=float)
    dx, dy = detector_offsets(xi, sources["eta"], roll_deg)
    target_x, target_y = _target_pixel(field_size, target_pixel)
    # BP decides visibility; G stands in where BP is missing (BP >= G for all
    # but the bluest stars, so this keeps rather than drops)
    limit_mag = np.where(np.isfinite(bp_mag), bp_mag, g_mag)
    kept = limit_mag < bp_limit
    return target_x + dx, target_y + dy, frame_counts(g_mag, bp_rp) * weight, kept


def render_frame(
    sources: dict,
    roll_deg: float,
    field_size=FIELD_SIZE,
    target_pixel=None,
    bp_limit: float = BP_LIMIT,
    add_noise: bool = True,
    seed=0,
):
    """Simulated single 0.2 s VISDA frame, bias subtracted.

    Parameters
    ----------
    sources : dict
        ``xi`` and ``eta`` (arcsec east and north of the target) and
        ``g_mag`` per source, optionally ``bp_mag``, ``bp_rp`` (NaN when
        unknown) and ``weight`` (share of the object's light, for objects
        sampled as many points, default 1). ``star_field.query_visible_field``
        returns this format.
    roll_deg : float
        Spacecraft roll in degrees.
    field_size : tuple of int
        Field (x, y) size in pixels.
    target_pixel : tuple of float, optional
        Target (x, y) pixel; the field center when omitted.
    bp_limit : float
        Sources with BP (G where BP is missing) at or above this are left out.
    add_noise : bool
        Add Gaussian read noise of ``VISDA_READ_NOISE``. Photon noise is left
        out: the detector gain is not in the data used here.
    seed : int or numpy.random.Generator, optional
        Seed for the noise.

    Returns
    -------
    np.ndarray
        (y, x) image in counts above bias, clipped at saturation.
    """
    x, y, counts, kept = _source_arrays(
        sources, roll_deg, field_size, target_pixel, bp_limit
    )
    # Only sources whose halo reaches the field
    reach = 5 * PSF_HALO_SIGMA
    kept &= (
        (x > -reach)
        & (x < field_size[0] + reach)
        & (y > -reach)
        & (y < field_size[1] + reach)
    )
    x, y, counts = x[kept], y[kept], counts[kept]

    # The PSF is a sum of separable Gaussians, so each part of the image is a
    # matrix product of per-source row and column profiles
    pixel_x = np.arange(field_size[0])
    pixel_y = np.arange(field_size[1])
    image = np.zeros((field_size[1], field_size[0]))
    for sigma, fraction in (
        (PSF_CORE_SIGMA, PSF_CORE_FRACTION),
        (PSF_HALO_SIGMA, 1.0 - PSF_CORE_FRACTION),
    ):
        norm = fraction / (2.0 * np.pi * sigma**2)
        for start in range(0, counts.size, _RENDER_CHUNK):
            part = slice(start, start + _RENDER_CHUNK)
            profile_x = np.exp(-0.5 * ((pixel_x - x[part, np.newaxis]) / sigma) ** 2)
            profile_y = np.exp(-0.5 * ((pixel_y - y[part, np.newaxis]) / sigma) ** 2)
            image += profile_y.T @ (norm * counts[part, np.newaxis] * profile_x)
    if add_noise:
        image += np.random.default_rng(seed).normal(0.0, VISDA_READ_NOISE, image.shape)
    return np.minimum(image, VISDA_SATURATION - VISDA_BIAS)


def plot_visible_field(
    sources: dict,
    roll,
    field_size=FIELD_SIZE,
    target_pixel=None,
    target_box: int = TARGET_BOX,
    center_box: int = CENTER_BOX,
    bp_limit: float = BP_LIMIT,
    label_bp_limit: float = 12.0,
    add_noise: bool = True,
    seed=0,
    ax=None,
    title=None,
):
    """Plot the simulated VISDA field in one uncoadded 0.2 s frame.

    The image is from ``render_frame`` on a log stretch from the read noise to
    saturation, in the VisSci detector frame (see the module docstring). A red
    box of ``target_box`` pixels is centered on the target, and a blue box of
    ``center_box`` pixels on the field center. Stars whose brightest pixel
    likely saturates (counts x ``VISDA_PEAK_FRACTION`` above saturation) are
    circled, and white contours mark where the smooth model itself saturates
    (extended bodies). Arrows give north (N) and east (E).

    Sources are labeled with their ``label`` entry when ``sources`` has a
    ``label`` array (empty strings are left unlabeled), otherwise with their
    BP when brighter than ``label_bp_limit``.

    Parameters
    ----------
    sources : dict
        See ``render_frame``.
    roll : astropy.units.Quantity
        Spacecraft roll (angle).
    field_size : tuple of int
        Field (x, y) size in pixels.
    target_pixel : tuple of float, optional
        Target (x, y) pixel; the field center when omitted.
    target_box, center_box : int
        Side in pixels of the red box on the target and the blue box on the
        field center.
    bp_limit : float
        Faintest BP drawn; see ``render_frame``.
    label_bp_limit : float or None
        Label sources brighter than this BP (None for no BP labels).
    add_noise : bool
        Add single-frame read noise.
    seed : int or numpy.random.Generator, optional
        Seed for the noise.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is made when omitted.
    title : str, optional
        Text placed before the roll in the title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    try:
        import matplotlib.patheffects as patheffects
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from matplotlib.patches import Circle, Rectangle
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )
    if not isinstance(roll, u.Quantity):
        raise TypeError(
            f"roll must be an astropy angle Quantity, got {type(roll).__name__}"
        )
    roll_deg = float(roll.to_value(u.deg))
    full_well = VISDA_SATURATION - VISDA_BIAS

    image = render_frame(
        sources, roll_deg, field_size, target_pixel, bp_limit, add_noise, seed
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 7.0))
    norm = LogNorm(vmin=VISDA_READ_NOISE, vmax=full_well)
    shown = ax.imshow(
        np.clip(image, VISDA_READ_NOISE, None),
        origin="lower",
        cmap="gray",
        norm=norm,
        interpolation="nearest",
    )
    ax.figure.colorbar(
        shown, ax=ax, fraction=0.046, pad=0.02, label="counts above bias (0.2 s)"
    )
    noiseless = render_frame(
        sources, roll_deg, field_size, target_pixel, bp_limit, add_noise=False
    )
    if noiseless.max() >= full_well:
        ax.contour(noiseless >= full_well, levels=[0.5], colors="white", linewidths=0.8)

    # Boxes: red on the target, blue on the field center
    target_x, target_y = _target_pixel(field_size, target_pixel)
    center_x, center_y = _target_pixel(field_size, None)
    for (box_x, box_y), side, color, name in (
        ((target_x, target_y), target_box, "red", f"target {target_box} px"),
        ((center_x, center_y), center_box, "tab:blue", f"center {center_box} px"),
    ):
        ax.add_patch(
            Rectangle(
                (box_x - side / 2, box_y - side / 2),
                side,
                side,
                fill=False,
                edgecolor=color,
                linewidth=1.2,
                label=name,
            )
        )

    # Saturation circles and labels for sources in the field
    x, y, counts, kept = _source_arrays(
        sources, roll_deg, field_size, target_pixel, bp_limit
    )
    inside = (
        kept
        & (x >= -0.5)
        & (x <= field_size[0] - 0.5)
        & (y >= -0.5)
        & (y <= field_size[1] - 0.5)
    )
    for index in np.flatnonzero(inside & (counts * VISDA_PEAK_FRACTION >= full_well)):
        ax.add_patch(
            Circle(
                (x[index], y[index]), 12, fill=False, edgecolor="orange", linewidth=1
            )
        )
    labels = sources.get("label")
    if labels is None:
        labels = [""] * len(x)
        if label_bp_limit is not None:
            bp_mag = np.asarray(sources.get("bp_mag", np.full(len(x), np.nan)))
            labels = [f"BP={bp:.1f}" if bp < label_bp_limit else "" for bp in bp_mag]
    outline = [patheffects.withStroke(linewidth=1.5, foreground="black")]
    for index in np.flatnonzero(inside):
        if labels[index]:
            ax.text(
                x[index] + 10,
                y[index] + 10,
                labels[index],
                color="yellow",
                fontsize=7,
                path_effects=outline,
            )

    # North and east arrows, 60 px long, in the lower left corner
    origin = np.array([0.08 * field_size[0], 0.08 * field_size[1]])
    for name, (d_xi, d_eta) in (("N", (0.0, 1.0)), ("E", (1.0, 0.0))):
        dx, dy = detector_offsets(d_xi, d_eta, roll_deg)
        tip = origin + 60 * np.array([dx, dy]) / np.hypot(dx, dy)
        ax.annotate(
            "",
            xy=tip,
            xytext=origin,
            arrowprops={"arrowstyle": "->", "color": "cyan", "linewidth": 1.2},
        )
        ax.text(
            *(origin + 1.35 * (tip - origin)),
            name,
            color="cyan",
            ha="center",
            va="center",
            fontsize=8,
        )

    ax.set_xlim(-0.5, field_size[0] - 0.5)
    ax.set_ylim(-0.5, field_size[1] - 0.5)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
    prefix = f"{title}\n" if title else ""
    ax.set_title(
        f"{prefix}roll {roll_deg:.1f}°, one 0.2 s frame, BP < {bp_limit:g}",
        fontsize=9,
    )
    ax.set_xlabel("x (column, px)")
    ax.set_ylabel("y (row, px)")
    return ax
