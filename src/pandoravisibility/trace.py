"""NIRDA trace contamination as a function of spacecraft roll.

NIRDA is a slitless prism, so every star in the field makes a trace of the
same length and orientation on the detector. The spacecraft holds the target
on its reference pixel (the pivot), so a roll rotates the sky about the pivot:
each neighbor moves on a circle of radius (separation / pixel scale) and
carries its parallel trace with it. A roll is unsafe when enough neighbor
light lands in the target's extraction aperture.

The contamination in extraction row (spectral channel) ``r`` is

    c(r) = sum_k F_k / F_t * [trace k covers row r] * E(v_k) / E(0)

with F_k / F_t the neighbor's J-band flux relative to the target, v_k its
offset across the dispersion and E the fraction of a trace's cross-dispersion
light inside the aperture. The score of a roll is the worst row, max_r c(r),
taken over every roll within ``ROLL_UNCERTAINTY`` of it (``robust_contamination``),
so a roll only scores well if a small roll error cannot spoil it.

The model and every constant below come from the "NIRDA trace contamination
vs. spacecraft roll" notebook (nirda_trace_roll_safety_v2, 2026-10), whose
on-orbit check against a V1298 Tau frame fixed the roll mapping and the
line-spread function. Angles are plain degrees and detector positions are
pixels relative to the pivot throughout; the field comes from
``star_field.query_field``.
"""

from math import erf

import numpy as np
from astropy import units as u

__all__ = [
    "trace_position_angle",
    "detector_offsets",
    "lsf_cdf",
    "roi_window",
    "contamination_vs_roll",
    "robust_contamination",
    "render_roi",
    "plot_trace",
]

# =====================================================================================
# Detector and roll convention
# =====================================================================================
NIRDA_PIXEL_SCALE = 1.19  # arcsec per pixel
# Trace position angle (east of north) of the detector +row axis, along the
# dispersion: ROLL_TO_TRACE_PA_SIGN * roll + ROLL_TO_TRACE_PA_OFFSET, with the
# roll in the Visibility convention (+X from projected celestial north,
# right-handed about the boresight). Measured on orbit (V1298 Tau, 2026-10-15,
# roll -80 deg): the 6.339 deg detector twist is as in pandoraref 2.1.0, but the
# roll's rotation sense is reversed.
ROLL_TO_TRACE_PA_SIGN = -1
ROLL_TO_TRACE_PA_OFFSET = -6.339  # deg
# Parity of the detector column axis (measured on orbit; as in pandoraref).
COLUMN_SIGN = +1
# Target pivot (row, column) inside the NIRDA ROI, measured in the 2026-10-15
# V1298 Tau frame (pandoraref CRPIX predicts (125, 43)).
PIVOT_IN_ROI = (124, 39)

# =====================================================================================
# Trace and extraction geometry (pixels from the pivot)
# =====================================================================================
# Trace extent along rows; +row is blue on orbit (pandoraref PRF extent, flipped).
TRACE_ROW_MIN, TRACE_ROW_MAX = -101.592, 106.408
# PSF light beyond each trace end, about 3 sigma of the core.
TRACE_END_PAD = 5.0
# Science rows (pandoraref spectrum_normalization edges) and the half-width of
# the extraction aperture across the dispersion.
EXTRACTION_ROW_MIN, EXTRACTION_ROW_MAX = -97.874, 97.874
APERTURE_HALF_WIDTH = 5.0

# =====================================================================================
# Cross-dispersion line-spread function (LSF)
# =====================================================================================
# Two equal Gaussian lobes (measured on orbit) plus an optional Lorentzian wing.
LSF_CORE_SIGMA = 1.0
LSF_LOBE_SEPARATION = 3.5
LSF_WING_FRACTION = 0.0
LSF_WING_HWHM = 1.0

# =====================================================================================
# Roll uncertainty
# =====================================================================================
# Every roll within +-ROLL_UNCERTAINTY (deg) of a candidate must also be clean,
# sampled every ROLL_UNCERTAINTY_STEP (deg).
ROLL_UNCERTAINTY = 1.0
ROLL_UNCERTAINTY_STEP = 0.1

# Standard normal CDF tabulated once: numpy has no erf, and interpolating this
# 0.005-sigma grid is accurate to ~1e-6, far below any contamination tolerance.
_NORMAL_GRID = np.linspace(-12.0, 12.0, 4801)
_NORMAL_CDF = 0.5 * (1.0 + np.array([erf(x / np.sqrt(2.0)) for x in _NORMAL_GRID]))


def trace_position_angle(roll_deg):
    """Trace position angle (deg east of north, in [0, 360)) for roll(s) in deg."""
    return (
        ROLL_TO_TRACE_PA_SIGN * np.asarray(roll_deg, dtype=float)
        + ROLL_TO_TRACE_PA_OFFSET
    ) % 360.0


def detector_offsets(xi, eta, trace_pa_deg):
    """Pixel offsets of stars from the pivot for a trace position angle.

    Parameters
    ----------
    xi, eta : array_like
        Sky offsets from the target in arcsec, east and north.
    trace_pa_deg : array_like
        Trace position angle(s) in degrees; broadcasts against ``xi``.

    Returns
    -------
    row, col : np.ndarray
        Offsets along (row) and across (col) the dispersion, in pixels.
    """
    pa = np.deg2rad(trace_pa_deg)
    row = (xi * np.sin(pa) + eta * np.cos(pa)) / NIRDA_PIXEL_SCALE
    col = COLUMN_SIGN * (xi * np.cos(pa) - eta * np.sin(pa)) / NIRDA_PIXEL_SCALE
    return row, col


def _normal_cdf(x):
    return np.interp(x, _NORMAL_GRID, _NORMAL_CDF)


def lsf_cdf(col_offset):
    """Fraction of a trace's cross-dispersion light below a column offset.

    Parameters
    ----------
    col_offset : array_like
        Column offset from the trace center, in pixels.

    Returns
    -------
    np.ndarray
        Cumulative LSF in [0, 1].
    """
    col_offset = np.asarray(col_offset, dtype=float)
    half_separation = LSF_LOBE_SEPARATION / 2.0
    core = 0.5 * (
        _normal_cdf((col_offset + half_separation) / LSF_CORE_SIGMA)
        + _normal_cdf((col_offset - half_separation) / LSF_CORE_SIGMA)
    )
    wing = 0.5 + np.arctan(col_offset / LSF_WING_HWHM) / np.pi
    return (1.0 - LSF_WING_FRACTION) * core + LSF_WING_FRACTION * wing


def _lsf_density(col_offset):
    """Cross-dispersion LSF per pixel, used only to render images."""
    col_offset = np.asarray(col_offset, dtype=float)
    half_separation = LSF_LOBE_SEPARATION / 2.0
    core = sum(
        np.exp(-0.5 * ((col_offset + lobe) / LSF_CORE_SIGMA) ** 2)
        for lobe in (-half_separation, half_separation)
    ) / (2.0 * np.sqrt(2.0 * np.pi) * LSF_CORE_SIGMA)
    wing = LSF_WING_HWHM / (np.pi * (LSF_WING_HWHM**2 + col_offset**2))
    return (1.0 - LSF_WING_FRACTION) * core + LSF_WING_FRACTION * wing


def roi_window(roi_x: int, roi_y: int):
    """Extraction rows and aperture column limits, clipped to the read-out ROI.

    Parameters
    ----------
    roi_x, roi_y : int
        NIRDA ROI size in columns (cross-dispersion) and rows (dispersion).
        The pivot sits at ``PIVOT_IN_ROI`` inside it.

    Returns
    -------
    rows : np.ndarray
        Integer extraction rows relative to the pivot.
    col_low, col_high : float
        Aperture edges across the dispersion relative to the pivot.

    Raises
    ------
    ValueError
        If the ROI does not contain the pivot.
    """
    pivot_row, pivot_col = PIVOT_IN_ROI
    if roi_y <= pivot_row or roi_x <= pivot_col:
        raise ValueError(
            f"a {roi_x} x {roi_y} NIRDA ROI does not contain the target pivot at "
            f"(row {pivot_row}, column {pivot_col})"
        )
    row_low = max(EXTRACTION_ROW_MIN, -pivot_row)
    row_high = min(EXTRACTION_ROW_MAX, roi_y - 1 - pivot_row)
    rows = np.arange(np.ceil(row_low), np.floor(row_high) + 1.0)
    # Pixel edges, half a pixel beyond the outermost pixel centers
    col_low = max(-APERTURE_HALF_WIDTH, -pivot_col - 0.5)
    col_high = min(APERTURE_HALF_WIDTH, roi_x - pivot_col - 0.5)
    return rows, col_low, col_high


def contamination_vs_roll(roll_degs, field: dict, roi_x: int, roi_y: int):
    """Worst-channel contamination of the target aperture at each roll.

    No roll uncertainty is applied; see ``robust_contamination``.

    Parameters
    ----------
    roll_degs : array_like
        Rolls in degrees.
    field : dict
        Neighbors from ``star_field.query_field`` (``xi``, ``eta``,
        ``flux_ratio``).
    roi_x, roi_y : int
        NIRDA ROI size; see ``roi_window``.

    Returns
    -------
    np.ndarray
        (n_roll,) contaminating / target flux in the worst extraction row.
    """
    rolls = np.atleast_1d(np.asarray(roll_degs, dtype=float))
    rows, col_low, col_high = roi_window(roi_x, roi_y)
    xi = np.asarray(field["xi"], dtype=float)
    eta = np.asarray(field["eta"], dtype=float)
    flux_ratio = np.asarray(field["flux_ratio"], dtype=float)
    if xi.size == 0 or rows.size == 0:
        return np.zeros(rolls.size)

    star_row, star_col = detector_offsets(
        xi[np.newaxis, :],
        eta[np.newaxis, :],
        trace_position_angle(rolls)[:, np.newaxis],
    )
    target_capture = lsf_cdf(col_high) - lsf_cdf(col_low)
    weight = (
        flux_ratio[np.newaxis, :]
        * (lsf_cdf(col_high - star_col) - lsf_cdf(col_low - star_col))
        / target_capture
    )

    # Each (padded) trace covers a contiguous run of extraction rows, so the
    # per-row sum is a cumulative sum of +weight at the first row it covers and
    # -weight past the last, instead of a (roll, row, star) array.
    n_rows = rows.size
    first = np.clip(
        np.ceil(star_row + TRACE_ROW_MIN - TRACE_END_PAD - rows[0]), 0, n_rows
    )
    last = np.clip(
        np.floor(star_row + TRACE_ROW_MAX + TRACE_END_PAD - rows[0]), -1, n_rows - 1
    )
    first, last = first.astype(int), last.astype(int)
    covers = last >= first
    roll_index = np.broadcast_to(np.arange(rolls.size)[:, np.newaxis], star_row.shape)
    edges = np.zeros((rolls.size, n_rows + 1))
    np.add.at(edges, (roll_index[covers], first[covers]), weight[covers])
    np.add.at(edges, (roll_index[covers], last[covers] + 1), -weight[covers])
    return np.cumsum(edges[:, :n_rows], axis=1).max(axis=1)


def robust_contamination(roll_degs, field: dict, roi_x: int, roi_y: int):
    """Worst-channel contamination within the roll uncertainty of each roll.

    Thresholding this is the same as requiring every roll within
    +-``ROLL_UNCERTAINTY`` of the candidate to be clean.

    Parameters
    ----------
    roll_degs : array_like
        Candidate rolls in degrees.
    field : dict
        Neighbors from ``star_field.query_field``.
    roi_x, roi_y : int
        NIRDA ROI size; see ``roi_window``.

    Returns
    -------
    np.ndarray
        (n_roll,) worst contamination over the uncertainty window.
    """
    offsets = np.arange(
        -ROLL_UNCERTAINTY,
        ROLL_UNCERTAINTY + ROLL_UNCERTAINTY_STEP / 2,
        ROLL_UNCERTAINTY_STEP,
    )
    rolls = np.atleast_1d(np.asarray(roll_degs, dtype=float))[:, np.newaxis] + offsets
    return (
        contamination_vs_roll(rolls.ravel(), field, roi_x, roi_y)
        .reshape(rolls.shape)
        .max(axis=1)
    )


def render_roi(field: dict, roll_deg: float, roi_x: int, roi_y: int):
    """Approximate NIRDA ROI image at one roll, for illustration only.

    Top-hat traces along rows (with Gaussian edges) times the LSF across them,
    scaled by flux ratio, the target's trace normalized to unit flux per row.

    Parameters
    ----------
    field : dict
        Neighbors from ``star_field.query_field``; the target is added at the
        pivot.
    roll_deg : float
        Roll in degrees.
    roi_x, roi_y : int
        NIRDA ROI size; see ``roi_window``.

    Returns
    -------
    np.ndarray
        (roi_y, roi_x) image, row index first.
    """
    roi_window(roi_x, roi_y)  # validates that the ROI holds the pivot
    star_row, star_col = detector_offsets(
        np.append(0.0, field["xi"]),
        np.append(0.0, field["eta"]),
        trace_position_angle(roll_deg),
    )
    flux_ratio = np.append(1.0, field["flux_ratio"])
    pixel_row = np.arange(roi_y)[:, np.newaxis, np.newaxis] - PIVOT_IN_ROI[0]
    pixel_col = np.arange(roi_x)[np.newaxis, :, np.newaxis] - PIVOT_IN_ROI[1]
    along = _normal_cdf(
        (pixel_row - (star_row + TRACE_ROW_MIN)) / LSF_CORE_SIGMA
    ) - _normal_cdf((pixel_row - (star_row + TRACE_ROW_MAX)) / LSF_CORE_SIGMA)
    across = _lsf_density(pixel_col - star_col)
    return (flux_ratio * along * across).sum(axis=2)


def plot_trace(field: dict, roll, roi_x: int, roi_y: int, ax=None, title=None):
    """Plot the simulated NIRDA ROI at one roll, with the extraction aperture.

    The image is from ``render_roi`` on a log stretch; the dashed box is the
    extraction aperture and the title gives the roll and its robust
    worst-channel contamination (``robust_contamination``).

    Parameters
    ----------
    field : dict
        Neighbors from ``star_field.query_field``.
    roll : astropy.units.Quantity
        Spacecraft roll (angle).
    roi_x, roi_y : int
        NIRDA ROI size; see ``roi_window``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is made when omitted.
    title : str, optional
        Text placed before the roll and contamination in the title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from matplotlib.patches import Rectangle
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )
    if not isinstance(roll, u.Quantity):
        raise TypeError(
            f"roll must be an astropy angle Quantity, got {type(roll).__name__}"
        )
    roll_deg = float(roll.to_value(u.deg))

    image = render_roi(field, roll_deg, roi_x, roi_y)
    contamination = float(robust_contamination([roll_deg], field, roi_x, roi_y)[0])
    if ax is None:
        _, ax = plt.subplots(figsize=(3.0, 7.0))
    ax.imshow(
        np.clip(image, 1e-6 * image.max(), None),
        origin="lower",
        cmap="magma",
        norm=LogNorm(vmin=1e-6 * image.max(), vmax=image.max()),
        aspect="equal",
    )
    rows, col_low, col_high = roi_window(roi_x, roi_y)
    ax.add_patch(
        Rectangle(
            (PIVOT_IN_ROI[1] + col_low, PIVOT_IN_ROI[0] + rows[0] - 0.5),
            col_high - col_low,
            rows[-1] - rows[0] + 1.0,
            fill=False,
            edgecolor="cyan",
            linestyle="--",
        )
    )
    prefix = f"{title}\n" if title else ""
    ax.set_title(f"{prefix}roll {roll_deg:.1f}°, c = {contamination:.1e}", fontsize=9)
    ax.set_xlabel("column")
    ax.set_ylabel("row (dispersion, blue up)")
    return ax
