"""
Tests for the simulated VISDA star field (visda.py).

The star field query feeding it, ``star_field.query_visible_field``, is
tested in ``test_trace.py`` beside the other star_field tests.
"""

import numpy as np
import pytest
from astropy import units as u

from pandoravisibility import visda


def _sources(xi, eta, g_mag, bp_mag=None, bp_rp=None):
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    sources = {"xi": xi, "eta": np.atleast_1d(eta), "g_mag": np.atleast_1d(g_mag)}
    if bp_mag is not None:
        sources["bp_mag"] = np.atleast_1d(bp_mag)
    if bp_rp is not None:
        sources["bp_rp"] = np.atleast_1d(bp_rp)
    return sources


class TestGeometry:
    """Roll and pixel mapping."""

    def test_matches_wasp52_astrometric_solution(self):
        """WASP-52 at calendar roll 176 deg: the onboard solution's pixel axes."""
        # Linear fit of the 2026-09-10 AstSoln star list, pixels per arcsec:
        # x = 0.0520 xi - 1.2673 eta, y = 1.2673 xi + 0.0520 eta
        x, y = visda.detector_offsets([1.0, 0.0], [0.0, 1.0], 176.0)
        np.testing.assert_allclose(x, [0.0520, -1.2673], atol=0.003)
        np.testing.assert_allclose(y, [1.2673, 0.0520], atol=0.003)

    def test_roll_rotates_about_target(self):
        """A roll keeps distances from the target and turns +y opposite to it."""
        x0, y0 = visda.detector_offsets(30.0, 40.0, 0.0)
        x1, y1 = visda.detector_offsets(30.0, 40.0, 90.0)
        assert np.hypot(x0, y0) == pytest.approx(np.hypot(x1, y1))
        # PA of +y goes from OFFSET to OFFSET - 90: north moves from +y toward -x
        north_x, north_y = visda.detector_offsets(0.0, 1.0, visda.VISDA_ROLL_OFFSET)
        assert north_y == pytest.approx(1.0 / visda.VISDA_PIXEL_SCALE)
        assert north_x == pytest.approx(0.0, abs=1e-12)

    def test_field_radius_covers_corners(self):
        """The cone reaches the farthest corner, also for an off-center target."""
        corner = np.hypot(528.0, 528.0) * visda.VISDA_PIXEL_SCALE / 60.0
        assert visda.field_radius_arcmin((1056, 1056)) > corner
        assert visda.field_radius_arcmin(
            (1056, 1056), target_pixel=(100, 100)
        ) > visda.field_radius_arcmin((1056, 1056))


class TestPhotometry:
    """Counts per frame."""

    def test_zero_point(self):
        """A star at the zero point (with its color term) gives one count."""
        color = 1.0
        g_mag = visda.VISDA_ZERO_POINT - visda.VISDA_COLOR_TERM * color
        assert visda.frame_counts(g_mag, color) == pytest.approx(1.0)
        assert visda.frame_counts(g_mag - 5.0, color) == pytest.approx(100.0)

    def test_missing_color_uses_default(self):
        np.testing.assert_allclose(
            visda.frame_counts(12.0, np.nan),
            visda.frame_counts(12.0, visda.VISDA_DEFAULT_COLOR),
        )


class TestRendering:
    """Frame image and plot."""

    def test_star_at_target_pixel_conserves_counts(self):
        """A lone star lands on the target pixel with all its counts."""
        image = visda.render_frame(
            _sources(0.0, 0.0, 12.0, 12.5, 1.0),
            30.0,
            field_size=(120, 80),
            target_pixel=(40, 50),
            add_noise=False,
        )
        assert image.shape == (80, 120)
        assert np.unravel_index(np.argmax(image), image.shape) == (50, 40)
        assert image.sum() == pytest.approx(visda.frame_counts(12.0, 1.0), rel=1e-3)

    def test_bp_limit_and_g_fallback(self):
        """Stars at or past the BP limit are dropped; G stands in for missing BP."""
        sources = _sources(
            [0.0, 20.0, -20.0],
            [0.0, 0.0, 0.0],
            [12.0, 15.5, 15.5],
            [12.5, 16.2, np.nan],
        )
        image = visda.render_frame(sources, 0.0, (100, 100), add_noise=False)
        alone = visda.render_frame(
            _sources(0.0, 0.0, 12.0, 12.5), 0.0, (100, 100), add_noise=False
        )
        with_g = visda.render_frame(
            {**sources, "bp_mag": np.array([12.5, 16.2, 15.5])},
            0.0,
            (100, 100),
            add_noise=False,
        )
        np.testing.assert_allclose(image, with_g)
        assert image.sum() > alone.sum() * 1.01

    def test_saturation_clip_and_noise(self):
        """Bright stars clip at saturation; noise has the read-noise spread."""
        image = visda.render_frame(_sources(0.0, 0.0, 4.0), 0.0, (64, 64), seed=1)
        assert image.max() == pytest.approx(visda.VISDA_SATURATION - visda.VISDA_BIAS)
        empty = visda.render_frame(_sources([], [], []), 0.0, (200, 200), seed=1)
        assert empty.std() == pytest.approx(visda.VISDA_READ_NOISE, rel=0.02)

    def test_plot_visible_field(self):
        """Boxes, labels and the Quantity check."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        sources = _sources([0.0, 100.0], [0.0, 50.0], [10.0, 11.0], [10.4, 11.4])
        ax = visda.plot_visible_field(
            sources, 10 * u.deg, field_size=(400, 300), target_box=40, title="test"
        )
        boxes = [patch for patch in ax.patches if isinstance(patch, Rectangle)]
        assert sorted(box.get_width() for box in boxes) == [40, 256]
        texts = [text.get_text() for text in ax.texts]
        assert "BP=10.4" in texts and "BP=11.4" in texts
        assert "roll 10.0" in ax.get_title()
        ax = visda.plot_visible_field(
            {**sources, "label": ["", "Moon"]}, 10 * u.deg, field_size=(400, 300)
        )
        assert "Moon" in [text.get_text() for text in ax.texts]
        with pytest.raises(TypeError, match="Quantity"):
            visda.plot_visible_field(sources, 10.0)
        plt.close("all")
