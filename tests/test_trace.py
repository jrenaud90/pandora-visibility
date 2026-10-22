"""
Tests for the NIRDA trace contamination model (trace.py) and its star field
(star_field.py).

Everything here runs without the gaiaoffline database except
``test_real_database_hd189733``, which is skipped when it is not installed.
"""

import os
import warnings

import gaiaoffline
import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from pandoravisibility import Visibility, star_field, trace

# Roll at which the trace (+row) points due north, so a star at eta > 0 lies
# straight along the dispersion.
_NORTH_ROLL = -trace.ROLL_TO_TRACE_PA_OFFSET / trace.ROLL_TO_TRACE_PA_SIGN

# The real field lookup, captured before conftest.py swaps in an empty field
_REAL_TRACE_FIELD = Visibility._trace_field

_LINE1 = "1 67395U 80229J   26057.99991898  .00000000  00000-0  37770-3 0    03"
_LINE2 = "2 67395  97.8009  58.3973 0006599 121.8878 132.9207 14.87804761    04"


def _field(xi, eta, flux_ratio):
    """Minimal field dict, as star_field.query_field returns it."""
    xi, eta, flux_ratio = (
        np.atleast_1d(np.asarray(a, dtype=float)) for a in (xi, eta, flux_ratio)
    )
    return {
        "target_source_id": 0,
        "target_jmag": 8.0,
        "xi": xi,
        "eta": eta,
        "jmag": 8.0 - 2.5 * np.log10(flux_ratio),
        "flux_ratio": flux_ratio,
    }


def _brute_force(roll_degs, field, roi_x, roi_y):
    """The notebook's (roll, row, star) formulation, for checking the fast path."""
    rows, col_low, col_high = trace.roi_window(roi_x, roi_y)
    pa = trace.trace_position_angle(roll_degs)[:, None]
    row, col = trace.detector_offsets(field["xi"][None], field["eta"][None], pa)
    weight = (
        field["flux_ratio"][None]
        * (trace.lsf_cdf(col_high - col) - trace.lsf_cdf(col_low - col))
        / (trace.lsf_cdf(col_high) - trace.lsf_cdf(col_low))
    )
    relative = rows[None, :, None] - row[:, None, :]
    inside = (relative >= trace.TRACE_ROW_MIN - trace.TRACE_END_PAD) & (
        relative <= trace.TRACE_ROW_MAX + trace.TRACE_END_PAD
    )
    return np.einsum("rus,rs->ru", inside, weight).max(axis=1)


class _FakeGaia:
    """Stands in for gaiaoffline.Gaia, returning a fixed catalog."""

    catalog = None

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def conesearch(self, ra, dec, radius):
        return self.catalog.copy()


@pytest.fixture
def fake_gaia(monkeypatch):
    """Route star_field queries to _FakeGaia and skip the database check."""
    monkeypatch.setattr(star_field, "Gaia", _FakeGaia)
    monkeypatch.setattr(star_field, "_check_database", lambda: None)
    return _FakeGaia


def _catalog(rows):
    """Gaia-like DataFrame from (source_id, ra, dec, pmra, pmdec, g, bp_rp) rows."""
    frame = pd.DataFrame(
        rows, columns=["source_id", "ra", "dec", "pmra", "pmdec", "g", "bp_rp"]
    )
    return pd.DataFrame(
        {
            "source_id": frame["source_id"],
            "ra": frame["ra"],
            "dec": frame["dec"],
            "pmra": frame["pmra"],
            "pmdec": frame["pmdec"],
            "phot_g_mean_mag": frame["g"],
            "phot_bp_mean_mag": frame["g"] + frame["bp_rp"] / 2,
            "phot_rp_mean_mag": frame["g"] - frame["bp_rp"] / 2,
        }
    )


class TestStarField:
    """Gaia field query, J estimate and sky offsets."""

    def test_estimate_jmag_matches_2mass(self):
        """HD 189733 A from Gaia lands within 0.1 mag of its 2MASS J (6.073)."""
        jmag = star_field.estimate_jmag(7.428371, 7.893073 - 6.797157)
        assert abs(float(jmag) - 6.073) < 0.1

    def test_estimate_jmag_without_color(self):
        """A star without BP - RP uses the conservative fixed G - J."""
        jmag = star_field.estimate_jmag([12.0], [np.nan])
        np.testing.assert_allclose(jmag, 12.0 - star_field.G_MINUS_J_NO_COLOR)

    def test_gnomonic_offsets_east_and_north(self):
        """xi points east and eta north, in arcsec."""
        dec0 = 30.0
        xi, eta = star_field.gnomonic_offsets(
            [10.0 + 1.0 / 3600 / np.cos(np.deg2rad(dec0)), 10.0],
            [dec0, dec0 + 1.0 / 3600],
            10.0,
            dec0,
        )
        # Curvature of the projection leaves ~1e-6 arcsec at 1 arcsec offsets
        np.testing.assert_allclose(xi, [1.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(eta, [0.0, 1.0], atol=1e-5)

    def test_missing_database_raises_without_creating_it(self, monkeypatch, tmp_path):
        """No database: a clear error, and no empty file left behind by sqlite."""
        missing = tmp_path / "gaiaoffline.db"
        monkeypatch.setattr(gaiaoffline, "DATABASEPATH", str(missing))
        with pytest.raises(FileNotFoundError, match="use_trace_sweep=False"):
            star_field.query_field(SkyCoord(10.0, 20.0, unit="deg"), Time("2026-01-01"))
        assert not missing.exists()

    def test_query_field_selects_and_scales_neighbors(self, fake_gaia):
        """Target excluded, faint and distant stars dropped, flux scaled to the target."""
        ra0, dec0 = 120.0, -10.0
        cos_dec = np.cos(np.deg2rad(dec0))
        fake_gaia.catalog = _catalog(
            [
                (1, ra0, dec0, 0.0, 0.0, 9.0, 1.0),  # target
                (2, ra0 + 30 / 3600 / cos_dec, dec0, 0.0, 0.0, 10.0, 1.0),  # 30" east
                (3, ra0, dec0 + 60 / 3600, 0.0, 0.0, 17.5, 1.0),  # too faint in J
                (4, ra0, dec0 + 6.1 / 60, 0.0, 0.0, 9.0, 1.0),  # beyond 6 arcmin
            ]
        )
        field = star_field.query_field(
            SkyCoord(ra0, dec0, unit="deg"), Time(2016.0, format="jyear")
        )
        assert field["target_source_id"] == 1
        np.testing.assert_allclose(field["xi"], [30.0], atol=1e-3)
        np.testing.assert_allclose(field["eta"], [0.0], atol=1e-3)
        np.testing.assert_allclose(field["flux_ratio"], [10 ** (-0.4)], rtol=1e-9)

    def test_query_field_follows_proper_motion(self, fake_gaia):
        """Offsets are measured at the observation epoch, not J2016."""
        ra0, dec0 = 50.0, 5.0
        fake_gaia.catalog = _catalog(
            [
                (1, ra0, dec0, 0.0, 1000.0, 9.0, 1.0),  # target moving 1"/yr north
                (2, ra0, dec0 + 20 / 3600, 0.0, 0.0, 10.0, 1.0),
            ]
        )
        field = star_field.query_field(
            SkyCoord(ra0, dec0, unit="deg"), Time(2026.0, format="jyear")
        )
        np.testing.assert_allclose(field["eta"], [10.0], atol=1e-3)

    def test_query_field_matches_fast_mover_at_j2000(self, fake_gaia):
        """A J2000 coordinate (e.g. SIMBAD) still finds a fast-moving target."""
        ra0, dec0 = 175.0, 26.0
        # Moves 1.2"/yr north: 19.2" from the J2000 coordinate by J2016
        fake_gaia.catalog = _catalog(
            [
                (1, ra0, dec0 + 19.2 / 3600, 0.0, 1200.0, 9.0, 2.5),
                (2, ra0 + 60 / 3600, dec0, 0.0, 0.0, 11.0, 1.0),
            ]
        )
        field = star_field.query_field(
            SkyCoord(ra0, dec0, unit="deg"), Time(2026.0, format="jyear")
        )
        assert field is not None and field["target_source_id"] == 1

    def test_query_field_without_target_warns(self, fake_gaia):
        """A bare coordinate with no Gaia star nearby gives None and a warning."""
        ra0, dec0 = 200.0, 40.0
        fake_gaia.catalog = _catalog([(2, ra0, dec0 + 30 / 3600, 0.0, 0.0, 10.0, 1.0)])
        with pytest.warns(UserWarning, match="trace check is skipped"):
            field = star_field.query_field(
                SkyCoord(ra0, dec0, unit="deg"), Time("2026-01-01")
            )
        assert field is None

    def test_query_visible_field_keeps_target_and_faint_stars(self, fake_gaia):
        """VISDA fields include the target and stars too faint in J for NIRDA."""
        ra0, dec0 = 120.0, -10.0
        fake_gaia.catalog = _catalog(
            [
                (1, ra0, dec0, 0.0, 1000.0, 9.0, 1.0),  # target moving 1"/yr north
                (2, ra0, dec0 + 60 / 3600, 0.0, 0.0, 15.8, 1.0),  # J ~ 14.6
                (3, ra0, dec0 + 9.0 / 60, 0.0, 0.0, 10.0, 1.0),  # beyond 8 arcmin
            ]
        )
        stars = star_field.query_visible_field(
            SkyCoord(ra0, dec0, unit="deg"), Time(2026.0, format="jyear"), 8.0
        )
        np.testing.assert_array_equal(stars["is_target"], [True, False])
        np.testing.assert_allclose(stars["eta"], [0.0, 50.0], atol=1e-3)
        np.testing.assert_allclose(stars["bp_rp"], [1.0, 1.0])
        np.testing.assert_allclose(stars["bp_mag"], [9.5, 16.3])

    def test_query_visible_field_without_target(self, fake_gaia):
        """With no Gaia target, offsets are from the coordinate and no warning."""
        ra0, dec0 = 200.0, 40.0
        fake_gaia.catalog = _catalog([(2, ra0, dec0 + 30 / 3600, 0.0, 0.0, 10.0, 1.0)])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            stars = star_field.query_visible_field(
                SkyCoord(ra0, dec0, unit="deg"), Time(2016.0, format="jyear"), 5.0
            )
        assert not stars["is_target"].any()
        np.testing.assert_allclose(stars["eta"], [30.0], atol=1e-3)

    def test_query_visible_field_match_target_off(self, fake_gaia):
        """A planet's coordinate is kept even with a star within the match radius."""
        ra0, dec0 = 200.0, 40.0
        fake_gaia.catalog = _catalog([(2, ra0, dec0 + 5 / 3600, 0.0, 0.0, 10.0, 1.0)])
        stars = star_field.query_visible_field(
            SkyCoord(ra0, dec0, unit="deg"),
            Time(2016.0, format="jyear"),
            5.0,
            match_target=False,
        )
        assert not stars["is_target"].any()
        np.testing.assert_allclose(stars["eta"], [5.0], atol=1e-3)

    @pytest.mark.skipif(
        not os.path.isfile(gaiaoffline.DATABASEPATH)
        or os.path.getsize(gaiaoffline.DATABASEPATH) == 0,
        reason="gaiaoffline database not installed",
    )
    def test_real_database_hd189733(self):
        """HD 189733 B (11.5", ~2-4% of A in J) is in the field."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            field = star_field.query_field(
                SkyCoord(300.1821223, 22.7097759, unit="deg"), Time("2026-09-20")
            )
        separation = np.hypot(field["xi"], field["eta"])
        companion = np.argmin(separation)
        assert 10.5 < separation[companion] < 12.5
        assert 0.015 < field["flux_ratio"][companion] < 0.05


class TestTraceGeometry:
    """Roll to detector mapping and the line-spread function."""

    def test_trace_position_angle_on_orbit(self):
        """Roll -80 deg puts the trace at the 73.7 deg measured on orbit."""
        assert trace.trace_position_angle(-80.0) == pytest.approx(73.661)

    def test_detector_offsets_along_and_across(self):
        """A star along the trace PA is all row; 90 deg away it is all column."""
        separation = 10 * trace.NIRDA_PIXEL_SCALE
        row, col = trace.detector_offsets(0.0, separation, 0.0)
        assert row == pytest.approx(10.0) and col == pytest.approx(0.0, abs=1e-12)
        row, col = trace.detector_offsets(separation, 0.0, 0.0)
        assert row == pytest.approx(0.0, abs=1e-12) and abs(col) == pytest.approx(10.0)

    def test_lsf_cdf_limits(self):
        """The cumulative LSF runs from 0 to 1 and is symmetric about the center."""
        np.testing.assert_allclose(
            trace.lsf_cdf([-50.0, 0.0, 50.0]), [0.0, 0.5, 1.0], atol=1e-9
        )

    def test_roi_must_contain_pivot(self):
        """An ROI too small to hold the pivot is rejected."""
        row, col = trace.PIVOT_IN_ROI
        with pytest.raises(ValueError, match="pivot"):
            trace.roi_window(80, row)
        with pytest.raises(ValueError, match="pivot"):
            trace.roi_window(col, 256)

    def test_roi_clips_extraction_rows(self):
        """Rows outside the read-out ROI are not extracted."""
        rows, _, _ = trace.roi_window(80, 150)
        assert rows[-1] == 150 - 1 - trace.PIVOT_IN_ROI[0]
        rows, col_low, col_high = trace.roi_window(80, 256)
        assert (rows[0], rows[-1]) == (-97.0, 97.0)
        assert (col_low, col_high) == (
            -trace.APERTURE_HALF_WIDTH,
            trace.APERTURE_HALF_WIDTH,
        )


class TestContamination:
    """Worst-channel contamination as a function of roll."""

    def test_matches_brute_force(self):
        """The cumulative-sum fast path equals the notebook's direct sum."""
        rng = np.random.default_rng(3)
        field = _field(
            rng.uniform(-250, 250, 60),
            rng.uniform(-250, 250, 60),
            rng.uniform(0, 0.5, 60),
        )
        rolls = np.arange(0.0, 360.0, 1.0)
        np.testing.assert_allclose(
            trace.contamination_vs_roll(rolls, field, 80, 256),
            _brute_force(rolls, field, 80, 256),
            atol=1e-12,
        )

    def test_neighbor_on_trace_counts_its_flux(self):
        """A neighbor straight along the trace leaks its full flux ratio."""
        field = _field(0.0, 20 * trace.NIRDA_PIXEL_SCALE, 0.01)
        aligned = trace.contamination_vs_roll([_NORTH_ROLL], field, 80, 256)
        np.testing.assert_allclose(aligned, [0.01], rtol=1e-6)
        across = trace.contamination_vs_roll([_NORTH_ROLL + 90.0], field, 80, 256)
        assert across[0] < 1e-9

    def test_empty_field_is_clean(self):
        """No neighbors means no contamination at any roll."""
        field = _field([], [], [])
        np.testing.assert_array_equal(
            trace.robust_contamination([0.0, 90.0], field, 80, 256), 0.0
        )

    def test_robust_is_worst_of_window(self):
        """The robust score is never below the plain one and catches near misses."""
        # 10 px across at the candidate roll: both LSF lobes clear the 5 px
        # aperture, but a 1 deg roll error swings a star 100 px along the trace
        # 1.7 px toward it.
        field = _field(10 * trace.NIRDA_PIXEL_SCALE, 100 * trace.NIRDA_PIXEL_SCALE, 1.0)
        rolls = np.array([_NORTH_ROLL, _NORTH_ROLL + 40.0])
        plain = trace.contamination_vs_roll(rolls, field, 80, 256)
        robust = trace.robust_contamination(rolls, field, 80, 256)
        assert np.all(robust >= plain)
        assert robust[0] > 10 * plain[0]

    def test_roi_clip_removes_unread_rows(self):
        """A trace that only covers rows outside a short ROI does not count."""
        # Pivot 150 rows up the dispersion: covers rows 43 to 261 from the pivot
        field = _field(0.0, 150 * trace.NIRDA_PIXEL_SCALE, 1.0)
        assert trace.contamination_vs_roll([_NORTH_ROLL], field, 80, 256)[0] > 0.5
        assert trace.contamination_vs_roll([_NORTH_ROLL], field, 80, 160)[0] == 0.0


class TestRendering:
    """ROI image and plot helper."""

    def test_render_roi_shape_and_target(self):
        """The image is (rows, columns) with the target trace through the pivot."""
        image = trace.render_roi(_field([], [], []), 0.0, 80, 256)
        assert image.shape == (256, 80)
        row, col = trace.PIVOT_IN_ROI
        assert np.argmax(image[row]) in (col - 2, col - 1, col, col + 1, col + 2)
        assert image[row].sum() == pytest.approx(1.0, rel=1e-3)

    def test_plot_trace(self):
        """plot_trace draws on an axes and requires an angle Quantity."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        field = _field(30.0, 40.0, 0.2)
        ax = trace.plot_trace(field, 45 * u.deg, 80, 256, title="test")
        assert "roll 45.0" in ax.get_title()
        with pytest.raises(TypeError, match="Quantity"):
            trace.plot_trace(field, 45.0, 80, 256)
        plt.close("all")

    def test_plot_trace_labels_neighbors_in_roi(self):
        """Neighbor traces inside the ROI get their J; ones outside do not."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # One neighbor 30 px along the trace (in the ROI, J = 8 + 1.75), one
        # 300 px across it (outside the 80 column ROI)
        field = _field(
            [0.0, 300 * trace.NIRDA_PIXEL_SCALE],
            [30 * trace.NIRDA_PIXEL_SCALE, 0.0],
            [0.2, 0.5],
        )
        roll = _NORTH_ROLL * u.deg
        ax = trace.plot_trace(field, roll, 80, 256)
        assert [text.get_text() for text in ax.texts] == ["J=9.7"]
        ax = trace.plot_trace(field, roll, 80, 256, label_neighbors=False)
        assert len(ax.texts) == 0
        plt.close("all")


class TestRollSearchWithTrace:
    """get_visibility(optimize_roll=True) with the trace check."""

    @pytest.fixture
    def target_coord(self):
        return SkyCoord(79.17305002, 45.99514569, frame="icrs", unit="deg")

    @pytest.fixture
    def times(self):
        return Time("2025-01-01T00:00:00") + np.arange(300) * u.min

    @staticmethod
    def _use_field(monkeypatch, field):
        monkeypatch.setattr(Visibility, "_trace_field", lambda self, coord, time: field)

    @staticmethod
    def _along_trace(roll_deg, separation_px, flux_ratio):
        """A neighbor straight along the trace at this roll."""
        pa = np.deg2rad(trace.trace_position_angle(roll_deg))
        separation = separation_px * trace.NIRDA_PIXEL_SCALE
        return _field(separation * np.sin(pa), separation * np.cos(pa), flux_ratio)

    def test_empty_field_matches_trace_off(self, target_coord, times):
        """With no neighbors the search returns what it did without the check."""
        on = Visibility(_LINE1, _LINE2).get_visibility(
            target_coord, times, optimize_roll=True
        )
        off = Visibility(_LINE1, _LINE2, use_trace_sweep=False).get_visibility(
            target_coord, times, optimize_roll=True
        )
        np.testing.assert_array_equal(on["roll_deg"], off["roll_deg"])
        np.testing.assert_array_equal(on["visible"], off["visible"])
        np.testing.assert_array_equal(on["trace_contamination"], 0.0)
        assert np.all(on["trace_ok"])
        assert np.all(np.isnan(off["trace_contamination"]))
        assert np.all(np.isnan(off["trace_min_roll_deg"]))

    def test_contaminated_default_steps_to_best_clean_roll(
        self, monkeypatch, target_coord, times
    ):
        """The held roll is the best-observing roll whose trace is clean."""
        vis = Visibility(_LINE1, _LINE2)
        default = vis.get_visibility(target_coord, times, optimize_roll=True)
        default_roll = float(default["roll_deg"][0])
        field = self._along_trace(default_roll, 30.0, 1.0)
        self._use_field(monkeypatch, field)

        result = vis.get_visibility(target_coord, times, optimize_roll=True)
        chosen = float(result["roll_deg"][0])
        assert chosen != default_roll
        assert np.all(result["trace_ok"])
        assert result["trace_contamination"][0] <= vis.trace_contamination_tol
        assert result["n_visible"] <= default["n_visible"]

        # No clean roll of the sweep observes more timesteps than the held one
        rolls = np.arange(0.0, 360.0, 2.0)
        clean = trace.robust_contamination(rolls, field, 80, 256) <= 1e-3
        counts = [
            vis.get_visibility(target_coord, times, roll=roll * u.deg)["n_visible"]
            for roll in rolls[clean]
        ]
        assert result["n_visible"] == max(counts)

    def test_no_clean_roll_means_not_visible(self, monkeypatch, target_coord, times):
        """A neighbor on top of the target spoils every roll."""
        vis = Visibility(_LINE1, _LINE2)
        default = vis.get_visibility(target_coord, times, optimize_roll=True)
        self._use_field(monkeypatch, _field(0.5, 0.5, 1.0))

        result = vis.get_visibility(target_coord, times, optimize_roll=True)
        assert result["n_visible"] == 0
        assert not np.any(result["visible"])
        assert not np.any(result["trace_ok"])
        np.testing.assert_array_equal(result["roll_deg"], default["roll_deg"])
        assert np.all(result["trace_min_contamination"] > vis.trace_contamination_tol)
        assert np.all(np.isnan(result["solar_power_frac"]))

    def test_target_without_gaia_star_skips_check(
        self, monkeypatch, target_coord, times
    ):
        """No field (no Gaia star at the target) behaves as the check off."""
        self._use_field(monkeypatch, None)
        result = Visibility(_LINE1, _LINE2).get_visibility(
            target_coord, times, optimize_roll=True
        )
        off = Visibility(_LINE1, _LINE2, use_trace_sweep=False).get_visibility(
            target_coord, times, optimize_roll=True
        )
        np.testing.assert_array_equal(result["visible"], off["visible"])
        assert np.all(np.isnan(result["trace_contamination"]))
        assert np.all(result["trace_ok"])

    def test_trace_off_never_queries(self, monkeypatch, target_coord, times):
        """use_trace_sweep=False does not touch the star field at all."""

        def fail(*args):
            raise AssertionError("field queried with the trace check off")

        monkeypatch.setattr(Visibility, "_trace_field", fail)
        Visibility(_LINE1, _LINE2, use_trace_sweep=False).get_visibility(
            target_coord, times, optimize_roll=True
        )

    def test_field_cached_per_target(self, monkeypatch, target_coord, times):
        """Repeated calls on one target query the catalog once."""
        calls = []

        def query(coord, time):
            calls.append(coord)
            return _field([], [], [])

        monkeypatch.setattr(Visibility, "_trace_field", _REAL_TRACE_FIELD)
        monkeypatch.setattr(star_field, "query_field", query)
        vis = Visibility(_LINE1, _LINE2)
        vis.get_visibility(target_coord, times[:100], optimize_roll=True)
        vis.get_visibility(target_coord, times[100:], optimize_roll=True)
        assert len(calls) == 1

    def test_multi_target_shapes(self, target_coord, times):
        """The trace fields stack like every other per-timestep field."""
        other = SkyCoord(10.0, -45.0, frame="icrs", unit="deg")
        result = Visibility(_LINE1, _LINE2).get_visibility(
            [target_coord, other], times[:50], optimize_roll=True
        )
        for key in (
            "trace_contamination",
            "trace_ok",
            "trace_min_contamination",
            "trace_min_roll_deg",
        ):
            assert result[key].shape == (2, 50), key

    def test_kwarg_validation(self):
        """A non-positive tolerance or an ROI without the pivot is rejected."""
        with pytest.raises(ValueError, match="trace_contamination_tol"):
            Visibility(_LINE1, _LINE2, trace_contamination_tol=0.0)
        with pytest.raises(ValueError, match="pivot"):
            Visibility(_LINE1, _LINE2, nirda_roi_y=100)
        vis = Visibility(_LINE1, _LINE2, nirda_roi_x=64, nirda_roi_y=200)
        assert (vis.nirda_roi_x, vis.nirda_roi_y) == (64, 200)


class TestExtendedSources:
    """Fields around non-Gaia targets, extended bodies and custom labels."""

    def test_query_stars_measures_from_center(self, fake_gaia):
        """No target is matched: every bright star in the cone is returned."""
        ra0, dec0 = 10.0, 5.0
        cos_dec = np.cos(np.deg2rad(dec0))
        fake_gaia.catalog = _catalog(
            [
                (1, ra0 + 20 / 3600 / cos_dec, dec0, 0.0, 0.0, 9.0, 1.0),
                (2, ra0, dec0 - 40 / 3600, 0.0, 0.0, 17.5, 1.0),  # too faint in J
            ]
        )
        stars = star_field.query_stars(
            SkyCoord(ra0, dec0, unit="deg"), Time(2016.0, format="jyear")
        )
        np.testing.assert_allclose(stars["xi"], [20.0], atol=1e-3)
        np.testing.assert_allclose(stars["eta"], [0.0], atol=1e-3)
        assert stars["jmag"].shape == (1,)

    def test_point_target_shape_matches_default(self):
        """A one-point target shape at the pivot is the default point target."""
        field = _field([3.0, -20.0], [30.0, 10.0], [0.3, 0.1])
        shaped = {**field, "target_shape": {"xi": [0.0], "eta": [0.0], "weight": [1.0]}}
        rolls = np.arange(0.0, 360.0, 10.0)
        np.testing.assert_allclose(
            trace.contamination_vs_roll(rolls, shaped, 80, 256),
            trace.contamination_vs_roll(rolls, field, 80, 256),
        )

    def test_extended_target_light_outside_aperture(self):
        """Half the target's light far across the dispersion doubles the ratio."""
        field = _field(0.0, 30 * trace.NIRDA_PIXEL_SCALE, 0.01)
        far = 40 * trace.NIRDA_PIXEL_SCALE
        shaped = {
            **field,
            "target_shape": {"xi": [0.0, far], "eta": [0.0, 0.0], "weight": [0.5, 0.5]},
        }
        point = trace.contamination_vs_roll([_NORTH_ROLL], field, 80, 256)
        extended = trace.contamination_vs_roll([_NORTH_ROLL], shaped, 80, 256)
        np.testing.assert_allclose(extended, 2 * point, rtol=1e-6)

    def test_render_roi_is_linear_across_chunks(self):
        """Many faint copies of a source render as one brighter source."""
        position = (5 * trace.NIRDA_PIXEL_SCALE, 20 * trace.NIRDA_PIXEL_SCALE)
        many = _field([position[0]] * 130, [position[1]] * 130, [0.01] * 130)
        one = _field(position[0], position[1], 1.3)
        np.testing.assert_allclose(
            trace.render_roi(many, 30.0, 80, 256),
            trace.render_roi(one, 30.0, 80, 256),
            rtol=1e-9,
            atol=1e-12,  # far tails reach subnormal floats, where rtol is noise
        )

    def test_plot_trace_uses_labels(self):
        """A label array replaces the J labels; empty entries stay unlabeled."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        field = _field(
            [0.0, 1.0],
            [30 * trace.NIRDA_PIXEL_SCALE, 30 * trace.NIRDA_PIXEL_SCALE],
            [0.2, 0.2],
        )
        field["label"] = ["Titan", ""]
        ax = trace.plot_trace(field, _NORTH_ROLL * u.deg, 80, 256)
        assert [text.get_text() for text in ax.texts] == ["Titan"]
        plt.close("all")
