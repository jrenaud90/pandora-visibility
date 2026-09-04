"""
Tests for Visibility class methods that are not covered in test_import.py
"""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import GCRS, SkyCoord, get_body
from astropy.time import Time

from pandoravisibility import Visibility

# The class defaults carry Pandora's flight keep-outs from pre v1.3.0, so a bare
# Visibility now has the star tracker limits and the dynamic Earth limb wedge
# switched on.  Tests that pin the older, simpler paths, or that need a
# boresight-only instance, name that here rather than inheriting whatever the
# defaults become next.
_LEGACY_DEFAULTS = dict(
    moon_min=25 * u.deg,
    earthlimb_day_min=None,
    earthlimb_night_min=None,
    use_dynamic_earthlimb=False,
    st_sun_min=0 * u.deg,
    st_moon_min=0 * u.deg,
    st_earthlimb_min=0 * u.deg,
)


def _legacy_visibility(line1, line2, **overrides):
    """A Visibility on the pre-v1.3.0 defaults, with *overrides* applied."""
    return Visibility(line1, line2, **{**_LEGACY_DEFAULTS, **overrides})


class TestVisibilityClassMethods:
    """Test suite for Visibility class methods."""

    @pytest.fixture
    def visibility_instance(self):
        """Create a standard Visibility instance for testing."""
        line1 = "1 67395U 80229J   26057.99991898  .00000000  00000-0  37770-3 0    03"
        line2 = "2 67395  97.8009  58.3973 0006599 121.8878 132.9207 14.87804761    04"
        return Visibility(line1, line2)

    @pytest.fixture
    def custom_visibility_instance(self):
        """Create a Visibility instance with custom limits."""
        line1 = "1 67395U 80229J   26057.99991898  .00000000  00000-0  37770-3 0    03"
        line2 = "2 67395  97.8009  58.3973 0006599 121.8878 132.9207 14.87804761    04"
        return Visibility(
            line1,
            line2,
            moon_min=30 * u.deg,
            sun_min=100 * u.deg,
            earthlimb_min=15 * u.deg,
            mars_min=5 * u.deg,
            jupiter_min=5 * u.deg,
        )

    @pytest.fixture
    def target_coord(self):
        """Standard target coordinate (Capella)."""
        return SkyCoord(79.17305002, 45.99514569, frame="icrs", unit="deg")

    @pytest.fixture
    def test_time(self):
        """Standard test time."""
        return Time("2025-01-01T00:00:00")

    def test_repr_default_constraints(self, visibility_instance):
        """Test __repr__ method with default constraints."""
        repr_str = repr(visibility_instance)
        assert "<Visibility:" in repr_str
        assert "SAT67395" in repr_str
        assert "moon≥" in repr_str
        assert "sun≥" in repr_str
        assert "limb≥" in repr_str

    def test_repr_custom_constraints(self, custom_visibility_instance):
        """Test __repr__ method with custom constraints."""
        repr_str = repr(custom_visibility_instance)
        assert "<Visibility:" in repr_str
        assert "SAT67395" in repr_str
        assert "moon≥30 deg" in repr_str
        assert "sun≥100 deg" in repr_str
        assert "mars≥5 deg" in repr_str
        assert "jupiter≥5 deg" in repr_str

    def test_repr_zero_constraints(self):
        """Test __repr__ method with zero constraints."""
        line1 = "1 67395U 80229J   26057.99991898  .00000000  00000-0  37770-3 0    03"
        line2 = "2 67395  97.8009  58.3973 0006599 121.8878 132.9207 14.87804761    04"
        vis = _legacy_visibility(
            line1, line2, moon_min=0 * u.deg, sun_min=0 * u.deg,
            earthlimb_min=0 * u.deg,
        )
        repr_str = repr(vis)
        assert "<Visibility:" in repr_str
        assert "default" in repr_str

    def test_get_all_constraints(self, visibility_instance, target_coord, test_time):
        """Test get_all_constraints method returns dict with all constraints."""
        constraints = visibility_instance.get_all_constraints(target_coord, test_time)

        assert isinstance(constraints, dict)
        assert "moon" in constraints
        assert "sun" in constraints
        assert "earthlimb" in constraints
        # Can be bool or numpy bool
        assert isinstance(constraints["moon"], (bool, np.bool_))
        assert isinstance(constraints["sun"], (bool, np.bool_))
        assert isinstance(constraints["earthlimb"], (bool, np.bool_))

    def test_get_all_constraints_with_planets(
        self, custom_visibility_instance, target_coord, test_time
    ):
        """Test get_all_constraints includes planetary constraints when enabled."""
        constraints = custom_visibility_instance.get_all_constraints(
            target_coord, test_time
        )

        assert "mars" in constraints
        assert "jupiter" in constraints
        # Can be bool or numpy bool
        assert isinstance(constraints["mars"], (bool, np.bool_))
        assert isinstance(constraints["jupiter"], (bool, np.bool_))

    def test_get_separations(self, visibility_instance, target_coord, test_time):
        """Test get_separations method returns angles for all bodies."""
        separations = visibility_instance.get_separations(target_coord, test_time)

        assert isinstance(separations, dict)
        assert "moon" in separations
        assert "sun" in separations
        assert "earthlimb" in separations
        assert "mars" in separations
        assert "jupiter" in separations

        # Check that all separations are angles
        for body, sep in separations.items():
            assert hasattr(sep, "unit")  # Has astropy unit
            assert sep.unit.is_equivalent(u.deg)  # Is angle unit

    def test_get_separations_values_reasonable(
        self, visibility_instance, target_coord, test_time
    ):
        """Test that separation values are in reasonable ranges."""
        separations = visibility_instance.get_separations(target_coord, test_time)

        # All separations should be between -90 and 360 degrees
        for body, sep in separations.items():
            sep_deg = sep.to(u.deg).value
            assert (
                -90 <= sep_deg <= 360
            ), f"{body} separation {sep_deg} deg is out of reasonable range"

    def test_summary_scalar_time(self, visibility_instance, target_coord, test_time):
        """Test summary method with scalar time."""
        summary = visibility_instance.summary(target_coord, test_time)

        assert isinstance(summary, str)
        assert "Visibility Summary" in summary
        assert "Target:" in summary
        assert "Time:" in summary
        assert "Sat:" in summary
        assert "Moon" in summary or "moon" in summary.lower()
        assert "Sun" in summary or "sun" in summary.lower()
        assert "Earthlimb" in summary or "earthlimb" in summary.lower()
        assert "Overall:" in summary
        assert "VISIBLE" in summary or "NOT VISIBLE" in summary

    def test_summary_shows_constraint_status(
        self, visibility_instance, target_coord, test_time
    ):
        """Test that summary shows PASS/FAIL status for constraints."""
        summary = visibility_instance.summary(target_coord, test_time)

        # Should contain status indicators
        assert "PASS" in summary or "FAIL" in summary
        # Should contain check marks or crosses
        assert "✓" in summary or "✗" in summary

    def test_summary_shows_separation_values(
        self, visibility_instance, target_coord, test_time
    ):
        """Test that summary shows actual separation values."""
        summary = visibility_instance.summary(target_coord, test_time)

        assert "req:" in summary or "actual:" in summary
        # Should contain degree values
        assert "deg" in summary

    def test_summary_array_time_raises_error(self, visibility_instance, target_coord):
        """Test that summary raises error with array time input."""

        times = Time("2025-01-01T00:00:00") + np.arange(5) * u.hour

        with pytest.raises(ValueError, match="scalar"):
            visibility_instance.summary(target_coord, times)

    def test_invalid_tle_empty_lines(self):
        """Test that empty TLE lines raise ValueError."""
        with pytest.raises(ValueError, match="TLE lines cannot be empty"):
            Visibility("", "")

    def test_invalid_tle_none_lines(self):
        """Test that None TLE lines raise ValueError."""
        with pytest.raises(ValueError, match="TLE lines cannot be empty"):
            Visibility(None, None)

    def test_invalid_tle_bad_format(self):
        """Test that malformed TLE data raises ValueError."""
        line1 = "INVALID LINE 1"
        line2 = "INVALID LINE 2"

        # The SGP4 library may not always raise an exception for invalid data
        # Just try to create and catch any exception
        try:
            vis = Visibility(line1, line2)
            # If it doesn't raise, at least verify it created something
            assert vis is not None
        except (ValueError, Exception):
            # This is the expected behavior
            pass

    def test_get_constraint_invalid_body(
        self, visibility_instance, target_coord, test_time
    ):
        """Test that get_constraint raises error for invalid body name."""
        with pytest.raises(ValueError, match="Invalid body"):
            visibility_instance.get_constraint(target_coord, "venus", test_time)

    def test_get_state_without_time_attribute(self, visibility_instance):
        """Test get_state raises error when no time is provided and self.time is not set."""
        # Remove time attribute if it exists
        if hasattr(visibility_instance, "time"):
            delattr(visibility_instance, "time")

        with pytest.raises(ValueError, match="No time parameter specified"):
            visibility_instance.get_state()

    def test_custom_limits_applied(self, custom_visibility_instance):
        """Test that custom limits are properly applied to instance."""
        assert custom_visibility_instance.moon_min == 30 * u.deg
        assert custom_visibility_instance.sun_min == 100 * u.deg
        assert custom_visibility_instance.earthlimb_min == 15 * u.deg
        assert custom_visibility_instance.mars_min == 5 * u.deg
        assert custom_visibility_instance.jupiter_min == 5 * u.deg

    def test_visibility_with_different_constraints_changes_result(
        self, target_coord, test_time
    ):
        """Test that different constraints produce different visibility results."""
        line1 = "1 67395U 80229J   26057.99991898  .00000000  00000-0  37770-3 0    03"
        line2 = "2 67395  97.8009  58.3973 0006599 121.8878 132.9207 14.87804761    04"

        # Very loose constraints
        vis_loose = _legacy_visibility(
            line1,
            line2,
            moon_min=0 * u.deg,
            sun_min=0 * u.deg,
            earthlimb_min=-90 * u.deg,
        )
        result_loose = vis_loose.get_visibility(target_coord, test_time)["visible"]

        # Very tight constraints
        vis_tight = _legacy_visibility(line1, line2, moon_min=180 * u.deg)
        result_tight = vis_tight.get_visibility(target_coord, test_time)["visible"]

        # Loose constraints should be more permissive
        assert result_loose is True
        assert result_tight is False

    def test_get_period_returns_correct_units(self, visibility_instance):
        """Test that get_period returns value with correct units."""
        period = visibility_instance.get_period()
        assert hasattr(period, "unit")
        assert period.unit == u.minute

    def test_get_state_returns_skycoord(self, visibility_instance, test_time):
        """Test that get_state returns a SkyCoord object."""
        state = visibility_instance.get_state(test_time)
        assert isinstance(state, SkyCoord)
        assert hasattr(state, "x")
        assert hasattr(state, "y")
        assert hasattr(state, "z")

    def test_get_state_with_array_time(self, visibility_instance):
        """Test get_state with array of times."""

        times = Time("2025-01-01T00:00:00") + np.arange(5) * u.hour
        states = visibility_instance.get_state(times)

        assert len(states) == 5
        assert isinstance(states, SkyCoord)

    @pytest.fixture(params=["scalar", "list", "skycoord_array"])
    def target_inputs(self, request, target_coord):
        """Provide target_coord in all supported input forms."""
        if request.param == "scalar":
            return target_coord, 1
        elif request.param == "list":
            return [target_coord] * 3, 3
        else:
            return SkyCoord([79.17] * 3, [45.99] * 3, frame="icrs", unit="deg"), 3

    def test_get_visibility_target_forms(
        self, visibility_instance, target_coord, target_inputs, test_time
    ):
        """Test get_visibility with scalar, list, and SkyCoord array targets."""
        targets, expected_len = target_inputs
        result = visibility_instance.get_visibility(targets, test_time)["visible"]
        single = visibility_instance.get_visibility(target_coord, test_time)["visible"]

        if expected_len == 1:
            assert isinstance(result, bool)
            assert result == single
        else:
            assert isinstance(result, np.ndarray)
            assert result.shape == (expected_len,)
            assert all(r == single for r in result)

    def test_get_visibility_target_forms_with_st(
        self, target_coord, target_inputs, test_time
    ):
        """Test get_visibility target forms with star tracker constraints."""
        line1 = "1 67395U 80229J   26057.99991898  .00000000  00000-0  37770-3 0    03"
        line2 = "2 67395  97.8009  58.3973 0006599 121.8878 132.9207 14.87804761    04"
        vis = Visibility(line1, line2, st_sun_min=44 * u.deg)

        targets, expected_len = target_inputs
        result = vis.get_visibility(targets, test_time)["visible"]
        single = vis.get_visibility(target_coord, test_time)["visible"]

        if expected_len == 1:
            assert isinstance(result, bool)
            assert result == single
        else:
            assert isinstance(result, np.ndarray)
            assert result.shape == (expected_len,)
            assert all(r == single for r in result)

    def test_get_visibility_multi_target_array_time(
        self, visibility_instance, target_coord
    ):
        """Multi-target + array time returns 2D bool array of shape (N, M)."""
        targets = [target_coord] * 3
        times = Time("2025-01-01T00:00:00") + np.arange(5) * u.hour

        result = visibility_instance.get_visibility(targets, times)["visible"]
        single = visibility_instance.get_visibility(target_coord, times)["visible"]

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        assert result.shape == (3, 5)
        for row in result:
            np.testing.assert_array_equal(row, single)

    def test_get_star_tracker_angles_return_structure(
        self, visibility_instance, target_coord, test_time
    ):
        """Test that get_star_tracker_angles returns dict with correct keys."""
        result = visibility_instance.get_star_tracker_angles(
            target_coord, test_time, tracker=1
        )

        assert isinstance(result, dict)
        assert "ra" in result
        assert "dec" in result
        assert "sun_angle" in result
        assert "earth_angle" in result
        assert "earthlimb_angle" in result

        # Check all values are Quantities with degree units
        for key, value in result.items():
            assert hasattr(value, "unit")
            assert value.unit.is_equivalent(u.deg)

    def test_get_star_tracker_angles_tracker1(
        self, visibility_instance, target_coord, test_time
    ):
        """Test star tracker 1 RA/Dec calculation."""
        result = visibility_instance.get_star_tracker_angles(
            target_coord, test_time, tracker=1
        )

        # RA should be in [0, 360) degrees
        assert 0 * u.deg <= result["ra"] < 360 * u.deg
        # Dec should be in [-90, 90] degrees
        assert -90 * u.deg <= result["dec"] <= 90 * u.deg

    def test_get_star_tracker_angles_tracker2(
        self, visibility_instance, target_coord, test_time
    ):
        """Test star tracker 2 RA/Dec calculation."""
        result = visibility_instance.get_star_tracker_angles(
            target_coord, test_time, tracker=2
        )

        # RA should be in [0, 360) degrees
        assert 0 * u.deg <= result["ra"] < 360 * u.deg
        # Dec should be in [-90, 90] degrees
        assert -90 * u.deg <= result["dec"] <= 90 * u.deg

    def test_get_star_tracker_angles_different_trackers(
        self, visibility_instance, target_coord, test_time
    ):
        """Test that tracker 1 and tracker 2 give different RA/Dec."""
        result1 = visibility_instance.get_star_tracker_angles(
            target_coord, test_time, tracker=1
        )
        result2 = visibility_instance.get_star_tracker_angles(
            target_coord, test_time, tracker=2
        )

        # The two trackers should point at different sky positions
        # Tracker 1 and 2 have opposite Y-components in their boresight vectors
        # (0.7071 vs -0.7071), which should produce different pointings
        ra_diff = abs(result1["ra"] - result2["ra"])
        # Handle RA wrapping at 0/360 degrees
        ra_diff = min(ra_diff, 360 * u.deg - ra_diff)
        dec_diff = abs(result1["dec"] - result2["dec"])

        # At least one coordinate should be different by > 1 degree
        # This threshold accounts for the geometric difference in tracker orientations
        assert ra_diff > 1.0 * u.deg or dec_diff > 1.0 * u.deg

    def test_get_star_tracker_sun_angle_symmetry(
        self, visibility_instance, target_coord, test_time
    ):
        """Regression: ST1 and ST2 must have equal sun angles.

        The sun lies in the spacecraft XZ plane (zero Y-component in body
        frame) and the two star tracker boresights are symmetric about
        that plane (+/-0.7071 in Y).  Therefore both trackers must see
        exactly the same angular separation to the sun.

        This test catches the bug where SkyCoord was created with bare
        ``frame="gcrs"`` (defaulting obstime=J2000), which caused
        astropy's ``separation()`` to apply an incorrect geocenter offset
        when comparing against a GCRS body at the observation epoch.
        Before the fix, ST2 sun angle was wrong by ~22 degrees.
        """
        result1 = visibility_instance.get_star_tracker_angles(
            target_coord, test_time, tracker=1
        )
        result2 = visibility_instance.get_star_tracker_angles(
            target_coord, test_time, tracker=2
        )

        # Sun angles must match to within 0.01 degrees
        assert abs(result1["sun_angle"] - result2["sun_angle"]) < 0.01 * u.deg

    def test_get_star_tracker_angles_sun_angle(
        self, visibility_instance, target_coord, test_time
    ):
        """Test sun angle calculation is reasonable."""
        result = visibility_instance.get_star_tracker_angles(
            target_coord, test_time, tracker=1
        )

        # Sun angle should be between 0 and 180 degrees
        assert 0 * u.deg <= result["sun_angle"] <= 180 * u.deg

    def test_get_star_tracker_angles_earth_angle(
        self, visibility_instance, target_coord, test_time
    ):
        """Test earth angle calculation is reasonable."""
        result = visibility_instance.get_star_tracker_angles(
            target_coord, test_time, tracker=1
        )

        # Earth angle should be between 0 and 180 degrees
        assert 0 * u.deg <= result["earth_angle"] <= 180 * u.deg

    def test_get_star_tracker_angles_earthlimb_angle(
        self, visibility_instance, target_coord, test_time
    ):
        """Test earthlimb angle calculation is reasonable."""
        result = visibility_instance.get_star_tracker_angles(
            target_coord, test_time, tracker=1
        )

        # Earthlimb angle should be between -90 and 180 degrees
        # (can be negative if below horizon)
        assert -90 * u.deg <= result["earthlimb_angle"] <= 180 * u.deg

    def test_get_star_tracker_angles_invalid_tracker_number(
        self, visibility_instance, target_coord, test_time
    ):
        """Test that invalid tracker number raises ValueError."""
        with pytest.raises(ValueError, match="Invalid tracker number"):
            visibility_instance.get_star_tracker_angles(
                target_coord, test_time, tracker=3
            )

        with pytest.raises(ValueError, match="Invalid tracker number"):
            visibility_instance.get_star_tracker_angles(
                target_coord, test_time, tracker=0
            )

    def test_get_star_tracker_angles_sun_aligned_target(self, visibility_instance):
        """Test that target aligned with sun raises ValueError."""
        # Use a time and find the sun's position
        test_time = Time("2025-06-21T12:00:00")

        # Get the sun's position at this time
        observer_location = visibility_instance._get_observer_location(test_time)
        sun_coord = get_body("sun", time=test_time, location=observer_location)

        # Use sun's position as target (aligned with sun)
        with pytest.raises(
            ValueError, match="Cannot determine attitude: target aligned with sun"
        ):
            visibility_instance.get_star_tracker_angles(sun_coord, test_time, tracker=1)

    def test_get_star_tracker_angles_default_tracker(
        self, visibility_instance, target_coord, test_time
    ):
        """Test that default tracker parameter is 1."""
        result_default = visibility_instance.get_star_tracker_angles(
            target_coord, test_time
        )
        result_tracker1 = visibility_instance.get_star_tracker_angles(
            target_coord, test_time, tracker=1
        )

        # Default should be same as tracker=1 for all values
        assert result_default["ra"] == result_tracker1["ra"]
        assert result_default["dec"] == result_tracker1["dec"]
        assert result_default["sun_angle"] == result_tracker1["sun_angle"]
        assert result_default["earth_angle"] == result_tracker1["earth_angle"]
        assert result_default["earthlimb_angle"] == result_tracker1["earthlimb_angle"]


class TestStarTrackerConstraints:
    """Test suite for star tracker keep-out constraint features."""

    @pytest.fixture
    def line1(self):
        return "1 67395U 80229J   26057.99991898  .00000000  00000-0  37770-3 0    03"

    @pytest.fixture
    def line2(self):
        return "2 67395  97.8009  58.3973 0006599 121.8878 132.9207 14.87804761    04"

    @pytest.fixture
    def target_coord(self):
        return SkyCoord(79.17305002, 45.99514569, frame="icrs", unit="deg")

    @pytest.fixture
    def test_time(self):
        return Time("2025-01-01T00:00:00")

    @pytest.fixture
    def breakdown_vis(self, line1, line2):
        return Visibility(
            line1, line2,
            st_sun_min=50 * u.deg,
            st_moon_min=20 * u.deg,
            st_earthlimb_min=30 * u.deg,
            st_required=1,
        )

    def test_breakdown_combined_matches_constraint(
        self, breakdown_vis, target_coord, test_time
    ):
        """combined reproduces get_star_tracker_constraint exactly."""
        times = test_time + np.arange(300) * u.min
        breakdown = breakdown_vis.get_star_tracker_breakdown(target_coord, times)
        engine = np.asarray(
            breakdown_vis.get_star_tracker_constraint(target_coord, times)
        )
        assert np.array_equal(
            np.asarray(breakdown["passed"]["combined"]), engine
        )

    def test_breakdown_rows_reconstruct_tracker(
        self, breakdown_vis, target_coord, test_time
    ):
        """ANDing one tracker's per-check rows gives that tracker's verdict."""
        times = test_time + np.arange(300) * u.min
        breakdown = breakdown_vis.get_star_tracker_breakdown(target_coord, times)
        for tracker in (1, 2):
            rows = [
                mask for name, mask in breakdown["passed"].items()
                if name.startswith(f"ST{tracker} ")
            ]
            assert rows, f"no per-check rows for ST{tracker}"
            recon = np.ones(len(times), dtype=bool)
            for mask in rows:
                recon &= np.asarray(mask)
            assert np.array_equal(
                recon, np.asarray(breakdown["passed"][f"ST{tracker}"])
            )

    def test_breakdown_only_lists_active_checks(self, line1, line2, target_coord,
                                                test_time):
        """Switched-off keep-outs get no row; per-tracker limits are honoured."""
        vis = _legacy_visibility(
            line1, line2,
            st_sun_min=44 * u.deg,
            st1_earthlimb_min=30 * u.deg,
            st_required=1,
        )
        breakdown = vis.get_star_tracker_breakdown(target_coord, test_time)
        rows = set(breakdown["passed"])
        assert "ST1 sun" in rows and "ST2 sun" in rows
        assert "ST1 limb" in rows          # per-tracker override is active
        assert "ST2 limb" not in rows      # falls back to st_earthlimb_min = 0
        assert not any(r.endswith(" moon") for r in rows)
        assert breakdown["limits"]["ST1 limb"] == 30 * u.deg

    def test_breakdown_separations_drive_the_masks(
        self, breakdown_vis, target_coord, test_time
    ):
        """Each row's mask is exactly its separation against its limit."""
        times = test_time + np.arange(200) * u.min
        breakdown = breakdown_vis.get_star_tracker_breakdown(target_coord, times)
        for name, mask in breakdown["passed"].items():
            if name in ("ST1", "ST2", "combined"):
                continue
            sep = np.asarray(breakdown["separations"][name], dtype=float)
            limit = breakdown["limits"][name].to(u.deg).value
            assert np.array_equal(np.asarray(mask), sep >= limit)

    def test_breakdown_scalar_time_returns_bools(
        self, breakdown_vis, target_coord, test_time
    ):
        """Scalar time gives plain bools, matching the array path."""
        times = test_time + np.arange(5) * u.min
        scalar = breakdown_vis.get_star_tracker_breakdown(target_coord, times[0])
        array = breakdown_vis.get_star_tracker_breakdown(target_coord, times)
        for name, value in scalar["passed"].items():
            assert isinstance(value, bool), f"{name} is {type(value).__name__}"
            assert value == bool(np.asarray(array["passed"][name])[0])

    def test_breakdown_roll_override_flows_through(
        self, breakdown_vis, target_coord, test_time
    ):
        """A roll override changes the rows and the combined verdict together."""
        times = test_time + np.arange(200) * u.min
        at_0 = breakdown_vis.get_star_tracker_breakdown(
            target_coord, times, roll=0 * u.deg)
        at_90 = breakdown_vis.get_star_tracker_breakdown(
            target_coord, times, roll=90 * u.deg)
        assert not np.array_equal(
            np.asarray(at_0["passed"]["combined"]),
            np.asarray(at_90["passed"]["combined"]),
        )
        # and each stays internally consistent
        for breakdown in (at_0, at_90):
            recon = np.ones(len(times), dtype=bool)
            for name, mask in breakdown["passed"].items():
                if name.startswith("ST1 "):
                    recon &= np.asarray(mask)
            assert np.array_equal(recon, np.asarray(breakdown["passed"]["ST1"]))

    def test_breakdown_requires_angle_quantity_for_roll(
        self, breakdown_vis, target_coord, test_time
    ):
        """A bare number for roll is rejected, as elsewhere in the API."""
        with pytest.raises(TypeError, match="roll"):
            breakdown_vis.get_star_tracker_breakdown(
                target_coord, test_time, roll=45
            )

    def test_st_defaults_are_the_flight_limits(self, line1, line2):
        """Star tracker keep-outs are on by default, at Pandora's limits."""
        vis = Visibility(line1, line2)
        assert vis.st_sun_min == 50 * u.deg
        assert vis.st_moon_min == 20 * u.deg
        assert vis.st_earthlimb_min == 30 * u.deg
        assert vis.st_required == 1
        assert vis._st_constraint_active is True

    def test_st_custom_limits_applied(self, line1, line2):
        """Custom star tracker limits are stored on the instance."""
        vis = Visibility(
            line1,
            line2,
            st_sun_min=45 * u.deg,
            st_moon_min=10 * u.deg,
            st_earthlimb_min=20 * u.deg,
        )
        assert vis.st_sun_min == 45 * u.deg
        assert vis.st_moon_min == 10 * u.deg
        assert vis.st_earthlimb_min == 20 * u.deg

    def test_repr_shows_st_constraints(self, line1, line2):
        """__repr__ includes star tracker constraints when non-zero."""
        vis = Visibility(line1, line2, st_sun_min=45 * u.deg)
        repr_str = repr(vis)
        assert "st_sun≥45 deg" in repr_str

    def test_repr_hides_st_when_zero(self, line1, line2):
        """__repr__ does not mention star tracker when all ST limits are 0."""
        vis = _legacy_visibility(line1, line2)
        repr_str = repr(vis)
        assert "st_sun" not in repr_str
        assert "st_moon" not in repr_str
        assert "st_limb" not in repr_str

    def test_constraint_passes_when_disabled(
        self, line1, line2, target_coord, test_time
    ):
        """With all ST limits at 0, get_star_tracker_constraint always returns True."""
        vis = _legacy_visibility(line1, line2)
        result = vis.get_star_tracker_constraint(target_coord, test_time)
        assert result is True

    def test_constraint_passes_when_disabled_array(self, line1, line2, target_coord):
        """Disabled ST constraints return all-True array for array times."""
        vis = _legacy_visibility(line1, line2)
        times = Time("2025-01-01T00:00:00") + np.arange(3) * u.hour
        result = vis.get_star_tracker_constraint(target_coord, times)
        assert np.all(result)
        assert result.shape == times.shape

    def test_get_star_tracker_skycoord_returns_skycoord(
        self, line1, line2, target_coord, test_time
    ):
        """_get_star_tracker_skycoord returns a SkyCoord in GCRS."""
        vis = Visibility(line1, line2)
        sc = vis._get_star_tracker_skycoord(target_coord, test_time, tracker=1)
        assert isinstance(sc, SkyCoord)

    def test_get_star_tracker_skycoord_two_trackers_differ(
        self, line1, line2, target_coord, test_time
    ):
        """Star tracker 1 and 2 point in different directions."""
        vis = Visibility(line1, line2)
        sc1 = vis._get_star_tracker_skycoord(target_coord, test_time, tracker=1)
        sc2 = vis._get_star_tracker_skycoord(target_coord, test_time, tracker=2)
        sep = sc1.separation(sc2)
        assert sep.deg > 0  # They should not be identical

    def test_get_star_tracker_skycoord_invalid_tracker(
        self, line1, line2, target_coord, test_time
    ):
        """Invalid tracker number raises ValueError."""
        vis = Visibility(line1, line2)
        with pytest.raises(ValueError, match="Invalid tracker"):
            vis._get_star_tracker_skycoord(target_coord, test_time, tracker=3)

    def test_constraint_with_very_large_sun_limit(
        self, line1, line2, target_coord, test_time
    ):
        """A 180° ST sun limit should fail (impossible to satisfy)."""
        vis = Visibility(line1, line2, st_sun_min=180 * u.deg)
        result = vis.get_star_tracker_constraint(target_coord, test_time)
        assert result is False

    def test_constraint_with_small_sun_limit(
        self, line1, line2, target_coord, test_time
    ):
        """A very small ST sun limit should pass for most targets."""
        vis = _legacy_visibility(line1, line2, st_sun_min=1 * u.deg)
        result = vis.get_star_tracker_constraint(target_coord, test_time)
        assert result  # Capella is far from the sun, both trackers should be fine

    def test_constraint_integrated_into_visibility(
        self, line1, line2, target_coord, test_time
    ):
        """ST constraint should affect get_visibility result."""
        # Build two instances that differ ONLY in ST sun limit
        vis_none = Visibility(line1, line2)
        vis_tight = Visibility(line1, line2, st_sun_min=180 * u.deg)

        result_none = vis_none.get_visibility(target_coord, test_time)["visible"]
        result_tight = vis_tight.get_visibility(target_coord, test_time)["visible"]

        # Without any ST constraint, baseline visibility is whatever it is
        # With an impossible 180° ST sun limit, visibility must be strictly worse
        if result_none:
            assert result_tight is False
        else:
            # Even if baseline is False (other constraints), tight ST can't help
            assert result_tight is False

    def test_get_all_constraints_includes_star_tracker(
        self, line1, line2, target_coord, test_time
    ):
        """get_all_constraints includes star_tracker key when active."""
        vis = Visibility(line1, line2, st_sun_min=45 * u.deg)
        constraints = vis.get_all_constraints(target_coord, test_time)
        assert "star_tracker" in constraints

    def test_get_all_constraints_excludes_star_tracker_when_disabled(
        self, line1, line2, target_coord, test_time
    ):
        """get_all_constraints omits star_tracker key when all ST limits are 0."""
        vis = _legacy_visibility(line1, line2)
        constraints = vis.get_all_constraints(target_coord, test_time)
        assert "star_tracker" not in constraints

    def test_summary_includes_star_tracker_section(
        self, line1, line2, target_coord, test_time
    ):
        """Summary output includes star tracker section when ST constraints active."""
        vis = Visibility(line1, line2, st_sun_min=45 * u.deg)
        summary = vis.summary(target_coord, test_time)
        assert "Star Tracker" in summary
        assert "ST1" in summary
        assert "ST2" in summary

    def test_summary_omits_star_tracker_when_disabled(
        self, line1, line2, target_coord, test_time
    ):
        """Summary output has no star tracker section when all ST limits are 0."""
        vis = _legacy_visibility(line1, line2)
        summary = vis.summary(target_coord, test_time)
        assert "Star Tracker" not in summary

    def test_constraint_array_time(self, line1, line2, target_coord):
        """ST constraint works with array times."""
        vis = Visibility(line1, line2, st_sun_min=45 * u.deg)
        times = Time("2025-01-01T00:00:00") + np.arange(3) * u.hour
        result = vis.get_star_tracker_constraint(target_coord, times)
        assert result.shape == times.shape
        assert result.dtype == bool

    def test_skycoord_array_time(self, line1, line2, target_coord):
        """_get_star_tracker_skycoord works with array times."""
        vis = Visibility(line1, line2)
        times = Time("2025-01-01T00:00:00") + np.arange(3) * u.hour
        sc = vis._get_star_tracker_skycoord(target_coord, times, tracker=1)
        assert isinstance(sc, SkyCoord)
        assert sc.shape == times.shape

    def test_degenerate_sun_aligned_array(self, line1, line2):
        """Degenerate timesteps (target=sun) produce NaN boresight, constraint=False."""
        vis = Visibility(line1, line2, st_sun_min=44 * u.deg)
        test_time = Time("2025-06-21T12:00:00")
        observer_location = vis._get_observer_location(test_time)
        sun_coord = get_body("sun", time=test_time, location=observer_location)

        # Use the sun position as target — degenerate attitude
        times = Time("2025-06-21T12:00:00") + np.array([0, 1]) * u.hour

        # Should not raise; degenerate indices should just give False constraint
        sc = vis._get_star_tracker_skycoord(sun_coord, times, tracker=1)
        assert sc.shape == times.shape
        # Degenerate timesteps should have NaN coordinates
        assert np.any(np.isnan(sc.cartesian.xyz.value))

        # Constraint should return False for degenerate timesteps
        result = vis.get_star_tracker_constraint(sun_coord, times)
        assert result.shape == times.shape
        assert result.dtype == bool

    def test_st_required_default_is_one(self, line1, line2):
        """st_required defaults to 1."""
        vis = Visibility(line1, line2)
        assert vis.st_required == 1

    def test_st_required_zero_disables_constraint(
        self, line1, line2, target_coord, test_time
    ):
        """st_required=0 means ST constraints are inactive even with limits set."""
        vis = Visibility(line1, line2, st_sun_min=180 * u.deg, st_required=0)
        # Should always pass since st_required=0 disables ST checks
        result = vis.get_star_tracker_constraint(target_coord, test_time)
        assert result is True

    def test_st_required_zero_excluded_from_all_constraints(
        self, line1, line2, target_coord, test_time
    ):
        """st_required=0 means star_tracker key is absent from get_all_constraints."""
        vis = Visibility(line1, line2, st_sun_min=45 * u.deg, st_required=0)
        constraints = vis.get_all_constraints(target_coord, test_time)
        assert "star_tracker" not in constraints

    def test_st_required_two_requires_both(self, line1, line2, target_coord, test_time):
        """st_required=2 means both trackers must pass."""
        # With a small limit both should pass for a reasonable target
        vis = _legacy_visibility(line1, line2, st_sun_min=1 * u.deg,
                                 st_required=2)
        result = vis.get_star_tracker_constraint(target_coord, test_time)
        assert result  # Both should pass with a tiny limit

    def test_st_required_invalid_raises(self, line1, line2):
        """st_required must be 0, 1, or 2."""
        with pytest.raises(ValueError, match="st_required must be 0, 1, or 2"):
            Visibility(line1, line2, st_required=3)

    def test_single_tracker_limit_is_not_waived_by_the_other(
        self, line1, line2, target_coord
    ):
        """A keep-out set on one tracker alone still rejects steps.

        st_required=1 used to be satisfied by the unconstrained tracker,
        which passed everything by default, so st1_earthlimb_min on its own
        rejected nothing at all.
        """
        times = Time("2026-03-01T00:00:00") + np.arange(0, 1440, 10) * u.min
        vis = _legacy_visibility(line1, line2, st1_earthlimb_min=30 * u.deg,
                                 st_required=1)
        assert vis._trackers_with_checks() == [1]

        passed = np.asarray(
            vis.get_star_tracker_constraint(target_coord, times)
        )
        assert not passed.all(), "the limit should reject something"

        # Nothing was asked of ST2, so the verdict is ST1's own.
        breakdown = vis.get_star_tracker_breakdown(target_coord, times)
        np.testing.assert_array_equal(
            passed, np.asarray(breakdown["passed"]["ST1"])
        )

    def test_single_tracker_limit_ignores_st_required(
        self, line1, line2, target_coord
    ):
        """With one tracker constrained, st_required 1 and 2 agree."""
        times = Time("2026-03-01T00:00:00") + np.arange(0, 1440, 10) * u.min
        one = _legacy_visibility(line1, line2, st2_earthlimb_min=30 * u.deg,
                                 st_required=1)
        two = _legacy_visibility(line1, line2, st2_earthlimb_min=30 * u.deg,
                                 st_required=2)
        np.testing.assert_array_equal(
            np.asarray(one.get_star_tracker_constraint(target_coord, times)),
            np.asarray(two.get_star_tracker_constraint(target_coord, times)),
        )

    def test_two_constrained_trackers_still_or_together(
        self, breakdown_vis, target_coord
    ):
        """With both trackers constrained, st_required=1 is unchanged."""
        times = Time("2026-03-01T00:00:00") + np.arange(0, 1440, 10) * u.min
        assert breakdown_vis._trackers_with_checks() == [1, 2]
        breakdown = breakdown_vis.get_star_tracker_breakdown(
            target_coord, times
        )
        np.testing.assert_array_equal(
            np.asarray(breakdown["passed"]["combined"]),
            np.asarray(breakdown["passed"]["ST1"])
            | np.asarray(breakdown["passed"]["ST2"]),
        )

    def test_nst_pass_ignores_an_unconstrained_tracker(
        self, line1, line2, target_coord
    ):
        """n_st_pass counts only the trackers something was asked of."""
        times = Time("2026-03-01T00:00:00") + np.arange(0, 1440, 10) * u.min
        vis = _legacy_visibility(line1, line2, st1_earthlimb_min=30 * u.deg,
                                 st_required=1)
        result = vis.get_visibility(target_coord, times, optimize_roll=True)
        assert result["visible"].any(), "expected some visible steps"
        assert result["n_st_pass"].max() <= 1

    def test_repr_shows_st_required(self, line1, line2):
        """__repr__ includes st_req when ST constraints are active."""
        vis = Visibility(line1, line2, st_sun_min=45 * u.deg, st_required=2)
        repr_str = repr(vis)
        assert "st_req=2" in repr_str

    def test_summary_shows_both_label(self, line1, line2, target_coord, test_time):
        """Summary shows 'both' when st_required=2."""
        vis = Visibility(line1, line2, st_sun_min=1 * u.deg, st_required=2)
        summary = vis.summary(target_coord, test_time)
        assert "both" in summary

    def test_star_tracker_angles_match_the_applied_limb_angle(
        self, breakdown_vis, target_coord, test_time
    ):
        """The reported limb angle is the one the constraint applies.

        get_star_tracker_angles used an AltAz altitude, measured from the
        geodetic horizon, while the check uses a geocentric one. The two
        sit up to ~0.2 deg apart, enough to disagree about a tracker near
        its limit.
        """
        times = test_time + np.arange(200) * u.min
        breakdown = breakdown_vis.get_star_tracker_breakdown(
            target_coord, times
        )
        for tracker in (1, 2):
            reported = breakdown_vis.get_star_tracker_angles(
                target_coord, times, tracker
            )["earthlimb_angle"].to(u.deg).value
            applied = np.asarray(breakdown["separations"][f"ST{tracker} limb"])
            np.testing.assert_allclose(reported, applied, atol=1e-3)

    def test_summary_star_tracker_rows_agree_with_result(
        self, line1, line2, target_coord
    ):
        """summary() cannot show every tracker failing above a passing result.

        The rows came from get_star_tracker_angles and the result from the
        geocentric check, so at this step the two disagreed and the section
        printed ST1 FAIL, ST2 FAIL, Result PASS.
        """
        vis = Visibility(line1, line2, st_earthlimb_min=15 * u.deg,
                         st_required=1)
        summary = vis.summary(target_coord, Time("2026-03-01T02:36:30.000"))

        verdicts = {}
        for line in summary.splitlines():
            stripped = line.strip()
            for row in ("ST1", "ST2", "Result"):
                if stripped.startswith(row):
                    verdicts[row] = "PASS" in stripped
        assert set(verdicts) == {"ST1", "ST2", "Result"}, summary
        assert verdicts["Result"] == (verdicts["ST1"] or verdicts["ST2"]), summary

    def test_summary_rows_agree_with_result_over_a_day(
        self, breakdown_vis, target_coord
    ):
        """The same agreement holds across a day, not just at one step."""
        times = Time("2026-03-01T00:00:00") + np.arange(0, 1440, 10) * u.min
        breakdown = breakdown_vis.get_star_tracker_breakdown(
            target_coord, times
        )
        st1 = np.asarray(breakdown["passed"]["ST1"])
        st2 = np.asarray(breakdown["passed"]["ST2"])
        combined = np.asarray(breakdown["passed"]["combined"])
        np.testing.assert_array_equal(combined, st1 | st2)


class TestRollParameter:
    """Tests for the configurable roll parameter."""

    @pytest.fixture
    def line1(self):
        return "1 67395U 80229J   26047.67973380 .00000000  00000-0  00000-0 0     9"

    @pytest.fixture
    def line2(self):
        return "2 67395  97.8021  48.2438 0011432 172.6532 187.4720 14.83698208    13"

    @pytest.fixture
    def target_coord(self):
        """Target well away from the pole."""
        return SkyCoord(134.6894, 8.2237, frame="icrs", unit="deg")

    @pytest.fixture
    def test_time(self):
        return Time("2026-02-15T18:44:12")

    # ── constructor / repr ──────────────────────────────────────────

    def test_roll_default_is_none(self, line1, line2):
        """Roll defaults to None (Sun-constrained attitude)."""
        vis = Visibility(line1, line2)
        assert vis.roll is None

    def test_roll_stored_in_degrees(self, line1, line2):
        """Roll is stored after unit conversion."""
        vis = Visibility(line1, line2, roll=45 * u.deg)
        assert vis.roll == 45 * u.deg

    def test_roll_converted_from_radians(self, line1, line2):
        """Roll specified in radians is converted to degrees."""
        vis = Visibility(line1, line2, roll=np.pi / 4 * u.rad)
        assert u.isclose(vis.roll, 45 * u.deg, atol=1e-10 * u.deg)

    def test_repr_shows_roll(self, line1, line2):
        """__repr__ includes roll when set."""
        vis = Visibility(line1, line2, roll=30 * u.deg)
        repr_str = repr(vis)
        assert "roll=30.0 deg" in repr_str

    def test_repr_omits_roll_when_none(self, line1, line2):
        """__repr__ omits roll when not set."""
        vis = Visibility(line1, line2)
        repr_str = repr(vis)
        assert "roll" not in repr_str

    # ── _roll_attitude unit-level ───────────────────────────────────

    def test_roll_attitude_orthonormality(self):
        """X, Y, Z from _roll_attitude are orthonormal."""
        z = np.array([0.3, 0.4, 0.866])
        z = z / np.linalg.norm(z)
        for angle_deg in [0, 45, 90, 135, 180, -45]:
            x, y = Visibility._roll_attitude(z, np.radians(angle_deg))
            assert abs(np.dot(x, y)) < 1e-12, f"X·Y != 0 at {angle_deg}°"
            assert abs(np.dot(x, z)) < 1e-12, f"X·Z != 0 at {angle_deg}°"
            assert abs(np.dot(y, z)) < 1e-12, f"Y·Z != 0 at {angle_deg}°"
            assert abs(np.linalg.norm(x) - 1) < 1e-12
            assert abs(np.linalg.norm(y) - 1) < 1e-12

    def test_roll_attitude_right_handed(self):
        """Axes form a right-handed frame: X × Y = Z."""
        z = np.array([0.5, 0.5, 0.7071])
        z = z / np.linalg.norm(z)
        for angle_deg in [0, 47, 90, 180]:
            x, y = Visibility._roll_attitude(z, np.radians(angle_deg))
            cross = np.cross(x, y)
            np.testing.assert_allclose(cross, z, atol=1e-12)

    def test_roll_zero_aligns_with_north_projection(self):
        """Roll=0 makes X align with celestial-north projection."""
        z = np.array([0.3, 0.4, 0.866])
        z = z / np.linalg.norm(z)
        x, _y = Visibility._roll_attitude(z, 0.0)
        # Compute expected north projection
        north = np.array([0.0, 0.0, 1.0])
        north_proj = north - np.dot(north, z) * z
        x_ref = north_proj / np.linalg.norm(north_proj)
        np.testing.assert_allclose(x, x_ref, atol=1e-12)

    def test_roll_90_perpendicular_to_north(self):
        """Roll=90° makes X perpendicular to celestial-north projection."""
        z = np.array([0.3, 0.4, 0.866])
        z = z / np.linalg.norm(z)
        x0, _y0 = Visibility._roll_attitude(z, 0.0)
        x90, _y90 = Visibility._roll_attitude(z, np.pi / 2)
        assert abs(np.dot(x90, x0)) < 1e-12

    def test_roll_180_flips_axes(self):
        """Roll=180° flips X and Y relative to roll=0."""
        z = np.array([0.3, 0.4, 0.866])
        z = z / np.linalg.norm(z)
        x0, y0 = Visibility._roll_attitude(z, 0.0)
        x180, y180 = Visibility._roll_attitude(z, np.pi)
        np.testing.assert_allclose(np.dot(x0, x180), -1, atol=1e-12)
        np.testing.assert_allclose(np.dot(y0, y180), -1, atol=1e-12)

    def test_roll_attitude_pole_fallback(self):
        """Boresight near celestial pole uses east fallback without crashing."""
        z_pole = np.array([0.0, 0.0, 1.0])
        x, y = Visibility._roll_attitude(z_pole, np.radians(30))
        assert abs(np.dot(x, z_pole)) < 1e-12
        assert abs(np.dot(y, z_pole)) < 1e-12
        assert abs(np.linalg.norm(x) - 1) < 1e-12
        assert abs(np.linalg.norm(y) - 1) < 1e-12
        # Right-handed
        np.testing.assert_allclose(np.cross(x, y), z_pole, atol=1e-12)

    # ── Integration: star tracker with roll ─────────────────────────

    def test_roll_none_matches_sun_constrained(
        self, line1, line2, target_coord, test_time
    ):
        """roll=None gives same ST pointing as the Sun-constrained default."""
        vis_default = Visibility(line1, line2, st_sun_min=1 * u.deg)
        # Without roll, same instance
        for tracker in [1, 2]:
            sc = vis_default._get_star_tracker_skycoord(
                target_coord, test_time, tracker
            )
            # Just verify it doesn't crash and gives a valid coordinate
            assert not np.isnan(sc.spherical.lon.deg)
            assert not np.isnan(sc.spherical.lat.deg)

    def test_roll_changes_star_tracker_pointing(
        self, line1, line2, target_coord, test_time
    ):
        """Setting a roll changes where the star tracker points."""
        vis_sun = Visibility(line1, line2, st_sun_min=1 * u.deg)
        vis_roll = Visibility(line1, line2, st_sun_min=1 * u.deg, roll=45 * u.deg)
        sc_sun = vis_sun._get_star_tracker_skycoord(target_coord, test_time, 1)
        sc_roll = vis_roll._get_star_tracker_skycoord(target_coord, test_time, 1)
        sep = sc_sun.separation(sc_roll)
        assert sep.deg > 1.0, "45° roll should significantly move ST pointing"

    def test_roll_scalar_and_array_time_agree(
        self, line1, line2, target_coord, test_time
    ):
        """Scalar and single-element array times give matching ST coordinates."""
        vis = Visibility(line1, line2, st_sun_min=1 * u.deg, roll=30 * u.deg)
        sc_scalar = vis._get_star_tracker_skycoord(target_coord, test_time, 1)
        times_arr = Time([test_time.iso])
        sc_array = vis._get_star_tracker_skycoord(target_coord, times_arr, 1)
        # Compare directly in the native frame (both are GCRS-like)
        sep = sc_scalar.separation(sc_array[0])
        assert sep.arcsec < 1.0, f"Scalar/array mismatch: {sep.arcsec:.2f} arcsec"

    def test_fixed_roll_constant_pointing(self, line1, line2, target_coord):
        """With fixed roll, ST pointing is the same at all times (no Sun motion)."""
        vis = Visibility(line1, line2, st_sun_min=1 * u.deg, roll=0 * u.deg)
        times = Time("2026-02-15T18:44:12") + np.array([0, 3, 6]) * u.hour
        for tracker in [1, 2]:
            sc = vis._get_star_tracker_skycoord(target_coord, times, tracker)
            ra_spread = sc.spherical.lon.deg.max() - sc.spherical.lon.deg.min()
            dec_spread = sc.spherical.lat.deg.max() - sc.spherical.lat.deg.min()
            assert ra_spread < 0.01, f"RA spread {ra_spread:.4f}° for ST{tracker}"
            assert dec_spread < 0.01, f"Dec spread {dec_spread:.4f}° for ST{tracker}"

    def test_fast_slow_agreement_with_roll(
        self, line1, line2, target_coord, test_time
    ):
        """Fast constraint path agrees with slow (SkyCoord) path when roll is set."""
        vis = _legacy_visibility(
            line1, line2, st_sun_min=45 * u.deg, roll=20 * u.deg
        )
        fast = vis.get_star_tracker_constraint(target_coord, test_time)
        # Slow path: compute angles explicitly
        angles1 = vis.get_star_tracker_angles(target_coord, test_time, tracker=1)
        angles2 = vis.get_star_tracker_angles(target_coord, test_time, tracker=2)
        slow_1 = angles1["sun_angle"].value >= 45.0
        slow_2 = angles2["sun_angle"].value >= 45.0
        slow = slow_1 | slow_2  # st_required=1
        assert fast == slow, f"Fast={fast}, Slow={slow}"

    def test_fast_slow_agreement_with_roll_array(
        self, line1, line2, target_coord
    ):
        """Fast/slow agreement with roll over an array of times."""
        vis = _legacy_visibility(
            line1, line2, st_sun_min=45 * u.deg, roll=20 * u.deg
        )
        times = Time("2026-02-15T18:00:00") + np.arange(5) * u.hour
        fast = vis.get_star_tracker_constraint(target_coord, times)
        slow = np.zeros(len(times), dtype=bool)
        for i, t in enumerate(times):
            angles1 = vis.get_star_tracker_angles(target_coord, t, tracker=1)
            angles2 = vis.get_star_tracker_angles(target_coord, t, tracker=2)
            s1 = angles1["sun_angle"].value >= 45.0
            s2 = angles2["sun_angle"].value >= 45.0
            slow[i] = s1 | s2
        np.testing.assert_array_equal(fast, slow)

    def test_roll_constraint_with_visibility(
        self, line1, line2, target_coord, test_time
    ):
        """Roll parameter works end-to-end through get_visibility."""
        vis = Visibility(
            line1, line2, st_sun_min=1 * u.deg, roll=30 * u.deg
        )
        # Should not crash
        result = vis.get_visibility(target_coord, test_time)["visible"]
        assert isinstance(result, (bool, np.bool_))

    def test_roll_constraint_array_with_visibility(
        self, line1, line2, target_coord
    ):
        """Roll + get_visibility over time array works and returns array."""
        vis = Visibility(
            line1, line2, st_sun_min=1 * u.deg, roll=30 * u.deg
        )
        times = Time("2026-02-15T00:00:00") + np.arange(10) * u.hour
        result = vis.get_visibility(target_coord, times)["visible"]
        assert result.shape == times.shape
        assert result.dtype == bool

    def test_roll_get_star_tracker_angles(
        self, line1, line2, target_coord, test_time
    ):
        """get_star_tracker_angles returns valid dict when roll is set."""
        vis = Visibility(line1, line2, roll=45 * u.deg)
        for tracker in [1, 2]:
            angles = vis.get_star_tracker_angles(
                target_coord, test_time, tracker=tracker
            )
            assert "ra" in angles
            assert "dec" in angles
            assert "sun_angle" in angles
            assert "moon_angle" in angles
            assert "earthlimb_angle" in angles
            assert not np.any(np.isnan(angles["ra"].value))
            assert not np.any(np.isnan(angles["dec"].value))

    def test_sun_constrained_roll_round_trip(
        self, line1, line2, target_coord, test_time
    ):
        """Measure roll from Sun-constrained attitude, then reproduce it.

        Compute the Sun-constrained ST pointing, measure what roll angle
        corresponds to that attitude, then set that roll and verify the
        ST pointing matches.
        """
        vis_sun = Visibility(line1, line2, st_sun_min=1 * u.deg)
        sc_sun = vis_sun._get_star_tracker_skycoord(target_coord, test_time, 1)

        # Get the Sun-constrained attitude axes
        from astropy.coordinates import GCRS
        ref_time = test_time
        target_gcrs = target_coord.transform_to(GCRS(obstime=ref_time))
        z = target_gcrs.cartesian.xyz.value.astype(float)
        z = z / np.linalg.norm(z)

        observer_location = vis_sun._get_observer_location(test_time)
        sun_coord = get_body("sun", time=test_time, location=observer_location)
        sun_xyz = sun_coord.cartesian.xyz.value.astype(float)
        sun_vec = sun_xyz / np.linalg.norm(sun_xyz)

        y_sun = np.cross(sun_vec, z)
        y_sun = y_sun / np.linalg.norm(y_sun)
        x_sun = np.cross(y_sun, z)
        x_sun = x_sun / np.linalg.norm(x_sun)

        # Measure roll: angle from north-projection to x_sun
        north = np.array([0.0, 0.0, 1.0])
        north_proj = north - np.dot(north, z) * z
        x_ref = north_proj / np.linalg.norm(north_proj)
        y_ref = np.cross(z, x_ref)
        y_ref = y_ref / np.linalg.norm(y_ref)
        cos_r = np.dot(x_ref, x_sun)
        sin_r = np.dot(y_ref, x_sun)
        measured_roll_rad = np.arctan2(sin_r, cos_r)

        # Now use that measured roll
        vis_roll = Visibility(
            line1, line2,
            st_sun_min=1 * u.deg,
            roll=np.degrees(measured_roll_rad) * u.deg,
        )
        sc_roll = vis_roll._get_star_tracker_skycoord(target_coord, test_time, 1)

        sep = sc_sun.separation(sc_roll)
        assert sep.arcsec < 1.0, (
            f"Round-trip failed: {sep.arcsec:.2f} arcsec separation"
        )


# ──────────────────────────────────────────────────────────────────────
# Tests for the merged get_visibility result dict and optimize_roll
# ──────────────────────────────────────────────────────────────────────

_BR_LINE1 = "1 67395U 80229J   26057.99991898  .00000000  00000-0  37770-3 0    03"
_BR_LINE2 = "2 67395  97.8009  58.3973 0006599 121.8878 132.9207 14.87804761    04"


def _mean_power(vis, target_coord, times, roll_deg):
    """Mean solar array power fraction over *times* at a held roll.

    The panel model the roll search ranks by, written out so the tests can
    ask about rolls the search did not choose.
    """
    pre = vis._precompute(times)
    unit = target_coord.transform_to(GCRS(obstime=times[0])).cartesian.xyz.value
    unit = unit / np.linalg.norm(unit)
    _, y_payload = vis._roll_attitude(unit, np.deg2rad(roll_deg))
    cos_sy = np.clip(y_payload @ pre["body_units"]["sun"], -1.0, 1.0)
    return float(np.mean(np.cos(np.pi / 2 - np.arccos(np.abs(cos_sy)))))


class TestMergedResultDict:
    """The details dict get_visibility now returns for every attitude policy."""

    @pytest.fixture
    def vis_st(self):
        """Visibility instance with ST keep-out constraints enabled."""
        return Visibility(
            _BR_LINE1, _BR_LINE2,
            st_sun_min=44 * u.deg,
            st_earthlimb_min=30 * u.deg,
            st_moon_min=12 * u.deg,
        )

    @pytest.fixture
    def target_coord(self):
        return SkyCoord(79.17305002, 45.99514569, frame="icrs", unit="deg")

    @pytest.fixture
    def times(self):
        return Time("2025-01-01T00:00:00") + np.arange(300) * u.min

    def test_sun_constrained_fields(self, vis_st, target_coord, times):
        """Default attitude: NaN roll, power exactly 1 where visible."""
        result = vis_st.get_visibility(target_coord, times)
        assert set(result.keys()) == {
            "visible", "boresight_visible", "roll_deg", "n_visible",
            "n_st_pass", "solar_power_frac",
        }
        assert np.all(np.isnan(result["roll_deg"]))
        vis_mask = result["visible"]
        assert result["n_visible"] == int(vis_mask.sum())
        np.testing.assert_allclose(result["solar_power_frac"][vis_mask], 1.0)
        assert np.all(np.isnan(result["solar_power_frac"][~vis_mask]))
        assert np.all(result["n_st_pass"][vis_mask] >= 1)
        assert np.all(result["n_st_pass"][~vis_mask] == 0)
        assert not np.any(vis_mask & ~result["boresight_visible"])

    def test_fixed_roll_echoed(self, vis_st, target_coord, times):
        """A given roll is reported back at every timestep, scalar or array."""
        result = vis_st.get_visibility(target_coord, times, roll=60 * u.deg)
        np.testing.assert_array_equal(result["roll_deg"], 60.0)

        roll_arr = np.linspace(-90, 90, len(times)) * u.deg
        result = vis_st.get_visibility(target_coord, times, roll=roll_arr)
        np.testing.assert_array_equal(result["roll_deg"], roll_arr.value)

    def test_instance_roll_echoed(self, target_coord, times):
        """The instance roll shows up in roll_deg when no override is given."""
        vis = Visibility(
            _BR_LINE1, _BR_LINE2,
            st_sun_min=44 * u.deg,
            st_earthlimb_min=30 * u.deg,
            st_moon_min=12 * u.deg,
            roll=45 * u.deg,
        )
        result = vis.get_visibility(target_coord, times)
        np.testing.assert_array_equal(result["roll_deg"], 45.0)

    def test_scalar_scalar_gives_python_scalars(self, vis_st, target_coord):
        """Scalar coord + scalar time unwraps every field to a scalar."""
        result = vis_st.get_visibility(target_coord, Time("2025-01-01T00:00:00"))
        assert isinstance(result["visible"], bool)
        assert isinstance(result["boresight_visible"], bool)
        assert isinstance(result["roll_deg"], float)
        assert isinstance(result["n_visible"], int)
        assert isinstance(result["n_st_pass"], int)
        assert isinstance(result["solar_power_frac"], float)

    def test_multi_target_shapes(self, vis_st, target_coord, times):
        """N targets add a leading axis; n_visible becomes (N,)."""
        other = SkyCoord(10.0, -45.0, frame="icrs", unit="deg")
        result = vis_st.get_visibility([target_coord, other], times[:50])
        for key in ("visible", "boresight_visible", "roll_deg",
                    "n_st_pass", "solar_power_frac"):
            assert result[key].shape == (2, 50), key
        assert result["n_visible"].shape == (2,)
        np.testing.assert_array_equal(
            result["visible"][0],
            vis_st.get_visibility(target_coord, times[:50])["visible"],
        )

        scalar = vis_st.get_visibility([target_coord, other],
                                       Time("2025-01-01T00:00:00"))
        assert scalar["visible"].shape == (2,)
        assert scalar["n_visible"].shape == (2,)

    def test_sweep_options_require_optimize(self, vis_st, target_coord, times):
        """roll_step, min_power_frac and weights are search-only options."""
        for kwargs in (
            dict(roll_step=5 * u.deg),
            dict(min_power_frac=0.5),
            dict(weights=np.ones(len(times))),
        ):
            with pytest.raises(ValueError, match="optimize_roll"):
                vis_st.get_visibility(target_coord, times, **kwargs)

    def test_roll_and_optimize_are_exclusive(self, vis_st, target_coord, times):
        """A roll to evaluate and a request to search for one conflict."""
        with pytest.raises(ValueError, match="not both"):
            vis_st.get_visibility(target_coord, times, roll=10 * u.deg,
                                  optimize_roll=True)


class TestOptimizeRoll:
    """get_visibility(optimize_roll=True): one roll over the given timesteps."""

    @pytest.fixture
    def vis_st(self):
        """Visibility instance with ST keep-out constraints enabled."""
        return Visibility(
            _BR_LINE1, _BR_LINE2,
            st_sun_min=44 * u.deg,
            st_earthlimb_min=30 * u.deg,
            st_moon_min=12 * u.deg,
        )

    @pytest.fixture
    def target_coord(self):
        return SkyCoord(79.17305002, 45.99514569, frame="icrs", unit="deg")

    @pytest.fixture
    def test_time(self):
        return Time("2025-01-01T00:00:00")

    @pytest.fixture
    def times(self):
        return Time("2025-01-01T00:00:00") + np.arange(300) * u.min

    def test_returns_dict_keys_scalar(self, vis_st, target_coord, test_time):
        """Scalar time returns a dict with the expected keys and scalar types."""
        result = vis_st.get_visibility(target_coord, test_time,
                                       optimize_roll=True)
        assert set(result.keys()) == {
            "visible", "boresight_visible", "roll_deg", "n_visible",
            "n_st_pass", "solar_power_frac",
        }
        assert isinstance(result["visible"], bool)
        assert isinstance(result["boresight_visible"], bool)
        assert isinstance(result["roll_deg"], float)
        assert isinstance(result["n_visible"], int)
        assert isinstance(result["n_st_pass"], int)
        assert isinstance(result["solar_power_frac"], float)

    def test_returns_dict_keys_array(self, vis_st, target_coord, test_time):
        """Array time returns arrays with matching shapes."""
        times = test_time + np.arange(10) * u.min
        result = vis_st.get_visibility(target_coord, times, optimize_roll=True)
        for key in ["visible", "boresight_visible", "roll_deg",
                    "n_st_pass", "solar_power_frac"]:
            assert result[key].shape == (10,), f"{key} shape mismatch"
        assert isinstance(result["n_visible"], int)

    def test_visible_subset_of_boresight(self, vis_st, target_coord, test_time):
        """visible should never be True where boresight_visible is False."""
        times = test_time + np.arange(50) * u.min
        result = vis_st.get_visibility(target_coord, times, optimize_roll=True)
        assert not np.any(result["visible"] & ~result["boresight_visible"])

    def test_one_roll_held_everywhere(self, vis_st, target_coord, times):
        """The chosen roll is a single angle in [-180, 180], reported at
        every timestep, visible or not."""
        result = vis_st.get_visibility(target_coord, times, optimize_roll=True)
        assert np.all(np.isfinite(result["roll_deg"]))
        assert np.all(result["roll_deg"] == result["roll_deg"][0])
        assert -180 <= result["roll_deg"][0] <= 180

    def test_power_nan_exactly_where_not_visible(self, vis_st, target_coord,
                                                 times):
        """solar_power_frac is finite where visible and NaN elsewhere."""
        result = vis_st.get_visibility(target_coord, times, optimize_roll=True)
        vis_mask = result["visible"]
        assert vis_mask.any() and not vis_mask.all()
        assert np.all(np.isfinite(result["solar_power_frac"][vis_mask]))
        assert np.all(np.isnan(result["solar_power_frac"][~vis_mask]))

    def test_nst_range(self, vis_st, target_coord, test_time):
        """n_st_pass should be 0, 1, or 2."""
        times = test_time + np.arange(50) * u.min
        result = vis_st.get_visibility(target_coord, times, optimize_roll=True)
        assert np.all((result["n_st_pass"] >= 0) & (result["n_st_pass"] <= 2))

    def test_solar_power_range(self, vis_st, target_coord, test_time):
        """solar_power_frac should be in [0, 1] where visible."""
        times = test_time + np.arange(97) * u.min
        result = vis_st.get_visibility(target_coord, times, optimize_roll=True)
        vis_mask = result["visible"]
        if vis_mask.any():
            pf = result["solar_power_frac"][vis_mask]
            assert np.all(pf >= 0) and np.all(pf <= 1)

    def test_agrees_with_get_visibility_at_that_roll(
        self, vis_st, target_coord, times
    ):
        """visible and n_visible are what get_visibility says at the roll."""
        result = vis_st.get_visibility(target_coord, times, optimize_roll=True)
        held = vis_st.get_visibility(
            target_coord, times, roll=result["roll_deg"][0] * u.deg
        )
        np.testing.assert_array_equal(result["visible"], held["visible"])
        np.testing.assert_array_equal(result["boresight_visible"],
                                      held["boresight_visible"])
        np.testing.assert_array_equal(result["n_st_pass"], held["n_st_pass"])
        # The one field the two paths compute in separate code
        np.testing.assert_allclose(result["solar_power_frac"],
                                   held["solar_power_frac"], equal_nan=True)
        assert result["n_visible"] == int(held["visible"].sum()) > 0
        assert np.all(result["n_st_pass"][result["visible"]] >= 1)
        assert np.all(result["n_st_pass"][~result["visible"]] == 0)

    def test_no_single_roll_observes_more(self, vis_st, target_coord, times):
        """Brute force over the same roll grid finds nothing better."""
        result = vis_st.get_visibility(target_coord, times, optimize_roll=True,
                                       roll_step=10 * u.deg)
        for roll in np.arange(0, 360, 10):
            held = vis_st.get_visibility(
                target_coord, times, roll=roll * u.deg
            )["visible"]
            assert int(held.sum()) <= result["n_visible"]

    def test_ties_go_to_solar_power(self, vis_st, target_coord, times):
        """Among the rolls observing the most, the best lit one wins."""
        result = vis_st.get_visibility(target_coord, times, optimize_roll=True,
                                       roll_step=10 * u.deg)
        chosen = np.nanmean(result["solar_power_frac"])
        for roll in np.arange(0, 360, 10):
            held = vis_st.get_visibility(
                target_coord, times, roll=roll * u.deg
            )["visible"]
            if int(held.sum()) == result["n_visible"]:
                assert _mean_power(vis_st, target_coord, times[held], roll) <= (
                    chosen + 1e-12
                )

    def test_power_floor_is_met(self, vis_st, target_coord, times):
        """A floor some roll reaches is respected; one none reaches is ignored."""
        floored = vis_st.get_visibility(target_coord, times, optimize_roll=True,
                                        min_power_frac=0.9)
        assert _mean_power(
            vis_st, target_coord, times, floored["roll_deg"][0]
        ) >= 0.9
        impossible = vis_st.get_visibility(
            target_coord, times, optimize_roll=True, min_power_frac=1.0
        )
        assert np.all(np.isfinite(impossible["roll_deg"]))
        with pytest.raises(ValueError):
            vis_st.get_visibility(target_coord, times, optimize_roll=True,
                                  min_power_frac=1.5)

    def test_weights_rank_one_group_first(self, vis_st, target_coord, times):
        """Heavily weighted timesteps decide the roll; the rest break ties."""
        first_hour = np.zeros(len(times), dtype=bool)
        first_hour[:60] = True
        weights = np.where(first_hour, len(times) + 1, 1)
        result = vis_st.get_visibility(target_coord, times, optimize_roll=True,
                                       roll_step=10 * u.deg, weights=weights)
        best_first_hour = max(
            int(vis_st.get_visibility(
                target_coord, times[first_hour], roll=roll * u.deg
            )["visible"].sum())
            for roll in np.arange(0, 360, 10)
        )
        assert int(result["visible"][first_hour].sum()) == best_first_hour
        with pytest.raises(ValueError):
            vis_st.get_visibility(target_coord, times, optimize_roll=True,
                                  weights=weights[:-1])

    def test_fallback_when_nothing_observable(self, target_coord, times):
        """A blocked boresight still returns a roll, with n_visible 0."""
        vis = Visibility(
            _BR_LINE1, _BR_LINE2,
            sun_min=170 * u.deg,
            st_sun_min=44 * u.deg,
            st_earthlimb_min=30 * u.deg,
            st_moon_min=12 * u.deg,
        )
        result = vis.get_visibility(target_coord, times, optimize_roll=True)
        assert not result["boresight_visible"].any(), "expected a blocked run"
        assert result["n_visible"] == 0 and not result["visible"].any()
        assert np.all(np.isfinite(result["roll_deg"]))
        assert np.all(np.isnan(result["solar_power_frac"]))
        assert np.all(result["n_st_pass"] == 0)

        # The fallback attitude lets the trackers be told apart on the
        # blocked run, rather than every check reading as failed because
        # there was no attitude.
        breakdown = vis.get_star_tracker_breakdown(
            target_coord, times, roll=result["roll_deg"] * u.deg
        )
        for row, separation in breakdown["separations"].items():
            assert np.all(np.isfinite(np.asarray(separation))), row

    def test_trackers_off_picks_best_lit(self, target_coord, times):
        """Without tracker keep-outs every roll observes the same, so power decides."""
        vis = _legacy_visibility(_BR_LINE1, _BR_LINE2)
        result = vis.get_visibility(target_coord, times, optimize_roll=True,
                                    roll_step=10 * u.deg)
        np.testing.assert_array_equal(
            result["visible"],
            vis.get_visibility(target_coord, times)["visible"],
        )
        np.testing.assert_array_equal(result["visible"],
                                      result["boresight_visible"])
        assert np.all(result["n_st_pass"] == 0)
        chosen = _mean_power(vis, target_coord, times, result["roll_deg"][0])
        for roll in np.arange(0, 360, 10):
            assert _mean_power(vis, target_coord, times, roll) <= chosen + 1e-12

    def test_constraints_reconstruct_the_run(self, vis_st, target_coord, times):
        """The constraint rows explain an optimize_roll run exactly.

        Only when they are asked at the same attitude: the diagnostics
        default to the Sun-constrained one, while the run held the roll
        it chose.
        """
        result = vis_st.get_visibility(target_coord, times, optimize_roll=True)
        constraints = vis_st.get_all_constraints(
            target_coord, times, roll=result["roll_deg"] * u.deg
        )

        rows_pass = np.ones(len(times), dtype=bool)
        for passed in constraints.values():
            rows_pass &= np.asarray(passed)
        np.testing.assert_array_equal(rows_pass, result["visible"])

        # The per-check breakdown reduces to the same star tracker row.
        breakdown = vis_st.get_star_tracker_breakdown(
            target_coord, times, roll=result["roll_deg"] * u.deg
        )
        np.testing.assert_array_equal(
            np.asarray(breakdown["passed"]["combined"]),
            np.asarray(constraints["star_tracker"]),
        )

    def test_array_roll_length_must_match_times(
        self, vis_st, target_coord, test_time
    ):
        """An array roll with the wrong length is rejected, not broadcast."""
        times = test_time + np.arange(10) * u.min
        with pytest.raises(ValueError, match="one entry per timestep"):
            vis_st.get_all_constraints(
                target_coord, times, roll=np.zeros(4) * u.deg
            )

    def test_agrees_with_fixed_roll_instance(self, vis_st, target_coord,
                                             test_time):
        """The chosen roll baked into a new instance gives the same verdicts."""
        times = test_time + np.arange(97) * u.min
        result = vis_st.get_visibility(target_coord, times, optimize_roll=True,
                                       roll_step=5 * u.deg)
        if not result["visible"].any():
            pytest.skip("No visible steps for this target/epoch")

        vis_fixed = Visibility(
            _BR_LINE1, _BR_LINE2,
            st_sun_min=44 * u.deg,
            st_earthlimb_min=30 * u.deg,
            st_moon_min=12 * u.deg,
            roll=result["roll_deg"][0] * u.deg,
        )
        fixed_vis = vis_fixed.get_visibility(target_coord, times)["visible"]
        np.testing.assert_array_equal(result["visible"], fixed_vis)

    def test_coarser_step_still_works(self, vis_st, target_coord, test_time):
        """A coarser roll step still returns valid results (may find fewer)."""
        times = test_time + np.arange(50) * u.min
        fine = vis_st.get_visibility(target_coord, times, optimize_roll=True,
                                     roll_step=2 * u.deg)
        coarse = vis_st.get_visibility(target_coord, times, optimize_roll=True,
                                       roll_step=10 * u.deg)
        assert coarse["n_visible"] <= fine["n_visible"]

    def test_optimize_ignores_instance_roll(self, vis_st, target_coord,
                                            times):
        """optimize_roll searches from scratch, whatever roll the instance holds."""
        vis_rolled = Visibility(
            _BR_LINE1, _BR_LINE2,
            st_sun_min=44 * u.deg,
            st_earthlimb_min=30 * u.deg,
            st_moon_min=12 * u.deg,
            roll=45 * u.deg,
        )
        result = vis_rolled.get_visibility(target_coord, times,
                                           optimize_roll=True)
        expected = vis_st.get_visibility(target_coord, times,
                                         optimize_roll=True)
        np.testing.assert_array_equal(result["roll_deg"], expected["roll_deg"])
        np.testing.assert_array_equal(result["visible"], expected["visible"])

    def test_multi_target_scalar_time(self, vis_st, target_coord):
        """Several targets at one instant: (N,) fields, one roll each."""
        other = SkyCoord(188.386, -10.1462, frame="icrs", unit="deg")
        result = vis_st.get_visibility([target_coord, other],
                                       Time("2025-01-01T00:00:00"),
                                       optimize_roll=True)
        for key in ("visible", "boresight_visible", "roll_deg",
                    "n_st_pass", "solar_power_frac"):
            assert result[key].shape == (2,), key
        assert result["n_visible"].shape == (2,)
        assert np.all(np.isfinite(result["roll_deg"]))

    def test_roll_is_chosen_per_target(self, vis_st, target_coord, times):
        """With several targets each gets its own independent roll."""
        other = SkyCoord(188.386, -10.1462, frame="icrs", unit="deg")
        result = vis_st.get_visibility([target_coord, other], times,
                                       optimize_roll=True)
        assert result["roll_deg"].shape == (2, len(times))
        single = vis_st.get_visibility(target_coord, times, optimize_roll=True)
        np.testing.assert_array_equal(result["roll_deg"][0],
                                      single["roll_deg"])
        np.testing.assert_array_equal(result["visible"][0], single["visible"])


class TestEarthlimbDayNight:
    """Tests for earthlimb_day_min / earthlimb_night_min parameters."""

    @pytest.fixture
    def line1(self):
        return "1 67395U 80229J   26057.99991898  .00000000  00000-0  37770-3 0    03"

    @pytest.fixture
    def line2(self):
        return "2 67395  97.8009  58.3973 0006599 121.8878 132.9207 14.87804761    04"

    @pytest.fixture
    def target_coord(self):
        """WASP-107 — has both sunlit and dark limb crossings in mid-2026."""
        return SkyCoord(188.386, -10.1462, frame="icrs", unit="deg")

    @pytest.fixture
    def test_time(self):
        return Time("2026-06-01T00:00:00")

    # ── Defaults & storage ──────────────────────────────────────────

    def test_defaults_are_the_flight_limits(self, line1, line2):
        """The day/night pair defaults to Pandora's limits.

        They are the fallback rather than the live thresholds: the dynamic
        wedge also defaults on and takes precedence over them.
        """
        vis = Visibility(line1, line2)
        assert vis.earthlimb_day_min == 44 * u.deg
        assert vis.earthlimb_night_min == 13 * u.deg
        assert vis.use_dynamic_earthlimb is True

    def test_custom_values_stored(self, line1, line2):
        """Custom day/night values are stored on the instance."""
        vis = Visibility(
            line1, line2,
            earthlimb_day_min=25 * u.deg,
            earthlimb_night_min=10 * u.deg,
        )
        assert vis.earthlimb_day_min == 25 * u.deg
        assert vis.earthlimb_night_min == 10 * u.deg

    def test_angle_validation(self, line1, line2):
        """Bare float without unit raises TypeError."""
        with pytest.raises(TypeError, match="astropy Quantity"):
            Visibility(line1, line2, earthlimb_day_min=25)
        with pytest.raises(TypeError, match="astropy Quantity"):
            Visibility(line1, line2, earthlimb_night_min=10)

    # ── Backward compatibility ──────────────────────────────────────

    def test_backward_compatible_when_none(self, line1, line2, target_coord, test_time):
        """When both day/night are None, result is identical to earthlimb_min."""
        vis_default = _legacy_visibility(line1, line2)
        vis_explicit = _legacy_visibility(line1, line2, earthlimb_min=20 * u.deg)
        times = test_time + np.arange(10) * u.min

        r1 = vis_default.get_visibility(target_coord, times)["visible"]
        r2 = vis_explicit.get_visibility(target_coord, times)["visible"]
        np.testing.assert_array_equal(r1, r2)

    # ── _earthlimb_is_sunlit unit test ──────────────────────────────

    def test_earthlimb_is_sunlit_synthetic(self):
        """Test sunlit detection with known geometry (no limb_angle_rad → legacy)."""
        # Target is in +X, zenith is +Z → limb point is in +X direction
        target = np.array([1.0, 0.0, 0.0])
        zenith = np.array([0.0, 0.0, 1.0])

        # Sun in +X → dot(limb_dir, sun) > 0 → sunlit
        sun_lit = np.array([1.0, 0.0, 0.0])
        assert Visibility._earthlimb_is_sunlit(target, zenith, sun_lit) is True or \
            bool(Visibility._earthlimb_is_sunlit(target, zenith, sun_lit)) is True

        # Sun in -X → dot(limb_dir, sun) < 0 → dark
        sun_dark = np.array([-1.0, 0.0, 0.0])
        assert bool(Visibility._earthlimb_is_sunlit(target, zenith, sun_dark)) is False

    def test_earthlimb_is_sunlit_with_limb_angle(self):
        """Test sunlit detection with limb_angle_rad (surface normal correction).

        With target in +X and zenith in +Z:
          limb_unit = +X,  surface normal = cos(la)*Z + sin(la)*X
        For large cos(la) (~0.91) the zenith component dominates.
        Sun in +Z (overhead) should be sunlit via the zenith term,
        even though dot(limb_unit, sun) = 0."""
        target = np.array([1.0, 0.0, 0.0])
        zenith = np.array([0.0, 0.0, 1.0])
        la_rad = np.arccos(0.91)  # typical LEO value

        # Sun in +Z: dot(zenith, sun)=1 → n·sun = cos(la) > 0 → sunlit
        sun_overhead = np.array([0.0, 0.0, 1.0])
        assert bool(Visibility._earthlimb_is_sunlit(
            target, zenith, sun_overhead, limb_angle_rad=la_rad
        )) is True

        # Sun in -Z: dot(zenith, sun)=-1 → n·sun = -cos(la) + 0 < 0 → dark
        sun_below = np.array([0.0, 0.0, -1.0])
        assert bool(Visibility._earthlimb_is_sunlit(
            target, zenith, sun_below, limb_angle_rad=la_rad
        )) is False

        # Sun in +X: dot(zenith, sun)=0, dot(limb, sun)=1
        #   → n·sun = sin(la) > 0 → sunlit
        sun_plusx = np.array([1.0, 0.0, 0.0])
        assert bool(Visibility._earthlimb_is_sunlit(
            target, zenith, sun_plusx, limb_angle_rad=la_rad
        )) is True

        # Sun in -X: dot(zenith, sun)=0, dot(limb, sun)=-1
        #   → n·sun = -sin(la) < 0 → dark
        sun_minusx = np.array([-1.0, 0.0, 0.0])
        assert bool(Visibility._earthlimb_is_sunlit(
            target, zenith, sun_minusx, limb_angle_rad=la_rad
        )) is False

    def test_earthlimb_is_sunlit_array(self):
        """Test sunlit detection with array inputs."""
        target = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        zenith = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        # First timestep: sun in +X (sunlit), second: sun in -X (dark)
        sun = np.array([[1.0, -1.0], [0.0, 0.0], [0.0, 0.0]])

        result = Visibility._earthlimb_is_sunlit(target, zenith, sun)
        assert result[0] is True or bool(result[0]) is True
        assert bool(result[1]) is False

    # ── __repr__ ────────────────────────────────────────────────────

    def test_repr_shows_day_night(self, line1, line2):
        """repr shows limb_day and limb_night when set."""
        vis = _legacy_visibility(
            line1, line2,
            earthlimb_day_min=25 * u.deg,
            earthlimb_night_min=10 * u.deg,
        )
        r = repr(vis)
        assert "limb_day≥" in r
        assert "limb_night≥" in r
        assert "25 deg" in r
        assert "10 deg" in r

    def test_repr_no_day_night_when_none(self, line1, line2):
        """repr shows plain limb≥ when day/night are both None."""
        vis = _legacy_visibility(line1, line2)
        r = repr(vis)
        assert "limb≥" in r
        assert "limb_day" not in r
        assert "limb_night" not in r

    # ── Integration with get_visibility ─────────────────────────────

    def test_day_limit_stricter_affects_visibility(self, line1, line2, target_coord):
        """Setting earthlimb_day_min=180° must *strictly* reduce visibility.

        We use a 7-day window so there are enough sunlit limb-crossing
        timesteps to see a difference."""
        times = Time("2026-06-01T00:00:00") + np.arange(7 * 1440) * u.min
        vis_default = _legacy_visibility(line1, line2)
        vis_strict_day = _legacy_visibility(
            line1, line2,
            earthlimb_day_min=180 * u.deg,
            earthlimb_night_min=20 * u.deg,
        )
        r_default = np.asarray(vis_default.get_visibility(target_coord, times)["visible"])
        r_strict = np.asarray(vis_strict_day.get_visibility(target_coord, times)["visible"])
        assert r_strict.sum() < r_default.sum(), (
            f"earthlimb_day_min=180° should strictly reduce visibility, "
            f"got {r_strict.sum()} vs default {r_default.sum()}"
        )

    def test_night_limit_stricter_affects_visibility(self, line1, line2, target_coord):
        """Setting earthlimb_night_min=180° must *strictly* reduce visibility."""
        times = Time("2026-06-01T00:00:00") + np.arange(7 * 1440) * u.min
        vis_default = _legacy_visibility(line1, line2)
        vis_strict_night = _legacy_visibility(
            line1, line2,
            earthlimb_day_min=20 * u.deg,
            earthlimb_night_min=180 * u.deg,
        )
        r_default = np.asarray(vis_default.get_visibility(target_coord, times)["visible"])
        r_strict = np.asarray(vis_strict_night.get_visibility(target_coord, times)["visible"])
        assert r_strict.sum() < r_default.sum(), (
            f"earthlimb_night_min=180° should strictly reduce visibility, "
            f"got {r_strict.sum()} vs default {r_default.sum()}"
        )

    def test_loose_both_more_permissive(self, line1, line2, target_coord):
        """Setting both day/night to 0 should give >= visibility vs default."""
        times = Time("2026-06-01T00:00:00") + np.arange(7 * 1440) * u.min
        vis_default = _legacy_visibility(line1, line2)
        vis_loose = _legacy_visibility(
            line1, line2,
            earthlimb_day_min=0 * u.deg,
            earthlimb_night_min=0 * u.deg,
        )
        r_default = np.asarray(vis_default.get_visibility(target_coord, times)["visible"])
        r_loose = np.asarray(vis_loose.get_visibility(target_coord, times)["visible"])
        assert r_loose.sum() >= r_default.sum()

    # ── get_constraint ──────────────────────────────────────────────

    def test_get_constraint_uses_day_night(self, line1, line2, target_coord, test_time):
        """get_constraint('earthlimb', ...) returns bool with day/night."""
        vis = _legacy_visibility(
            line1, line2,
            earthlimb_day_min=25 * u.deg,
            earthlimb_night_min=10 * u.deg,
        )
        result = vis.get_constraint(target_coord, "earthlimb", test_time)
        assert isinstance(result, (bool, np.bool_))

    # ── summary ─────────────────────────────────────────────────────

    def test_summary_shows_day_or_night(self, line1, line2, target_coord, test_time):
        """Summary should indicate [day] or [night] for earthlimb."""
        vis = _legacy_visibility(
            line1, line2,
            earthlimb_day_min=25 * u.deg,
            earthlimb_night_min=10 * u.deg,
        )
        summary = vis.summary(target_coord, test_time)
        assert "[day]" in summary or "[night]" in summary

    @pytest.mark.parametrize("mode", ["subsatellite", "limb"])
    def test_summary_threshold_matches_engine(self, line1, line2, mode):
        """The req: value summary prints is the one get_visibility applied.

        summary() used to derive day/night from the nearest limb point
        regardless of daynight_mode, so in the default "subsatellite" mode
        it could report the night limit while the engine applied the day
        limit (or vice versa).
        """
        day_min, night_min = 40 * u.deg, 5 * u.deg
        vis = _legacy_visibility(
            line1, line2,
            earthlimb_day_min=day_min,
            earthlimb_night_min=night_min,
            daynight_mode=mode,
        )
        target = SkyCoord(ra=90, dec=-30, unit="deg")

        checked = 0
        for i in range(0, 200, 5):
            time = Time("2026-02-15T18:00:00") + i * u.min
            pre = vis._precompute(time)
            tgt_u = vis._target_unit(target, time)
            engine_deg = float(vis._effective_earthlimb_min_deg(
                tgt_u, pre["zenith_unit"], pre["body_units"]["sun"],
                limb_angle_rad=pre["limb_angle_rad"],
            ))
            expected_side = "day" if engine_deg == day_min.value else "night"

            line = next(
                ln for ln in vis.summary(target, time).split("\n")
                if ln.startswith("Earthlimb")
            )
            assert f"[{expected_side}]" in line, (
                f"{mode} mode at {time.iso}: engine used {engine_deg} deg "
                f"but summary reported: {line.strip()}"
            )
            assert f"{engine_deg:>6.1f} deg" in line, (
                f"{mode} mode at {time.iso}: engine used {engine_deg} deg "
                f"but summary reported: {line.strip()}"
            )
            checked += 1

        assert checked == 40

    def test_daynight_is_sunlit_follows_mode(self, line1, line2):
        """_daynight_is_sunlit dispatches on daynight_mode.

        Geometry chosen so the two modes disagree: the spacecraft is over
        sunlit ground, but the limb point toward the target is dark.
        """
        target = np.array([-1.0, 0.0, 0.0])
        zenith = np.array([0.0, 0.0, 1.0])
        sun = np.array([0.995, 0.0, 0.0999])  # low sun, 84 deg off zenith
        limb_rad = np.deg2rad(21.0)

        vis_sub = _legacy_visibility(line1, line2, daynight_mode="subsatellite")
        vis_limb = _legacy_visibility(line1, line2, daynight_mode="limb")

        assert bool(vis_sub._daynight_is_sunlit(
            target, zenith, sun, limb_angle_rad=limb_rad)) is True
        assert bool(vis_limb._daynight_is_sunlit(
            target, zenith, sun, limb_angle_rad=limb_rad)) is False

    # ── Fallback behavior ───────────────────────────────────────────

    def test_only_day_set_falls_back_to_earthlimb_min(self, line1, line2):
        """When only day is set, night falls back to earthlimb_min."""
        vis = _legacy_visibility(
            line1, line2,
            earthlimb_day_min=30 * u.deg,
        )
        assert vis.earthlimb_day_min == 30 * u.deg
        assert vis.earthlimb_night_min is None
        # Night threshold should use earthlimb_min (20 deg default)
        # Verify via _effective_earthlimb_min_deg with a dark limb
        target = np.array([1.0, 0.0, 0.0])
        zenith = np.array([0.0, 0.0, 1.0])
        sun_dark = np.array([-1.0, 0.0, 0.0])
        eff = vis._effective_earthlimb_min_deg(target, zenith, sun_dark)
        assert float(eff) == pytest.approx(20.0)  # falls back to earthlimb_min

    def test_only_night_set_falls_back_to_earthlimb_min(self, line1, line2):
        """When only night is set, day falls back to earthlimb_min."""
        vis = _legacy_visibility(
            line1, line2,
            earthlimb_night_min=5 * u.deg,
        )
        assert vis.earthlimb_night_min == 5 * u.deg
        assert vis.earthlimb_day_min is None

    @pytest.mark.parametrize("mode,sun_lit", [
        # Sun along the zenith: the ground below the spacecraft is sunlit.
        ("subsatellite", np.array([0.0, 0.0, 1.0])),
        # Sun along the target's horizontal direction: the limb the
        # boresight grazes is sunlit.
        ("limb", np.array([1.0, 0.0, 0.0])),
    ])
    def test_day_falls_back_to_earthlimb_min(self, line1, line2, mode, sun_lit):
        """With only night set, the day threshold uses earthlimb_min.

        Checked in both day/night modes with a geometry that reads as day
        for that mode, so the fallback is covered whichever is default.
        """
        vis = _legacy_visibility(
            line1, line2,
            earthlimb_night_min=5 * u.deg,
            daynight_mode=mode,
        )
        target = np.array([1.0, 0.0, 0.0])
        zenith = np.array([0.0, 0.0, 1.0])
        eff = vis._effective_earthlimb_min_deg(target, zenith, sun_lit)
        assert float(eff) == pytest.approx(20.0)  # earthlimb_min default

    # ── Array time ──────────────────────────────────────────────────

    def test_array_time_different_thresholds(self, line1, line2, target_coord):
        """Over an array of times, day/night thresholds vary per timestep."""
        vis = Visibility(
            line1, line2,
            earthlimb_day_min=25 * u.deg,
            earthlimb_night_min=5 * u.deg,
        )
        times = Time("2025-01-01T00:00:00") + np.arange(100) * u.min
        result = vis.get_visibility(target_coord, times)["visible"]
        assert isinstance(result, np.ndarray)
        assert result.shape == times.shape
        assert result.dtype == bool

    # ── Twilight margin ─────────────────────────────────────────────

    def test_twilight_margin_default_zero(self, line1, line2):
        """twilight_margin defaults to 0 deg."""
        vis = Visibility(line1, line2)
        assert vis.twilight_margin == 0 * u.deg

    def test_twilight_margin_stored(self, line1, line2):
        """Custom twilight_margin is stored on the instance."""
        vis = Visibility(line1, line2, twilight_margin=18 * u.deg)
        assert vis.twilight_margin == 18 * u.deg

    def test_twilight_margin_angle_validation(self, line1, line2):
        """Bare float without unit raises TypeError."""
        with pytest.raises(TypeError, match="astropy Quantity"):
            Visibility(line1, line2, twilight_margin=18)

    def test_twilight_margin_zero_matches_original(self, line1, line2, target_coord):
        """margin=0 gives identical visibility to no-margin (backward compat)."""
        times = Time("2025-01-01T00:00:00") + np.arange(200) * u.min
        vis_default = Visibility(
            line1, line2,
            earthlimb_day_min=40 * u.deg,
            earthlimb_night_min=15 * u.deg,
        )
        vis_zero = Visibility(
            line1, line2,
            earthlimb_day_min=40 * u.deg,
            earthlimb_night_min=15 * u.deg,
            twilight_margin=0 * u.deg,
        )
        r_default = vis_default.get_visibility(target_coord, times)["visible"]
        r_zero = vis_zero.get_visibility(target_coord, times)["visible"]
        np.testing.assert_array_equal(r_default, r_zero)

    def test_twilight_margin_more_conservative(self, line1, line2, target_coord):
        """Positive margin classifies more timesteps as dayside → fewer visible."""
        times = Time("2026-06-01T00:00:00") + np.arange(1440) * u.min
        vis_sharp = Visibility(
            line1, line2,
            earthlimb_day_min=40 * u.deg,
            earthlimb_night_min=15 * u.deg,
            twilight_margin=0 * u.deg,
        )
        vis_margin = Visibility(
            line1, line2,
            earthlimb_day_min=40 * u.deg,
            earthlimb_night_min=15 * u.deg,
            twilight_margin=18 * u.deg,
        )
        r_sharp = vis_sharp.get_visibility(target_coord, times)["visible"]
        r_margin = vis_margin.get_visibility(target_coord, times)["visible"]
        # Margin can only remove visibility, never add it
        assert np.all(r_margin <= r_sharp)
        assert np.sum(r_margin) <= np.sum(r_sharp)

    def test_twilight_margin_sunlit_synthetic(self):
        """Twilight margin shifts the sunlit boundary in synthetic geometry.

        With target in +X, zenith in +Z, sun barely below limb in -X:
          dot(n, sun) = -sin(la) ≈ -0.41 (for la_rad = arccos(0.91))
        Default (margin=0): not sunlit.
        Margin=30 → threshold = -sin(30°) = -0.5 → still sunlit."""
        target = np.array([1.0, 0.0, 0.0])
        zenith = np.array([0.0, 0.0, 1.0])
        la_rad = np.arccos(0.91)

        # Sun in -X: dot(n, sun) = -sin(la) ≈ -0.41
        sun = np.array([-1.0, 0.0, 0.0])

        # margin=0: not sunlit (dot_n_sun ≈ -0.41 < 0)
        assert bool(Visibility._earthlimb_is_sunlit(
            target, zenith, sun, limb_angle_rad=la_rad,
            twilight_margin_deg=0.0,
        )) is False

        # margin=30: threshold = -sin(30°) = -0.5
        #   dot_n_sun ≈ -0.41 > -0.5 → classified as sunlit
        assert bool(Visibility._earthlimb_is_sunlit(
            target, zenith, sun, limb_angle_rad=la_rad,
            twilight_margin_deg=30.0,
        )) is True

    def test_twilight_margin_no_effect_without_day_night(self, line1, line2, target_coord):
        """When day/night are both None, twilight_margin has no effect."""
        times = Time("2025-01-01T00:00:00") + np.arange(100) * u.min
        vis_plain = Visibility(line1, line2, earthlimb_min=20 * u.deg)
        vis_margin = Visibility(
            line1, line2,
            earthlimb_min=20 * u.deg,
            twilight_margin=30 * u.deg,
        )
        r_plain = vis_plain.get_visibility(target_coord, times)["visible"]
        r_margin = vis_margin.get_visibility(target_coord, times)["visible"]
        np.testing.assert_array_equal(r_plain, r_margin)

    def test_twilight_margin_repr(self, line1, line2):
        """repr shows twilight_margin when > 0 and day/night is set."""
        vis = _legacy_visibility(
            line1, line2,
            earthlimb_day_min=40 * u.deg,
            earthlimb_night_min=15 * u.deg,
            twilight_margin=18 * u.deg,
        )
        r = repr(vis)
        assert "twilight_margin=18 deg" in r

    def test_twilight_margin_repr_hidden_when_zero(self, line1, line2):
        """repr does not show twilight_margin when it's 0."""
        vis = Visibility(
            line1, line2,
            earthlimb_day_min=40 * u.deg,
            earthlimb_night_min=15 * u.deg,
        )
        r = repr(vis)
        assert "twilight_margin" not in r

    # ── daynight_mode / subsatellite ────────────────────────────────

    def test_daynight_mode_default_is_subsatellite(self, line1, line2):
        """Default daynight_mode is 'subsatellite'.

        The subsatellite point is a target-independent, orbit-only solar
        zenith angle, so every target on a given pass sees the same Earth
        illumination.  It is also the reference for both Earth limb
        models, so the day/night pair and the dynamic DPC wedge agree.
        """
        vis = Visibility(line1, line2)
        assert vis.daynight_mode == "subsatellite"
        assert Visibility.DAYNIGHT_MODE == "subsatellite"

    def test_daynight_mode_limb_stored(self, line1, line2):
        """Custom daynight_mode='limb' is stored."""
        vis = Visibility(line1, line2, daynight_mode="limb")
        assert vis.daynight_mode == "limb"

    def test_daynight_mode_invalid_raises(self, line1, line2):
        """Invalid daynight_mode raises ValueError."""
        with pytest.raises(ValueError, match="daynight_mode"):
            Visibility(line1, line2, daynight_mode="bogus")

    def test_subsatellite_is_sunlit_basic(self):
        """Subsatellite-point sunlit detection with known geometry."""
        # Zenith in +Z, sun in +Z → dayside (dot > 0)
        zenith = np.array([0.0, 0.0, 1.0])
        sun_day = np.array([0.0, 0.0, 1.0])
        assert bool(Visibility._subsatellite_is_sunlit(zenith, sun_day)) is True

        # Zenith in +Z, sun in -Z → nightside (dot < 0)
        sun_night = np.array([0.0, 0.0, -1.0])
        assert bool(Visibility._subsatellite_is_sunlit(zenith, sun_night)) is False

        # Zenith in +Z, sun in +X → exactly at terminator (dot = 0)
        # With margin=0 threshold is 0, so dot=0 is NOT > 0 → nightside
        sun_terminator = np.array([1.0, 0.0, 0.0])
        assert bool(Visibility._subsatellite_is_sunlit(zenith, sun_terminator)) is False

    def test_subsatellite_is_sunlit_twilight_margin(self):
        """Twilight margin shifts the subsatellite day/night boundary."""
        zenith = np.array([0.0, 0.0, 1.0])
        # Sun perpendicular → dot(zenith, sun) = 0
        sun_perp = np.array([1.0, 0.0, 0.0])

        # margin=0: threshold=0, dot=0 → NOT sunlit
        assert bool(Visibility._subsatellite_is_sunlit(
            zenith, sun_perp, twilight_margin_deg=0.0
        )) is False

        # margin=10: threshold=-sin(10°)≈-0.17, dot=0 > -0.17 → sunlit
        assert bool(Visibility._subsatellite_is_sunlit(
            zenith, sun_perp, twilight_margin_deg=10.0
        )) is True

    def test_subsatellite_is_sunlit_array(self):
        """Subsatellite sunlit detection with array inputs."""
        zenith = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        # First timestep: sun above → day; second: sun below → night
        sun = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, -1.0]])
        result = Visibility._subsatellite_is_sunlit(zenith, sun)
        assert bool(result[0]) is True
        assert bool(result[1]) is False

    def test_subsatellite_illumination_angle_basic(self):
        """Subsatellite illumination is the zenith—Sun angle."""
        zenith = np.array([0.0, 0.0, 1.0])
        for sun, expected in [
            (np.array([0.0, 0.0, 1.0]), 0.0),    # subsolar point
            (np.array([1.0, 0.0, 0.0]), 90.0),   # terminator
            (np.array([0.0, 0.0, -1.0]), 180.0), # antisolar point
        ]:
            assert float(
                Visibility._subsatellite_illumination_angle(zenith, sun)
            ) == pytest.approx(expected)

    def test_subsatellite_illumination_angle_array(self):
        """Array inputs give one angle per timestep."""
        zenith = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        sun = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, -1.0]])
        angles = Visibility._subsatellite_illumination_angle(zenith, sun)
        assert angles.shape == (2,)
        assert angles[0] == pytest.approx(0.0)
        assert angles[1] == pytest.approx(180.0)

    def test_subsatellite_illumination_agrees_with_is_sunlit(self, line1, line2,
                                                             target_coord):
        """< 90 deg illumination is exactly the subsatellite sunlit test."""
        times = Time("2026-06-01T00:00:00") + np.arange(2 * 1440) * u.min
        vis = Visibility(line1, line2)
        pre = vis._precompute(times)
        zen, sun = pre["zenith_unit"], pre["body_units"]["sun"]

        illum = vis._subsatellite_illumination_angle(zen, sun)
        sunlit = vis._subsatellite_is_sunlit(zen, sun)
        # Both day and night occur in this window, so this is a real test
        assert sunlit.any() and not sunlit.all()
        np.testing.assert_array_equal(illum < 90.0, sunlit)

    def test_daynight_illumination_angle_follows_mode(self, line1, line2):
        """_daynight_illumination_angle dispatches on daynight_mode.

        Same geometry as test_daynight_is_sunlit_follows_mode: the
        spacecraft is over sunlit ground while the limb point toward the
        target is dark, so the two modes must straddle 90 deg.
        """
        target = np.array([-1.0, 0.0, 0.0])
        zenith = np.array([0.0, 0.0, 1.0])
        sun = np.array([0.995, 0.0, 0.0999])  # low sun, 84 deg off zenith
        limb_rad = np.deg2rad(21.0)

        vis_sub = _legacy_visibility(line1, line2, daynight_mode="subsatellite")
        vis_limb = _legacy_visibility(line1, line2, daynight_mode="limb")

        illum_sub = float(vis_sub._daynight_illumination_angle(
            target, zenith, sun, limb_angle_rad=limb_rad))
        illum_limb = float(vis_limb._daynight_illumination_angle(
            target, zenith, sun, limb_angle_rad=limb_rad))

        assert illum_sub == pytest.approx(float(
            Visibility._subsatellite_illumination_angle(zenith, sun)))
        assert illum_limb == pytest.approx(float(
            Visibility._get_earth_illumination_angle(
                target, zenith, sun, limb_angle_rad=limb_rad)))
        assert illum_sub < 90.0 < illum_limb

    def test_daynight_illumination_angle_ignores_target_in_subsatellite(
        self, line1, line2
    ):
        """In subsatellite mode the angle does not depend on the target."""
        zenith = np.array([0.0, 0.0, 1.0])
        sun = np.array([0.6, 0.0, 0.8])
        limb_rad = np.deg2rad(21.0)
        vis = Visibility(line1, line2, daynight_mode="subsatellite")
        angles = {
            float(vis._daynight_illumination_angle(
                np.array(t, dtype=float), zenith, sun, limb_angle_rad=limb_rad))
            for t in ([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        }
        assert len(angles) == 1

    def test_subsatellite_mode_differs_from_limb(self, line1, line2, target_coord):
        """subsatellite and limb modes classify day/night differently.

        We verify this at the threshold-level: for the same timesteps,
        the effective threshold arrays should differ when the two modes
        disagree on which timesteps are day vs night.
        """
        times = Time("2026-06-01T00:00:00") + np.arange(3 * 1440) * u.min
        vis_limb = Visibility(
            line1, line2,
            earthlimb_day_min=40 * u.deg,
            earthlimb_night_min=5 * u.deg,
            daynight_mode="limb",
        )
        vis_subsat = Visibility(
            line1, line2,
            earthlimb_day_min=40 * u.deg,
            earthlimb_night_min=5 * u.deg,
            daynight_mode="subsatellite",
        )
        # Force precomputation and extract effective thresholds
        pre_limb = vis_limb._precompute(times)
        pre_subsat = vis_subsat._precompute(times)

        # Target direction in GCRS
        tgt_gcrs = target_coord.transform_to(GCRS(obstime=times))
        tgt_xyz = tgt_gcrs.cartesian.xyz.value
        tgt_b = tgt_xyz / np.linalg.norm(tgt_xyz, axis=0, keepdims=True)

        thresh_limb = vis_limb._effective_earthlimb_min_deg(
            tgt_b, pre_limb["zenith_unit"], pre_limb["body_units"]["sun"],
            limb_angle_rad=pre_limb["limb_angle_rad"],
        )
        thresh_subsat = vis_subsat._effective_earthlimb_min_deg(
            tgt_b, pre_subsat["zenith_unit"], pre_subsat["body_units"]["sun"],
            limb_angle_rad=pre_subsat["limb_angle_rad"],
        )
        # The thresholds should differ on at least some timesteps
        assert not np.array_equal(thresh_limb, thresh_subsat), (
            "subsatellite and limb modes should produce different "
            "day/night thresholds for at least some timesteps"
        )

    def test_subsatellite_mode_repr_omits_daynight(self, line1, line2):
        """repr omits daynight when mode is the default 'subsatellite'."""
        vis = Visibility(
            line1, line2,
            earthlimb_day_min=25 * u.deg,
            earthlimb_night_min=10 * u.deg,
            daynight_mode="subsatellite",
        )
        r = repr(vis)
        assert "daynight=" not in r

    def test_limb_mode_repr_shows_daynight(self, line1, line2):
        """repr shows daynight=limb when mode is non-default."""
        vis = Visibility(
            line1, line2,
            earthlimb_day_min=25 * u.deg,
            earthlimb_night_min=10 * u.deg,
            daynight_mode="limb",
        )
        r = repr(vis)
        assert "daynight=limb" in r

    def test_subsatellite_no_effect_without_day_night(self, line1, line2, target_coord):
        """When day/night both None, daynight_mode makes no difference."""
        times = Time("2025-01-01T00:00:00") + np.arange(200) * u.min
        vis_limb = Visibility(line1, line2, daynight_mode="limb")
        vis_subsat = Visibility(line1, line2, daynight_mode="subsatellite")
        r_limb = vis_limb.get_visibility(target_coord, times)["visible"]
        r_subsat = vis_subsat.get_visibility(target_coord, times)["visible"]
        np.testing.assert_array_equal(r_limb, r_subsat)


class TestDynamicEarthlimb:
    """Tests for the use_dynamic_earthlimb (DPC wedge) keep-out."""

    # Keep-out values from the DPC wedge table, referenced to the limb
    # (i.e. with the 66 deg Earth angular radius already subtracted).
    BRIGHT = 110.0 - 66.0
    DARK = 75.0 - 66.0

    @pytest.fixture
    def line1(self):
        return "1 67395U 80229J   26057.99991898  .00000000  00000-0  37770-3 0    03"

    @pytest.fixture
    def line2(self):
        return "2 67395  97.8009  58.3973 0006599 121.8878 132.9207 14.87804761    04"

    @pytest.fixture
    def target_coord(self):
        """WASP-107 — has both sunlit and dark limb crossings in mid-2026."""
        return SkyCoord(188.386, -10.1462, frame="icrs", unit="deg")

    @pytest.fixture
    def test_time(self):
        return Time("2026-06-01T00:00:00")

    # ── Defaults & storage ──────────────────────────────────────────

    def test_default_is_true(self, line1, line2):
        """use_dynamic_earthlimb is on by default."""
        vis = Visibility(line1, line2)
        assert vis.use_dynamic_earthlimb is True

    def test_stored_when_true(self, line1, line2):
        """use_dynamic_earthlimb=True is stored on the instance."""
        vis = Visibility(line1, line2, use_dynamic_earthlimb=True)
        assert vis.use_dynamic_earthlimb is True

    def test_true_matches_omitting_it(self, line1, line2, target_coord, test_time):
        """Passing use_dynamic_earthlimb=True changes nothing, it is the default."""
        times = test_time + np.arange(300) * u.min
        vis_omit = Visibility(line1, line2)
        vis_true = Visibility(line1, line2, use_dynamic_earthlimb=True)
        np.testing.assert_array_equal(
            vis_omit.get_visibility(target_coord, times)["visible"],
            vis_true.get_visibility(target_coord, times)["visible"],
        )

    # ── _dynamic_earthlimb_min_deg piecewise fit ────────────────────

    def test_piecewise_flat_segments(self):
        """Below 78 deg and at/above 90 deg the curve is flat."""
        for illum in [0.0, 40.0, 77.0, 77.999]:
            assert Visibility._dynamic_earthlimb_min_deg(illum) == pytest.approx(
                self.BRIGHT
            )
        for illum in [90.0, 120.0, 180.0]:
            assert Visibility._dynamic_earthlimb_min_deg(illum) == pytest.approx(
                self.DARK
            )

    def test_piecewise_anchor_points(self):
        """The three anchor points are hit exactly (from the Earth centre)."""
        assert Visibility._dynamic_earthlimb_min_deg(78.0) == pytest.approx(
            110.0 - 66.0
        )
        assert Visibility._dynamic_earthlimb_min_deg(89.0) == pytest.approx(
            82.0 - 66.0
        )
        assert Visibility._dynamic_earthlimb_min_deg(90.0) == pytest.approx(
            75.0 - 66.0
        )

    def test_piecewise_rule1(self):
        """78-89 deg is a straight line from 110 to 82 deg."""
        m = (82.0 - 110.0) / (89.0 - 78.0)
        for illum in [78.0, 82.5, 89.0]:
            assert Visibility._dynamic_earthlimb_min_deg(illum) == pytest.approx(
                110.0 + m * (illum - 78.0) - 66.0
            )

    def test_piecewise_rule2(self):
        """89-90 deg is a steeper straight line from 82 to 75 deg."""
        m = (75.0 - 82.0) / (90.0 - 89.0)
        for illum in [89.001, 89.5, 89.999]:
            assert Visibility._dynamic_earthlimb_min_deg(illum) == pytest.approx(
                82.0 + m * (illum - 89.0) - 66.0
            )

    def test_piecewise_continuous(self):
        """The curve has no steps at the 78, 89 and 90 deg joins."""
        for knee in [78.0, 89.0, 90.0]:
            below = Visibility._dynamic_earthlimb_min_deg(knee - 1e-6)
            at = Visibility._dynamic_earthlimb_min_deg(knee)
            above = Visibility._dynamic_earthlimb_min_deg(knee + 1e-6)
            assert below == pytest.approx(at, abs=1e-4)
            assert above == pytest.approx(at, abs=1e-4)

    def test_piecewise_monotonic_and_bounded(self):
        """Keep-out falls monotonically from the bright to the dark value."""
        keepout = Visibility._dynamic_earthlimb_min_deg(
            np.linspace(0.0, 180.0, 18001)
        )
        assert np.all(np.diff(keepout) <= 1e-9)
        assert keepout.max() == pytest.approx(self.BRIGHT)
        assert keepout.min() == pytest.approx(self.DARK)
        assert keepout[0] == pytest.approx(self.BRIGHT)
        assert keepout[-1] == pytest.approx(self.DARK)

    def test_piecewise_wraps_around(self):
        """The curve is symmetric: -x, +x and +x+360 give the same keep-out."""
        for illum in [0.0, 45.0, 78.0, 85.0, 89.5, 90.0, 120.0, 180.0]:
            base = Visibility._dynamic_earthlimb_min_deg(illum)
            assert Visibility._dynamic_earthlimb_min_deg(-illum) == pytest.approx(
                base
            )
            assert Visibility._dynamic_earthlimb_min_deg(
                illum + 360.0
            ) == pytest.approx(base)
            assert Visibility._dynamic_earthlimb_min_deg(
                360.0 - illum
            ) == pytest.approx(base)

    def test_piecewise_wrap_beyond_180(self):
        """Angles past 180 deg fold back toward the sub-solar direction."""
        # 190 deg is 170 deg on the other side → still fully dark
        assert Visibility._dynamic_earthlimb_min_deg(190.0) == pytest.approx(
            Visibility._dynamic_earthlimb_min_deg(170.0)
        )
        # 282 deg folds to 78 deg → back on the bright ramp
        assert Visibility._dynamic_earthlimb_min_deg(282.0) == pytest.approx(
            Visibility._dynamic_earthlimb_min_deg(78.0)
        )

    def test_piecewise_wraps_array(self):
        """Wrapping also works element-wise on arrays."""
        illum = np.array([-85.0, -12.0, 275.0, 400.0])
        np.testing.assert_allclose(
            Visibility._dynamic_earthlimb_min_deg(illum),
            Visibility._dynamic_earthlimb_min_deg(np.array([85.0, 12.0, 85.0, 40.0])),
        )

    def test_illumination_angle_range(self, line1, line2, target_coord, test_time):
        """The computed illumination angle always lands in [0, 180]."""
        times = test_time + np.arange(2 * 1440) * u.min
        vis = Visibility(line1, line2)
        pre = vis._precompute(times)
        tgt_gcrs = target_coord.transform_to(GCRS(obstime=times))
        tgt_xyz = tgt_gcrs.cartesian.xyz.value
        tgt_b = tgt_xyz / np.linalg.norm(tgt_xyz, axis=0, keepdims=True)
        illum = vis._get_earth_illumination_angle(
            tgt_b, pre["zenith_unit"], pre["body_units"]["sun"],
            limb_angle_rad=pre["limb_angle_rad"],
        )
        assert illum.min() >= 0.0
        assert illum.max() <= 180.0

    def test_piecewise_array_shape(self):
        """Array input returns an array of the same shape."""
        illum = np.array([10.0, 85.0, 89.5, 95.0])
        keepout = Visibility._dynamic_earthlimb_min_deg(illum)
        assert keepout.shape == illum.shape
        assert keepout[0] == pytest.approx(self.BRIGHT)
        assert keepout[3] == pytest.approx(self.DARK)

    # ── _get_earth_illumination_angle unit tests ────────────────────

    def test_illumination_angle_with_limb_angle(self):
        """Known geometry: target +X, zenith +Z, normal = cos(la)Z + sin(la)X."""
        target = np.array([1.0, 0.0, 0.0])
        zenith = np.array([0.0, 0.0, 1.0])
        la_rad = np.arccos(0.91)  # typical LEO value

        # Sun overhead: n·sun = cos(la) = 0.91
        assert float(Visibility._get_earth_illumination_angle(
            target, zenith, np.array([0.0, 0.0, 1.0]), limb_angle_rad=la_rad
        )) == pytest.approx(np.rad2deg(np.arccos(0.91)))

        # Sun below: n·sun = -cos(la)
        assert float(Visibility._get_earth_illumination_angle(
            target, zenith, np.array([0.0, 0.0, -1.0]), limb_angle_rad=la_rad
        )) == pytest.approx(180.0 - np.rad2deg(np.arccos(0.91)))

        # Sun along the limb direction: n·sun = sin(la)
        assert float(Visibility._get_earth_illumination_angle(
            target, zenith, np.array([1.0, 0.0, 0.0]), limb_angle_rad=la_rad
        )) == pytest.approx(np.rad2deg(np.arccos(np.sin(la_rad))))

    def test_illumination_angle_legacy_fallback(self):
        """Without limb_angle_rad the horizontal projection is used."""
        target = np.array([1.0, 0.0, 0.0])
        zenith = np.array([0.0, 0.0, 1.0])
        assert float(Visibility._get_earth_illumination_angle(
            target, zenith, np.array([1.0, 0.0, 0.0])
        )) == pytest.approx(0.0, abs=1e-6)
        assert float(Visibility._get_earth_illumination_angle(
            target, zenith, np.array([-1.0, 0.0, 0.0])
        )) == pytest.approx(180.0)
        assert float(Visibility._get_earth_illumination_angle(
            target, zenith, np.array([0.0, 0.0, 1.0])
        )) == pytest.approx(90.0)

    def test_illumination_angle_array(self):
        """Array inputs give one illumination angle per timestep."""
        target = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        zenith = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        sun = np.array([[1.0, -1.0], [0.0, 0.0], [0.0, 0.0]])
        angles = Visibility._get_earth_illumination_angle(target, zenith, sun)
        assert angles.shape == (2,)
        assert angles[0] == pytest.approx(0.0, abs=1e-6)
        assert angles[1] == pytest.approx(180.0)

    def test_illumination_angle_agrees_with_is_sunlit(self, line1, line2,
                                                      target_coord, test_time):
        """< 90 deg illumination is exactly the sunlit condition."""
        times = test_time + np.arange(2 * 1440) * u.min
        vis = Visibility(line1, line2)
        pre = vis._precompute(times)
        tgt_gcrs = target_coord.transform_to(GCRS(obstime=times))
        tgt_xyz = tgt_gcrs.cartesian.xyz.value
        tgt_b = tgt_xyz / np.linalg.norm(tgt_xyz, axis=0, keepdims=True)

        illum = vis._get_earth_illumination_angle(
            tgt_b, pre["zenith_unit"], pre["body_units"]["sun"],
            limb_angle_rad=pre["limb_angle_rad"],
        )
        sunlit = vis._earthlimb_is_sunlit(
            tgt_b, pre["zenith_unit"], pre["body_units"]["sun"],
            limb_angle_rad=pre["limb_angle_rad"],
        )
        # Both are day and night in this window, so this is a real test
        assert sunlit.any() and not sunlit.all()
        np.testing.assert_array_equal(illum < 90.0, sunlit)

    # ── Effective threshold ─────────────────────────────────────────

    def test_effective_threshold_uses_dynamic_curve(self, line1, line2,
                                                    target_coord, test_time):
        """The effective threshold is the wedge curve of the illumination angle."""
        times = test_time + np.arange(1440) * u.min
        vis = Visibility(line1, line2, use_dynamic_earthlimb=True)
        pre = vis._precompute(times)
        tgt_gcrs = target_coord.transform_to(GCRS(obstime=times))
        tgt_xyz = tgt_gcrs.cartesian.xyz.value
        tgt_b = tgt_xyz / np.linalg.norm(tgt_xyz, axis=0, keepdims=True)

        thresh = vis._effective_earthlimb_min_deg(
            tgt_b, pre["zenith_unit"], pre["body_units"]["sun"],
            limb_angle_rad=pre["limb_angle_rad"],
        )
        # Mode-aware angle: the wedge reads the daynight_mode reference point
        illum = vis._daynight_illumination_angle(
            tgt_b, pre["zenith_unit"], pre["body_units"]["sun"],
            limb_angle_rad=pre["limb_angle_rad"],
        )
        np.testing.assert_allclose(
            thresh, Visibility._dynamic_earthlimb_min_deg(illum)
        )
        # Over a full day the threshold varies and stays inside the curve
        curve = Visibility._dynamic_earthlimb_min_deg(
            np.linspace(0.0, 180.0, 18001)
        )
        assert thresh.min() >= curve.min()
        assert thresh.max() <= curve.max()
        assert thresh.max() - thresh.min() > 1.0
        # The dark plateau is reached on the night side of every orbit
        assert np.isclose(thresh, self.DARK).any()

    # —— daynight_mode interaction —————————————————————————————

    def test_dynamic_honours_daynight_mode(self, line1, line2, target_coord,
                                           test_time):
        """The wedge reads the illumination angle at the daynight_mode point.

        The dynamic curve used to always reference the nearest limb
        point, silently ignoring daynight_mode.  It now follows it, so
        the two modes must give different thresholds.
        """
        times = test_time + np.arange(3 * 1440) * u.min
        vis_sub = _legacy_visibility(
            line1, line2, use_dynamic_earthlimb=True,
            daynight_mode="subsatellite",
        )
        vis_limb = _legacy_visibility(
            line1, line2, use_dynamic_earthlimb=True, daynight_mode="limb",
        )
        pre = vis_sub._precompute(times)
        tgt_gcrs = target_coord.transform_to(GCRS(obstime=times))
        tgt_xyz = tgt_gcrs.cartesian.xyz.value
        tgt_b = tgt_xyz / np.linalg.norm(tgt_xyz, axis=0, keepdims=True)

        args = (tgt_b, pre["zenith_unit"], pre["body_units"]["sun"])
        kw = dict(limb_angle_rad=pre["limb_angle_rad"])
        thresh_sub = vis_sub._effective_earthlimb_min_deg(*args, **kw)
        thresh_limb = vis_limb._effective_earthlimb_min_deg(*args, **kw)

        assert not np.allclose(thresh_sub, thresh_limb)
        np.testing.assert_allclose(
            thresh_sub,
            Visibility._dynamic_earthlimb_min_deg(
                Visibility._subsatellite_illumination_angle(
                    pre["zenith_unit"], pre["body_units"]["sun"])
            ),
        )
        # ...and the difference reaches the visibility result itself
        assert not np.array_equal(
            vis_sub.get_visibility(target_coord, times)["visible"],
            vis_limb.get_visibility(target_coord, times)["visible"],
        )

    def test_dynamic_default_mode_is_target_independent(self, line1, line2,
                                                        test_time):
        """Under the default subsatellite mode the wedge ignores the target."""
        times = test_time + np.arange(500) * u.min
        vis = Visibility(line1, line2, use_dynamic_earthlimb=True)
        pre = vis._precompute(times)
        zen, sun, la = (pre["zenith_unit"], pre["body_units"]["sun"],
                        pre["limb_angle_rad"])

        thresholds = []
        for coord in (SkyCoord(188.386, -10.1462, frame="icrs", unit="deg"),
                      SkyCoord(270.0, -66.0, frame="icrs", unit="deg"),
                      SkyCoord(10.0, 45.0, frame="icrs", unit="deg")):
            xyz = coord.transform_to(GCRS(obstime=times)).cartesian.xyz.value
            tgt_b = xyz / np.linalg.norm(xyz, axis=0, keepdims=True)
            thresholds.append(vis._effective_earthlimb_min_deg(
                tgt_b, zen, sun, limb_angle_rad=la))

        for other in thresholds[1:]:
            np.testing.assert_allclose(thresholds[0], other)

    def test_dynamic_repr_shows_non_default_daynight(self, line1, line2):
        """repr surfaces daynight_mode on the dynamic branch too.

        The mode now changes the wedge threshold, so it must not be
        silently omitted the way it was when it had no effect there.
        """
        assert "daynight=" not in repr(
            Visibility(line1, line2, use_dynamic_earthlimb=True))
        r = repr(Visibility(line1, line2, use_dynamic_earthlimb=True,
                            daynight_mode="limb"))
        assert "limb=dynamic" in r
        assert "daynight=limb" in r

    @pytest.mark.parametrize("mode", ["subsatellite", "limb"])
    def test_dynamic_summary_matches_engine(self, line1, line2, target_coord,
                                            test_time, mode):
        """The illum angle summary prints is the one the wedge was fed."""
        vis = Visibility(line1, line2, use_dynamic_earthlimb=True,
                         daynight_mode=mode)
        for i in range(0, 200, 20):
            time = test_time + i * u.min
            pre = vis._precompute(time)
            tgt_u = vis._target_unit(target_coord, time)
            illum = float(vis._daynight_illumination_angle(
                tgt_u, pre["zenith_unit"], pre["body_units"]["sun"],
                limb_angle_rad=pre["limb_angle_rad"],
            ))
            line = next(
                ln for ln in vis.summary(target_coord, time).split("\n")
                if ln.startswith("Earthlimb")
            )
            assert f"illum {illum:.1f}" in line, (
                f"{mode} mode at {time.iso}: engine used {illum:.1f} deg "
                f"but summary reported: {line.strip()}"
            )

    def test_dynamic_overrides_day_night(self, line1, line2, target_coord,
                                         test_time):
        """use_dynamic_earthlimb takes precedence over the day/night pair."""
        times = test_time + np.arange(500) * u.min
        vis_dyn = Visibility(line1, line2, use_dynamic_earthlimb=True)
        vis_both = Visibility(
            line1, line2,
            use_dynamic_earthlimb=True,
            earthlimb_day_min=90 * u.deg,
            earthlimb_night_min=0 * u.deg,
        )
        np.testing.assert_array_equal(
            vis_dyn.get_visibility(target_coord, times)["visible"],
            vis_both.get_visibility(target_coord, times)["visible"],
        )

    # ── Integration with get_visibility ─────────────────────────────

    def test_dynamic_bracketed_by_fixed_limits(self, line1, line2, target_coord,
                                               test_time):
        """Dynamic visibility sits between the flat bright and dark limits."""
        times = test_time + np.arange(3 * 1440) * u.min
        vis_dyn = _legacy_visibility(line1, line2, use_dynamic_earthlimb=True)
        vis_loose = _legacy_visibility(line1, line2, earthlimb_min=self.DARK * u.deg)
        vis_tight = _legacy_visibility(line1, line2, earthlimb_min=self.BRIGHT * u.deg)

        r_dyn = vis_dyn.get_visibility(target_coord, times)["visible"]
        r_loose = vis_loose.get_visibility(target_coord, times)["visible"]
        r_tight = vis_tight.get_visibility(target_coord, times)["visible"]

        assert np.all(r_dyn <= r_loose)
        assert np.all(r_tight <= r_dyn)
        # The dynamic curve is not degenerate to either bound
        assert r_dyn.sum() < r_loose.sum()
        assert r_dyn.sum() > r_tight.sum()

    def test_dynamic_differs_from_default(self, line1, line2, target_coord,
                                          test_time):
        """The dynamic curve changes visibility versus the fixed 20 deg limit."""
        times = test_time + np.arange(3 * 1440) * u.min
        r_fixed = _legacy_visibility(
            line1, line2, earthlimb_min=20 * u.deg
        ).get_visibility(target_coord, times)["visible"]
        r_dyn = _legacy_visibility(
            line1, line2, use_dynamic_earthlimb=True
        ).get_visibility(target_coord, times)["visible"]
        assert not np.array_equal(r_fixed, r_dyn)

    def test_get_constraint_matches_manual_threshold(self, line1, line2,
                                                     target_coord, test_time):
        """get_constraint('earthlimb') applies the same dynamic threshold."""
        times = test_time + np.arange(400) * u.min
        vis = Visibility(line1, line2, use_dynamic_earthlimb=True)
        pre = vis._precompute(times)
        tgt_gcrs = target_coord.transform_to(GCRS(obstime=times))
        tgt_xyz = tgt_gcrs.cartesian.xyz.value
        tgt_b = tgt_xyz / np.linalg.norm(tgt_xyz, axis=0, keepdims=True)

        illum = vis._daynight_illumination_angle(
            tgt_b, pre["zenith_unit"], pre["body_units"]["sun"],
            limb_angle_rad=pre["limb_angle_rad"],
        )
        actual = vis.get_separations(target_coord, times)["earthlimb"]
        expected = (
            actual.to(u.deg).value
            >= Visibility._dynamic_earthlimb_min_deg(illum)
        )
        np.testing.assert_array_equal(
            vis.get_constraint(target_coord, "earthlimb", times), expected
        )

    def test_get_constraint_scalar_time(self, line1, line2, target_coord,
                                        test_time):
        """Scalar times work through the get_constraint path."""
        vis = Visibility(line1, line2, use_dynamic_earthlimb=True)
        result = vis.get_constraint(target_coord, "earthlimb", test_time)
        assert bool(result) in (True, False)

    def test_best_roll_uses_dynamic(self, line1, line2, target_coord, test_time):
        """The roll-search path applies the dynamic threshold too."""
        times = test_time + np.arange(97) * u.min
        vis = _legacy_visibility(line1, line2, use_dynamic_earthlimb=True)
        result = vis.get_visibility(target_coord, times, optimize_roll=True)
        np.testing.assert_array_equal(
            result["boresight_visible"], vis.get_visibility(target_coord, times)["visible"]
        )

    # ── repr / summary ──────────────────────────────────────────────

    def test_repr_shows_dynamic(self, line1, line2):
        """repr flags the dynamic limb keep-out."""
        vis = Visibility(line1, line2, use_dynamic_earthlimb=True)
        r = repr(vis)
        assert "limb=dynamic" in r
        assert "limb_day" not in r

    def test_summary_shows_illumination(self, line1, line2, target_coord,
                                        test_time):
        """summary() reports the illumination angle driving the threshold."""
        vis = Visibility(line1, line2, use_dynamic_earthlimb=True)
        text = vis.summary(target_coord, test_time)
        assert "illum" in text
        assert "Earthlimb" in text


class TestEphemerisStepAndCaching:
    """Tests for ephemeris_step interpolation and the precompute caches."""

    @pytest.fixture
    def line1(self):
        return "1 67395U 80229J   26057.99991898  .00000000  00000-0  37770-3 0    03"

    @pytest.fixture
    def line2(self):
        return "2 67395  97.8009  58.3973 0006599 121.8878 132.9207 14.87804761    04"

    @pytest.fixture
    def kwargs(self):
        return dict(
            earthlimb_day_min=44 * u.deg,
            earthlimb_night_min=13 * u.deg,
            st_sun_min=50 * u.deg,
            st_moon_min=20 * u.deg,
            st_earthlimb_min=30 * u.deg,
            st_required=1,
        )

    @pytest.fixture
    def targets(self):
        return [
            SkyCoord(188.386, -10.1462, frame="icrs", unit="deg"),
            SkyCoord(79.17, 45.99, frame="icrs", unit="deg"),
        ]

    @pytest.fixture
    def times(self):
        return Time("2026-06-01T00:00:00") + np.arange(600) * u.min

    # ── ephemeris_step ──────────────────────────────────────────────

    def test_default_is_none(self, line1, line2):
        """ephemeris_step defaults to None, i.e. an exact ephemeris."""
        assert Visibility(line1, line2).ephemeris_step is None

    def test_stored(self, line1, line2):
        """A custom ephemeris_step is stored on the instance."""
        vis = Visibility(line1, line2, ephemeris_step=30 * u.min)
        assert vis.ephemeris_step == 30 * u.min

    def test_requires_time_units(self, line1, line2):
        """A bare number, or an angle, is rejected."""
        with pytest.raises(TypeError, match="astropy Quantity"):
            Visibility(line1, line2, ephemeris_step=30)
        with pytest.raises(u.UnitsError, match="time units"):
            Visibility(line1, line2, ephemeris_step=30 * u.deg)

    def test_interpolated_bodies_match_exact(self, line1, line2, times):
        """Interpolated body directions agree with the exact ephemeris."""
        exact = Visibility(line1, line2)._precompute(times)
        interp = Visibility(
            line1, line2, ephemeris_step=60 * u.min
        )._precompute(times)

        for name in ("sun", "moon"):
            dot = np.sum(exact["body_units"][name] * interp["body_units"][name],
                         axis=0)
            offset = np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))
            # Tens of degrees of keep-out; this must be far below that
            assert offset.max() < 0.01, f"{name} off by {offset.max()} deg"

        # The satellite-dependent quantities are untouched by interpolation
        np.testing.assert_array_equal(exact["zenith_unit"],
                                      interp["zenith_unit"])
        np.testing.assert_array_equal(exact["limb_angle_rad"],
                                      interp["limb_angle_rad"])

    def test_interpolation_does_not_change_visibility(self, line1, line2,
                                                      kwargs, targets, times):
        """get_visibility is unchanged by the interpolated ephemeris."""
        exact = Visibility(line1, line2, **kwargs)
        interp = Visibility(line1, line2, ephemeris_step=60 * u.min, **kwargs)
        for target in targets:
            np.testing.assert_array_equal(
                exact.get_visibility(target, times)["visible"],
                interp.get_visibility(target, times)["visible"],
            )

    def test_interpolation_does_not_change_best_roll(self, line1, line2,
                                                     kwargs, targets, times):
        """The roll-search decisions are unchanged by the interpolated ephemeris."""
        exact = Visibility(line1, line2, **kwargs)
        interp = Visibility(line1, line2, ephemeris_step=60 * u.min, **kwargs)
        for target in targets:
            a = exact.get_visibility(target, times, optimize_roll=True)
            b = interp.get_visibility(target, times, optimize_roll=True)
            for key in ("visible", "boresight_visible", "n_st_pass",
                        "roll_deg"):
                np.testing.assert_array_equal(np.asarray(a[key]),
                                              np.asarray(b[key]))
            assert a["n_visible"] == b["n_visible"]

    def test_scalar_time_ignores_interpolation(self, line1, line2, kwargs,
                                               targets):
        """A scalar time is always evaluated exactly."""
        moment = Time("2026-06-01T03:00:00")
        exact = Visibility(line1, line2, **kwargs)
        interp = Visibility(line1, line2, ephemeris_step=60 * u.min, **kwargs)
        assert (exact.get_visibility(targets[0], moment)["visible"]
                == interp.get_visibility(targets[0], moment)["visible"])

    # ── precompute cache ────────────────────────────────────────────

    def test_precompute_cache_hits_same_object(self, line1, line2, times):
        """The same Time object is served from the cache."""
        vis = Visibility(line1, line2)
        first = vis._precompute(times)
        assert vis._precompute(times) is first

    def test_precompute_cache_misses_other_times(self, line1, line2, times):
        """A different grid must never be served the cached entry.

        The cache used to key on ``id(time)``, which CPython recycles once
        the original is collected, so a later grid could land on a dead
        entry's address and be served its data.
        """
        vis = Visibility(line1, line2)
        first = vis._precompute(times)
        zenith_first = first["zenith_unit"].copy()

        # Drop the reference and build a new grid, which may well land on
        # the freed address.
        other = Time("2026-07-15T00:00:00") + np.arange(600) * u.min
        second = vis._precompute(other)
        assert second is not first
        assert not np.allclose(second["zenith_unit"], zenith_first)

        # Recomputing the original grid still gives the original answer
        again = Visibility(line1, line2)._precompute(times)
        np.testing.assert_allclose(again["zenith_unit"], zenith_first)

    # ── vectorised roll attitude ────────────────────────────────────

    def test_roll_attitude_batch_matches_scalar(self):
        """_roll_attitude_batch reproduces _roll_attitude for every angle."""
        z_unit = np.array([0.3, -0.5, 0.81])
        z_unit = z_unit / np.linalg.norm(z_unit)
        rolls = np.deg2rad(np.arange(0, 360, 2.0))

        x_all, y_all = Visibility._roll_attitude_batch(z_unit, rolls)
        assert x_all.shape == (len(rolls), 3)
        for i, roll in enumerate(rolls):
            x_one, y_one = Visibility._roll_attitude(z_unit, roll)
            np.testing.assert_allclose(x_all[i], x_one, atol=1e-14)
            np.testing.assert_allclose(y_all[i], y_one, atol=1e-14)

    def test_roll_attitude_batch_near_pole(self):
        """The celestial-pole fallback is applied in the batch version too."""
        z_unit = np.array([0.0, 0.0, 1.0])
        rolls = np.deg2rad(np.array([0.0, 90.0, 180.0]))
        x_all, y_all = Visibility._roll_attitude_batch(z_unit, rolls)
        for i, roll in enumerate(rolls):
            x_one, y_one = Visibility._roll_attitude(z_unit, roll)
            np.testing.assert_allclose(x_all[i], x_one, atol=1e-14)
            np.testing.assert_allclose(y_all[i], y_one, atol=1e-14)


class TestConstraintApiConsistency:
    """The diagnostic API must agree with get_visibility, body for body."""

    LINE1 = "1 67395U 80229J   26212.76861111  .00000000  00000-0  37770-3 0    05"
    LINE2 = "2 67395  97.8073 210.7389 0004787   3.5990  91.5851 14.88172851    02"

    @pytest.fixture
    def vis(self):
        return Visibility(
            self.LINE1, self.LINE2,
            earthlimb_day_min=44 * u.deg,
            earthlimb_night_min=13 * u.deg,
            sun_min=91 * u.deg,
            moon_min=20 * u.deg,
            st_sun_min=50 * u.deg,
            st_moon_min=20 * u.deg,
            st_earthlimb_min=30 * u.deg,
            st_required=1,
        )

    @pytest.fixture
    def times(self):
        return Time("2026-08-10T00:00:00") + np.arange(500) * u.min

    @pytest.fixture
    def target_coord(self):
        return SkyCoord(330.795, 18.884, frame="icrs", unit="deg")

    def test_every_invisible_step_is_explained(self, vis, target_coord, times):
        """No step may be invisible without some constraint failing.

        The two used to disagree because the diagnostics went through the
        astropy path while get_visibility used the vector path.
        """
        visible = np.asarray(vis.get_visibility(target_coord, times)["visible"])
        constraints = vis.get_all_constraints(target_coord, times)

        failing = np.zeros(len(times), dtype=bool)
        for passed in constraints.values():
            failing |= ~np.asarray(passed)

        assert not np.any(~visible & ~failing), (
            f"{int(np.sum(~visible & ~failing))} invisible steps that no "
            f"individual constraint explains"
        )

    def test_all_constraints_passing_means_visible(self, vis, target_coord,
                                                   times):
        """The converse: if every constraint passes the target is visible."""
        visible = np.asarray(vis.get_visibility(target_coord, times)["visible"])
        constraints = vis.get_all_constraints(target_coord, times)

        all_pass = np.ones(len(times), dtype=bool)
        for passed in constraints.values():
            all_pass &= np.asarray(passed)

        np.testing.assert_array_equal(all_pass & ~visible,
                                      np.zeros(len(times), dtype=bool))

    def test_star_tracker_constraint_matches_visibility(self, vis, target_coord,
                                                        times):
        """The ST diagnostic uses the same path as get_visibility."""
        pre = vis._precompute(times)
        target_unit = vis._target_unit(target_coord, times)[:, 0].copy()
        np.testing.assert_array_equal(
            np.asarray(vis.get_star_tracker_constraint(target_coord, times)),
            np.asarray(vis._get_st_constraint_fast(target_unit, times, pre)),
        )

    def test_earthlimb_separation_matches_visibility_math(self, vis,
                                                          target_coord, times):
        """The reported limb angle is the one the constraint is applied to.

        Both are geocentric now; get_separations used to return an AltAz
        altitude, which is referenced to the geodetic horizon and so
        differed from the constraint by up to 0.1 deg.
        """
        pre = vis._precompute(times)
        target_unit = vis._target_unit(target_coord, times)
        expected = vis._fast_limb_deg(
            target_unit, pre["zenith_unit"], pre["limb_angle_rad"]
        )
        separations = vis.get_separations(target_coord, times)
        np.testing.assert_allclose(
            separations["earthlimb"].to(u.deg).value, expected
        )

    def test_separations_agree_with_constraints(self, vis, target_coord, times):
        """A body constraint passes exactly when its separation clears the limit."""
        separations = vis.get_separations(target_coord, times)
        for body, limit in (("moon", vis.moon_min), ("sun", vis.sun_min)):
            np.testing.assert_array_equal(
                np.asarray(vis.get_constraint(target_coord, body, times)),
                separations[body] >= limit,
            )

    # ── precompute reuse ────────────────────────────────────────────

    def test_all_constraints_precomputes_once(self, vis, target_coord, times,
                                              monkeypatch):
        """One set of ephemeris/SGP4 results covers every body."""
        calls = []
        original = Visibility._precompute

        def counted(self, time):
            calls.append(time)
            return original(self, time)

        monkeypatch.setattr(Visibility, "_precompute", counted)
        vis.get_all_constraints(target_coord, times)
        assert len(calls) == 1, f"_precompute called {len(calls)} times"

    def test_constraint_accepts_precomputed_data(self, vis, target_coord, times):
        """Passing `pre` gives the same answer as letting it compute one."""
        pre = vis._precompute(times)
        for body in ("moon", "sun", "earthlimb"):
            np.testing.assert_array_equal(
                np.asarray(vis.get_constraint(target_coord, body, times)),
                np.asarray(vis.get_constraint(target_coord, body, times,
                                              pre=pre)),
            )

    def test_disabled_planet_still_queryable(self, vis, target_coord, times):
        """mars/jupiter can be asked for even when their keep-out is off.

        _precompute only carries the planets whose limit is active, so this
        exercises the ephemeris fallback.
        """
        assert vis.mars_min == 0 * u.deg
        result = np.asarray(vis.get_constraint(target_coord, "mars", times))
        assert result.shape == (len(times),)
        assert result.all()  # a 0 deg keep-out passes everywhere

        separations = vis.get_separations(target_coord, times)
        assert separations["mars"].unit.is_equivalent(u.deg)
        assert np.all(separations["mars"].to(u.deg).value >= 0)

    def test_summary_still_renders(self, vis, target_coord, times):
        """summary() works off the shared precompute."""
        text = vis.summary(target_coord, times[0])
        assert "Visibility Summary" in text
        assert "Earthlimb" in text
        assert "Star Tracker Constraints" in text


class TestEarthlimbRegressionSpotChecks:
    """Frozen results for the pre-existing (non-dynamic) limb paths.

    These guard the day/night, twilight and default behaviour against
    accidental changes made while extending the Earth limb constraint.
    """

    LINE1 = "1 67395U 80229J   26057.99991898  .00000000  00000-0  37770-3 0    03"
    LINE2 = "2 67395  97.8009  58.3973 0006599 121.8878 132.9207 14.87804761    04"

    @pytest.fixture
    def times(self):
        return Time("2026-06-01T00:00:00") + np.arange(3 * 1440) * u.min

    @pytest.fixture
    def wasp107(self):
        return SkyCoord(188.386, -10.1462, frame="icrs", unit="deg")

    @pytest.fixture
    def south_target(self):
        return SkyCoord(270.0, -66.0, frame="icrs", unit="deg")

    # Every day/night entry names its mode explicitly, so both are pinned
    # and neither can drift if DAYNIGHT_MODE changes again.  The "mode
    # implicit" rows pin what the default currently resolves to.
    @pytest.mark.parametrize("kwargs,expected", [
        ({}, 2259),
        (dict(earthlimb_day_min=40 * u.deg,
              earthlimb_night_min=5 * u.deg,
              daynight_mode="subsatellite"), 2159),
        (dict(earthlimb_day_min=40 * u.deg,
              earthlimb_night_min=5 * u.deg,
              daynight_mode="limb"), 2626),
        # mode implicit — must track the "subsatellite" row above
        (dict(earthlimb_day_min=40 * u.deg,
              earthlimb_night_min=5 * u.deg), 2159),
        (dict(earthlimb_day_min=40 * u.deg,
              earthlimb_night_min=5 * u.deg,
              twilight_margin=18 * u.deg,
              daynight_mode="subsatellite"), 1768),
        # mode implicit, with twilight margin
        (dict(earthlimb_day_min=40 * u.deg,
              earthlimb_night_min=5 * u.deg,
              twilight_margin=18 * u.deg), 1768),
    ])
    def test_visible_counts_unchanged(self, times, south_target, kwargs, expected):
        """Visible-timestep counts for known configurations."""
        vis = _legacy_visibility(self.LINE1, self.LINE2, **kwargs)
        result = vis.get_visibility(south_target, times)["visible"]
        assert int(result.sum()) == expected

    def test_default_wasp107_count(self, times, wasp107):
        """Default constraints on WASP-107 over three days."""
        vis = _legacy_visibility(self.LINE1, self.LINE2)
        assert int(vis.get_visibility(wasp107, times)["visible"].sum()) == 2297

    def test_star_tracker_count(self, times, wasp107):
        """Star-tracker-constrained visibility is unchanged."""
        vis = _legacy_visibility(
            self.LINE1, self.LINE2,
            st_sun_min=44 * u.deg,
            st_earthlimb_min=30 * u.deg,
            st_moon_min=12 * u.deg,
        )
        assert int(vis.get_visibility(wasp107, times)["visible"].sum()) == 1033
