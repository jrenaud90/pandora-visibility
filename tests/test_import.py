import numpy as np
from astropy import units as u
from packaging import version

from pandoravisibility import Visibility

# The class defaults carry Pandora's flight keep-outs from v1.3.0.  These
# counts predate that, so they name the older, simpler configuration rather
# than tracking whatever the defaults become next.
_LEGACY_DEFAULTS = dict(
    moon_min=25 * u.deg,
    earthlimb_day_min=None,
    earthlimb_night_min=None,
    use_dynamic_earthlimb=False,
    st_sun_min=0 * u.deg,
    st_moon_min=0 * u.deg,
    st_earthlimb_min=0 * u.deg,
)


def test_numpy_compatibility():
    """Test that numpy version is >= 1.26 and imports work correctly."""
    numpy_version = version.parse(np.__version__)
    min_version = version.parse("1.26.0")
    assert (
        numpy_version >= min_version
    ), f"NumPy version {np.__version__} is less than {min_version}"

    # Verify numpy can be imported and basic operations work
    test_array = np.array([1, 2, 3])
    assert test_array.sum() == 6
    assert test_array.shape == (3,)


def test_visibility():
    # Example TLE lines (replace with actual TLE data)
    line1 = "1 67395U 80229J   26057.99991898  .00000000  00000-0  37770-3 0    03"
    line2 = "2 67395  97.8009  58.3973 0006599 121.8878 132.9207 14.87804761    04"

    vis = Visibility(line1, line2)

    # Test get_period method
    period = vis.get_period()
    assert period.value > 96.78
    assert period.value < 96.79

    # Test get_state method with a time input
    from astropy.time import Time

    time = Time("2025-01-01T00:00:00")
    state = vis.get_state(time)
    assert state is not None

    # test satnum
    assert int(vis.tle.satnum) == 67395

    from astropy import units as u
    from astropy.constants import R_earth

    assert (vis.tle.a * u.earthRad - R_earth).to(u.km).value > 601
    assert (vis.tle.a * u.earthRad - R_earth).to(u.km).value < 602


def test_target():
    # Example TLE lines
    line1 = "1 67395U 80229J   26057.99991898  .00000000  00000-0  37770-3 0    03"
    line2 = "2 67395  97.8009  58.3973 0006599 121.8878 132.9207 14.87804761    04"

    vis = Visibility(line1, line2, **_LEGACY_DEFAULTS)

    from astropy.time import Time, TimeDelta

    tstart = Time("2025-01-01T00:00:00.000")
    tstop = Time("2025-01-02T00:00:00.000")  # Example stop time

    from astropy import units as u

    dt = TimeDelta((1 / 144) * u.day, format="jd")  # 10 minutes in Julian days
    n_steps = int((tstop - tstart) / dt)  # Number of time steps
    time_deltas = TimeDelta(np.arange(n_steps) * dt.jd, format="jd")
    times = tstart + time_deltas

    # using Capella because it is visible in the time period
    from astropy.coordinates import SkyCoord

    target_coord = SkyCoord(79.17305002, 45.99514569, frame="icrs", unit="deg")
    targ_vis = vis.get_visibility(target_coord, times)["visible"]

    assert int(targ_vis.shape[0]) == 144
    assert targ_vis.astype(int).sum() == 78


def test_custom_limits():
    # Example TLE lines
    line1 = "1 67395U 80229J   26057.99991898  .00000000  00000-0  37770-3 0    03"
    line2 = "2 67395  97.8009  58.3973 0006599 121.8878 132.9207 14.87804761    04"

    from astropy import units as u

    vis = Visibility(
        line1,
        line2,
        **{**_LEGACY_DEFAULTS, "moon_min": 20 * u.deg},
        earthlimb_min=10 * u.deg,
        sun_min=90 * u.deg,
        jupiter_min=0 * u.deg,
        mars_min=0 * u.deg,
    )

    from astropy.time import Time, TimeDelta

    tstart = Time("2025-01-01T00:00:00.000")
    tstop = Time("2025-01-02T00:00:00.000")  # Example stop time

    dt = TimeDelta((1 / 144) * u.day, format="jd")  # 10 minutes in Julian days
    n_steps = int((tstop - tstart) / dt)  # Number of time steps
    time_deltas = TimeDelta(np.arange(n_steps) * dt.jd, format="jd")
    times = tstart + time_deltas

    # using Capella because it is visible in the time period
    from astropy.coordinates import SkyCoord

    target_coord = SkyCoord(79.17305002, 45.99514569, frame="icrs", unit="deg")
    targ_vis = vis.get_visibility(target_coord, times)["visible"]

    assert int(targ_vis.shape[0]) == 144
    assert targ_vis.astype(int).sum() == 90


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time

    line1 = "1 67395U 80229J   26057.99991898  .00000000  00000-0  37770-3 0    03"
    line2 = "2 67395  97.8009  58.3973 0006599 121.8878 132.9207 14.87804761    04"

    _ = Visibility(line1, line2)
    target_coord = SkyCoord(79.17305002, 45.99514569, frame="icrs", unit="deg")
    time = Time("2025-01-01T00:00:00")

    # Test with zero constraints
    vis_zero = Visibility(
        line1, line2,
        **{**_LEGACY_DEFAULTS, "moon_min": 0 * u.deg, "sun_min": 0 * u.deg,
           "earthlimb_min": -90 * u.deg},
    )
    result = vis_zero.get_visibility(target_coord, time)["visible"]
    assert isinstance(result, bool)
    assert result is True

    # Test with very high constraints (should always fail)
    vis_high = Visibility(line1, line2, moon_min=180 * u.deg)
    result = vis_high.get_visibility(target_coord, time)["visible"]
    assert result is False
