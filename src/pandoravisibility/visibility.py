import numpy as np
from astropy import units as u
from astropy.constants import R_earth
from astropy.coordinates import GCRS, TEME, EarthLocation, SkyCoord, get_body
from astropy.time import Time
from sgp4.api import SGP4_ERRORS, Satrec

__all__ = ["Visibility"]

_R_EARTH_M = R_earth.to(u.m).value

# DPC wedge keep-out: piecewise map from Earth illumination angle (deg) to
# a keep-out angle measured from the Earth *centre* (deg).  Subtracting the
# nominal Earth angular radius converts it to the limb-referenced angle used
# everywhere else in this module.
#
# The curve is anchored at three points — (78, 110), (89, 82) and (90, 75) —
# flat outside them and straight lines in between, so it is continuous.
_DYN_EARTH_ANGULAR_RADIUS_DEG = 66.0
_DYN_BRIGHT_ILLUM_DEG, _DYN_BRIGHT_KEEPOUT_DEG = 78.0, 110.0
_DYN_KNEE_ILLUM_DEG, _DYN_KNEE_KEEPOUT_DEG = 89.0, 82.0
_DYN_DARK_ILLUM_DEG, _DYN_DARK_KEEPOUT_DEG = 90.0, 75.0

# Rule 1 (78 - 89 deg) and rule 2 (89 - 90 deg) linear fits through them
_DYN_RULE1_M = (
    (_DYN_KNEE_KEEPOUT_DEG - _DYN_BRIGHT_KEEPOUT_DEG)
    / (_DYN_KNEE_ILLUM_DEG - _DYN_BRIGHT_ILLUM_DEG)
)
_DYN_RULE1_B = _DYN_BRIGHT_KEEPOUT_DEG - _DYN_RULE1_M * _DYN_BRIGHT_ILLUM_DEG
_DYN_RULE2_M = (
    (_DYN_DARK_KEEPOUT_DEG - _DYN_KNEE_KEEPOUT_DEG)
    / (_DYN_DARK_ILLUM_DEG - _DYN_KNEE_ILLUM_DEG)
)
_DYN_RULE2_B = _DYN_KNEE_KEEPOUT_DEG - _DYN_RULE2_M * _DYN_KNEE_ILLUM_DEG


def _validate_angle(value, name):
    """Raise TypeError if *value* is not an astropy angular Quantity."""
    if not isinstance(value, u.Quantity):
        raise TypeError(
            f"{name} must be an astropy Quantity with angular units "
            f"(e.g. {name}={value}*u.deg), got {type(value).__name__}"
        )
    if not value.unit.physical_type == "angle":
        raise u.UnitsError(
            f"{name} must have angular units (e.g. u.deg), "
            f"got {value.unit}"
        )


def _validate_time_quantity(value, name):
    """Raise TypeError if *value* is not an astropy time Quantity."""
    if not isinstance(value, u.Quantity):
        raise TypeError(
            f"{name} must be an astropy Quantity with time units "
            f"(e.g. {name}={value}*u.min), got {type(value).__name__}"
        )
    if not value.unit.physical_type == "time":
        raise u.UnitsError(
            f"{name} must have time units (e.g. u.min), "
            f"got {value.unit}"
        )


class Visibility:
    """
    A class to handle Two-Line Element (TLE) data and target visibility.

    This class provides functionality to:
    - Calculate satellite positions from TLE data
    - Determine visibility of astronomical targets based on constraints
    - Analyze observation windows and duty cycles
    - Support visualization of visibility data

    Examples:
    ---------
    >>> # Initialize with TLE data
    >>> vis = Visibility(line1, line2)
    >>>
    >>> # Check visibility for a single target and time
    >>> from astropy.coordinates import SkyCoord
    >>> from astropy.time import Time
    >>> target = SkyCoord(ra=79.17, dec=45.99, unit="deg")
    >>> time = Time("2026-01-15T00:00:00")
    >>> is_visible = vis.get_visibility(target, time)
    >>> print(vis.summary(target, time))
    >>>
    >>> # Analyze visibility over a time period
    >>> times = Time("2026-01-01") + np.arange(365) * u.day
    >>> visibility = vis.get_visibility(target, times)
    >>>
    >>> # Plot visibility timeline
    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(12,4))
    >>> plt.plot(times.utc, visibility)
    >>> plt.xlabel("Time")
    >>> plt.ylabel("Visibility")

    See Also:
    ---------
    sgp4.api.Satrec : Low-level access to SGP4 propagator
    """

    # Default constants - can be overridden per instance
    MOON_MIN = 25 * u.deg
    SUN_MIN = 91 * u.deg
    EARTHLIMB_MIN = 20 * u.deg
    EARTHLIMB_DAY_MIN = None    # None = use EARTHLIMB_MIN
    EARTHLIMB_NIGHT_MIN = None  # None = use EARTHLIMB_MIN
    TWILIGHT_MARGIN = 0 * u.deg  # 0 = sharp terminator (current behaviour)
    USE_DYNAMIC_EARTHLIMB = False  # True = DPC wedge keep-out vs illumination angle
    # "subsatellite" = ground below spacecraft; "limb" = nearest-limb-to-target.
    # "subsatellite" is the default: it is a target-independent, orbit-only
    # solar zenith angle, so every target on a given pass sees the same Earth
    # illumination. It governs both Earth limb models, the day/night step
    # pair (`earthlimb_day_min` / `earthlimb_night_min`) and the dynamic
    # DPC wedge (`use_dynamic_earthlimb`), so the two can never disagree
    # about where the reference point is. Set "limb" to reference the patch
    # of Earth the boresight actually grazes instead.
    DAYNIGHT_MODE = "subsatellite"
    MARS_MIN = 0 * u.deg
    JUPITER_MIN = 0 * u.deg

    # Star tracker keep-out defaults (0 = disabled)
    ST_SUN_MIN = 0 * u.deg
    ST_MOON_MIN = 0 * u.deg
    ST_EARTHLIMB_MIN = 0 * u.deg
    ST1_EARTHLIMB_MIN = None  # Per-tracker override (None = use ST_EARTHLIMB_MIN)
    ST2_EARTHLIMB_MIN = None  # Per-tracker override (None = use ST_EARTHLIMB_MIN)
    # Number of star trackers required to pass (0, 1, or 2). Only trackers
    # carrying an active keep-out are counted, so setting a limit on one
    # tracker alone constrains that tracker rather than being waived by the
    # unconstrained other one.
    ST_REQUIRED = 1
    ROLL = None  # Spacecraft roll about boresight (None = Maximum solar power)

    # Ephemeris sampling. None evaluates the Sun/Moon exactly at every
    # timestep; a time Quantity evaluates them on a grid of that spacing
    # and interpolates, which is much faster (see _precompute).
    EPHEMERIS_STEP = None

    def __init__(self, line1: str, line2: str, **custom_limits):
        """
        Initialize the TLE object with the two lines of TLE data.

        Parameters:
        line1 : str
            The first line of the TLE.
        line2 : str
            The second line of the TLE.
        **custom_limits : dict
            Optional custom limits (e.g., moon_min=30*u.deg).

            ``ephemeris_step`` is not a limit but a speed/accuracy control:
            when set to a time Quantity (e.g. ``60*u.min``) the Sun and Moon
            are evaluated on a grid of that spacing and interpolated, which
            makes long runs several times faster.  Their directions stay
            accurate to well under 0.01 deg against keep-outs measured in
            tens of degrees, and the spacecraft's own motion is never
            interpolated.  ``None`` (the default) evaluates them exactly at
            every timestep.
        """
        # Validate TLE lines
        if not line1 or not line2:
            raise ValueError("TLE lines cannot be empty")

        try:
            self.tle = Satrec.twoline2rv(line1, line2)
        except Exception as e:
            raise ValueError(f"Invalid TLE data: {e}")

        # Validate units on any user-supplied angle parameters
        _angle_params = [
            "moon_min", "sun_min", "earthlimb_min",
            "earthlimb_day_min", "earthlimb_night_min",
            "twilight_margin",
            "mars_min",
            "jupiter_min", "st_sun_min", "st_moon_min",
            "st_earthlimb_min", "st1_earthlimb_min", "st2_earthlimb_min",
            "roll",
        ]
        for key in _angle_params:
            if key in custom_limits and custom_limits[key] is not None:
                _validate_angle(custom_limits[key], key)

        # Set instance limits (use class defaults if not provided)
        self.moon_min = custom_limits.get("moon_min", self.MOON_MIN)
        self.sun_min = custom_limits.get("sun_min", self.SUN_MIN)
        self.earthlimb_min = custom_limits.get("earthlimb_min", self.EARTHLIMB_MIN)
        self.earthlimb_day_min = custom_limits.get(
            "earthlimb_day_min", self.EARTHLIMB_DAY_MIN
        )
        self.earthlimb_night_min = custom_limits.get(
            "earthlimb_night_min", self.EARTHLIMB_NIGHT_MIN
        )
        self.twilight_margin = custom_limits.get(
            "twilight_margin", self.TWILIGHT_MARGIN
        )
        self.use_dynamic_earthlimb = custom_limits.get(
            "use_dynamic_earthlimb", self.USE_DYNAMIC_EARTHLIMB
        )
        self.daynight_mode = custom_limits.get(
            "daynight_mode", self.DAYNIGHT_MODE
        )
        if self.daynight_mode not in ("limb", "subsatellite"):
            raise ValueError(
                f"daynight_mode must be 'limb' or 'subsatellite', "
                f"got {self.daynight_mode!r}"
            )
        self.mars_min = custom_limits.get("mars_min", self.MARS_MIN)
        self.jupiter_min = custom_limits.get("jupiter_min", self.JUPITER_MIN)

        # Star tracker limits
        self.st_sun_min = custom_limits.get("st_sun_min", self.ST_SUN_MIN)
        self.st_moon_min = custom_limits.get("st_moon_min", self.ST_MOON_MIN)
        self.st_earthlimb_min = custom_limits.get(
            "st_earthlimb_min", self.ST_EARTHLIMB_MIN
        )
        # Per-tracker Earth limb overrides (None = use shared st_earthlimb_min)
        self.st1_earthlimb_min = custom_limits.get(
            "st1_earthlimb_min", self.ST1_EARTHLIMB_MIN
        )
        self.st2_earthlimb_min = custom_limits.get(
            "st2_earthlimb_min", self.ST2_EARTHLIMB_MIN
        )
        self.st_required = custom_limits.get("st_required", self.ST_REQUIRED)
        if self.st_required not in (0, 1, 2):
            raise ValueError(f"st_required must be 0, 1, or 2, got {self.st_required}")

        # Spacecraft roll angle about boresight (None = Sun-constrained)
        self.roll = custom_limits.get("roll", self.ROLL)
        if self.roll is not None:
            self.roll = self.roll.to(u.deg)

        # Ephemeris interpolation spacing (None = exact at every timestep)
        self.ephemeris_step = custom_limits.get(
            "ephemeris_step", self.EPHEMERIS_STEP
        )
        if self.ephemeris_step is not None:
            _validate_time_quantity(self.ephemeris_step, "ephemeris_step")

        # One-entry cache for time-dependent quantities reused across calls.
        # The Time object itself is held so its identity stays meaningful:
        # keying on id() alone would let a recycled address serve stale data.
        self._precompute_cache_time = None
        self._precompute_cache_key = None
        self._precompute_cache_value = None

        # Cache of the orbit-sampling grids built by get_visibility_best_roll.
        # These depend only on the time grid and the orbit, not the target,
        # so every target in a run reuses them.
        self._orbit_grid_cache_time = None
        self._orbit_grid_cache_key = None
        self._orbit_grid_cache_value = None

    def __repr__(self) -> str:
        """Return a string representation of the TLE object for debugging."""
        constraints = []
        if self.moon_min > 0 * u.deg:
            constraints.append(f"moon≥{self.moon_min:.0f}")
        if self.sun_min > 0 * u.deg:
            constraints.append(f"sun≥{self.sun_min:.0f}")
        if self.use_dynamic_earthlimb:
            constraints.append("limb=dynamic")
            # The wedge curve reads the illumination angle at the
            # daynight_mode reference point, so the mode matters here too.
            if self.daynight_mode != self.DAYNIGHT_MODE:
                constraints.append(f"daynight={self.daynight_mode}")
        elif self.earthlimb_day_min is not None or self.earthlimb_night_min is not None:
            day_lim = self.earthlimb_day_min if self.earthlimb_day_min is not None else self.earthlimb_min
            night_lim = self.earthlimb_night_min if self.earthlimb_night_min is not None else self.earthlimb_min
            constraints.append(f"limb_day≥{day_lim:.0f}")
            constraints.append(f"limb_night≥{night_lim:.0f}")
            # Compare against the class default so this stays correct if the
            # default changes again, or a subclass picks a different one.
            if self.daynight_mode != self.DAYNIGHT_MODE or self.twilight_margin > 0 * u.deg:
                constraints.append(f"daynight={self.daynight_mode}")
            if self.twilight_margin > 0 * u.deg:
                constraints.append(f"twilight_margin={self.twilight_margin:.0f}")
        elif self.earthlimb_min > 0 * u.deg:
            constraints.append(f"limb≥{self.earthlimb_min:.0f}")
        if self.mars_min > 0 * u.deg:
            constraints.append(f"mars≥{self.mars_min:.0f}")
        if self.jupiter_min > 0 * u.deg:
            constraints.append(f"jupiter≥{self.jupiter_min:.0f}")
        if self.st_sun_min > 0 * u.deg:
            constraints.append(f"st_sun≥{self.st_sun_min:.0f}")
        if self.st_moon_min > 0 * u.deg:
            constraints.append(f"st_moon≥{self.st_moon_min:.0f}")
        if self.st_earthlimb_min > 0 * u.deg:
            if self.st1_earthlimb_min is not None or self.st2_earthlimb_min is not None:
                st1_lim = self._st_earthlimb_min_for(1)
                st2_lim = self._st_earthlimb_min_for(2)
                constraints.append(f"st1_limb≥{st1_lim:.0f}")
                constraints.append(f"st2_limb≥{st2_lim:.0f}")
            else:
                constraints.append(f"st_limb≥{self.st_earthlimb_min:.0f}")
        elif self.st1_earthlimb_min is not None or self.st2_earthlimb_min is not None:
            st1_lim = self._st_earthlimb_min_for(1)
            st2_lim = self._st_earthlimb_min_for(2)
            if st1_lim > 0 * u.deg:
                constraints.append(f"st1_limb≥{st1_lim:.0f}")
            if st2_lim > 0 * u.deg:
                constraints.append(f"st2_limb≥{st2_lim:.0f}")
        if self._st_constraint_active:
            constraints.append(f"st_req={self.st_required}")
        if self.roll is not None:
            constraints.append(f"roll={self.roll:.1f}")

        constraint_str = ", ".join(constraints) if constraints else "default"
        return f"<Visibility: SAT{self.tle.satnum} [{constraint_str}]>"

    def get_period(self) -> u.Quantity:
        """
        Calculate the orbital period at the epoch of the TLE.

        Returns:
        u.Quantity
            The orbital period in minutes.
        """
        return (2 * np.pi / self.tle.no) * u.minute

    def get_state(self, time: Time = None) -> SkyCoord:
        """
        Calculate position and velocity coordinates at a given time.

        Parameters:
        time : astropy.time.Time, optional
            The time at which to calculate the coordinates. If not provided,
            `self.time` is used if it exists.

        Returns:
        SkyCoord
            The ITRS coordinates and velocities at the given time.
        """
        if time is None:
            if not hasattr(self, "time"):
                raise ValueError(
                    "No time parameter specified and self.time is not defined."
                )
            time = self.time

        # Handle scalar and array-shaped times
        shape = time.shape
        time = time.ravel()
        time = time.utc

        # Compute satellite position and velocity using SGP4
        e, xyz, vxyz = self.tle.sgp4_array(time.jd1, time.jd2)
        x, y, z = xyz.T
        vx, vy, vz = vxyz.T

        # Handle SGP4 errors
        errors = e[e != 0]
        if errors.size > 0:
            raise RuntimeError(SGP4_ERRORS[errors[0]])

        # Construct SkyCoord in TEME frame and convert to ITRS
        state = SkyCoord(
            x=x * u.km,
            y=y * u.km,
            z=z * u.km,
            v_x=vx * u.km / u.s,
            v_y=vy * u.km / u.s,
            v_z=vz * u.km / u.s,
            frame=TEME(obstime=time),
        ).itrs

        # Restore original shape if necessary
        return state.reshape(shape) if shape else state[0]

    def get_constraint(self, target_coord: SkyCoord, body: str, time: Time,
                       pre: dict = None) -> bool:
        """
        Calculate whether the constraint for the specified body is met.

        Parameters:
        target_coord : SkyCoord
            The target coordinate to compare with.
        body : str
            The celestial body (e.g., "moon", "sun", "earthlimb", "mars", "jupiter").
        time : astropy.time.Time
            The time at which to calculate the constraint.
        pre : dict, optional
            Precomputed time-dependent data from ``_precompute``.  Supplied
            by ``get_all_constraints`` so that one set of ephemeris and
            SGP4 results covers every body.

        Returns:
        bool
            True if the constraint is satisfied, False otherwise.
        """
        # Map body names to the corresponding minimum separation
        body_min_map = {
            "moon": self.moon_min,
            "sun": self.sun_min,
            "earthlimb": self.earthlimb_min,
            "jupiter": self.jupiter_min,
            "mars": self.mars_min,
        }

        if body not in body_min_map:
            raise ValueError(
                f"Invalid body: {body}. Choose from: {', '.join(body_min_map.keys())}."
            )

        min_angle = body_min_map[body]
        if pre is None:
            pre = self._precompute(time)
        target_unit = self._target_unit(target_coord, time)

        if body in ["moon", "sun", "mars", "jupiter"]:
            # Angular separation between the body and the target
            body_unit = self._body_unit(body, time, pre)
            return (
                self._fast_sep_deg(body_unit, target_unit)
                >= min_angle.to(u.deg).value
            )

        elif body == "earthlimb":
            limb_angle = self._fast_limb_deg(
                target_unit, pre["zenith_unit"], pre["limb_angle_rad"]
            )
            return limb_angle >= self._effective_earthlimb_min_deg(
                target_unit, pre["zenith_unit"], pre["body_units"]["sun"],
                limb_angle_rad=pre["limb_angle_rad"],
            )

    def _get_observer_location(self, time: Time) -> EarthLocation:
        """Helper method to get observer location without side effects."""
        # Temporarily set time for state calculation
        original_time = getattr(self, "time", None)
        self.time = time
        try:
            state = self.get_state()
            return EarthLocation.from_geocentric(state.x, state.y, state.z)
        finally:
            if original_time is not None:
                self.time = original_time
            elif hasattr(self, "time"):
                delattr(self, "time")

    # ------------------------------------------------------------------
    # Internal fast-path helpers: precompute time-dependent data once,
    # then evaluate each target cheaply using numpy dot products.
    # ------------------------------------------------------------------

    @staticmethod
    def _target_unit(target_coord: SkyCoord, time: Time, gcrs_frame=None):
        """Target direction unit vector(s) in GCRS.

        Parameters
        ----------
        target_coord : SkyCoord
            Target coordinate (scalar).
        time : astropy.time.Time
            Observation time(s).
        gcrs_frame : GCRS, optional
            Prebuilt frame to transform into, saving its construction.

        Returns
        -------
        np.ndarray
            (3,) for a scalar time, else (3, N).
        """
        frame = gcrs_frame if gcrs_frame is not None else GCRS(obstime=time)
        xyz = target_coord.transform_to(frame).cartesian.xyz.value
        if time.isscalar:
            return xyz / np.linalg.norm(xyz)
        return xyz / np.linalg.norm(xyz, axis=0, keepdims=True)

    def _body_unit(self, body: str, time: Time, pre: dict):
        """Body direction unit vector(s), from *pre* when it holds them.

        ``_precompute`` only carries the planets whose keep-out is active,
        so a direct request for a switched-off planet falls back to an
        ephemeris lookup.
        """
        if body in pre["body_units"]:
            return pre["body_units"][body]

        body_coord = get_body(
            body, time=time, location=pre["observer_location"]
        )
        xyz = body_coord.cartesian.xyz.value
        if time.isscalar:
            return xyz / np.linalg.norm(xyz)
        return xyz / np.linalg.norm(xyz, axis=0, keepdims=True)

    @staticmethod
    def _fast_sep_deg(a, b):
        """Angular separation in degrees between (3,...) unit vectors.

        Supports shapes (3,)·(3,), (3,N)·(3,N), or (3,1)·(3,N).
        """
        dot = np.sum(a * b, axis=0)
        return np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))

    @staticmethod
    def _fast_limb_deg(target_unit, zenith_unit, limb_angle_rad):
        """Earth limb angle in degrees via geometric calculation.

        elev = arcsin(dot(target, zenith))  [altitude above local horizon]
        limb = arccos(R_earth / observer_dist)
        result = elev + limb
        """
        dot = np.sum(target_unit * zenith_unit, axis=0)
        elev = np.arcsin(np.clip(dot, -1.0, 1.0))
        return np.rad2deg(elev + limb_angle_rad)

    @staticmethod
    def _earthlimb_is_sunlit(target_unit, zenith_unit, sun_unit,
                             limb_angle_rad=None,
                             twilight_margin_deg=0.0):
        """Whether the nearest Earth limb point to the target is sunlit.

        The nearest limb point's outward surface normal is:

            n = cos(limb_angle) * zenith  +  sin(limb_angle) * limb_dir

        where *limb_dir* is the projection of the target direction onto
        the plane perpendicular to the zenith, and *limb_angle* =
        arccos(R_earth / observer_distance).  The limb point is sunlit
        when ``dot(n, sun) > -sin(twilight_margin)``.

        Parameters
        ----------
        target_unit : ndarray, shape (3,) or (3, N)
            Target direction unit vector(s) in GCRS.
        zenith_unit : ndarray, shape (3,) or (3, N)
            Observer zenith direction unit vector(s).
        sun_unit : ndarray, shape (3,) or (3, N)
            Sun direction unit vector(s).
        limb_angle_rad : float or ndarray or None
            Earth-limb half-angle in radians (``arccos(R_earth / d)``).
            When *None*, falls back to a simple horizontal projection
            (ignoring the zenith component of the surface normal).
        twilight_margin_deg : float
            Degrees past the geometric terminator to still classify as
            sunlit.  0 (default) reproduces the original sharp
            terminator.  18 is analogous to astronomical twilight.

        Returns
        -------
        bool or ndarray of bool
            True where the nearest limb point is sunlit.
        """
        dot_tz = np.sum(target_unit * zenith_unit, axis=0)
        if target_unit.ndim == 1:
            proj = target_unit - zenith_unit * dot_tz
        else:
            proj = target_unit - zenith_unit * dot_tz[np.newaxis, :]
        proj_norm = np.linalg.norm(proj, axis=0, keepdims=True)
        limb_unit = proj / np.where(proj_norm < 1e-12, 1.0, proj_norm)

        threshold = -np.sin(np.deg2rad(twilight_margin_deg))

        if limb_angle_rad is None:
            # Legacy fallback: horizontal projection only
            return np.sum(limb_unit * sun_unit, axis=0) > threshold

        cos_la = np.cos(limb_angle_rad)
        sin_la = np.sin(limb_angle_rad)
        # Surface normal of the limb point
        dot_n_sun = cos_la * np.sum(zenith_unit * sun_unit, axis=0) + \
                    sin_la * np.sum(limb_unit * sun_unit, axis=0)
        return dot_n_sun > threshold

    @staticmethod
    def _get_earth_illumination_angle(target_unit, zenith_unit, sun_unit,
                                      limb_angle_rad=None):
        """Earth illumination angle at the nearest limb point, in degrees.

        This is the solar zenith angle at the Earth surface point the
        boresight grazes: the angle between that point's outward surface
        normal and the direction to the Sun.  0 deg is the subsolar point
        (brightest limb), 90 deg is the terminator, 180 deg is the
        antisolar point (fully dark limb).

        The `daynight_mode="limb"` half of
        `_daynight_illumination_angle`; call that instead of this to
        respect the configured reference point.

        The surface normal is built exactly as in ``_earthlimb_is_sunlit``,

            n = cos(limb_angle) * zenith  +  sin(limb_angle) * limb_dir

        Parameters
        ----------
        target_unit : ndarray, shape (3,) or (3, N)
            Target direction unit vector(s) in GCRS.
        zenith_unit : ndarray, shape (3,) or (3, N)
            Observer zenith direction unit vector(s).
        sun_unit : ndarray, shape (3,) or (3, N)
            Sun direction unit vector(s).
        limb_angle_rad : float or ndarray or None
            Earth-limb half-angle in radians (``arccos(R_earth / d)``).
            When *None*, falls back to a simple horizontal projection
            (ignoring the zenith component of the surface normal).

        Returns
        -------
        float or ndarray
            Illumination angle in degrees, in [0, 180].
        """
        dot_tz = np.sum(target_unit * zenith_unit, axis=0)
        if target_unit.ndim == 1:
            proj = target_unit - zenith_unit * dot_tz
        else:
            proj = target_unit - zenith_unit * dot_tz[np.newaxis, :]
        proj_norm = np.linalg.norm(proj, axis=0, keepdims=True)
        limb_unit = proj / np.where(proj_norm < 1e-12, 1.0, proj_norm)

        if limb_angle_rad is None:
            # Legacy fallback: horizontal projection only
            dot_n_sun = np.sum(limb_unit * sun_unit, axis=0)
        else:
            dot_n_sun = (
                np.cos(limb_angle_rad) * np.sum(zenith_unit * sun_unit, axis=0)
                + np.sin(limb_angle_rad) * np.sum(limb_unit * sun_unit, axis=0)
            )
        return np.rad2deg(np.arccos(np.clip(dot_n_sun, -1.0, 1.0)))

    @staticmethod
    def _dynamic_earthlimb_min_deg(illumination_deg):
        """DPC wedge keep-out in degrees above the limb.

        Piecewise function of the Earth illumination angle (see
        ``_get_earth_illumination_angle``), given from the Earth centre as

        ===================  ===========================
        Illumination angle   Keep-out from Earth centre
        ===================  ===========================
        <= 78 deg            110 deg
        78 - 89 deg          linear rule 1 (110 → 82 deg)
        89 - 90 deg          linear rule 2 (82 → 75 deg)
        >= 90 deg            75 deg
        ===================  ===========================

        Both rules are straight lines through the anchor points, so the
        curve is continuous at 78, 89 and 90 deg.

        The nominal Earth angular radius (66 deg) is subtracted so the
        result is referenced to the limb like every other Earth limb
        angle in this class.

        The input is wrapped into [0, 180] first, so the curve is
        symmetric about the sub-solar and anti-solar directions: -78,
        +78 and +282 deg all give the same keep-out.

        Parameters
        ----------
        illumination_deg : float or ndarray
            Earth illumination angle(s) in degrees.  Any angle is
            accepted; it is folded into [0, 180] before evaluation.

        Returns
        -------
        float or ndarray
            Minimum allowed angle above the Earth limb, in degrees.
        """
        # Fold onto [0, 180]: the keep-out depends only on how far the
        # limb point is from the sub-solar direction, not on which side.
        illum = np.abs(
            (np.asarray(illumination_deg, dtype=float) + 180.0) % 360.0 - 180.0
        )
        keepout = np.where(
            illum < _DYN_BRIGHT_ILLUM_DEG,
            _DYN_BRIGHT_KEEPOUT_DEG,
            np.where(
                illum <= _DYN_KNEE_ILLUM_DEG,
                _DYN_RULE1_M * illum + _DYN_RULE1_B,
                np.where(
                    illum < _DYN_DARK_ILLUM_DEG,
                    _DYN_RULE2_M * illum + _DYN_RULE2_B,
                    _DYN_DARK_KEEPOUT_DEG,
                ),
            ),
        )
        return keepout - _DYN_EARTH_ANGULAR_RADIUS_DEG

    @staticmethod
    def _subsatellite_is_sunlit(zenith_unit, sun_unit,
                                twilight_margin_deg=0.0):
        """Whether the subsatellite point (ground below spacecraft) is sunlit.

        The subsatellite point is the point on Earth's surface directly
        below the spacecraft.  It is sunlit when the angle between the
        zenith direction (observer → away from Earth centre) and the Sun
        direction is less than 90° (plus an optional twilight margin).

        Geometrically: ``dot(zenith, sun) > -sin(twilight_margin)``.

        Parameters
        ----------
        zenith_unit : ndarray, shape (3,) or (3, N)
            Observer zenith direction unit vector(s).
        sun_unit : ndarray, shape (3,) or (3, N)
            Sun direction unit vector(s).
        twilight_margin_deg : float
            Degrees past the geometric terminator to still classify as
            sunlit.  0 (default) gives a sharp day/night boundary.

        Returns
        -------
        bool or ndarray of bool
            True where the subsatellite point is sunlit.
        """
        threshold = -np.sin(np.deg2rad(twilight_margin_deg))
        dot_zs = np.sum(zenith_unit * sun_unit, axis=0)
        return dot_zs > threshold

    @staticmethod
    def _subsatellite_illumination_angle(zenith_unit, sun_unit):
        """Earth illumination angle at the subsatellite point, in degrees.

        The solar zenith angle on the ground directly below the
        spacecraft: the angle between that point's outward surface
        normal, which is just the observer zenith, and the direction
        to the Sun. 0 deg is the subsolar point, 90 deg the terminator,
        180 deg the antisolar point.

        The subsatellite counterpart of
        `_get_earth_illumination_angle`. Like that method it agrees
        with its own sunlit test: this angle is < 90 deg
        exactly where `_subsatellite_is_sunlit` is True at twilight.

        Parameters
        ----------
        zenith_unit : ndarray, shape (3,) or (3, N)
            Observer zenith direction unit vector(s).
        sun_unit : ndarray, shape (3,) or (3, N)
            Sun direction unit vector(s).

        Returns
        -------
        float or ndarray
            Illumination angle in degrees, in [0, 180].
        """
        dot_zs = np.sum(zenith_unit * sun_unit, axis=0)
        return np.rad2deg(np.arccos(np.clip(dot_zs, -1.0, 1.0)))

    def _daynight_is_sunlit(self, target_unit, zenith_unit, sun_unit,
                            limb_angle_rad=None):
        """Whether the Earth below counts as sunlit, per ``daynight_mode``.

        The single source of truth for the day/night split, so the
        threshold applied by ``_effective_earthlimb_min_deg`` and the
        ``[day]``/``[night]`` label printed by ``summary`` can never
        disagree.

        * ``"subsatellite"`` (default): the ground directly below the
          spacecraft, independent of where the boresight points.
        * ``"limb"``: the nearest limb point to the target direction,
          the patch of Earth the boresight actually grazes.

        Parameters
        ----------
        target_unit : ndarray, shape (3,) or (3, N)
            Target direction unit vector(s) in GCRS.  Unused in
            ``"subsatellite"`` mode.
        zenith_unit : ndarray, shape (3,) or (3, N)
            Observer zenith direction unit vector(s).
        sun_unit : ndarray, shape (3,) or (3, N)
            Sun direction unit vector(s).
        limb_angle_rad : float or ndarray or None
            Earth-limb half-angle in radians.  Only used in ``"limb"`` mode.

        Returns
        -------
        bool or ndarray of bool
            True where the relevant Earth point is sunlit.
        """
        twilight_deg = self.twilight_margin.to(u.deg).value
        if self.daynight_mode == "subsatellite":
            return self._subsatellite_is_sunlit(
                zenith_unit, sun_unit,
                twilight_margin_deg=twilight_deg,
            )
        return self._earthlimb_is_sunlit(
            target_unit, zenith_unit, sun_unit,
            limb_angle_rad=limb_angle_rad,
            twilight_margin_deg=twilight_deg,
        )

    def _daynight_illumination_angle(self, target_unit, zenith_unit, sun_unit,
                                     limb_angle_rad=None):
        """Earth illumination angle in degrees, per `daynight_mode`.

        The single source of truth for the continuous Earth
        illumination angle, the way `_daynight_is_sunlit` is for the
        binary day/night split. Both read the same
        `self.daynight_mode`, so the dynamic DPC wedge
        (`use_dynamic_earthlimb`) and the day/night step pair always
        measure the Sun at the same point on Earth.

        * `"subsatellite"` (default): the ground directly below the
          spacecraft, independent of where the boresight points.
        * `"limb"`: the nearest limb point to the target direction,
          the patch of Earth the boresight actually grazes.

        `twilight_margin` is deliberately not applied here. It shifts
        a hard day/night boundary, and this angle is continuous, the
        DPC wedge curve does its own smooth roll-off near the terminator.

        Parameters
        ----------
        target_unit : ndarray, shape (3,) or (3, N)
            Target direction unit vector(s) in GCRS.  Unused in
            ``"subsatellite"`` mode.
        zenith_unit : ndarray, shape (3,) or (3, N)
            Observer zenith direction unit vector(s).
        sun_unit : ndarray, shape (3,) or (3, N)
            Sun direction unit vector(s).
        limb_angle_rad : float or ndarray or None
            Earth-limb half-angle in radians.  Only used in ``"limb"`` mode.

        Returns
        -------
        float or ndarray
            Illumination angle in degrees, in [0, 180].
        """
        if self.daynight_mode == "subsatellite":
            return self._subsatellite_illumination_angle(zenith_unit, sun_unit)
        return self._get_earth_illumination_angle(
            target_unit, zenith_unit, sun_unit,
            limb_angle_rad=limb_angle_rad,
        )

    def _effective_earthlimb_min_deg(self, target_unit, zenith_unit, sun_unit,
                                     limb_angle_rad=None):
        """Per-timestep effective Earth limb threshold in degrees.

        When ``use_dynamic_earthlimb`` is True, the threshold follows the
        DPC wedge keep-out curve as a function of the Earth illumination
        angle and the day/night pair below is bypassed.

        Otherwise, when `earthlimb_day_min` or `earthlimb_night_min` is set,
        returns a scalar or array of thresholds that depend on whether
        the observer is over sunlit or shadowed Earth.

        Both models take their Earth reference point from
        `self.daynight_mode`:

        * `"subsatellite"` (default): subsatellite point directly below
          the spacecraft.
        * `"limb"`: nearest limb point to the target direction.

        Otherwise returns a plain scalar from ``earthlimb_min``.

        Parameters
        ----------
        limb_angle_rad : float or ndarray or None
            Earth-limb half-angle in radians, forwarded to
            ``_earthlimb_is_sunlit``.
        """
        if self.use_dynamic_earthlimb:
            # DPC wedge: continuous threshold from the illumination angle
            # at the daynight_mode reference point. Takes precedence
            # over the day/night pair.
            return self._dynamic_earthlimb_min_deg(
                self._daynight_illumination_angle(
                    target_unit, zenith_unit, sun_unit,
                    limb_angle_rad=limb_angle_rad,
                )
            )

        if self.earthlimb_day_min is None and self.earthlimb_night_min is None:
            return self.earthlimb_min.to(u.deg).value

        day_deg = (
            self.earthlimb_day_min.to(u.deg).value
            if self.earthlimb_day_min is not None
            else self.earthlimb_min.to(u.deg).value
        )
        night_deg = (
            self.earthlimb_night_min.to(u.deg).value
            if self.earthlimb_night_min is not None
            else self.earthlimb_min.to(u.deg).value
        )

        sunlit = self._daynight_is_sunlit(
            target_unit, zenith_unit, sun_unit,
            limb_angle_rad=limb_angle_rad,
        )
        return np.where(sunlit, day_deg, night_deg)

    def _active_bodies(self) -> list:
        """Names of the bodies whose directions the constraints need."""
        bodies = ["moon", "sun"]
        if self.mars_min > 0 * u.deg:
            bodies.append("mars")
        if self.jupiter_min > 0 * u.deg:
            bodies.append("jupiter")
        return bodies

    def _interpolated_body_units(self, time: Time, obs_xyz, bodies) -> dict:
        """Body direction unit vectors from an interpolated ephemeris.

        The *geocentric* body vectors are smooth and slow-moving, so they
        interpolate well over hours.  The fast-moving part — the
        spacecraft's parallax, which swings with the orbital period — is
        applied exactly by subtracting the true spacecraft position, so
        the orbital signal is never interpolated.

        With ``ephemeris_step`` at one hour this agrees with the exact
        ephemeris to better than 0.001 deg for the Moon and 0.0001 deg
        for the Sun, against keep-outs measured in tens of degrees.

        Parameters
        ----------
        time : astropy.time.Time
            Observation times (array).
        obs_xyz : ndarray, shape (3, N)
            Spacecraft GCRS position in metres at those times.
        bodies : list of str
            Body names to evaluate.

        Returns
        -------
        dict
            Body name → (3, N) unit vector array.
        """
        jd = time.jd
        step = self.ephemeris_step.to(u.day).value
        span = jd.max() - jd.min()
        n_coarse = max(int(np.ceil(span / step)) + 3, 4)
        # Pad by one step so every requested time is interpolated, never
        # extrapolated.
        coarse_jd = np.linspace(jd.min() - step, jd.max() + step, n_coarse)
        coarse = Time(coarse_jd, format="jd", scale=time.scale)

        body_units = {}
        for name in bodies:
            geocentric = get_body(name, time=coarse).cartesian.xyz.to(u.m).value
            topocentric = np.stack(
                [np.interp(jd, coarse_jd, row) for row in geocentric]
            ) - obs_xyz
            body_units[name] = topocentric / np.linalg.norm(
                topocentric, axis=0, keepdims=True
            )
        return body_units

    def _precompute(self, time: Time) -> dict:
        """Precompute time-dependent quantities shared across targets.

        Everything in the returned dict depends only on the observation
        time(s) and satellite orbit, not on the science target.  Passing
        this dict to ``_get_visibility_single`` avoids redundant SGP4
        propagation, ephemeris lookups, and coordinate transforms.

        Astropy's ephemeris and frame machinery has a large per-call
        overhead, so this is much cheaper called once on a long time
        array than repeatedly on short ones.

        When ``ephemeris_step`` is set the Sun and Moon are evaluated on
        a grid of that spacing and interpolated; see
        ``_interpolated_body_units``.
        """
        # Cache on the identity of the Time object: common workflows reuse the
        # same grid across calls (e.g. many targets on one time grid).  The
        # object is held in _precompute_cache_time so that its address cannot
        # be recycled by a later Time while this entry is live.
        cache_key = (
            bool(self.mars_min > 0 * u.deg),
            bool(self.jupiter_min > 0 * u.deg),
        )

        if (self._precompute_cache_time is time and
                cache_key == self._precompute_cache_key and
                self._precompute_cache_value is not None):
            return self._precompute_cache_value

        observer_location = self._get_observer_location(time)

        # Observer GCRS position → zenith direction + Earth limb angle
        obs_gcrs = observer_location.get_gcrs(obstime=time)
        obs_xyz = obs_gcrs.cartesian.xyz.to(u.m).value  # (3,) or (3, N)
        if time.isscalar:
            obs_dist = np.linalg.norm(obs_xyz)
            zenith_unit = obs_xyz / obs_dist
        else:
            obs_dist = np.linalg.norm(obs_xyz, axis=0)  # (N,)
            zenith_unit = obs_xyz / obs_dist[np.newaxis, :]  # (3, N)

        with np.errstate(invalid="ignore"):
            limb_angle_rad = np.arccos(_R_EARTH_M / obs_dist)  # scalar or (N,)

        # Body direction unit vectors (normalised cartesian xyz)
        bodies = self._active_bodies()
        if (self.ephemeris_step is not None and not time.isscalar
                and time.size >= 8):
            body_units = self._interpolated_body_units(time, obs_xyz, bodies)
        else:
            body_units = {}
            for name in bodies:
                body = get_body(name, time=time, location=observer_location)
                xyz = body.cartesian.xyz.value
                if time.isscalar:
                    body_units[name] = xyz / np.linalg.norm(xyz)
                else:
                    body_units[name] = xyz / np.linalg.norm(
                        xyz, axis=0, keepdims=True
                    )

        pre = {
            "observer_location": observer_location,
            "body_units": body_units,
            "zenith_unit": zenith_unit,
            "limb_angle_rad": limb_angle_rad,
        }
        self._precompute_cache_time = time
        self._precompute_cache_key = cache_key
        self._precompute_cache_value = pre
        return pre

    def _get_visibility_single(
        self, target_coord: SkyCoord, time: Time, pre: dict,
        effective_roll=None, gcrs_frame=None,
    ):
        """Visibility for one scalar target using precomputed time data."""
        body_units = pre["body_units"]
        zenith_unit = pre["zenith_unit"]
        limb_rad = pre["limb_angle_rad"]

        # Target direction unit vector(s) in GCRS.
        frame = gcrs_frame if gcrs_frame is not None else GCRS(obstime=time)
        tgt_gcrs = target_coord.transform_to(frame)
        tgt_xyz = tgt_gcrs.cartesian.xyz.value
        if time.isscalar:
            tgt_unit = tgt_xyz / np.linalg.norm(tgt_xyz)  # (3,)
            tgt_b = tgt_unit
        else:
            tgt_b = tgt_xyz / np.linalg.norm(
                tgt_xyz, axis=0, keepdims=True
            )  # (3, N)
            tgt_unit = tgt_b[:, 0].copy()

        # Boresight body constraints via fast dot-product separation
        moon_deg = self.moon_min.to(u.deg).value
        sun_deg = self.sun_min.to(u.deg).value
        limb_threshold = self._effective_earthlimb_min_deg(
            tgt_b, zenith_unit, body_units["sun"], limb_angle_rad=limb_rad
        )

        result = self._fast_sep_deg(body_units["moon"], tgt_b) >= moon_deg
        result &= self._fast_sep_deg(body_units["sun"], tgt_b) >= sun_deg
        result &= self._fast_limb_deg(tgt_b, zenith_unit, limb_rad) >= limb_threshold

        if self.mars_min > 0 * u.deg:
            result &= (
                self._fast_sep_deg(body_units["mars"], tgt_b)
                >= self.mars_min.to(u.deg).value
            )
        if self.jupiter_min > 0 * u.deg:
            result &= (
                self._fast_sep_deg(body_units["jupiter"], tgt_b)
                >= self.jupiter_min.to(u.deg).value
            )

        # Star tracker constraints
        if self._st_constraint_active:
            st_result = self._get_st_constraint_fast(
                tgt_unit, time, pre, effective_roll=effective_roll,
            )
            result = result & st_result

        if time.isscalar:
            return bool(result)
        return np.asarray(result)

    def _st_tracker_separations(self, tgt_unit, time, pre, *,
                                effective_roll=None):
        """Keep-out separations for both star trackers, in degrees.

        The single place the tracker attitude and geometry are evaluated.
        ``_get_st_constraint_fast`` reduces this to a pass/fail verdict and
        ``get_star_tracker_breakdown`` reports it check by check, so a
        breakdown can never disagree with the verdict it explains.

        All three separations are returned for each tracker whether or not
        the corresponding keep-out is switched on; they are dot products
        over vectors already in hand, so computing the unused ones is free.

        Parameters
        ----------
        tgt_unit : np.ndarray
            Target direction as (3,) unit vector in GCRS.
        time : Time
            Observation time (scalar or array).
        pre : dict
            Precomputed data from ``_precompute()``.
        effective_roll : Quantity or None
            Roll angle to use.  Scalar, or one angle per timestep to
            evaluate a changing attitude, as ``get_visibility_best_roll``
            returns.  If ``None``, falls back to ``self.roll``.

        Returns
        -------
        separations : dict
            ``{1: {"sun_angle": ..., "moon_angle": ..., "earthlimb_angle": ...},
            2: {...}}``, each value a float or (N,) array of degrees.  NaN
            wherever the attitude is degenerate, which compares False
            against any threshold.
        degenerate : bool or np.ndarray of bool
            True where the attitude is undefined (target aligned with the
            Sun, so ``Sun x Z`` does not define a payload +Y).
        """
        roll = effective_roll if effective_roll is not None else self.roll
        body_units = pre["body_units"]
        zenith_unit = pre["zenith_unit"]
        limb_rad = pre["limb_angle_rad"]
        sun_vec = body_units["sun"]

        # Compute payload attitude ONCE for both trackers
        if roll is not None:
            roll_rad = np.asarray(roll.to(u.rad).value, dtype=float)
            if roll_rad.ndim == 0:
                x_payload, y_payload = self._roll_attitude(
                    tgt_unit, float(roll_rad)
                )
                if time.isscalar:
                    z_col = tgt_unit
                    degenerate = False
                else:
                    N = len(time)
                    z_col = np.tile(tgt_unit.reshape(3, 1), (1, N))
                    x_payload = x_payload[:, np.newaxis]  # (3,) → (3,1)
                    y_payload = y_payload[:, np.newaxis]  # (3,) → (3,1)
                    degenerate = np.zeros(N, dtype=bool)
            else:
                # One roll per timestep, the shape get_visibility_best_roll
                # returns. The attitude changes from step to step, so build
                # every one of them in a single pass.
                if time.isscalar or roll_rad.shape != time.shape:
                    raise ValueError(
                        "an array-valued roll needs one entry per timestep, "
                        f"got shape {roll_rad.shape} for {time.shape} times"
                    )
                N = len(time)
                z_col = np.tile(tgt_unit.reshape(3, 1), (1, N))
                # _roll_attitude_batch returns (N, 3); transpose so each
                # column pairs with the timestep its roll came from.
                x_all, y_all = self._roll_attitude_batch(tgt_unit, roll_rad)
                x_payload, y_payload = x_all.T, y_all.T
                # A NaN roll means no roll angle was found for that step, so
                # the attitude is as undefined there as a degenerate
                # Sun-constrained one.
                degenerate = ~np.isfinite(roll_rad)
        elif time.isscalar:
            z_col = tgt_unit
            y_payload = np.cross(sun_vec, tgt_unit)
            y_norm = np.linalg.norm(y_payload)
            degenerate = bool(y_norm < 1e-10)
            y_payload = y_payload / (1.0 if degenerate else y_norm)
            x_payload = np.cross(y_payload, tgt_unit)
            x_norm = np.linalg.norm(x_payload)
            x_payload = x_payload / (1.0 if x_norm < 1e-10 else x_norm)
        else:
            N = len(time)
            z_col = np.tile(tgt_unit.reshape(3, 1), (1, N))
            y_payload = np.cross(sun_vec, z_col, axis=0)
            y_norms = np.linalg.norm(y_payload, axis=0, keepdims=True)
            degenerate = (y_norms < 1e-10).ravel()
            y_payload = y_payload / np.where(y_norms < 1e-10, 1.0, y_norms)
            x_payload = np.cross(y_payload, z_col, axis=0)
            x_norms = np.linalg.norm(x_payload, axis=0, keepdims=True)
            x_payload = x_payload / np.where(x_norms < 1e-10, 1.0, x_norms)

        separations = {}
        for tracker in [1, 2]:
            st_body = np.array(self._get_star_tracker_body_xyz(tracker))

            # Rotate body-frame vector to ECI
            st_eci = (
                x_payload * st_body[0]
                + y_payload * st_body[1]
                + z_col * st_body[2]
            )

            if time.isscalar:
                st_norm = np.linalg.norm(st_eci)
                if st_norm < 1e-10 or degenerate:
                    st_eci = np.full(3, np.nan)
                else:
                    st_eci = st_eci / st_norm
            else:
                st_eci = st_eci / np.linalg.norm(st_eci, axis=0, keepdims=True)
                st_eci[:, degenerate] = np.nan

            with np.errstate(invalid="ignore"):
                separations[tracker] = {
                    "sun_angle": self._fast_sep_deg(st_eci, body_units["sun"]),
                    "moon_angle": self._fast_sep_deg(st_eci, body_units["moon"]),
                    "earthlimb_angle": self._fast_limb_deg(
                        st_eci, zenith_unit, limb_rad
                    ),
                }

        return separations, degenerate

    def _get_st_constraint_fast(self, tgt_unit, time, pre, *,
                                effective_roll=None):
        """Star tracker constraint check using pure numpy.

        Computes the payload attitude matrix once and applies it to both
        tracker boresight vectors.  Angular separations use dot products
        instead of SkyCoord.separation().

        Parameters
        ----------
        tgt_unit : np.ndarray
            Target direction as (3,) unit vector in GCRS.
        time : Time
            Observation time (scalar or array).
        pre : dict
            Precomputed data from ``_precompute()``.
        effective_roll : Quantity or None
            Roll angle to use.  If ``None``, falls back to ``self.roll``.
        """
        separations, degenerate = self._st_tracker_separations(
            tgt_unit, time, pre, effective_roll=effective_roll,
        )

        if time.isscalar and degenerate:
            return False  # degenerate: both trackers fail

        tracker_results = []
        for tracker in [1, 2]:
            if time.isscalar:
                tracker_ok = True
            else:
                tracker_ok = np.ones(time.shape, dtype=bool)

            for _, limit, key in self._st_checks_for(tracker):
                sep = separations[tracker].get(key)
                if sep is None:
                    continue
                tracker_ok = tracker_ok & (sep >= limit.to(u.deg).value)

            tracker_results.append(tracker_ok)

        combined = self._combine_tracker_results(*tracker_results)

        if time.isscalar:
            return bool(combined)
        return combined

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_visibility(self, target_coord: SkyCoord, time: Time, roll=None):
        """
        Calculate whether the target is visible based on all constraints.

        Parameters:
        -----------
        target_coord : SkyCoord or list of SkyCoord
            The target coordinate(s) to compare with. If a list is provided,
            visibility is computed for each target independently and an array
            of results is returned.
        time : astropy.time.Time
            The time at which to calculate the constraint. Can be scalar or array.
        roll : Quantity, optional
            Spacecraft roll angle about boresight.  Overrides the instance
            ``roll`` for this call only.  ``None`` (default) keeps the
            instance value (which itself defaults to Sun-constrained when
            not set at construction time).

        Returns:
        --------
        bool or np.ndarray
            True if the target is visible, False otherwise.
            - Scalar coord + scalar time → bool
            - Scalar coord + array time (M,) → np.ndarray of bool, shape (M,)
            - N coords (list or array) + scalar time → np.ndarray of bool, shape (N,)
            - N coords (list or array) + array time (M,) → np.ndarray of bool, shape (N, M)
        """
        # Resolve effective roll without mutating instance state
        if roll is not None:
            _validate_angle(roll, "roll")
            effective_roll = roll.to(u.deg)
        else:
            effective_roll = self.roll
        return self._get_visibility_inner(target_coord, time, effective_roll)

    def _get_visibility_inner(self, target_coord: SkyCoord, time: Time,
                               effective_roll=None):
        """Core visibility logic (called by get_visibility).

        Parameters
        ----------
        effective_roll : Quantity or None
            Roll angle to use for this evaluation.  Passed through to
            ``_get_visibility_single`` → ``_get_st_constraint_fast``
            so that instance state is never mutated.
        """
        # Precompute satellite state and body positions once for all targets
        pre = self._precompute(time)
        gcrs_frame = GCRS(obstime=time)

        # Handle multiple target coordinates (list or array SkyCoord)
        # Each target defines a different boresight, so must be evaluated independently
        if isinstance(target_coord, list):
            return np.array(
                [self._get_visibility_single(tc, time, pre, effective_roll,
                                             gcrs_frame)
                 for tc in target_coord]
            )
        if hasattr(target_coord, "shape") and target_coord.shape != ():
            return np.array(
                [
                    self._get_visibility_single(target_coord[i], time, pre,
                                                effective_roll, gcrs_frame)
                    for i in range(len(target_coord))
                ]
            )

        return self._get_visibility_single(target_coord, time, pre,
                                           effective_roll, gcrs_frame)

    def _orbit_sampling_grid(self, time: Time, orbit_time_step) -> dict:
        """Orbit grouping and sampled orbit windows for ``get_visibility_best_roll``.

        Splits `time` into orbits, builds the sampled window used to search
        for the best roll in each, and precomputes the time-dependent
        quantities for both the input grid and every orbit window.

        Astropy's ephemeris and frame code is dominated by per-call
        overhead, so both precomputes are done as a single vectorised call
        instead of once per orbit.  Nothing here depends on the science
        target, so the result is cached and reused by every target
        evaluated against the same time grid.

        Parameters
        ----------
        time : astropy.time.Time
            Input observation times (array).
        orbit_time_step : Quantity
            Sampling interval within each orbit window.

        Returns
        -------
        dict
            orbit_ids : ndarray of the distinct orbit numbers.
            chunk_indices : list of index arrays into *time*, one per orbit.
            centers : Time array of orbit-window centres.
            n_orbit_samp : samples per orbit window.
            pre_orbit_all : precompute over every orbit window, laid out
                orbit by orbit (orbit *i* occupies samples
                ``i * n_orbit_samp`` to ``(i + 1) * n_orbit_samp``).
            pre_input_all : precompute over the input grid.
        """
        cache_key = (
            orbit_time_step.to(u.min).value,
            bool(self.mars_min > 0 * u.deg),
            bool(self.jupiter_min > 0 * u.deg),
            self.ephemeris_step,
        )
        if (self._orbit_grid_cache_time is time and
                cache_key == self._orbit_grid_cache_key and
                self._orbit_grid_cache_value is not None):
            return self._orbit_grid_cache_value

        period = self.get_period()
        period_day = period.to(u.day).value
        period_min = period.to(u.min).value
        half_p_min = period_min / 2

        t0_jd = np.min(time.jd)
        orbit_id = np.floor((time.jd - t0_jd) / period_day).astype(int)
        orbit_ids = np.unique(orbit_id)
        chunk_indices = [np.where(orbit_id == oid)[0] for oid in orbit_ids]

        center_jd = np.array([
            (time.jd[idx].min() + time.jd[idx].max()) / 2
            for idx in chunk_indices
        ])
        centers = Time(center_jd, format="jd", scale=time.scale)

        orb_step_min = orbit_time_step.to(u.min).value
        n_orbit_samp = int(np.ceil(period_min / orb_step_min)) + 1
        offsets = np.linspace(-half_p_min, half_p_min, n_orbit_samp) * u.min

        # (n_orbits, n_orbit_samp) flattened orbit by orbit
        orbit_times = (centers.reshape(-1, 1) + offsets).ravel()

        grid = {
            "orbit_ids": orbit_ids,
            "chunk_indices": chunk_indices,
            "centers": centers,
            "n_orbit_samp": n_orbit_samp,
            "pre_orbit_all": self._precompute(orbit_times),
            "pre_input_all": self._precompute(time),
        }
        self._orbit_grid_cache_time = time
        self._orbit_grid_cache_key = cache_key
        self._orbit_grid_cache_value = grid
        return grid

    def get_visibility_best_roll(
        self, target_coord: SkyCoord, time: Time, roll_step=2 * u.deg,
        orbit_time_step=1 * u.min,
    ) -> dict:
        """
        Calculate visibility using the optimal roll angle for each orbit.

        For each input time, determines which orbit it falls in, finds the
        best fixed roll angle for that orbit by sweeping ``roll_step``-spaced
        angles over a full orbital period sampled every ``orbit_time_step``,
        then evaluates visibility at the specific input time using that roll.

        The best roll is the one satisfying all star-tracker keep-out
        constraints at the greatest number of boresight-visible orbit
        timesteps, with solar array power as tiebreaker.

        Parameters
        ----------
        target_coord : SkyCoord
            The science target coordinate (+Z boresight direction).
        time : Time
            Observation time(s).  Scalar or array.
        roll_step : Quantity, optional
            Roll sweep resolution (default 2 deg).
        orbit_time_step : Quantity, optional
            Time step for the internal orbit sampling used to determine
            the optimal roll (default 1 min).

        Returns
        -------
        dict
            visible : bool or np.ndarray
                True where all constraints (boresight + ST with orbit-best
                roll) pass.
            boresight_visible : bool or np.ndarray
                True where boresight constraints alone pass (before ST/roll).
            roll_deg : float or np.ndarray
                Orbit-optimal roll angle in degrees (NaN where not visible).
                Constant within each orbit.
            n_st_pass : int or np.ndarray
                Number of star trackers passing at the chosen roll, counting
                only those with an active keep-out, so 0 to 2.
            solar_power_frac : float or np.ndarray
                Solar panel power fraction at the chosen roll (NaN if not
                visible).

        Notes
        -----
        The orbit grouping and the sampled orbit windows do not depend on
        the target, so they are cached on the instance: evaluating many
        targets against the same time grid reuses one set of ephemeris and
        SGP4 results.  Setting ``ephemeris_step`` at construction cuts that
        shared cost further.

        Examples
        --------
        >>> vis = Visibility(line1, line2,
        ...                  st_sun_min=44*u.deg,
        ...                  st_earthlimb_min=30*u.deg,
        ...                  st_moon_min=12*u.deg)
        >>> target = SkyCoord(ra=79.17, dec=45.99, unit="deg")
        >>> times = Time("2026-02-15T18:00:00") + np.arange(97) * u.min
        >>> result = vis.get_visibility_best_roll(target, times)
        >>> print(result['visible'].sum(), "visible time steps")
        >>> print("Roll angles used:", result['roll_deg'])
        """
        _validate_angle(roll_step, "roll_step")
        _validate_time_quantity(orbit_time_step, "orbit_time_step")
        if not roll_step.isscalar:
            raise ValueError("roll_step must be a scalar Quantity")
        if not orbit_time_step.isscalar:
            raise ValueError("orbit_time_step must be a scalar Quantity")

        is_scalar = time.isscalar
        if is_scalar:
            time = Time([time])
        N_input = len(time)

        # Target direction in GCRS for input boresight checks.
        tgt_gcrs = target_coord.transform_to(GCRS(obstime=time))
        tgt_xyz = tgt_gcrs.cartesian.xyz.value
        tgt_b_all = tgt_xyz / np.linalg.norm(
            tgt_xyz, axis=0, keepdims=True
        )  # (3, N_input)

        # Roll setup
        step_deg = roll_step.to(u.deg).value
        roll_degs = np.arange(0, 360, step_deg)

        st1_body = np.array(self._get_star_tracker_body_xyz(1))
        st2_body = np.array(self._get_star_tracker_body_xyz(2))
        st1_checks = self._st_checks_for(1)
        st2_checks = self._st_checks_for(2)

        # Output arrays
        out_visible = np.zeros(N_input, dtype=bool)
        out_boresight = np.zeros(N_input, dtype=bool)
        out_roll = np.full(N_input, np.nan)
        out_nst = np.zeros(N_input, dtype=int)
        out_power = np.full(N_input, np.nan)

        # ── Fast path: no ST constraints ───────────────────────────
        if not self._st_constraint_active:
            pre = self._precompute(time)
            bu = pre["body_units"]
            bs = self._boresight_ok(
                tgt_b_all, bu, pre["zenith_unit"], pre["limb_angle_rad"]
            )
            out_visible = bs.copy()
            out_boresight = bs.copy()
            if is_scalar:
                return {
                    "visible": bool(out_visible[0]),
                    "boresight_visible": bool(out_boresight[0]),
                    "roll_deg": float(out_roll[0]),
                    "n_st_pass": int(out_nst[0]),
                    "solar_power_frac": float(out_power[0]),
                }
            return {
                "visible": out_visible,
                "boresight_visible": out_boresight,
                "roll_deg": out_roll,
                "n_st_pass": out_nst,
                "solar_power_frac": out_power,
            }

        # Orbit grouping and orbit sampling
        # This is all target-independent, so it is built once per time grid
        # and reused by every target evaluated against the same grid.
        grid = self._orbit_sampling_grid(time, orbit_time_step)
        orbit_ids = grid["orbit_ids"]
        chunk_indices = grid["chunk_indices"]
        centers = grid["centers"]
        n_orbit_samp = grid["n_orbit_samp"]
        pre_orbit_all = grid["pre_orbit_all"]
        pre_input_all = grid["pre_input_all"]

        # Per-orbit representative target direction at each orbit center,
        # in one vectorised transform rather than one per orbit.
        # Aberration shift within one orbit (~97 min) is <0.1", so a single
        # direction is fine for the roll sweep and orbit-sample boresight
        # constraints.
        center_xyz = target_coord.transform_to(
            GCRS(obstime=centers)
        ).cartesian.xyz.value
        center_units = center_xyz / np.linalg.norm(
            center_xyz, axis=0, keepdims=True
        )  # (3, n_orbits)

        roll_rads = np.deg2rad(roll_degs)

        for position, oid in enumerate(orbit_ids):
            idx = chunk_indices[position]

            tgt_unit = center_units[:, position]
            tgt_b = tgt_unit[:, np.newaxis]  # (3, 1) for orbit sampling

            # Per-timestep target directions for input boresight checks
            chunk_tgt_b = tgt_b_all[:, idx]  # (3, N_chunk)

            # ── Find best roll from orbit window ──────────────────
            samples = slice(position * n_orbit_samp,
                            (position + 1) * n_orbit_samp)
            bu_orb = {name: unit[:, samples]
                      for name, unit in pre_orbit_all["body_units"].items()}
            zen_orb = pre_orbit_all["zenith_unit"][:, samples]
            limb_orb = pre_orbit_all["limb_angle_rad"][samples]

            # Boresight constraints on orbit
            bs_orb = self._boresight_ok(tgt_b, bu_orb, zen_orb, limb_orb)

            best_orbit_roll = np.nan

            # A roll is only worth searching for where the boresight is
            # clear somewhere in the orbit. get_orbit_roll_angles drops that
            # guard when a diagnostic needs an attitude regardless.
            if self._st_constraint_active and bs_orb.any():
                st_ok_orb, solar_orb = self._orbit_roll_sweep(
                    tgt_unit, roll_rads, bu_orb, zen_orb, limb_orb,
                    n_orbit_samp,
                )
                best_orbit_roll = self._pick_orbit_roll(
                    roll_degs, st_ok_orb, solar_orb, bs_orb
                )

            # ── Evaluate at input times with orbit-optimal roll ───
            bu_inp = {name: unit[:, idx]
                      for name, unit in pre_input_all["body_units"].items()}
            zen_inp = pre_input_all["zenith_unit"][:, idx]
            limb_inp = pre_input_all["limb_angle_rad"][idx]
            N_chunk = len(idx)

            # Boresight at input times (per-timestep target direction)
            bs_inp = self._boresight_ok(
                chunk_tgt_b, bu_inp, zen_inp, limb_inp
            )
            out_boresight[idx] = bs_inp

            if np.isnan(best_orbit_roll):
                # No roll satisfied ST constraints for this orbit;
                # out_visible remains False, out_roll stays NaN.
                continue

            # ST constraints at input times with the orbit-optimal roll
            roll_rad = np.deg2rad(best_orbit_roll)
            x_pay, y_pay = self._roll_attitude(tgt_unit, roll_rad)
            z_col_inp = np.tile(tgt_unit.reshape(3, 1), (1, N_chunk))

            st1_eci = (
                x_pay[:, np.newaxis] * st1_body[0]
                + y_pay[:, np.newaxis] * st1_body[1]
                + z_col_inp * st1_body[2]
            )
            st1_eci = st1_eci / np.linalg.norm(
                st1_eci, axis=0, keepdims=True
            )
            st2_eci = (
                x_pay[:, np.newaxis] * st2_body[0]
                + y_pay[:, np.newaxis] * st2_body[1]
                + z_col_inp * st2_body[2]
            )
            st2_eci = st2_eci / np.linalg.norm(
                st2_eci, axis=0, keepdims=True
            )

            t1_ok = np.ones(N_chunk, dtype=bool)
            for _, limit, key in st1_checks:
                lim = limit.to(u.deg).value
                if key == "sun_angle":
                    sep = self._fast_sep_deg(st1_eci, bu_inp["sun"])
                elif key == "moon_angle":
                    sep = self._fast_sep_deg(st1_eci, bu_inp["moon"])
                elif key == "earthlimb_angle":
                    sep = self._fast_limb_deg(st1_eci, zen_inp, limb_inp)
                else:
                    continue
                t1_ok &= sep >= lim

            t2_ok = np.ones(N_chunk, dtype=bool)
            for _, limit, key in st2_checks:
                lim = limit.to(u.deg).value
                if key == "sun_angle":
                    sep = self._fast_sep_deg(st2_eci, bu_inp["sun"])
                elif key == "moon_angle":
                    sep = self._fast_sep_deg(st2_eci, bu_inp["moon"])
                elif key == "earthlimb_angle":
                    sep = self._fast_limb_deg(st2_eci, zen_inp, limb_inp)
                else:
                    continue
                t2_ok &= sep >= lim

            st_ok_inp = self._combine_tracker_results(t1_ok, t2_ok)

            vis_inp = bs_inp & st_ok_inp
            out_visible[idx] = vis_inp
            # Count only the trackers something was asked of, to match the
            # rule _combine_tracker_results applies just above.
            passing = sum(
                (t1_ok if tracker == 1 else t2_ok).astype(int)
                for tracker in self._trackers_with_checks()
            )
            out_nst[idx] = np.where(vis_inp, passing, 0)

            # Solar power at input times
            cos_sy = np.sum(y_pay[:, np.newaxis] * bu_inp["sun"], axis=0)
            cos_sy = np.clip(cos_sy, -1.0, 1.0)
            theta_sy = np.arccos(np.abs(cos_sy))
            incidence = np.pi / 2 - theta_sy
            power = np.cos(incidence)
            out_power[idx] = np.where(vis_inp, power, np.nan)
            out_roll[idx] = np.where(vis_inp, best_orbit_roll, np.nan)

        if is_scalar:
            return {
                "visible": bool(out_visible[0]),
                "boresight_visible": bool(out_boresight[0]),
                "roll_deg": float(out_roll[0]),
                "n_st_pass": int(out_nst[0]),
                "solar_power_frac": float(out_power[0]),
            }
        return {
            "visible": out_visible,
            "boresight_visible": out_boresight,
            "roll_deg": out_roll,
            "n_st_pass": out_nst,
            "solar_power_frac": out_power,
        }

    def get_orbit_roll_angles(self, target_coord: SkyCoord, time: Time,
                              roll_step=2 * u.deg,
                              orbit_time_step=1 * u.min):
        """Roll angle held during each orbit, one value per timestep.

        The same choice ``get_visibility_best_roll`` makes, with one
        difference: an orbit whose boresight is blocked the whole way round
        still gets a roll here. That method skips the sweep for a roll it
        could never use, which is the cheaper thing to do when all it owes
        the caller is a visibility flag. A diagnostic needs more, because
        without an attitude there is no way to say which star tracker would
        have failed on an orbit the boresight had already lost. This sweeps
        either way.

        Wherever a roll could serve the target this returns exactly the
        angle ``get_visibility_best_roll`` used, so feeding the result to
        ``get_all_constraints`` or ``get_star_tracker_breakdown`` explains
        that run step for step.

        Assumes the target direction is fixed across an orbit, as
        ``get_visibility_best_roll`` does: aberration moves it by well
        under an arcsecond over one period.

        Parameters
        ----------
        target_coord : SkyCoord
            The science target coordinate (+Z boresight direction).
        time : Time
            Observation time(s), scalar or array.
        roll_step : Quantity, optional
            Roll sweep resolution (default 2 deg). Give the same value used
            for ``get_visibility_best_roll``, or the two searches can land
            on different angles.
        orbit_time_step : Quantity, optional
            Sampling interval within each orbit window (default 1 min).
            Match it for the same reason.

        Returns
        -------
        float or np.ndarray
            Roll in degrees in [-180, 180], constant within each orbit. NaN
            only when the star tracker keep-outs are switched off, since no
            roll is chosen at all then.

        Examples
        --------
        >>> roll = vis.get_orbit_roll_angles(target, times)
        >>> constraints = vis.get_all_constraints(target, times,
        ...                                       roll=roll * u.deg)
        """
        _validate_angle(roll_step, "roll_step")
        _validate_time_quantity(orbit_time_step, "orbit_time_step")
        if not roll_step.isscalar:
            raise ValueError("roll_step must be a scalar Quantity")
        if not orbit_time_step.isscalar:
            raise ValueError("orbit_time_step must be a scalar Quantity")

        is_scalar = time.isscalar
        if is_scalar:
            time = Time([time])

        out_roll = np.full(len(time), np.nan)
        if not self._st_constraint_active:
            # No tracker keep-outs, so nothing selects a roll.
            return float(out_roll[0]) if is_scalar else out_roll

        roll_degs = np.arange(0, 360, roll_step.to(u.deg).value)
        roll_rads = np.deg2rad(roll_degs)

        # Shares the orbit grouping and its ephemeris with
        # get_visibility_best_roll, so calling both on one time grid pays
        # for the expensive part once.
        grid = self._orbit_sampling_grid(time, orbit_time_step)
        n_orbit_samp = grid["n_orbit_samp"]
        pre_orbit_all = grid["pre_orbit_all"]

        center_xyz = target_coord.transform_to(
            GCRS(obstime=grid["centers"])
        ).cartesian.xyz.value
        center_units = center_xyz / np.linalg.norm(
            center_xyz, axis=0, keepdims=True
        )  # (3, n_orbits)

        all_samples = np.ones(n_orbit_samp, dtype=bool)

        for position, idx in enumerate(grid["chunk_indices"]):
            target_unit = center_units[:, position]
            samples = slice(position * n_orbit_samp,
                            (position + 1) * n_orbit_samp)
            body_units = {name: unit[:, samples]
                          for name, unit in pre_orbit_all["body_units"].items()}
            zenith_unit = pre_orbit_all["zenith_unit"][:, samples]
            limb_angle_rad = pre_orbit_all["limb_angle_rad"][samples]

            st_ok, solar = self._orbit_roll_sweep(
                target_unit, roll_rads, body_units, zenith_unit,
                limb_angle_rad, n_orbit_samp,
            )

            # Ask the question get_visibility_best_roll asks first, so the
            # two agree on every orbit where it found an answer.
            boresight_ok = self._boresight_ok(
                target_unit[:, np.newaxis], body_units, zenith_unit,
                limb_angle_rad,
            )
            roll = self._pick_orbit_roll(roll_degs, st_ok, solar,
                                         boresight_ok)

            if np.isnan(roll):
                # Nothing was observable this orbit. Rank the rolls on the
                # trackers alone: they do not care why the boresight was
                # lost, and the diagnostic still has to name the tracker
                # that would have failed.
                roll = self._pick_orbit_roll(roll_degs, st_ok, solar,
                                             all_samples)

            if np.isnan(roll):
                # No roll passes a tracker anywhere in the orbit, so there
                # is no tie for solar power to break. Keep the arrays best
                # lit instead, which at least leaves the attitude defined
                # and physically sensible.
                best = roll_degs[np.argmax(solar.mean(axis=1))]
                roll = float((best + 180) % 360 - 180)

            out_roll[idx] = roll

        return float(out_roll[0]) if is_scalar else out_roll

    @property
    def _st_constraint_active(self) -> bool:
        """Whether any star tracker constraints are active."""
        if self.st_required == 0:
            return False
        if self.st_sun_min > 0 * u.deg or self.st_moon_min > 0 * u.deg:
            return True
        # Check if any tracker has an active Earth limb constraint
        for t in [1, 2]:
            if self._st_earthlimb_min_for(t) > 0 * u.deg:
                return True
        return False

    def _trackers_with_checks(self) -> list:
        """Tracker numbers carrying at least one active keep-out.

        A tracker with an empty check list is not modelled at all rather
        than modelled as unconstrained, which is what keeps it out of the
        ``st_required`` count in ``_combine_tracker_results``.
        """
        return [tracker for tracker in (1, 2)
                if self._st_checks_for(tracker)]

    def _combine_tracker_results(self, st1_ok, st2_ok):
        """Reduce the two tracker verdicts to the ``st_required`` answer.

        The one place ``st_required`` is applied, so the constraint check,
        the breakdown and both roll searches cannot disagree about what it
        means.

        A tracker with no active keep-out is left out of the count instead
        of counting as a pass. Otherwise "at least one tracker" would be
        satisfied by a tracker nothing was ever asked of, and a per-tracker
        limit set on its own, ``st1_earthlimb_min`` with nothing for ST2,
        would silently reject nothing. With ``st_required=2`` this changes
        no result, since ANDing against an always-passing tracker already
        reduced to the constrained one.

        Parameters
        ----------
        st1_ok, st2_ok : bool or np.ndarray of bool
            Whether each tracker met all of its own active keep-outs.

        Returns
        -------
        bool or np.ndarray of bool
            Whether the star tracker requirement is met.
        """
        active = self._trackers_with_checks()
        if len(active) == 1:
            return st1_ok if active[0] == 1 else st2_ok
        if self.st_required == 1:
            return st1_ok | st2_ok
        return st1_ok & st2_ok

    def _st_earthlimb_min_for(self, tracker: int):
        """Effective Earth limb keep-out for a specific tracker.

        Returns the per-tracker override if set, otherwise the shared value.
        """
        if tracker == 1 and self.st1_earthlimb_min is not None:
            return self.st1_earthlimb_min
        elif tracker == 2 and self.st2_earthlimb_min is not None:
            return self.st2_earthlimb_min
        return self.st_earthlimb_min

    def _st_checks_for(self, tracker: int) -> list:
        """Active ST constraint checks for a specific tracker.

        Parameters
        ----------
        tracker : int
            Star tracker number (1 or 2).

        Returns
        -------
        list of (name, limit, key) tuples
        """
        checks = []
        if self.st_sun_min > 0 * u.deg:
            checks.append(("sun", self.st_sun_min, "sun_angle"))
        if self.st_moon_min > 0 * u.deg:
            checks.append(("moon", self.st_moon_min, "moon_angle"))
        limb_min = self._st_earthlimb_min_for(tracker)
        if limb_min > 0 * u.deg:
            checks.append(("limb", limb_min, "earthlimb_angle"))
        return checks

    @staticmethod
    def _get_star_tracker_body_xyz(tracker: int) -> tuple:
        """
        Get the star tracker boresight direction in body frame coordinates.

        Parameters:
        tracker : int
            Star tracker number (1 or 2)

        Returns:
        tuple
            (x, y, z) unit vector in body frame
        """
        if tracker == 1:
            vec = np.array([0.6804, -0.7071, -0.1923], dtype=float)
        elif tracker == 2:
            vec = np.array([0.6804, 0.7071, -0.1923], dtype=float)
        else:
            raise ValueError(f"Invalid tracker number: {tracker}. Must be 1 or 2.")

        norm = np.linalg.norm(vec)
        if norm == 0.0:
            raise ValueError("Star tracker boresight vector has zero magnitude.")

        vec_unit = vec / norm
        return tuple(vec_unit.tolist())

    @staticmethod
    def _roll_attitude(z_unit, roll_rad):
        """Compute payload X, Y axes from boresight Z and a fixed roll angle.

        The roll angle is measured from the projection of celestial north
        onto the plane perpendicular to the boresight, rotating toward
        ``cross(Z, north_proj)`` (right-hand rule about Z).

        Parameters
        ----------
        z_unit : np.ndarray
            (3,) unit vector along boresight (+Z payload).
        roll_rad : float
            Roll angle in radians.

        Returns
        -------
        x_payload, y_payload : np.ndarray
            (3,) unit vectors for payload +X and +Y axes.
        """
        north = np.array([0.0, 0.0, 1.0])
        north_proj = north - np.dot(north, z_unit) * z_unit
        north_norm = np.linalg.norm(north_proj)
        if north_norm < 1e-8:
            # Boresight near celestial pole — use east as fallback
            east = np.array([1.0, 0.0, 0.0])
            north_proj = east - np.dot(east, z_unit) * z_unit
            north_norm = np.linalg.norm(north_proj)
        x_ref = north_proj / north_norm
        y_ref = np.cross(z_unit, x_ref)
        y_ref = y_ref / np.linalg.norm(y_ref)

        cos_r = np.cos(roll_rad)
        sin_r = np.sin(roll_rad)
        x_payload = cos_r * x_ref + sin_r * y_ref
        y_payload = -sin_r * x_ref + cos_r * y_ref
        return x_payload, y_payload

    @staticmethod
    def _roll_attitude_batch(z_unit, roll_rads):
        """``_roll_attitude`` evaluated for many roll angles at once.

        Parameters
        ----------
        z_unit : np.ndarray
            (3,) unit vector along boresight (+Z payload).
        roll_rads : np.ndarray
            (N_roll,) roll angles in radians.

        Returns
        -------
        x_payload, y_payload : np.ndarray
            (N_roll, 3) payload +X and +Y axes, one row per roll angle.
        """
        north = np.array([0.0, 0.0, 1.0])
        north_proj = north - np.dot(north, z_unit) * z_unit
        north_norm = np.linalg.norm(north_proj)
        if north_norm < 1e-8:
            # Boresight near celestial pole — use east as fallback
            east = np.array([1.0, 0.0, 0.0])
            north_proj = east - np.dot(east, z_unit) * z_unit
            north_norm = np.linalg.norm(north_proj)
        x_ref = north_proj / north_norm
        y_ref = np.cross(z_unit, x_ref)
        y_ref = y_ref / np.linalg.norm(y_ref)

        cos_r = np.cos(roll_rads)[:, np.newaxis]
        sin_r = np.sin(roll_rads)[:, np.newaxis]
        return (cos_r * x_ref + sin_r * y_ref, -sin_r * x_ref + cos_r * y_ref)

    def _boresight_ok(self, target_b, body_units, zenith_unit,
                      limb_angle_rad):
        """Boresight keep-out verdict over a set of samples.

        Where the boresight visibility is checked during a roll
        search, so ``get_visibility_best_roll`` and
        ``get_orbit_roll_angles`` can never disagree about which samples a
        roll could be used at.

        Parameters
        ----------
        target_b : np.ndarray
            Target direction, (3, 1) to broadcast one direction over every
            sample, or (3, N) for a direction per sample.
        body_units : dict
            Body direction unit vectors from ``_precompute``.
        zenith_unit : np.ndarray
            Observer zenith directions, (3, N).
        limb_angle_rad : float or np.ndarray
            Earth limb half-angle in radians for those samples.

        Returns
        -------
        np.ndarray
            (N,) boolean, True where every active boresight keep-out passes.
        """
        ### BORESIGHT-MOON KEEPOUT
        ok = (
            self._fast_sep_deg(body_units["moon"], target_b)
            >= self.moon_min.to(u.deg).value
        )

        ### BORESIGHT-SUN KEEPOUT
        ok &= (
            self._fast_sep_deg(body_units["sun"], target_b)
            >= self.sun_min.to(u.deg).value
        )

        ### BORESIGHT-EARTHLIMB KEEPOUT
        ok &= (
            self._fast_limb_deg(target_b, zenith_unit, limb_angle_rad)
            >= self._effective_earthlimb_min_deg(
                target_b, zenith_unit, body_units["sun"],
                limb_angle_rad=limb_angle_rad,
            )
        )

        ### BORESIGHT-MARS KEEPOUT
        if self.mars_min > 0 * u.deg:
            ok &= (
                self._fast_sep_deg(body_units["mars"], target_b)
                >= self.mars_min.to(u.deg).value
            )

        ### BORESIGHT-JUPITER KEEPOUT
        if self.jupiter_min > 0 * u.deg:
            ok &= (
                self._fast_sep_deg(body_units["jupiter"], target_b)
                >= self.jupiter_min.to(u.deg).value
            )
        return np.asarray(ok).ravel()

    def _orbit_roll_sweep(self, target_unit, roll_rads, body_units,
                          zenith_unit, limb_angle_rad, n_samples):
        """Star tracker verdict and solar power for every roll over one orbit.

        Parameters
        ----------
        target_unit : np.ndarray
            (3,) boresight direction, held fixed across the orbit.
        roll_rads : np.ndarray
            (n_rolls,) roll angles to try, in radians.
        body_units, zenith_unit, limb_angle_rad
            Precomputed quantities for this orbit's samples.
        n_samples : int
            Number of samples in the orbit window.

        Returns
        -------
        st_ok : np.ndarray
            (n_rolls, n_samples) boolean, True where the ``st_required``
            combination of trackers passes.
        solar : np.ndarray
            (n_rolls, n_samples) solar array power fraction.
        """
        z_col = np.tile(target_unit.reshape(3, 1), (1, n_samples))
        x_payload, y_payload = self._roll_attitude_batch(
            target_unit, roll_rads
        )

        tracker_ok = [
            self._sweep_tracker(
                x_payload, y_payload, z_col,
                np.array(self._get_star_tracker_body_xyz(tracker)),
                self._st_checks_for(tracker),
                body_units, zenith_unit, limb_angle_rad,
            )
            for tracker in (1, 2)
        ]
        st_ok = self._combine_tracker_results(*tracker_ok)

        cos_sy = np.clip(
            np.sum(
                y_payload[:, :, np.newaxis] * body_units["sun"][np.newaxis],
                axis=1,
            ),
            -1.0, 1.0,
        )
        solar = np.cos(np.pi / 2 - np.arccos(np.abs(cos_sy)))
        return st_ok, solar

    @staticmethod
    def _pick_orbit_roll(roll_degs, st_ok, solar, usable):
        """Roll with the most usable samples, average solar power breaking ties.

        Parameters
        ----------
        roll_degs : np.ndarray
            (n_rolls,) candidate roll angles in degrees.
        st_ok, solar : np.ndarray
            (n_rolls, n_samples) arrays from ``_orbit_roll_sweep``.
        usable : np.ndarray
            (n_samples,) boolean marking samples a roll could be used at.
            Pass the boresight verdict to pick the roll that observes the
            most, or all True to ask which roll suits the star trackers
            alone, ignoring whether the boresight was clear.

        Returns
        -------
        float
            Roll in degrees normalized to [-180, 180], or NaN when no roll
            has a single usable sample.
        """
        good = usable[np.newaxis, :] & st_ok
        counts = good.sum(axis=1)
        if counts.max() == 0:
            return np.nan
        candidates = np.where(counts == counts.max())[0]
        avg_power = np.array([solar[r, good[r]].mean() for r in candidates])
        best = roll_degs[candidates[np.argmax(avg_power)]]
        return float((best + 180) % 360 - 180)

    def _sweep_tracker(self, x_payload, y_payload, z_col, st_body, checks,
                       body_units, zenith_unit, limb_rad):
        """Star tracker keep-out check for every roll angle at once.

        Parameters
        ----------
        x_payload, y_payload : np.ndarray
            (N_roll, 3) payload axes from ``_roll_attitude_batch``.
        z_col : np.ndarray
            (3, N_samp) boresight direction, repeated over the samples.
        st_body : np.ndarray
            (3,) tracker boresight in body coordinates.
        checks : list
            Active checks from ``_st_checks_for``.
        body_units, zenith_unit, limb_rad
            Precomputed time-dependent quantities for the same samples.

        Returns
        -------
        np.ndarray
            (N_roll, N_samp) boolean array, True where the tracker passes.
        """
        # (N_roll, 3, N_samp)
        st_eci = (
            x_payload[:, :, np.newaxis] * st_body[0]
            + y_payload[:, :, np.newaxis] * st_body[1]
            + z_col[np.newaxis] * st_body[2]
        )
        st_eci = st_eci / np.linalg.norm(st_eci, axis=1, keepdims=True)

        ok = np.ones(st_eci.shape[::2], dtype=bool)  # (N_roll, N_samp)
        for _, limit, key in checks:
            limit_deg = limit.to(u.deg).value
            if key == "sun_angle":
                dot = np.sum(st_eci * body_units["sun"][np.newaxis], axis=1)
                sep = np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))
            elif key == "moon_angle":
                dot = np.sum(st_eci * body_units["moon"][np.newaxis], axis=1)
                sep = np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))
            elif key == "earthlimb_angle":
                dot = np.sum(st_eci * zenith_unit[np.newaxis], axis=1)
                sep = np.rad2deg(
                    np.arcsin(np.clip(dot, -1.0, 1.0)) + limb_rad
                )
            else:
                continue
            ok &= sep >= limit_deg
        return ok

    def get_star_tracker_angles(
        self, target_coord: SkyCoord, time: Time, tracker: int = 1
    ) -> dict:
        """
        Calculate the star tracker sun and Earth angles.

        When ``roll`` is None (default), the payload attitude is
        Sun-constrained: +Y = Sun × Z, +X = Y × Z.
        When ``roll`` is set, the attitude is determined by rotating
        from the celestial-north projection by that angle about Z.

        Parameters:
        target_coord : SkyCoord
            The science target coordinate (+Z direction)
        time : Time
            The observation time (scalar or array)
        tracker : int
            Star tracker number (1 or 2)

        Returns:
        dict
            Dictionary with 'ra', 'dec', 'sun_angle', 'moon_angle',
            'earth_angle', and 'earthlimb_angle' as Quantities in degrees.
            Values are scalar or array depending on time input.

        Raises:
        ValueError
            If target is too close to the sun (degenerate attitude)
        """
        observer_location = self._get_observer_location(time)

        # Reuse _get_star_tracker_skycoord for the boresight direction
        st_coord = self._get_star_tracker_skycoord(target_coord, time, tracker)

        # Sun angle
        sun_coord = get_body("sun", time=time, location=observer_location)
        sun_angle = st_coord.separation(sun_coord)

        # Moon angle
        moon_coord = get_body("moon", time=time, location=observer_location)
        moon_angle = st_coord.separation(moon_coord)

        # Earth center angle (nadir direction).
        # Use the same topocentric frame as st_coord so separation() does not
        # need a frame translation (which would distort nearby unit-distance
        # SkyCoords).
        obs_gcrs = observer_location.get_gcrs(obstime=time)
        obs_eci = obs_gcrs.cartesian.xyz
        if time.isscalar:
            earth_eci = -obs_eci / np.linalg.norm(obs_eci)
        else:
            earth_eci = -obs_eci / np.linalg.norm(obs_eci, axis=0, keepdims=True)
        earth_coord = SkyCoord(
            x=earth_eci.value[0],
            y=earth_eci.value[1],
            z=earth_eci.value[2],
            representation_type="cartesian",
            frame=st_coord.frame,
        )
        earth_angle = st_coord.separation(earth_coord)

        # Earth limb angle, geocentric like the constraint check, so this
        # reports the number get_star_tracker_constraint applies rather
        # than one that merely resembles it. An AltAz altitude is measured
        # from the geodetic horizon, which sits up to ~0.2 deg away from
        # the geocentric one, enough to disagree about a tracker sitting
        # near its limit.
        pre = self._precompute(time)
        st_xyz = st_coord.cartesian.xyz.value
        if time.isscalar:
            st_unit = st_xyz / np.linalg.norm(st_xyz)
        else:
            st_unit = st_xyz / np.linalg.norm(st_xyz, axis=0, keepdims=True)
        earthlimb_angle = self._fast_limb_deg(
            st_unit, pre["zenith_unit"], pre["limb_angle_rad"]
        ) * u.deg

        return {
            "ra": st_coord.spherical.lon.to(u.deg),
            "dec": st_coord.spherical.lat.to(u.deg),
            "sun_angle": sun_angle.to(u.deg),
            "moon_angle": moon_angle.to(u.deg),
            "earth_angle": earth_angle.to(u.deg),
            "earthlimb_angle": earthlimb_angle.to(u.deg),
        }

    def _get_star_tracker_skycoord(
        self, target_coord: SkyCoord, time: Time, tracker: int
    ) -> SkyCoord:
        """
        Calculate the sky coordinate where a star tracker boresight points.

        The payload +Z points at the science target.  When ``self.roll``
        is None, the attitude is Sun-constrained (+Y = Sun × Z).  When
        ``self.roll`` is an angle, the attitude is set by rotating from
        celestial-north projection by that roll about Z.  The star
        tracker body-frame vector is then rotated into the ECI frame.

        Parameters:
        -----------
        target_coord : SkyCoord
            The science target coordinate (+Z payload direction)
        time : Time
            The observation time (scalar or array)
        tracker : int
            Star tracker number (1 or 2)

        Returns:
        --------
        SkyCoord
            The GCRS coordinate of the star tracker boresight

        Raises:
        -------
        ValueError
            If target is aligned with the sun (scalar time only)
        """
        observer_location = self._get_observer_location(time)

        # Satellite GCRS frame (topocentric: obsgeoloc = satellite position).
        # Body SkyCoords from get_body(location=observer_location) carry the
        # same obsgeoloc, so separation() won't apply a spurious origin
        # translation that shifts the Moon direction by up to ~1 deg.
        obs_gcrs = observer_location.get_gcrs(obstime=time)
        sat_gcrs_frame = GCRS(
            obstime=time,
            obsgeoloc=obs_gcrs.cartesian.without_differentials(),
            obsgeovel=obs_gcrs.velocity.d_xyz,
        )

        # Target direction unit vector(s) in GCRS at each observation time.
        # Using obstime=time (not just time[0]) correctly accounts for
        # aberration and precession over long time arrays.
        target_gcrs = target_coord.transform_to(GCRS(obstime=time))
        z_payload_raw = target_gcrs.cartesian.xyz.value

        if time.isscalar:
            z_payload = z_payload_raw / np.linalg.norm(z_payload_raw)
        else:
            z_payload = z_payload_raw / np.linalg.norm(
                z_payload_raw, axis=0, keepdims=True
            )  # (3, N)

        st_body = np.array(self._get_star_tracker_body_xyz(tracker))

        if self.roll is not None:
            # Fixed-roll attitude: no Sun dependency.
            # Use representative direction for attitude frame; the
            # per-timestep z_payload is used for the final rotation.
            roll_rad = self.roll.to(u.rad).value
            z_rep = z_payload if time.isscalar else z_payload[:, 0]
            x_payload, y_payload = self._roll_attitude(z_rep, roll_rad)

            if time.isscalar:
                R = np.column_stack([x_payload, y_payload, z_payload])
                st_eci = R @ st_body
                st_eci = st_eci / np.linalg.norm(st_eci)
            else:
                st_eci = (
                    x_payload[:, np.newaxis] * st_body[0]
                    + y_payload[:, np.newaxis] * st_body[1]
                    + z_payload * st_body[2]
                )
                st_eci = st_eci / np.linalg.norm(st_eci, axis=0, keepdims=True)

            return SkyCoord(
                x=st_eci[0],
                y=st_eci[1],
                z=st_eci[2],
                representation_type="cartesian",
                frame=sat_gcrs_frame,
            )

        # Sun-constrained attitude (default)
        sun_coord = get_body("sun", time=time, location=observer_location)
        sun_xyz = sun_coord.cartesian.xyz.value

        if time.isscalar:
            sun_vec = sun_xyz / np.linalg.norm(sun_xyz)

            y_payload = np.cross(sun_vec, z_payload)
            y_norm = np.linalg.norm(y_payload)
            if y_norm < 1e-10:
                raise ValueError("Cannot determine attitude: target aligned with sun")
            y_payload = y_payload / y_norm
            x_payload = np.cross(y_payload, z_payload)
            x_payload = x_payload / np.linalg.norm(x_payload)

            R = np.column_stack([x_payload, y_payload, z_payload])
            st_eci = R @ st_body
            st_eci = st_eci / np.linalg.norm(st_eci)

            return SkyCoord(
                x=st_eci[0],
                y=st_eci[1],
                z=st_eci[2],
                representation_type="cartesian",
                frame=sat_gcrs_frame,
            )
        else:
            # Array case: sun_xyz shape is (3, N)
            sun_vec = sun_xyz / np.linalg.norm(sun_xyz, axis=0, keepdims=True)
            # z_payload is already (3, N) from per-timestep GCRS transform

            y_payload = np.cross(sun_vec, z_payload, axis=0)
            y_norms = np.linalg.norm(y_payload, axis=0, keepdims=True)

            # Detect degenerate timesteps where target is aligned with sun
            degenerate = (y_norms < 1e-10).ravel()

            # Safe-divide: set degenerate norms to 1 to avoid division by zero,
            # then overwrite those columns with NaN so they propagate cleanly
            y_norms_safe = np.where(y_norms < 1e-10, 1.0, y_norms)
            y_payload = y_payload / y_norms_safe

            x_payload = np.cross(y_payload, z_payload, axis=0)
            x_norms = np.linalg.norm(x_payload, axis=0, keepdims=True)
            x_norms_safe = np.where(x_norms < 1e-10, 1.0, x_norms)
            x_payload = x_payload / x_norms_safe

            # Transform star tracker body vector to ECI for each timestep
            st_eci = (
                x_payload * st_body[0] + y_payload * st_body[1] + z_payload * st_body[2]
            )
            st_eci = st_eci / np.linalg.norm(st_eci, axis=0, keepdims=True)

            # Mark degenerate timesteps with NaN so downstream separations
            # return NaN (which compares as False against any threshold)
            st_eci[:, degenerate] = np.nan

            return SkyCoord(
                x=st_eci[0],
                y=st_eci[1],
                z=st_eci[2],
                representation_type="cartesian",
                frame=sat_gcrs_frame,
            )

    def get_star_tracker_constraint(self, target_coord: SkyCoord, time: Time,
                                    pre: dict = None, roll=None):
        """
        Check if the required number of star trackers satisfy all keep-out constraints.

        Evaluates sun, moon, and Earth limb keep-out angles for both star
        trackers and returns True if self.st_required trackers meet all active
        constraints (0 = disabled, 1 = at least one, 2 = both).

        Parameters:
        -----------
        target_coord : SkyCoord
            The science target coordinate
        time : Time
            The observation time (scalar or array)
        pre : dict, optional
            Precomputed time-dependent data from ``_precompute``.
        roll : Quantity, optional
            Roll angle about the boresight for this call only. Scalar, or
            one angle per timestep to reproduce a run whose attitude
            changed, as ``get_orbit_roll_angles`` returns. ``None`` keeps
            the instance value, which itself defaults to the
            Sun-constrained attitude.

        Returns:
        --------
        bool or np.ndarray
            True if the required number of star trackers meet all constraints

        Notes:
        ------
        This shares its implementation with ``get_visibility``, so the two
        always agree.  ``get_star_tracker_angles`` remains available for the
        per-timestep angles themselves.
        """
        if not self._st_constraint_active:
            if time.isscalar:
                return True
            return np.ones(time.shape, dtype=bool)

        if roll is not None:
            _validate_angle(roll, "roll")
            effective_roll = roll.to(u.deg)
        else:
            effective_roll = self.roll

        if pre is None:
            pre = self._precompute(time)
        target_unit = self._target_unit(target_coord, time)
        if not time.isscalar:
            target_unit = target_unit[:, 0].copy()

        return self._get_st_constraint_fast(
            target_unit, time, pre, effective_roll=effective_roll
        )

    def get_star_tracker_breakdown(self, target_coord: SkyCoord, time: Time,
                                   roll=None, pre: dict = None) -> dict:
        """Which star tracker fails which keep-out, check by check.

        ``get_star_tracker_constraint`` answers "did the trackers pass?".
        This answers "and if not, which one, on what?" — useful for
        diagnosing why a target dropped out.

        Shares ``_st_tracker_separations`` with the constraint check
        itself, so the per-check masks always reconstruct the verdict:
        ``result["passed"]["combined"]`` equals
        ``get_star_tracker_constraint(...)`` exactly.

        Parameters
        ----------
        target_coord : SkyCoord
            The science target coordinate (+Z boresight direction).
        time : Time
            Observation time(s), scalar or array.
        roll : Quantity, optional
            Roll angle about the boresight for this call only.  ``None``
            keeps the instance value, which itself defaults to the
            Sun-constrained attitude.
        pre : dict, optional
            Precomputed data from ``_precompute``.

        Returns
        -------
        dict
            passed : dict
                Boolean *pass* masks keyed ``"ST1 sun"``, ``"ST1 moon"``,
                ``"ST1 limb"`` and the ST2 equivalents, one entry per
                *active* check per tracker; plus ``"ST1"`` / ``"ST2"`` for
                each tracker overall and ``"combined"`` for the
                ``st_required`` verdict.
            separations : dict
                The angle behind each per-check entry, in degrees.  NaN
                where the attitude is degenerate.
            limits : dict
                The threshold applied to each per-check entry, a Quantity.

        Examples
        --------
        >>> br = vis.get_star_tracker_breakdown(target, times)
        >>> for name, ok in br["passed"].items():
        ...     print(f"{name:<10} fails at {int((~ok).sum()):>4} steps")
        """
        if roll is not None:
            _validate_angle(roll, "roll")
            effective_roll = roll.to(u.deg)
        else:
            effective_roll = self.roll

        if pre is None:
            pre = self._precompute(time)
        target_unit = self._target_unit(target_coord, time)
        if not time.isscalar:
            target_unit = target_unit[:, 0].copy()

        separations, degenerate = self._st_tracker_separations(
            target_unit, time, pre, effective_roll=effective_roll,
        )

        # "limb" rather than "earthlimb" keeps the row labels short; the
        # names come from _st_checks_for so only active checks appear.
        passed, seps_out, limits_out = {}, {}, {}
        tracker_overall = {}
        for tracker in [1, 2]:
            if time.isscalar:
                tracker_ok = True
            else:
                tracker_ok = np.ones(time.shape, dtype=bool)

            for name, limit, key in self._st_checks_for(tracker):
                sep = separations[tracker][key]
                ok = sep >= limit.to(u.deg).value
                row = f"ST{tracker} {name}"
                passed[row] = bool(ok) if time.isscalar else np.asarray(ok)
                seps_out[row] = sep
                limits_out[row] = limit
                tracker_ok = tracker_ok & ok

            tracker_overall[tracker] = (
                bool(tracker_ok) if time.isscalar else np.asarray(tracker_ok)
            )

        passed["ST1"] = tracker_overall[1]
        passed["ST2"] = tracker_overall[2]

        # Reduce exactly as _get_st_constraint_fast does, rather than
        # calling get_star_tracker_constraint, which would ignore a roll
        # override and disagree with the per-check rows above.
        if not self._st_constraint_active:
            passed["combined"] = (
                True if time.isscalar else np.ones(time.shape, dtype=bool)
            )
        elif time.isscalar and degenerate:
            passed["combined"] = False
        else:
            passed["combined"] = self._combine_tracker_results(
                tracker_overall[1], tracker_overall[2]
            )

        return {
            "passed": passed,
            "separations": seps_out,
            "limits": limits_out,
        }

    def get_all_constraints(self, target_coord: SkyCoord, time: Time,
                            roll=None) -> dict:
        """Get status of all active constraints.

        Every constraint is evaluated from a single set of precomputed
        ephemeris and orbit data, so the results agree with
        ``get_visibility`` body for body.

        Parameters
        ----------
        target_coord : SkyCoord
            The target coordinate to check.
        time : Time
            Observation time(s), scalar or array.
        roll : Quantity, optional
            Roll angle about the boresight for this call only, forwarded to
            the star tracker check. Only the star tracker constraint
            depends on it; the boresight ones do not.  Pass the array
            from ``get_orbit_roll_angles`` to explain a
            ``get_visibility_best_roll`` run, since the default
            Sun-constrained attitude is not the one that run held.

        Returns
        -------
        dict
            One boolean or boolean array per active constraint.
        """
        pre = self._precompute(time)
        constraints = {
            "moon": self.get_constraint(target_coord, "moon", time, pre=pre),
            "sun": self.get_constraint(target_coord, "sun", time, pre=pre),
            "earthlimb": self.get_constraint(
                target_coord, "earthlimb", time, pre=pre
            ),
        }

        if self.mars_min > 0 * u.deg:
            constraints["mars"] = self.get_constraint(
                target_coord, "mars", time, pre=pre
            )

        if self.jupiter_min > 0 * u.deg:
            constraints["jupiter"] = self.get_constraint(
                target_coord, "jupiter", time, pre=pre
            )

        if self._st_constraint_active:
            constraints["star_tracker"] = self.get_star_tracker_constraint(
                target_coord, time, pre=pre, roll=roll
            )

        return constraints

    def get_separations(self, target_coord: SkyCoord, time: Time) -> dict:
        """Get actual separation angles from all bodies."""
        pre = self._precompute(time)
        target_unit = self._target_unit(target_coord, time)
        separations = {}

        for body in ["moon", "sun", "mars", "jupiter"]:
            separations[body] = self._fast_sep_deg(
                self._body_unit(body, time, pre), target_unit
            ) * u.deg

        separations["earthlimb"] = self._fast_limb_deg(
            target_unit, pre["zenith_unit"], pre["limb_angle_rad"]
        ) * u.deg
        return separations

    def summary(self, target_coord: SkyCoord, time: Time) -> str:
        """
        Get a human-readable summary of visibility constraints.

        Parameters:
        -----------
        target_coord : SkyCoord
            The target coordinate to analyze.
        time : Time
            The observation time. Must be scalar (single time point).

        Returns:
        --------
        str
            Formatted summary of all visibility constraints.

        Raises:
        -------
        ValueError
            If time is not scalar (array inputs not supported).
        """
        # Enforce scalar time restriction
        if not time.isscalar:
            raise ValueError(
                "summary() only supports scalar time inputs. "
                "Use get_visibility() or get_all_constraints() for array inputs."
            )

        try:
            constraints = self.get_all_constraints(target_coord, time)
            separations = self.get_separations(target_coord, time)
        except Exception as e:
            return f"Error calculating visibility: {e}"

        # Better coordinate formatting
        coord_str = target_coord.to_string("hmsdms", precision=1)
        if len(coord_str) > 35:
            coord_str = coord_str[:32] + "..."

        lines = [
            "Visibility Summary",
            f"Target: {coord_str}",
            f"Time:   {time.iso}",
            f"Sat:    {self.tle.satnum}",
            "=" * 60,
        ]

        for body in constraints:
            if body == "star_tracker":
                continue  # handled in dedicated section below
            status = "PASS" if constraints[body] else "FAIL"
            status_symbol = "✓" if constraints[body] else "✗"
            actual_sep = separations[body]

            if body == "earthlimb" and (
                self.use_dynamic_earthlimb
                or self.earthlimb_day_min is not None
                or self.earthlimb_night_min is not None
            ):
                # Show the active threshold and how it was chosen
                day_lim = (
                    self.earthlimb_day_min
                    if self.earthlimb_day_min is not None
                    else self.earthlimb_min
                )
                night_lim = (
                    self.earthlimb_night_min
                    if self.earthlimb_night_min is not None
                    else self.earthlimb_min
                )
                # Determine whether limb point is sunlit at this time
                pre = self._precompute(time)
                zenith_u = pre["zenith_unit"]
                sun_u = pre["body_units"]["sun"]
                la_rad = pre["limb_angle_rad"]
                tgt_u = self._target_unit(target_coord, time)
                if self.use_dynamic_earthlimb:
                    # Must go through _daynight_illumination_angle, not
                    # _get_earth_illumination_angle directly, so the
                    # reported angle is measured at the same point
                    # _effective_earthlimb_min_deg used.
                    illum = float(self._daynight_illumination_angle(
                        tgt_u, zenith_u, sun_u, limb_angle_rad=la_rad,
                    ))
                    side = f"illum {illum:.1f}°"
                    eff_lim = float(
                        self._dynamic_earthlimb_min_deg(illum)
                    ) * u.deg
                else:
                    # Must go through _daynight_is_sunlit, not
                    # _earthlimb_is_sunlit directly, so the reported
                    # threshold is the one get_visibility actually applied
                    # under the active daynight_mode.
                    is_sunlit = bool(self._daynight_is_sunlit(
                        tgt_u, zenith_u, sun_u, limb_angle_rad=la_rad,
                    ))
                    side = "day" if is_sunlit else "night"
                    eff_lim = day_lim if is_sunlit else night_lim
                lines.append(
                    f"{body.capitalize():<10} {status_symbol} {status:<4} "
                    f"(req: {eff_lim:>6.1f} [{side}], actual: {actual_sep:>6.1f})"
                )
            else:
                min_sep = getattr(self, f"{body}_min")
                lines.append(
                    f"{body.capitalize():<10} {status_symbol} {status:<4} "
                    f"(req: {min_sep:>6.1f}, actual: {actual_sep:>6.1f})"
                )

        # Star tracker constraints section
        if self._st_constraint_active:
            lines.append("-" * 60)
            req_label = "both" if self.st_required == 2 else "≥1"
            lines.append(
                f"Star Tracker Constraints (need {req_label} tracker passing):"
            )

            # Rows and result both come from the breakdown, which shares
            # its geometry with the constraint check itself. Deriving the
            # rows from get_star_tracker_angles let them contradict the
            # result printed underneath, because that reported an AltAz
            # limb angle while the check applied a geocentric one.
            breakdown = self.get_star_tracker_breakdown(target_coord, time)

            for tracker in [1, 2]:
                tracker_pass = breakdown["passed"][f"ST{tracker}"]
                symbol = "✓" if tracker_pass else "✗"
                status = "PASS" if tracker_pass else "FAIL"
                lines.append(f"  ST{tracker:<8}{symbol} {status}")

                for name, limit, _ in self._st_checks_for(tracker):
                    row = f"ST{tracker} {name}"
                    sym = "✓" if breakdown["passed"][row] else "✗"
                    actual = breakdown["separations"][row]
                    # NaN means the attitude itself is undefined, which
                    # happens when the target lies along the Sun and
                    # Sun x Z stops defining a payload +Y.
                    shown = (
                        "  undefined" if np.isnan(actual)
                        else f"{actual * u.deg:>6.1f}"
                    )
                    lines.append(
                        f"    {name}:{sym} req:{limit:>6.1f} act:{shown}"
                    )

            st_combined = breakdown["passed"]["combined"]
            st_sym = "✓" if st_combined else "✗"
            st_stat = "PASS" if st_combined else "FAIL"
            lines.append(f"  {'Result':<9}{st_sym} {st_stat}")

        overall_status = (
            "VISIBLE" if self.get_visibility(target_coord, time) else "NOT VISIBLE"
        )
        overall_symbol = "✓" if overall_status == "VISIBLE" else "✗"

        lines.extend(["=" * 60, f"Overall: {overall_symbol} {overall_status}"])

        return "\n".join(lines)
