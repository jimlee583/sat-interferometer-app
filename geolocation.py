"""LEO interferometer geolocation on WGS84.

Coordinate frames
-----------------
ECEF
    Earth-centered, Earth-fixed. +X through the equator at longitude 0,
    +Y through the equator at longitude 90° E, +Z through the north pole.
    Units are meters.

Geodetic (WGS84)
    Latitude, longitude, and ellipsoidal height. Longitude is east-positive.
    The ellipsoid normal is the geodetic "up" direction. A satellite's
    geodetic subpoint is the ellipsoid point with the same latitude and
    longitude and zero height. The body +Z axis points at the geocenter, so
    the geocentric nadir intercept is not that subpoint except on the equator
    and at the poles. In LEO the ground separation is a few kilometers; it
    grows with altitude toward the ~21 km surface offset between geodetic and
    geocentric latitude.

VVLH (this project)
    Local orbital frame for a spherical, circular orbit. Right-handed:
        +X along-track (circular-orbit velocity),
        +Y cross-track,
        +Z geocentric nadir (toward Earth's center, −r̂).
    +Y = +Z × +X = −ĥ, where ĥ is the orbit angular-momentum direction.
    This is not the zenith-positive LVLH convention, and it is not a
    geodetic-nadir frame. Velocity heading comes from the inclination and
    from whether the pass is ascending or descending; a subpoint alone does
    not determine it.

Body
    Coincident with VVLH when roll = pitch = yaw = 0.
    Attitude is an intrinsic 3-2-1 sequence (yaw, then pitch, then roll).
    Positive angles follow the right-hand rule:
        yaw about +Z, nose (+X) toward +Y (nose right with +Z down),
        pitch about +Y, nose toward −Z (nose up, away from Earth),
        roll about +X, right wing (+Y) toward +Z (right wing down).
    ``v_vvlh = R_z(yaw) @ R_y(pitch) @ R_x(roll) @ v_body``.

Antenna
    Aligned with the body frame. The Version 1–2 interferometer axes are
    used unchanged: elements in the XY plane, +Z boresight, azimuth
    counterclockwise from +X, off-boresight measured from +Z. With the
    default nadir-pointing attitude, +Z looks at geocentric nadir and Earth
    lies in the front hemisphere.

The arrival unit vector passed to ``interferometer.phase_difference`` points
from the satellite toward the emitter, expressed in the antenna frame.
"""
from dataclasses import dataclass, replace
import numpy as np
from pyproj import Geod
from scipy.optimize import least_squares

from interferometer import (
    ambiguity_surface,
    candidate_solutions,
    direction,
    measured_phases,
    phase_difference,
    wavelength,
    wrap_rad,
)

WGS84_A = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_EP2 = WGS84_E2 / (1.0 - WGS84_E2)

_GEOD = Geod(ellps='WGS84')
# Extra wrapped-phase RMS, in degrees, that still counts as the same ambiguity family.
AMBIGUITY_MARGIN_DEG = 1.0


class OrbitFrameError(ValueError):
    """Orbit inputs cannot define the spherical circular-orbit VVLH frame."""


@dataclass(frozen=True)
class OrbitSpec:
    """Satellite geodetic subpoint, altitude, and the orbit plane's orientation."""
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    inclination_deg: float
    ascending: bool = True


@dataclass(frozen=True)
class Attitude:
    """Body attitude relative to VVLH, in degrees. Zero is nadir-pointing."""
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0


@dataclass
class OrbitFrame:
    spec: OrbitSpec
    sat_ecef: np.ndarray
    r_hat: np.ndarray
    v_hat: np.ndarray
    geocentric_lat_deg: float
    geocentric_lon_deg: float
    heading_deg: float
    argument_of_latitude_deg: float | None
    x_vvlh: np.ndarray
    y_vvlh: np.ndarray
    z_vvlh: np.ndarray
    # Columns are VVLH basis vectors expressed in ECEF.
    R_ecef_from_vvlh: np.ndarray


@dataclass(frozen=True)
class DoaEstimate:
    ux: float
    uy: float
    uz: float
    azimuth_deg: float
    off_boresight_deg: float
    phase_rms_deg: float

    @property
    def unit(self) -> np.ndarray:
        return np.array([self.ux, self.uy, self.uz], dtype=float)


@dataclass(frozen=True)
class GroundSolution:
    doa: DoaEstimate
    status: str
    lat_deg: float | None = None
    lon_deg: float | None = None
    height_m: float | None = None
    ecef: np.ndarray | None = None
    slant_range_m: float | None = None
    u_ecef: np.ndarray | None = None


@dataclass
class InverseResult:
    """DOA and Earth-intercept solutions. Ranking uses phase residual only."""
    solutions: list
    ambiguous: bool
    underdetermined: bool
    n_doa_candidates: int
    n_valid_intercepts: int

    @property
    def primary(self) -> GroundSolution | None:
        valid = [s for s in self.solutions if s.status == 'ok']
        if not valid:
            return None
        return min(valid, key=lambda s: s.doa.phase_rms_deg)


@dataclass
class SceneResult:
    frame: OrbitFrame
    attitude: Attitude
    emitter_lat_deg: float | None
    emitter_lon_deg: float | None
    emitter_ecef: np.ndarray | None
    visible: bool
    visibility_message: str | None
    front_hemisphere: bool
    azimuth_deg: float | None
    off_nadir_deg: float | None
    geometric_off_nadir_deg: float | None
    u_body: np.ndarray | None
    u_ecef: np.ndarray | None
    true_phases_rad: np.ndarray | None
    measured_phases_rad: np.ndarray | None
    slant_range_m: float | None
    inverse: InverseResult | None


@dataclass
class SensitivityCurve:
    altitude_km: float
    off_nadir_deg: np.ndarray
    ground_error_m: np.ndarray
    slant_range_km: np.ndarray
    elevation_deg: np.ndarray
    missed: np.ndarray
    horizon_deg: float


@dataclass
class MonteCarloResult:
    n_trials: int
    errors_m: np.ndarray
    success: np.ndarray
    ambiguous: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    fixed_channel_bias_deg: np.ndarray
    fixed_pair_bias_deg: np.ndarray
    pct_success: float
    pct_ambiguous: float
    median_unambiguous_m: float
    p95_unambiguous_m: float
    rms_unambiguous_m: float
    n_unambiguous: int
    median_primary_m: float
    p95_primary_m: float
    rms_primary_m: float


def max_reachable_latitude_deg(inclination_deg: float) -> float:
    """Largest |geocentric latitude| on a spherical circular orbit."""
    inc = np.deg2rad(inclination_deg)
    return float(np.rad2deg(np.arccos(min(1.0, abs(np.cos(inc))))))


def geodetic_to_ecef(lat_deg, lon_deg, h_m) -> np.ndarray:
    """WGS84 geodetic latitude, longitude, and height to ECEF meters.

    Scalar inputs return shape (3,). Array inputs return shape (..., 3).
    """
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sl * sl)
    h = np.asarray(h_m, dtype=float)
    x = (n + h) * cl * np.cos(lon)
    y = (n + h) * cl * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h) * sl
    return np.stack((x, y, z), axis=-1)


def geodetic_normal(lat_deg, lon_deg) -> np.ndarray:
    """Outward WGS84 ellipsoid normal. Scalar inputs return shape (3,)."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    return np.stack((
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ), axis=-1)


def ecef_to_geodetic(xyz) -> tuple[float, float, float]:
    """ECEF meters to WGS84 latitude (deg), longitude (deg), and height (m).

    Inverse uses Bowring's reduced-latitude formula plus one latitude update.
    """
    x, y, z = (float(v) for v in np.asarray(xyz, dtype=float).reshape(3))
    lon = float(np.arctan2(y, x))
    p = float(np.hypot(x, y))
    if p < 1e-9:
        lat = float(np.copysign(np.pi / 2.0, z))
        h = abs(z) - WGS84_B
        return float(np.rad2deg(lat)), float(np.rad2deg(lon)), float(h)
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)
    st, ct = np.sin(theta), np.cos(theta)
    lat = np.arctan2(z + WGS84_EP2 * WGS84_B * st**3, p - WGS84_E2 * WGS84_A * ct**3)
    for _ in range(2):
        sl, cl = np.sin(lat), np.cos(lat)
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sl * sl)
        h = p / cl - n
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + h)))
    sl, cl = np.sin(lat), np.cos(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sl * sl)
    h = p / cl - n
    return float(np.rad2deg(lat)), float(np.rad2deg(lon)), float(h)


def geodesic_distance_m(lat1_deg, lon1_deg, lat2_deg, lon2_deg) -> float:
    """WGS84 geodesic distance on the ellipsoid, in meters."""
    _az12, _az21, dist = _GEOD.inv(lon1_deg, lat1_deg, lon2_deg, lat2_deg)
    return float(dist)


def geodesic_arc(lat1_deg, lon1_deg, lat2_deg, lon2_deg, n: int = 24, h_m: float = 0.0) -> np.ndarray:
    """ECEF samples along the WGS84 geodesic, including both endpoints."""
    if n < 2:
        raise ValueError('Need at least two geodesic samples')
    mids = _GEOD.npts(lon1_deg, lat1_deg, lon2_deg, lat2_deg, n - 2, initial_idx=0, terminus_idx=0)
    lons = [lon1_deg] + [p[0] for p in mids] + [lon2_deg]
    lats = [lat1_deg] + [p[1] for p in mids] + [lat2_deg]
    return geodetic_to_ecef(np.asarray(lats), np.asarray(lons), h_m)


def _rotation_x(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def vvlh_from_body(attitude: Attitude) -> np.ndarray:
    """Map body-frame components into VVLH. Yaw, then pitch, then roll."""
    return _rotation_z(attitude.yaw_deg) @ _rotation_y(attitude.pitch_deg) @ _rotation_x(attitude.roll_deg)


def build_orbit_frame(spec: OrbitSpec) -> OrbitFrame:
    """Spherical circular-orbit VVLH basis at a WGS84 satellite position.

    Inclination is the angle between the orbit normal and ECEF +Z, in
    [0, 180]. Ascending means geocentric latitude is increasing.
    """
    if not 0.0 <= spec.inclination_deg <= 180.0:
        raise OrbitFrameError('Inclination must be between 0° and 180°.')
    if abs(spec.latitude_deg) > 90.0:
        raise OrbitFrameError('Satellite latitude must be between −90° and 90°.')
    if spec.altitude_m <= 0.0:
        raise OrbitFrameError('Altitude must be above the WGS84 ellipsoid.')
    sat = np.asarray(geodetic_to_ecef(spec.latitude_deg, spec.longitude_deg, spec.altitude_m), dtype=float)
    r_norm = float(np.linalg.norm(sat))
    r_hat = sat / r_norm
    lon = float(np.arctan2(sat[1], sat[0]))
    lat = float(np.arctan2(sat[2], np.hypot(sat[0], sat[1])))
    cos_lat = float(np.cos(lat))
    if cos_lat < 1e-8:
        raise OrbitFrameError('The satellite is too close to a pole to build a stable East-North basis.')
    inc = float(np.deg2rad(spec.inclination_deg))
    v_east = float(np.cos(inc) / cos_lat)
    if abs(v_east) > 1.0 + 1e-9:
        reached = max_reachable_latitude_deg(spec.inclination_deg)
        raise OrbitFrameError(
            f'Geocentric latitude {np.rad2deg(lat):.2f}° is outside this orbit. '
            f'At inclination {spec.inclination_deg:.2f}° the maximum |latitude| is {reached:.2f}°.'
        )
    v_east = float(np.clip(v_east, -1.0, 1.0))
    v_north = float(np.sqrt(max(0.0, 1.0 - v_east * v_east)))
    if not spec.ascending:
        v_north = -v_north
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    v_hat = v_east * east + v_north * north
    v_hat = v_hat - np.dot(v_hat, r_hat) * r_hat
    v_hat = v_hat / np.linalg.norm(v_hat)
    z_vvlh = -r_hat
    x_vvlh = v_hat
    y_vvlh = np.cross(z_vvlh, x_vvlh)
    y_vvlh = y_vvlh / np.linalg.norm(y_vvlh)
    x_vvlh = np.cross(y_vvlh, z_vvlh)
    x_vvlh = x_vvlh / np.linalg.norm(x_vvlh)
    if abs(np.sin(inc)) < 1e-10:
        arg_lat = None
    else:
        sin_u = float(np.clip(np.sin(lat) / np.sin(inc), -1.0, 1.0))
        cos_u = float(np.sqrt(max(0.0, 1.0 - sin_u * sin_u)))
        if not spec.ascending:
            cos_u = -cos_u
        arg_lat = float(np.rad2deg(np.arctan2(sin_u, cos_u)) % 360.0)
    heading = float(np.rad2deg(np.arctan2(v_east, v_north)) % 360.0)
    return OrbitFrame(
        spec=spec,
        sat_ecef=sat,
        r_hat=r_hat,
        v_hat=x_vvlh.copy(),
        geocentric_lat_deg=float(np.rad2deg(lat)),
        geocentric_lon_deg=float(np.rad2deg(lon)),
        heading_deg=heading,
        argument_of_latitude_deg=arg_lat,
        x_vvlh=x_vvlh,
        y_vvlh=y_vvlh,
        z_vvlh=z_vvlh,
        R_ecef_from_vvlh=np.column_stack((x_vvlh, y_vvlh, z_vvlh)),
    )


def body_to_ecef(v_body, frame: OrbitFrame, attitude: Attitude) -> np.ndarray:
    v_vvlh = vvlh_from_body(attitude) @ np.asarray(v_body, dtype=float)
    return frame.R_ecef_from_vvlh @ v_vvlh


def ecef_to_body(v_ecef, frame: OrbitFrame, attitude: Attitude) -> np.ndarray:
    v_vvlh = frame.R_ecef_from_vvlh.T @ np.asarray(v_ecef, dtype=float)
    return vvlh_from_body(attitude).T @ v_vvlh


def angles_from_unit(u) -> tuple[float, float]:
    """Azimuth (deg, CCW from +X) and off-boresight (deg from +Z)."""
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)
    off = float(np.rad2deg(np.arctan2(np.hypot(u[0], u[1]), u[2])))
    az = float(np.rad2deg(np.arctan2(u[1], u[0])) % 360.0)
    return az, off


def angle_between_deg(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.rad2deg(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def horizontal_baseline_rank(positions) -> int:
    pos = np.asarray(positions, dtype=float)
    if len(pos) < 2:
        return 0
    baselines = pos[1:, :2] - pos[0, :2]
    if baselines.size == 0:
        return 0
    return int(np.linalg.matrix_rank(baselines, tol=1e-9))


def intersect_ray_wgs84(origin, direction) -> tuple[np.ndarray | None, float | None, str]:
    """First forward intersection of a ray with the WGS84 ellipsoid.

    Returns ``(point_ecef, slant_range_m, status)``. Status is ``ok``,
    ``miss`` (no real intersection), or ``behind`` (intersections only
    opposite the ray direction).
    """
    origin = np.asarray(origin, dtype=float).reshape(3)
    direction = np.asarray(direction, dtype=float).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError('Ray direction must be a finite non-zero vector')
    d_ecef = direction / norm
    # Scale onto the unit sphere so the quadratic coefficients stay O(1).
    # A direct ECEF quadratic has coefficients ~1/a^2 and confuses misses with grazes.
    scale = np.array([WGS84_A, WGS84_A, WGS84_B])
    o = origin / scale
    d = d_ecef / scale
    d_norm = float(np.linalg.norm(d))
    u = d / d_norm
    b_coef = 2.0 * float(np.dot(o, u))
    c_coef = float(np.dot(o, o) - 1.0)
    disc = b_coef * b_coef - 4.0 * c_coef
    if disc < 0.0:
        if disc > -1e-10:
            disc = 0.0
        else:
            return None, None, 'miss'
    root = float(np.sqrt(disc))
    taus = ((-b_coef - root) / 2.0, (-b_coef + root) / 2.0)
    forward = [tau / d_norm for tau in taus if tau / d_norm > 1.0]
    if not forward:
        return None, None, 'behind'
    t = float(min(forward))
    return origin + t * d_ecef, t, 'ok'


def _inside_ellipsoid(point) -> bool:
    p = np.asarray(point, dtype=float)
    return bool((p[0] * p[0] + p[1] * p[1]) / (WGS84_A * WGS84_A) + (p[2] * p[2]) / (WGS84_B * WGS84_B) < 1.0)


def above_horizon(sat_ecef, emitter_ecef, lat_deg, lon_deg, tol_m: float = 10.0) -> bool:
    """True when the satellite is on or above the emitter's local horizon."""
    normal = geodetic_normal(lat_deg, lon_deg)
    return float(np.dot(normal, np.asarray(sat_ecef) - np.asarray(emitter_ecef))) >= -tol_m


def look_geometry(frame: OrbitFrame, attitude: Attitude, emitter_ecef):
    """Line of sight from the satellite to an emitter ECEF position."""
    los = np.asarray(emitter_ecef, dtype=float) - frame.sat_ecef
    slant = float(np.linalg.norm(los))
    if slant < 1.0:
        raise ValueError('Emitter is coincident with the satellite')
    u_ecef = los / slant
    u_body = ecef_to_body(u_ecef, frame, attitude)
    u_body = u_body / np.linalg.norm(u_body)
    az, off = angles_from_unit(u_body)
    geometric = angle_between_deg(u_ecef, -frame.r_hat)
    return u_ecef, u_body, az, off, geometric, slant


def emitter_on_ray(frame: OrbitFrame, attitude: Attitude, azimuth_deg: float, off_nadir_deg: float):
    """Ground point where a body-frame look direction meets WGS84.

    Returns ``(lat_deg, lon_deg, height_m, ecef, slant_range_m, status)``.
    """
    u_body = np.asarray(direction(azimuth_deg, off_nadir_deg), dtype=float)
    u_ecef = body_to_ecef(u_body, frame, attitude)
    point, slant, status = intersect_ray_wgs84(frame.sat_ecef, u_ecef)
    if status != 'ok':
        return None, None, None, None, None, status
    lat, lon, h = ecef_to_geodetic(point)
    return lat, lon, h, point, slant, status


def pair_biases_from_channels(channel_bias_deg) -> np.ndarray:
    """Reference-pair biases (antenna j minus antenna 0), in degrees."""
    bias = np.asarray(channel_bias_deg, dtype=float)
    if bias.ndim != 1 or bias.size < 2:
        raise ValueError('Channel biases must include the reference antenna and at least one other')
    return bias[1:] - bias[0]


def _phase_rms_deg(positions, frequency_hz, measured_rad, azimuth_deg, off_deg) -> float:
    pred = phase_difference(positions, frequency_hz, azimuth_deg, off_deg)
    residual = wrap_rad(pred - np.asarray(measured_rad, dtype=float))
    score = float(np.sqrt(np.mean((2.0 * np.sin(residual / 2.0)) ** 2)))
    return float(np.rad2deg(2.0 * np.arcsin(np.clip(score / 2.0, 0.0, 1.0))))


def _unit_from_horizontal(ux: float, uy: float) -> np.ndarray:
    r2 = ux * ux + uy * uy
    if r2 > 1.0:
        scale = np.sqrt(r2)
        ux, uy = ux / scale, uy / scale
        uz = 0.0
    else:
        uz = float(np.sqrt(1.0 - r2))
    return np.array([ux, uy, uz], dtype=float)


def refine_doa(positions, frequency_hz, measured_rad, step_deg: float = 2.0, max_candidates: int = 12) -> list[DoaEstimate]:
    """Front-hemisphere DOA minima from wrapped phases, refined off the search grid.

    The search reuses the Version 1–2 ambiguity surface. Refinement minimizes
    the circular phase residual and does not receive a true direction.
    """
    positions = np.asarray(positions, dtype=float)
    measured = np.asarray(measured_rad, dtype=float).reshape(-1)
    xx, yy, valid, score = ambiguity_surface(positions, frequency_hz, measured, step_deg)
    seeds = candidate_solutions(
        xx, yy, valid, score,
        max_candidates=max_candidates,
        separation=max(0.035, 2.0 * np.sin(np.deg2rad(step_deg))),
    )
    baselines = positions[1:, :2] - positions[0, :2]
    k = 2.0 * np.pi / wavelength(frequency_hz)

    def residual(u):
        pred = k * (baselines @ u)
        phase_r = 2.0 * np.sin(wrap_rad(pred - measured) / 2.0)
        overflow = max(0.0, float(u[0] * u[0] + u[1] * u[1]) - 1.0)
        return np.concatenate((phase_r, np.array([5.0 * np.sqrt(overflow)])))

    refined = []
    for seed in seeds:
        fit = least_squares(
            residual,
            np.array([seed.ux, seed.uy], dtype=float),
            method='trf',
            bounds=([-1.0, -1.0], [1.0, 1.0]),
            xtol=1e-12,
            ftol=1e-12,
            max_nfev=40,
        )
        unit = _unit_from_horizontal(float(fit.x[0]), float(fit.x[1]))
        az, off = angles_from_unit(unit)
        refined.append(DoaEstimate(
            ux=float(unit[0]), uy=float(unit[1]), uz=float(unit[2]),
            azimuth_deg=az, off_boresight_deg=off,
            phase_rms_deg=_phase_rms_deg(positions, frequency_hz, measured, az, off),
        ))
    refined.sort(key=lambda item: item.phase_rms_deg)
    kept: list[DoaEstimate] = []
    for item in refined:
        if any(angle_between_deg(item.unit, prior.unit) < 0.4 for prior in kept):
            continue
        kept.append(item)
    return kept


def estimate_emitter(measured_rad, positions, frequency_hz, frame: OrbitFrame, attitude: Attitude,
                     step_deg: float = 2.0) -> InverseResult:
    """Invert wrapped phases to Earth intercepts.

    The true emitter coordinates and the true line of sight are not inputs.
    The reported location is the lowest phase-residual candidate that meets
    the ellipsoid in front of the satellite. Other intercepts are retained.
    """
    doas = refine_doa(positions, frequency_hz, measured_rad, step_deg=step_deg)
    solutions: list[GroundSolution] = []
    for doa in doas:
        u_ecef = body_to_ecef(doa.unit, frame, attitude)
        u_ecef = u_ecef / np.linalg.norm(u_ecef)
        point, slant, status = intersect_ray_wgs84(frame.sat_ecef, u_ecef)
        if status != 'ok':
            solutions.append(GroundSolution(doa=doa, status=status, u_ecef=u_ecef))
            continue
        lat, lon, h = ecef_to_geodetic(point)
        solutions.append(GroundSolution(
            doa=doa, status='ok', lat_deg=lat, lon_deg=lon, height_m=h,
            ecef=point, slant_range_m=slant, u_ecef=u_ecef,
        ))
    valid = [s for s in solutions if s.status == 'ok']
    ambiguous = False
    if valid:
        best = min(s.doa.phase_rms_deg for s in valid)
        plausible = [
            s for s in valid
            if s.doa.phase_rms_deg <= best + AMBIGUITY_MARGIN_DEG
        ]
        distinct = []
        for item in plausible:
            u = item.doa.unit
            if any(angle_between_deg(u, prior.doa.unit) < 1.0 for prior in distinct):
                continue
            distinct.append(item)
        ambiguous = len(distinct) >= 2
    return InverseResult(
        solutions=solutions,
        ambiguous=ambiguous or horizontal_baseline_rank(positions) < 2,
        underdetermined=horizontal_baseline_rank(positions) < 2,
        n_doa_candidates=len(solutions),
        n_valid_intercepts=len(valid),
    )


def solve_scene(spec: OrbitSpec, attitude: Attitude, positions, frequency_hz: float, *,
                emitter_lat_deg: float | None = None, emitter_lon_deg: float | None = None,
                emitter_azimuth_deg: float | None = None, emitter_off_nadir_deg: float | None = None,
                noise_std_deg: float = 0.0, pair_biases_deg=None, rng=None,
                step_deg: float = 2.0, invert: bool = True) -> SceneResult:
    """Forward geometry plus an inverse that sees only the wrapped phases."""
    frame = build_orbit_frame(spec)
    positions = np.asarray(positions, dtype=float)
    by_angles = emitter_azimuth_deg is not None or emitter_off_nadir_deg is not None
    by_latlon = emitter_lat_deg is not None or emitter_lon_deg is not None
    if by_angles and by_latlon:
        raise ValueError('Specify the emitter by latitude/longitude or by azimuth/off-nadir, not both')
    if not by_angles and not by_latlon:
        raise ValueError('An emitter latitude/longitude or azimuth/off-nadir is required')

    if by_angles:
        lat, lon, _h, emitter, _slant, status = emitter_on_ray(
            frame, attitude, float(emitter_azimuth_deg), float(emitter_off_nadir_deg))
        if status != 'ok':
            reason = ('The look direction points away from Earth.' if status == 'behind'
                      else 'The look direction misses the Earth ellipsoid.')
            return SceneResult(
                frame, attitude, None, None, None, False, reason, False,
                float(emitter_azimuth_deg), float(emitter_off_nadir_deg), None,
                None, None, None, None, None, None,
            )
        emitter_lat_deg, emitter_lon_deg = lat, lon
        emitter_ecef = emitter
    else:
        if abs(emitter_lat_deg) > 90.0:
            raise ValueError('Emitter latitude must be between −90° and 90°.')
        emitter_ecef = np.asarray(geodetic_to_ecef(emitter_lat_deg, emitter_lon_deg, 0.0), dtype=float)

    visible = above_horizon(frame.sat_ecef, emitter_ecef, emitter_lat_deg, emitter_lon_deg)
    message = None if visible else 'Emitter is beyond the horizon. The Earth blocks this line of sight.'
    u_ecef, u_body, az, off, geometric, slant = look_geometry(frame, attitude, emitter_ecef)
    front = off <= 90.0 + 1e-6
    if visible and not front:
        message = 'Emitter is outside the antenna front hemisphere (+Z). No interferometer phase is formed.'
    phases_true = None
    phases_meas = None
    inverse = None
    if visible and front:
        phases_true = phase_difference(positions, frequency_hz, az, off)
        if pair_biases_deg is None:
            pair_biases_deg = np.zeros(phases_true.shape[-1])
        phases_meas = measured_phases(phases_true, noise_std_deg, pair_biases_deg, rng)
        if invert:
            inverse = estimate_emitter(phases_meas, positions, frequency_hz, frame, attitude, step_deg=step_deg)
    return SceneResult(
        frame=frame, attitude=attitude,
        emitter_lat_deg=float(emitter_lat_deg), emitter_lon_deg=float(emitter_lon_deg),
        emitter_ecef=np.asarray(emitter_ecef, dtype=float),
        visible=visible and front, visibility_message=message, front_hemisphere=front,
        azimuth_deg=az, off_nadir_deg=off, geometric_off_nadir_deg=geometric,
        u_body=u_body, u_ecef=u_ecef,
        true_phases_rad=None if phases_true is None else np.asarray(phases_true, dtype=float),
        measured_phases_rad=None if phases_meas is None else np.asarray(phases_meas, dtype=float),
        slant_range_m=slant, inverse=inverse,
    )


def horizon_off_boresight_deg(frame: OrbitFrame, attitude: Attitude, azimuth_deg: float) -> float | None:
    """Largest front-hemisphere off-boresight angle whose ray still meets Earth."""
    def hits(off: float) -> bool:
        _lat, _lon, _h, _ecef, _slant, status = emitter_on_ray(frame, attitude, azimuth_deg, off)
        return status == 'ok'

    if not hits(0.0):
        return None
    if hits(90.0):
        return 90.0
    lo, hi = 0.0, 90.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if hits(mid):
            lo = mid
        else:
            hi = mid
    return lo


def horizon_ring(frame: OrbitFrame, attitude: Attitude, n: int = 180) -> np.ndarray | None:
    """ECEF points on the satellite's Earth horizon, in the antenna azimuth order."""
    azimuths = np.linspace(0.0, 360.0, n, endpoint=False)
    points = []
    for azimuth in azimuths:
        limit = horizon_off_boresight_deg(frame, attitude, float(azimuth))
        if limit is None:
            return None
        _lat, _lon, _h, point, _slant, status = emitter_on_ray(
            frame, attitude, float(azimuth), max(0.0, limit - 0.02))
        if status != 'ok':
            continue
        points.append(point)
    if len(points) < 8:
        return None
    return np.asarray(points, dtype=float)


def off_nadir_sensitivity(latitude_deg: float, longitude_deg: float, inclination_deg: float,
                          ascending: bool, attitude: Attitude, azimuth_deg: float,
                          altitudes_m, delta_deg: float = 0.1, step_deg: float = 1.0) -> list[SensitivityCurve]:
    """Ground-location shift caused by a small off-nadir perturbation.

    The unperturbed ray defines the reference ground point. The perturbation
    is an increase in off-boresight along the same antenna azimuth. Rays that
    miss the Earth after the perturbation are marked and omitted from the
    error curve.
    """
    if delta_deg <= 0.0:
        raise ValueError('Angular perturbation must be positive')
    curves = []
    for altitude_m in altitudes_m:
        spec = OrbitSpec(latitude_deg, longitude_deg, float(altitude_m), inclination_deg, ascending)
        frame = build_orbit_frame(spec)
        horizon = horizon_off_boresight_deg(frame, attitude, azimuth_deg)
        if horizon is None or horizon <= 0.0:
            curves.append(SensitivityCurve(
                altitude_km=altitude_m / 1000.0,
                off_nadir_deg=np.array([]), ground_error_m=np.array([]),
                slant_range_km=np.array([]), elevation_deg=np.array([]),
                missed=np.array([], dtype=bool), horizon_deg=0.0,
            ))
            continue
        offs = np.arange(0.0, horizon + 1e-9, step_deg)
        if offs.size == 0 or offs[-1] < horizon:
            offs = np.append(offs, horizon)
        errors, slants, elevs, missed = [], [], [], []
        for off in offs:
            lat, lon, _h, _ecef, slant, status = emitter_on_ray(frame, attitude, azimuth_deg, float(off))
            if status != 'ok':
                errors.append(np.nan)
                slants.append(np.nan)
                elevs.append(np.nan)
                missed.append(True)
                continue
            plat, plon, _ph, _pecef, _pslant, pstatus = emitter_on_ray(
                frame, attitude, azimuth_deg, float(off) + delta_deg)
            slants.append(slant / 1000.0)
            elevs.append(_elevation_deg(frame.sat_ecef, lat, lon))
            if pstatus != 'ok':
                errors.append(np.nan)
                missed.append(True)
            else:
                errors.append(geodesic_distance_m(lat, lon, plat, plon))
                missed.append(False)
        curves.append(SensitivityCurve(
            altitude_km=float(altitude_m) / 1000.0,
            off_nadir_deg=np.asarray(offs, dtype=float),
            ground_error_m=np.asarray(errors, dtype=float),
            slant_range_km=np.asarray(slants, dtype=float),
            elevation_deg=np.asarray(elevs, dtype=float),
            missed=np.asarray(missed, dtype=bool),
            horizon_deg=float(horizon),
        ))
    return curves


def _elevation_deg(sat_ecef, lat_deg, lon_deg) -> float:
    surface = geodetic_to_ecef(lat_deg, lon_deg, 0.0)
    los = np.asarray(sat_ecef, dtype=float) - surface
    normal = geodetic_normal(lat_deg, lon_deg)
    sine = float(np.dot(normal, los) / np.linalg.norm(los))
    return float(np.rad2deg(np.arcsin(np.clip(sine, -1.0, 1.0))))


def _population_stats(errors: np.ndarray) -> tuple[float, float, float, int]:
    good = errors[np.isfinite(errors)]
    if good.size == 0:
        return float('nan'), float('nan'), float('nan'), 0
    return (
        float(np.median(good)),
        float(np.percentile(good, 95)),
        float(np.sqrt(np.mean(good ** 2))),
        int(good.size),
    )


def monte_carlo_geolocation(spec: OrbitSpec, attitude: Attitude, emitter_lat_deg: float,
                            emitter_lon_deg: float, positions, frequency_hz: float, *,
                            n_trials: int = 200, phase_noise_deg: float = 1.0,
                            channel_bias_sigma_deg: float = 0.0, attitude_sigma_deg: float = 0.0,
                            position_sigma_m: float = 0.0, seed: int = 42,
                            step_deg: float = 2.0) -> MonteCarloResult:
    """Repeated geolocation with separate systematic and random errors.

    Channel calibration biases are drawn once and held fixed. Phase noise,
    attitude-knowledge error, and position-knowledge error are independent
    on every trial. Position error shifts the assumed ECEF origin in the
    true VVLH frame (+X along-track, +Y cross-track, +Z nadir) and does not
    rebuild that frame. Attitude error is added to the estimator's roll,
    pitch, and yaw. The estimator never receives the true emitter location.
    """
    if n_trials < 1:
        raise ValueError('Need at least one Monte Carlo trial')
    positions = np.asarray(positions, dtype=float)
    scene = solve_scene(
        spec, attitude, positions, frequency_hz,
        emitter_lat_deg=emitter_lat_deg, emitter_lon_deg=emitter_lon_deg,
        noise_std_deg=0.0, rng=np.random.default_rng(0), step_deg=step_deg, invert=False,
    )
    if scene.true_phases_rad is None:
        raise ValueError(scene.visibility_message or 'Emitter is not observable')
    bias_rng = np.random.default_rng(seed)
    trial_rng = np.random.default_rng(seed + 10_007)
    channel_bias = bias_rng.normal(0.0, channel_bias_sigma_deg, size=len(positions))
    pair_bias = pair_biases_from_channels(channel_bias)
    errors = np.full(n_trials, np.nan)
    success = np.zeros(n_trials, dtype=bool)
    ambiguous = np.zeros(n_trials, dtype=bool)
    lats = np.full(n_trials, np.nan)
    lons = np.full(n_trials, np.nan)
    for i in range(n_trials):
        measured = measured_phases(scene.true_phases_rad, phase_noise_deg, pair_bias, trial_rng)
        att_est = Attitude(
            attitude.roll_deg + float(trial_rng.normal(0.0, attitude_sigma_deg)),
            attitude.pitch_deg + float(trial_rng.normal(0.0, attitude_sigma_deg)),
            attitude.yaw_deg + float(trial_rng.normal(0.0, attitude_sigma_deg)),
        )
        delta = trial_rng.normal(0.0, position_sigma_m, size=3)
        origin = scene.frame.sat_ecef + scene.frame.R_ecef_from_vvlh @ delta
        if _inside_ellipsoid(origin):
            continue
        frame_est = replace(scene.frame, sat_ecef=np.asarray(origin, dtype=float))
        inverse = estimate_emitter(measured, positions, frequency_hz, frame_est, att_est, step_deg=step_deg)
        primary = inverse.primary
        if primary is None:
            continue
        success[i] = True
        ambiguous[i] = inverse.ambiguous
        lats[i] = primary.lat_deg
        lons[i] = primary.lon_deg
        errors[i] = geodesic_distance_m(scene.emitter_lat_deg, scene.emitter_lon_deg, primary.lat_deg, primary.lon_deg)
    unambiguous_errors = np.where(success & ~ambiguous, errors, np.nan)
    primary_errors = np.where(success, errors, np.nan)
    med_u, p95_u, rms_u, n_u = _population_stats(unambiguous_errors)
    med_p, p95_p, rms_p, _n_p = _population_stats(primary_errors)
    return MonteCarloResult(
        n_trials=n_trials,
        errors_m=errors,
        success=success,
        ambiguous=ambiguous,
        lat_deg=lats,
        lon_deg=lons,
        fixed_channel_bias_deg=channel_bias,
        fixed_pair_bias_deg=pair_bias,
        pct_success=100.0 * float(np.mean(success)),
        pct_ambiguous=100.0 * float(np.mean(ambiguous)),
        median_unambiguous_m=med_u,
        p95_unambiguous_m=p95_u,
        rms_unambiguous_m=rms_u,
        n_unambiguous=n_u,
        median_primary_m=med_p,
        p95_primary_m=p95_p,
        rms_primary_m=rms_p,
    )
