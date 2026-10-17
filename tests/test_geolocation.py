import inspect

import numpy as np
import pytest
from pyproj import Transformer

from interferometer import array_positions, direction, phase_difference, wrap_rad
from geolocation import (
    WGS84_A,
    Attitude,
    OrbitFrameError,
    OrbitSpec,
    above_horizon,
    angles_from_unit,
    body_to_ecef,
    build_orbit_frame,
    ecef_to_body,
    ecef_to_geodetic,
    emitter_on_ray,
    estimate_emitter,
    geodesic_distance_m,
    geodetic_to_ecef,
    horizontal_baseline_rank,
    intersect_ray_wgs84,
    max_reachable_latitude_deg,
    monte_carlo_geolocation,
    off_nadir_sensitivity,
    solve_scene,
    vvlh_from_body,
)


_TO_ECEF = Transformer.from_crs('EPSG:4979', 'EPSG:4978', always_xy=True)
_TO_GEOD = Transformer.from_crs('EPSG:4978', 'EPSG:4979', always_xy=True)


def _unique_array():
    return array_positions('3-L', 0.05, 0.05)


def _ellipsoid_residual(point):
    return (point[0] ** 2 + point[1] ** 2) / WGS84_A ** 2 + point[2] ** 2 / (6_356_752.314245179 ** 2) - 1.0


def test_geodetic_ecef_known_points():
    np.testing.assert_allclose(geodetic_to_ecef(0, 0, 0), [WGS84_A, 0, 0], atol=1e-6)
    np.testing.assert_allclose(geodetic_to_ecef(0, 90, 0), [0, WGS84_A, 0], atol=1e-6)
    np.testing.assert_allclose(geodetic_to_ecef(0, 0, 500_000), [WGS84_A + 500_000, 0, 0], atol=1e-4)
    pole = geodetic_to_ecef(90, 0, 0)
    assert abs(pole[0]) < 1e-6 and abs(pole[1]) < 1e-6
    assert pole[2] > 6.3e6


def test_geodetic_round_trip_matches_pyproj():
    samples = [(0, 0, 0), (20, -100, 500_000), (-35, 140, 10_000), (45, 10, 800_000),
               (89.5, -20, 300_000), (-89, 170, 0)]
    for lat, lon, h in samples:
        ecef = geodetic_to_ecef(lat, lon, h)
        px, py, pz = _TO_ECEF.transform(lon, lat, h)
        np.testing.assert_allclose(ecef, [px, py, pz], atol=1e-4)
        got_lat, got_lon, got_h = ecef_to_geodetic(ecef)
        assert got_lat == pytest.approx(lat, abs=1e-8)
        assert got_lon == pytest.approx(lon, abs=1e-8)
        assert got_h == pytest.approx(h, abs=1e-3)
        rlon, rlat, rh = _TO_GEOD.transform(ecef[0], ecef[1], ecef[2])
        # PROJ's own inverse leaves a few millimeters at LEO altitude; the ECEF match above is exact.
        assert got_lat == pytest.approx(rlat, abs=1e-6)
        assert got_lon == pytest.approx(rlon, abs=1e-6)
        assert got_h == pytest.approx(rh, abs=0.02)


def test_equatorial_headings_and_nadir_axis():
    polar = build_orbit_frame(OrbitSpec(0, 0, 500_000, 90, True))
    np.testing.assert_allclose(polar.v_hat, [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(polar.z_vvlh, -polar.r_hat, atol=1e-12)
    np.testing.assert_allclose(polar.y_vvlh, [0, 1, 0], atol=1e-12)
    equatorial = build_orbit_frame(OrbitSpec(0, 0, 500_000, 0, True))
    np.testing.assert_allclose(equatorial.v_hat, [0, 1, 0], atol=1e-12)
    inclined = build_orbit_frame(OrbitSpec(0, 0, 500_000, 53, True))
    expected = np.array([0.0, np.cos(np.deg2rad(53)), np.sin(np.deg2rad(53))])
    np.testing.assert_allclose(inclined.v_hat, expected, atol=1e-12)
    assert inclined.argument_of_latitude_deg == pytest.approx(0.0, abs=1e-8)
    retrograde = build_orbit_frame(OrbitSpec(0, 0, 500_000, 98, True))
    assert retrograde.v_hat[1] < 0
    rotation = polar.R_ecef_from_vvlh
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_descending_reverses_north_component():
    up = build_orbit_frame(OrbitSpec(20, -100, 500_000, 53, True))
    down = build_orbit_frame(OrbitSpec(20, -100, 500_000, 53, False))
    assert up.argument_of_latitude_deg < 90
    assert down.argument_of_latitude_deg > 90
    # Latitude rate changes sign; the vertical ECEF component of velocity flips relative to north.
    assert np.dot(up.v_hat, down.v_hat) < 0.2


def test_unreachable_latitude_raises():
    with pytest.raises(OrbitFrameError):
        build_orbit_frame(OrbitSpec(60, 0, 500_000, 53, True))
    assert max_reachable_latitude_deg(53) == pytest.approx(53, abs=1e-6)
    assert max_reachable_latitude_deg(98) == pytest.approx(82, abs=0.1)


def test_attitude_sign_convention_and_round_trip():
    nose = vvlh_from_body(Attitude(0, 10, 0)) @ np.array([1.0, 0.0, 0.0])
    assert nose[2] < 0
    wing = vvlh_from_body(Attitude(10, 0, 0)) @ np.array([0.0, 1.0, 0.0])
    assert wing[2] > 0
    yawed = vvlh_from_body(Attitude(0, 0, 20)) @ np.array([1.0, 0.0, 0.0])
    assert yawed[1] > 0
    frame = build_orbit_frame(OrbitSpec(20, -100, 500_000, 53, True))
    attitude = Attitude(8, -5, 15)
    vector = np.array([0.2, -0.4, 0.7])
    vector = vector / np.linalg.norm(vector)
    recovered = ecef_to_body(body_to_ecef(vector, frame, attitude), frame, attitude)
    np.testing.assert_allclose(recovered, vector, atol=1e-12)


def test_ray_hits_geodetic_subpoint_along_the_normal():
    lat, lon, h = 37.0, -122.0, 500_000.0
    surface = geodetic_to_ecef(lat, lon, 0)
    sat = geodetic_to_ecef(lat, lon, h)
    normal = (sat - surface) / np.linalg.norm(sat - surface)
    hit, slant, status = intersect_ray_wgs84(sat, -normal)
    assert status == 'ok'
    np.testing.assert_allclose(hit, surface, atol=1e-4)
    assert slant == pytest.approx(h, rel=1e-12)
    assert abs(_ellipsoid_residual(hit)) < 1e-12


def test_ray_miss_and_ray_pointing_away():
    sat = geodetic_to_ecef(0, 0, 500_000)
    _hit, _slant, miss = intersect_ray_wgs84(sat, [0, 1, 0])
    assert miss == 'miss'
    _hit, _slant, away = intersect_ray_wgs84(sat, [1, 0, 0])
    assert away == 'behind'


def test_geocentric_nadir_matches_subpoint_only_at_equator():
    frame = build_orbit_frame(OrbitSpec(0, -100, 500_000, 53, True))
    hit, _slant, status = intersect_ray_wgs84(frame.sat_ecef, -frame.r_hat)
    assert status == 'ok'
    lat, lon, _h = ecef_to_geodetic(hit)
    assert geodesic_distance_m(0, -100, lat, lon) < 1.0
    mid = build_orbit_frame(OrbitSpec(45, 10, 500_000, 53, True))
    hit, _slant, status = intersect_ray_wgs84(mid.sat_ecef, -mid.r_hat)
    lat, lon, _h = ecef_to_geodetic(hit)
    separation = geodesic_distance_m(45, 10, lat, lon)
    assert 1_000 < separation < 5_000


def test_nadir_emitter_has_zero_horizontal_error():
    spec = OrbitSpec(0, -100, 500_000, 53, True)
    scene = solve_scene(
        spec, Attitude(), _unique_array(), 2.2e9,
        emitter_lat_deg=0.0, emitter_lon_deg=-100.0,
        noise_std_deg=0.0, rng=np.random.default_rng(1), step_deg=2.0,
    )
    assert scene.visible
    assert scene.off_nadir_deg == pytest.approx(0.0, abs=1e-6)
    assert scene.inverse.n_valid_intercepts == 1
    assert scene.inverse.ambiguous is False
    primary = scene.inverse.primary
    error = geodesic_distance_m(0, -100, primary.lat_deg, primary.lon_deg)
    assert error < 1.0
    assert scene.slant_range_m == pytest.approx(500_000, rel=1e-6)


def test_satellite_directly_above_emitter_off_equator():
    spec = OrbitSpec(27, 15, 600_000, 53, True)
    frame = build_orbit_frame(spec)
    lat, lon, _h, _ecef, _slant, status = emitter_on_ray(frame, Attitude(), 0, 0)
    assert status == 'ok'
    scene = solve_scene(
        spec, Attitude(), _unique_array(), 2.2e9,
        emitter_lat_deg=lat, emitter_lon_deg=lon,
        noise_std_deg=0.0, rng=np.random.default_rng(2), step_deg=2.0,
    )
    primary = scene.inverse.primary
    assert geodesic_distance_m(lat, lon, primary.lat_deg, primary.lon_deg) < 1.0
    assert scene.off_nadir_deg == pytest.approx(0.0, abs=1e-5)


def test_known_off_nadir_round_trip():
    spec = OrbitSpec(20, -100, 500_000, 53, True)
    attitude = Attitude(0, 0, 0)
    scene = solve_scene(
        spec, attitude, _unique_array(), 2.2e9,
        emitter_azimuth_deg=35, emitter_off_nadir_deg=18,
        noise_std_deg=0.0, rng=np.random.default_rng(3), step_deg=1.0,
    )
    assert scene.visible
    assert scene.azimuth_deg == pytest.approx(35, abs=1e-6)
    assert scene.off_nadir_deg == pytest.approx(18, abs=1e-6)
    primary = scene.inverse.primary
    assert angle_error(scene.u_body, primary.doa.unit) < 0.01
    assert geodesic_distance_m(scene.emitter_lat_deg, scene.emitter_lon_deg, primary.lat_deg, primary.lon_deg) < 20.0
    # The same ground point, specified by latitude and longitude, reproduces the look angles.
    again = solve_scene(
        spec, attitude, _unique_array(), 2.2e9,
        emitter_lat_deg=scene.emitter_lat_deg, emitter_lon_deg=scene.emitter_lon_deg,
        noise_std_deg=0.0, rng=np.random.default_rng(3), step_deg=1.0, invert=False,
    )
    assert again.azimuth_deg == pytest.approx(35, abs=1e-4)
    assert again.off_nadir_deg == pytest.approx(18, abs=1e-4)


def test_phase_forward_inverse_consistency():
    positions = _unique_array()
    measured = wrap_rad(phase_difference(positions, 2.2e9, 35, 40))
    spec = OrbitSpec(0, 0, 500_000, 90, True)
    frame = build_orbit_frame(spec)
    inverse = estimate_emitter(measured, positions, 2.2e9, frame, Attitude(), step_deg=1.0)
    assert inverse.primary is not None
    assert angle_error(direction(35, 40), inverse.primary.doa.unit) < 0.02
    # Truth is not an input to the estimator.
    assert 'emitter_lat_deg' not in inspect.signature(estimate_emitter).parameters


def test_ambiguous_doa_keeps_multiple_intercepts():
    positions = array_positions('3-L', 0.07, 0.32)
    assert horizontal_baseline_rank(positions) == 2
    scene = solve_scene(
        OrbitSpec(10, 20, 500_000, 53, True), Attitude(), positions, 2.2e9,
        emitter_azimuth_deg=40, emitter_off_nadir_deg=25,
        noise_std_deg=0.0, rng=np.random.default_rng(4), step_deg=1.0,
    )
    assert scene.inverse.n_doa_candidates >= 2
    assert scene.inverse.n_valid_intercepts >= 2
    assert scene.inverse.ambiguous
    costs = [s.doa.phase_rms_deg for s in scene.inverse.solutions if s.status == 'ok']
    assert min(costs) < 0.05
    # At least one intercept is the true direction; selection is by phase cost.
    separations = [angle_error(scene.u_body, s.doa.unit) for s in scene.inverse.solutions]
    assert min(separations) < 1.0


def test_collinear_array_is_underdetermined():
    positions = array_positions('2-element', 0.07, 0.5)
    scene = solve_scene(
        OrbitSpec(0, 0, 500_000, 53, True), Attitude(), positions, 2.2e9,
        emitter_azimuth_deg=10, emitter_off_nadir_deg=12,
        noise_std_deg=0.0, rng=np.random.default_rng(5), step_deg=2.0,
    )
    assert scene.inverse.underdetermined
    assert scene.inverse.ambiguous


def test_controlled_phase_noise_moves_the_fix():
    positions = _unique_array()
    spec = OrbitSpec(15, 40, 500_000, 53, True)
    quiet = solve_scene(
        spec, Attitude(), positions, 2.2e9,
        emitter_azimuth_deg=20, emitter_off_nadir_deg=12,
        noise_std_deg=0.0, rng=np.random.default_rng(9), step_deg=2.0,
    )
    noisy = solve_scene(
        spec, Attitude(), positions, 2.2e9,
        emitter_azimuth_deg=20, emitter_off_nadir_deg=12,
        noise_std_deg=2.0, rng=np.random.default_rng(9), step_deg=2.0,
    )
    quiet_error = geodesic_distance_m(
        quiet.emitter_lat_deg, quiet.emitter_lon_deg, quiet.inverse.primary.lat_deg, quiet.inverse.primary.lon_deg)
    noisy_error = geodesic_distance_m(
        noisy.emitter_lat_deg, noisy.emitter_lon_deg, noisy.inverse.primary.lat_deg, noisy.inverse.primary.lon_deg)
    assert quiet_error < 5.0
    assert noisy_error > quiet_error
    assert noisy_error < 80_000


def test_horizon_and_sensitivity_miss():
    spec = OrbitSpec(0, 0, 500_000, 53, True)
    assert above_horizon(geodetic_to_ecef(0, 0, 500_000), geodetic_to_ecef(0, 0, 0), 0, 0)
    assert not above_horizon(geodetic_to_ecef(0, 0, 500_000), geodetic_to_ecef(0, 90, 0), 0, 90)
    curves = off_nadir_sensitivity(0, 0, 53, True, Attitude(), 0.0, [500_000, 1_000_000], delta_deg=0.1, step_deg=5)
    low, high = curves
    assert low.horizon_deg > 60
    assert high.horizon_deg < low.horizon_deg
    finite = np.isfinite(low.ground_error_m)
    assert finite[0]
    assert low.ground_error_m[finite][-1] > low.ground_error_m[finite][0]
    assert np.any(low.missed)
    frame = build_orbit_frame(spec)
    _lat, _lon, _h, _ecef, _slant, status = emitter_on_ray(frame, Attitude(), 0, low.horizon_deg + 1.0)
    assert status == 'miss'


def test_monte_carlo_separates_bias_from_noise():
    positions = _unique_array()
    spec = OrbitSpec(5, 5, 500_000, 53, True)
    kwargs = dict(spec=spec, attitude=Attitude(), emitter_lat_deg=8.0, emitter_lon_deg=7.0,
                  positions=positions, frequency_hz=2.2e9, n_trials=6, step_deg=2.0, seed=11)
    systematic = monte_carlo_geolocation(
        phase_noise_deg=0.0, channel_bias_sigma_deg=4.0, attitude_sigma_deg=0.0, position_sigma_m=0.0, **kwargs)
    assert systematic.pct_success == 100
    assert systematic.pct_ambiguous == 0
    assert np.nanstd(systematic.errors_m) < 1e-6
    assert np.nanmean(systematic.errors_m) > 100
    random = monte_carlo_geolocation(
        phase_noise_deg=1.5, channel_bias_sigma_deg=0.0, attitude_sigma_deg=0.0, position_sigma_m=0.0, **kwargs)
    assert random.pct_success == 100
    assert np.nanstd(random.errors_m) > 1.0
    assert random.fixed_channel_bias_deg.shape == (len(positions),)
    assert np.allclose(random.fixed_pair_bias_deg, 0)


def test_body_angles_match_interferometer_direction():
    az, off = angles_from_unit(direction(35, 40))
    assert az == pytest.approx(35)
    assert off == pytest.approx(40)


def test_earth_figure_uses_equal_data_scale():
    from satellite_viz import GlobeLayers, earth_figure
    sat = geodetic_to_ecef(20, -100, 500_000)
    sub = geodetic_to_ecef(20, -100, 0)
    emitter = geodetic_to_ecef(25, -95, 0)
    other = geodetic_to_ecef(25.2, -94.8, 0)
    fig = earth_figure(
        sat, sub, emitter, estimates=[{'ecef': other, 'primary': True}],
        true_latlon=(25, -95), primary_latlon=(25.2, -94.8), layers=GlobeLayers(),
    )
    assert fig.layout.scene.aspectmode == 'data'
    assert any(getattr(trace, 'type', None) == 'surface' for trace in fig.data)
    assert any(getattr(trace, 'name', None) == 'Satellite' for trace in fig.data)


def angle_error(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.rad2deg(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))
