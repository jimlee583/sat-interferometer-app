"""Streamlit satellite RF interferometer: array lab (versions 1–2) and LEO geolocation (version 3)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from interferometer import (wavelength, array_positions, pairs, phase_difference,
                            measured_phases, wrap_rad, ambiguity_surface,
                            candidate_solutions, direction)
from geolocation import (Attitude, OrbitFrameError, OrbitSpec, angle_between_deg,
                         geodesic_distance_m, geodetic_to_ecef, horizon_ring,
                         monte_carlo_geolocation, off_nadir_sensitivity,
                         pair_biases_from_channels, solve_scene)
from satellite_viz import GlobeLayers, earth_figure


def _quantity(value, unit, digits=3):
    if value is None or not np.isfinite(value):
        return '—', unit
    return f'{value:.{digits}f}', unit


def _distance_text(meters):
    if meters is None or not np.isfinite(meters):
        return '—', 'm'
    if abs(meters) >= 1000:
        return f'{meters / 1000:.3f}', 'km'
    return f'{meters:.1f}', 'm'


def render_satellite_tab(positions, frequency_hz, noise_deg, common_bias_deg, seed, step_deg, layout, lam, long_m):
    """LEO interferometer geolocation. Versions 1–2 stay on the other tabs."""
    st.subheader('Satellite geolocation')
    st.caption('Sidebar frequency, array, phase noise, common pair bias, seed, and direction-map resolution are shared. '
               'The sidebar azimuth and off-boresight sliders apply only to the interferometer tabs.')
    with st.expander('Coordinate conventions'):
        st.markdown(
            '**ECEF** is Earth-centered, Earth-fixed. **Geodetic** latitude, longitude, and height use WGS84. '
            'The satellite subpoint is the ellipsoid point with the satellite’s geodetic latitude and longitude.\n\n'
            '**VVLH** is a spherical circular-orbit frame, not a zenith-positive LVLH frame:\n'
            '- +X along-track (velocity)\n'
            '- +Y cross-track, equal to −orbit normal\n'
            '- +Z geocentric nadir, toward Earth’s center\n\n'
            'Heading comes from inclination and ascending or descending motion. '
            'A subpoint alone does not set the velocity direction.\n\n'
            '**Body** matches VVLH at zero attitude. Rotations are applied yaw, then pitch, then roll (3-2-1). '
            'Positive yaw moves the nose right, positive pitch moves the nose up (away from Earth), '
            'and positive roll moves the right wing toward nadir.\n\n'
            '**Antenna** axes match the body. The interferometer +Z boresight is body +Z. '
            'With zero attitude that is geocentric nadir, which is a few kilometers from the geodetic subpoint at mid-latitudes.'
        )
        st.latex(r'\mathbf p(t)=\mathbf r_s+t\hat{\mathbf u},\qquad \frac{x^2+y^2}{a^2}+\frac{z^2}{b^2}=1')
        st.latex(r'\mathbf v_{\mathrm{VVLH}}=R_z(\psi)\,R_y(\theta)\,R_x(\phi)\,\mathbf v_{\mathrm{body}}')

    orbit, attitude_col, emitter_col = st.columns(3)
    with orbit:
        st.markdown('**Orbit and position**')
        alt_km = st.slider('Satellite altitude (km)', 300.0, 1500.0, 500.0, 10.0)
        inclination = st.slider('Orbit inclination (°)', 0.0, 180.0, 53.0, 0.5)
        sub_lat = st.slider('Subpoint latitude (°)', -90.0, 90.0, 20.0, 0.1)
        sub_lon = st.slider('Subpoint longitude (°)', -180.0, 180.0, -100.0, 0.1)
        ascending = st.radio('Orbital direction', ['Ascending', 'Descending'], horizontal=True) == 'Ascending'
    with attitude_col:
        st.markdown('**Attitude relative to VVLH**')
        roll = st.slider('Roll (°)', -20.0, 20.0, 0.0, 0.1)
        pitch = st.slider('Pitch (°)', -20.0, 20.0, 0.0, 0.1)
        yaw = st.slider('Yaw (°)', -20.0, 20.0, 0.0, 0.1)
        st.caption('Order: yaw, then pitch, then roll. Zero is nadir pointing. '
                   'These are the true angles. Monte Carlo can add a separate knowledge error.')
    with emitter_col:
        st.markdown('**Emitter**')
        by_angles = st.radio('Specify emitter by', ['Latitude / longitude', 'Azimuth and off-nadir']) == 'Azimuth and off-nadir'
        if by_angles:
            emitter_az = st.slider('Emitter azimuth from +X (°)', 0.0, 360.0, 45.0, 1.0)
            emitter_off = st.slider('Emitter off-nadir from +Z (°)', 0.0, 90.0, 20.0, 0.5)
            emitter_lat = emitter_lon = None
        else:
            emitter_lat = st.slider('Emitter latitude (°)', -90.0, 90.0, 25.0, 0.1)
            emitter_lon = st.slider('Emitter longitude (°)', -180.0, 180.0, -95.0, 0.1)
            emitter_az = emitter_off = None
        use_cal = st.checkbox('Channel phase calibration errors')
        channel = np.zeros(len(positions))
        if use_cal:
            st.caption('Fixed per-antenna bias, unknown to the estimator. Pair bias is antenna j minus A0.')
            for index in range(1, len(positions)):
                channel[index] = st.number_input(
                    f'A{index} channel bias (°)', -30.0, 30.0, 0.0, 0.5, key=f'chan_bias_{index}')

    if layout == '3-L' and long_m > 0.5 * lam:
        st.info(f'The 3-L layout puts both arms on the long baseline. {long_m:.2f} m is {long_m / lam:.1f} wavelengths '
                f'at this frequency, so the wrapped phases are usually ambiguous. '
                f'A baseline near {0.5 * lam:.3f} m is about half a wavelength.')

    spec = OrbitSpec(sub_lat, sub_lon, alt_km * 1000.0, inclination, ascending)
    attitude = Attitude(roll, pitch, yaw)
    pair_bias = pair_biases_from_channels(channel) + common_bias_deg
    try:
        scene = solve_scene(
            spec, attitude, positions, frequency_hz,
            emitter_lat_deg=emitter_lat, emitter_lon_deg=emitter_lon,
            emitter_azimuth_deg=emitter_az, emitter_off_nadir_deg=emitter_off,
            noise_std_deg=noise_deg, pair_biases_deg=pair_bias,
            rng=np.random.default_rng(seed), step_deg=step_deg,
        )
    except OrbitFrameError as exc:
        st.error(str(exc))
        return
    except ValueError as exc:
        st.error(str(exc))
        return

    if scene.visibility_message:
        st.warning(scene.visibility_message)
    inverse = scene.inverse
    if inverse is not None and inverse.underdetermined:
        st.warning('The array is collinear, so one direction cosine is unobserved. The geolocation is ambiguous.')
    elif inverse is not None and inverse.ambiguous:
        st.warning('Geolocation is ambiguous. More than one direction fits the wrapped phases. '
                   'The highlighted fix is the lowest phase residual, not a disambiguated location.')
    if inverse is not None and inverse.solutions:
        best = min(inverse.solutions, key=lambda item: item.doa.phase_rms_deg)
        if best.status != 'ok':
            st.info('The lowest phase-residual direction does not meet the Earth in front of the satellite. '
                    'Any highlighted fix is the best residual among rays that do.')
    if inverse is not None and inverse.n_doa_candidates == 0:
        st.warning('No direction-of-arrival candidate was found. Use a finer direction-map resolution in the sidebar.')

    primary = None if inverse is None else inverse.primary
    angular = None
    ground_m = None
    if primary is not None and scene.u_body is not None:
        angular = angle_between_deg(scene.u_body, primary.doa.unit)
        ground_m = geodesic_distance_m(scene.emitter_lat_deg, scene.emitter_lon_deg, primary.lat_deg, primary.lon_deg)

    ground_value, ground_unit = _distance_text(ground_m)
    slant_value, slant_unit = _distance_text(scene.slant_range_m)
    rows = [
        ('Satellite altitude', f'{alt_km:.1f}', 'km'),
        ('Satellite latitude/longitude', f'{sub_lat:.3f}, {sub_lon:.3f}', 'degrees'),
        ('True emitter latitude/longitude',
         '—' if scene.emitter_lat_deg is None else f'{scene.emitter_lat_deg:.5f}, {scene.emitter_lon_deg:.5f}', 'degrees'),
        ('Estimated emitter latitude/longitude',
         '—' if primary is None else f'{primary.lat_deg:.5f}, {primary.lon_deg:.5f}', 'degrees'),
        ('Emitter azimuth', '—' if scene.azimuth_deg is None else f'{scene.azimuth_deg:.3f}', 'degrees'),
        ('Emitter off-nadir angle', '—' if scene.off_nadir_deg is None else f'{scene.off_nadir_deg:.3f}', 'degrees'),
        ('Estimated azimuth/off-nadir',
         '—' if primary is None else f'{primary.doa.azimuth_deg:.3f} / {primary.doa.off_boresight_deg:.3f}', 'degrees'),
        ('Slant range', slant_value, slant_unit),
        ('Angular estimation error', '—' if angular is None else f'{angular:.4f}', 'degrees'),
        ('Ground geolocation error', ground_value, ground_unit),
        ('Number of DOA candidates', '—' if inverse is None else str(inverse.n_doa_candidates), 'count'),
        ('Number of valid Earth intercepts', '—' if inverse is None else str(inverse.n_valid_intercepts), 'count'),
    ]
    headline = st.columns(4)
    headline[0].metric('Ground error', f'{ground_value} {ground_unit}')
    headline[1].metric('Angular error', '—' if angular is None else f'{angular:.4f}°')
    headline[2].metric('Off-nadir', '—' if scene.off_nadir_deg is None else f'{scene.off_nadir_deg:.2f}°')
    headline[3].metric('Valid Earth intercepts', '—' if inverse is None else str(inverse.n_valid_intercepts))
    st.dataframe(pd.DataFrame(rows, columns=['Metric', 'Value', 'Units']), hide_index=True, use_container_width=True)
    arg_lat = 'n/a (equatorial)' if scene.frame.argument_of_latitude_deg is None else f'{scene.frame.argument_of_latitude_deg:.1f}°'
    geo_off = '—' if scene.geometric_off_nadir_deg is None else f'{scene.geometric_off_nadir_deg:.2f}°'
    st.caption(f'Ground-track heading {scene.frame.heading_deg:.1f}° from north toward east. '
               f'Argument of latitude {arg_lat}. Geocentric latitude {scene.frame.geocentric_lat_deg:.2f}°. '
               f'Angle from geocentric nadir {geo_off}. '
               f'Body off-nadir is measured from +Z; it equals the geocentric off-nadir angle only at zero attitude.')

    if scene.true_phases_rad is not None:
        phase_table = pd.DataFrame({
            'Pair': [f'A{i} → A{j}' for i, j in pairs(positions)],
            'True unwrapped (°)': np.rad2deg(scene.true_phases_rad),
            'Measured wrapped (°)': np.rad2deg(scene.measured_phases_rad),
        })
        st.dataframe(phase_table.style.format({'True unwrapped (°)': '{:.2f}', 'Measured wrapped (°)': '{:.2f}'}),
                     hide_index=True, use_container_width=True)

    if inverse is not None and inverse.solutions:
        candidate_rows = []
        for index, sol in enumerate(inverse.solutions, start=1):
            error_km = None
            if sol.status == 'ok' and scene.emitter_lat_deg is not None:
                error_km = geodesic_distance_m(scene.emitter_lat_deg, scene.emitter_lon_deg, sol.lat_deg, sol.lon_deg) / 1000.0
            candidate_rows.append({
                'Candidate': index,
                'Azimuth (°)': sol.doa.azimuth_deg,
                'Off-nadir (°)': sol.doa.off_boresight_deg,
                'Phase RMS (°)': sol.doa.phase_rms_deg,
                'Intercept': sol.status,
                'Latitude (°)': sol.lat_deg,
                'Longitude (°)': sol.lon_deg,
                'Geodesic error (km)': error_km,
                'Estimator choice': 'lowest-cost estimate' if sol is primary else '',
            })
        st.dataframe(pd.DataFrame(candidate_rows), hide_index=True, use_container_width=True)
        st.caption('Candidates are ranked by wrapped-phase residual. Geodesic error uses the simulated truth for display only. '
                   'The estimator never receives the true emitter coordinates. A long baseline can have more aliases than this list shows.')

    layer_labels = [
        ('earth', 'Earth', True), ('coastlines', 'Coastlines', True), ('graticule', 'Graticule', True),
        ('satellite', 'Satellite', True), ('subpoint', 'Subpoint', True), ('true_emitter', 'True emitter', True),
        ('estimated', 'Estimates', True), ('true_los', 'True line of sight', True),
        ('estimated_los', 'Estimated line of sight', True), ('horizon', 'Horizon', False),
        ('error_vector', 'Ground error', True),
    ]
    toggles = {}
    for row in (layer_labels[:6], layer_labels[6:]):
        toggle_cols = st.columns(len(row))
        for column, (key, label, default) in zip(toggle_cols, row):
            toggles[key] = column.checkbox(label, value=default, key=f'layer_{key}')
    layers = GlobeLayers(**toggles)
    estimates = []
    if inverse is not None:
        for sol in inverse.solutions:
            if sol.status == 'ok':
                estimates.append({'ecef': sol.ecef, 'primary': sol is primary})
    horizon = None
    if layers.horizon:
        with st.spinner('Tracing the Earth horizon'):
            horizon = horizon_ring(scene.frame, attitude, n=90)
    subpoint = geodetic_to_ecef(spec.latitude_deg, spec.longitude_deg, 0.0)
    primary_latlon = None if primary is None else (primary.lat_deg, primary.lon_deg)
    true_latlon = None if scene.emitter_lat_deg is None else (scene.emitter_lat_deg, scene.emitter_lon_deg)
    st.plotly_chart(earth_figure(
        scene.frame.sat_ecef, subpoint, scene.emitter_ecef, estimates, horizon,
        true_latlon=true_latlon, primary_latlon=primary_latlon, layers=layers,
    ), use_container_width=True)
    st.caption('Drag to rotate and scroll to zoom. Axis units are kilometers and the scale is equal, so the ellipsoid stays round. '
               'Orange is the satellite, green is the true emitter, red is the lowest-cost estimate, and purple marks other intercepts.')

    st.subheader('Off-nadir sensitivity')
    st.write('Each point is the ground shift caused by increasing the off-nadir angle a small amount, along one antenna azimuth. '
             'Near the limb the ray is almost tangent, so the same angle error walks a long distance across the surface and some perturbed rays miss the Earth.')
    sens_left, sens_right = st.columns(2)
    sens_az = sens_left.slider('Sensitivity azimuth from body +X (°)', 0.0, 360.0, 0.0, 5.0)
    delta = sens_right.number_input('Angular perturbation (°)', 0.01, 1.0, 0.1, 0.01, format='%.2f')
    compare_alts = st.checkbox('Compare 400, 800, and 1,200 km', value=True)
    altitudes = [spec.altitude_m]
    if compare_alts:
        for extra in (400_000.0, 800_000.0, 1_200_000.0):
            if abs(extra - spec.altitude_m) > 1_000:
                altitudes.append(extra)
    try:
        curves = off_nadir_sensitivity(
            spec.latitude_deg, spec.longitude_deg, spec.inclination_deg, spec.ascending,
            attitude, sens_az, altitudes, delta_deg=delta, step_deg=1.0)
    except OrbitFrameError as exc:
        st.warning(str(exc))
        curves = []
    if curves:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                            subplot_titles=(f'Ground shift for a {delta:.2f}° off-nadir increase', 'Slant range'))
        for curve in curves:
            name = f'{curve.altitude_km:.0f} km'
            custom = np.column_stack((curve.elevation_deg, curve.missed.astype(float)))
            fig.add_trace(go.Scatter(
                x=curve.off_nadir_deg, y=curve.ground_error_m / 1000.0, mode='lines', name=name,
                customdata=custom,
                hovertemplate='off-nadir %{x:.1f}°<br>error %{y:.2f} km<br>elevation %{customdata[0]:.1f}°<extra>' + name + '</extra>',
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=curve.off_nadir_deg, y=curve.slant_range_km, mode='lines', name=name,
                showlegend=False, hovertemplate='off-nadir %{x:.1f}°<br>slant %{y:.0f} km<extra>' + name + '</extra>',
            ), row=2, col=1)
        fig.update_xaxes(title_text='Off-nadir angle (°)', row=2, col=1)
        fig.update_yaxes(title_text='Ground error (km)', row=1, col=1)
        fig.update_yaxes(title_text='Slant range (km)', row=2, col=1)
        fig.update_layout(
            height=680, margin=dict(t=90, b=48, l=72, r=24),
            legend=dict(orientation='h', yanchor='bottom', y=1.16, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption('Gaps are perturbations whose ray misses the Earth. '
                   'Ground error grows roughly as slant range × angle error / sin(elevation). Elevation goes to zero at the limb, so the error grows sharply. '
                   'A higher orbit sees a smaller Earth disk (smaller horizon angle) and a longer slant range.')

    st.subheader('Monte Carlo geolocation error')
    st.write('Channel calibration biases are drawn once from the seed and then held fixed. '
             'Phase noise, attitude-knowledge error, and position-knowledge error are new on every trial. '
             'Statistics labeled unambiguous exclude trials that still have more than one plausible Earth intercept, '
             'so an ambiguity-selection failure is not reported as ordinary noise.')
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    n_trials = int(mc1.number_input('Trials', 20, 2000, 200, 20))
    mc_noise = mc2.number_input('Phase noise (° RMS)', 0.0, 30.0, float(noise_deg), 0.5)
    mc_cal = mc3.number_input('Fixed channel bias σ (°)', 0.0, 30.0, 0.0, 0.5)
    mc_att = mc4.number_input('Attitude knowledge σ (°)', 0.0, 5.0, 0.0, 0.05, format='%.2f')
    mc_pos = mc5.number_input('Position knowledge σ (m)', 0.0, 50_000.0, 0.0, 50.0)
    st.caption('Position error is applied independently on VVLH +X (along-track), +Y (cross-track), and +Z (nadir). '
               'It moves the assumed satellite origin and does not rebuild the orbital frame. '
               'Attitude error is added to the estimator’s roll, pitch, and yaw.')
    mc_signature = (
        round(alt_km, 3), round(inclination, 3), round(sub_lat, 3), round(sub_lon, 3), bool(ascending),
        round(roll, 3), round(pitch, 3), round(yaw, 3), layout, round(float(frequency_hz), 1),
        tuple(np.round(positions.ravel(), 5)), n_trials, round(mc_noise, 3), round(mc_cal, 3),
        round(mc_att, 3), round(mc_pos, 3), int(seed), float(step_deg),
        None if scene.emitter_lat_deg is None else round(scene.emitter_lat_deg, 5),
        None if scene.emitter_lon_deg is None else round(scene.emitter_lon_deg, 5),
    )
    if not scene.visible:
        st.warning('Monte Carlo needs an emitter inside the Earth horizon and the antenna front hemisphere.')
    elif st.button('Run Monte Carlo', type='primary'):
        with st.spinner('Running geolocation trials'):
            st.session_state['mc_result'] = monte_carlo_geolocation(
                spec, attitude, scene.emitter_lat_deg, scene.emitter_lon_deg, positions, frequency_hz,
                n_trials=n_trials, phase_noise_deg=mc_noise, channel_bias_sigma_deg=mc_cal,
                attitude_sigma_deg=mc_att, position_sigma_m=mc_pos, seed=seed, step_deg=step_deg,
            )
            st.session_state['mc_truth'] = (scene.emitter_lat_deg, scene.emitter_lon_deg, sub_lat, sub_lon)
            st.session_state['mc_signature'] = mc_signature
    result = st.session_state.get('mc_result')
    if result is not None and scene.visible:
        if st.session_state.get('mc_signature') != mc_signature:
            st.warning('Inputs changed after this run. The plots below are from the previous Monte Carlo.')
        truth = st.session_state.get('mc_truth', (scene.emitter_lat_deg, scene.emitter_lon_deg, sub_lat, sub_lon))
        bias_text = ', '.join(f'A{i}={value:.2f}°' for i, value in enumerate(result.fixed_channel_bias_deg))
        st.caption(f'Fixed channel biases for this run: {bias_text}. Pair biases are differences from A0.')
        s1, s2, s3, s4 = st.columns(4)
        s1.metric('Successful trials', f'{result.pct_success:.0f}%')
        s2.metric('Unresolved ambiguity', f'{result.pct_ambiguous:.0f}%')
        uniq_med, uniq_unit = _distance_text(result.median_unambiguous_m)
        uniq_p95, uniq_p95_unit = _distance_text(result.p95_unambiguous_m)
        s3.metric('Median error, unambiguous', f'{uniq_med} {uniq_unit}')
        s4.metric('95th percentile, unambiguous', f'{uniq_p95} {uniq_p95_unit}')
        rms_u, rms_u_unit = _distance_text(result.rms_unambiguous_m)
        med_p, med_p_unit = _distance_text(result.median_primary_m)
        p95_p, p95_p_unit = _distance_text(result.p95_primary_m)
        rms_p, rms_p_unit = _distance_text(result.rms_primary_m)
        st.caption(f'Unambiguous trials: {result.n_unambiguous} of {result.n_trials}. '
                   f'RMS unambiguous {rms_u} {rms_u_unit}. '
                   f'Lowest-cost candidate including ambiguous trials: median {med_p} {med_p_unit}, '
                   f'95th {p95_p} {p95_p_unit}, RMS {rms_p} {rms_p_unit}.')
        if result.n_unambiguous == 0:
            st.warning('Every successful trial was ambiguous, so there is no noise-only error distribution. '
                       'The lowest-cost numbers above include ambiguity selection.')
        ok = result.success
        map_fig = go.Figure()
        if np.any(ok):
            map_fig.add_trace(go.Scattergeo(
                lon=result.lon_deg[ok], lat=result.lat_deg[ok], mode='markers', name='Trials',
                marker=dict(
                    size=7, color=result.errors_m[ok] / 1000.0, colorscale='Viridis',
                    colorbar=dict(title='Error (km)'),
                    symbol=['diamond' if flag else 'circle' for flag in result.ambiguous[ok]],
                ),
                hovertemplate='lat %{lat:.3f}<br>lon %{lon:.3f}<extra></extra>',
            ))
        map_fig.add_trace(go.Scattergeo(
            lon=[truth[1]], lat=[truth[0]], mode='markers', name='Truth',
            marker=dict(size=12, color='#16a34a', symbol='star'),
        ))
        map_fig.update_geos(
            projection_type='orthographic', projection_rotation=dict(lat=truth[2], lon=truth[3]),
            showcoastlines=True, showland=True, landcolor='#e5e7eb', showocean=True, oceancolor='#dbeafe',
        )
        map_fig.update_layout(height=460, margin=dict(l=0, r=0, t=30, b=0), title='Estimated ground locations')
        hist = go.Figure()
        uniq = ok & ~result.ambiguous
        amb = ok & result.ambiguous
        if np.any(uniq):
            hist.add_trace(go.Histogram(x=result.errors_m[uniq] / 1000.0, name='Unambiguous', nbinsx=30, opacity=0.75))
        if np.any(amb):
            hist.add_trace(go.Histogram(x=result.errors_m[amb] / 1000.0, name='Ambiguous lowest-cost', nbinsx=30, opacity=0.55))
        hist.update_layout(barmode='overlay', height=460, margin=dict(l=0, r=0, t=30, b=0),
                           title='Ground error', xaxis_title='Error (km)', yaxis_title='Trials')
        left, right = st.columns(2)
        left.plotly_chart(map_fig, use_container_width=True)
        right.plotly_chart(hist, use_container_width=True)
        st.caption('Circles are unambiguous fixes. Diamonds are lowest-cost picks from trials that remained ambiguous.')


st.set_page_config(page_title='Satellite RF Interferometer Lab', layout='wide', page_icon='📡')
st.title('📡 Satellite RF Interferometer Lab')
st.caption('Versions 1–3 · Narrowband interferometry, 2D direction of arrival, and LEO emitter geolocation')
st.info('Array frame: antennas lie in the XY plane and +Z is boresight. Azimuth is counterclockwise from +X. '
        'On the Satellite Geolocation tab that boresight is the satellite +Z nadir axis unless roll, pitch, or yaw is applied.')

with st.sidebar:
    st.header('RF and array')
    f_ghz = st.number_input('Frequency (GHz)', min_value=0.1, max_value=40.0, value=2.2, step=0.1)
    lam = wavelength(f_ghz*1e9)
    layout = st.selectbox('Layout', ['3-L', '2-element', '3-linear', '3-triangle', '4-dual-axis', '5-cross'])
    short_m = st.number_input('Short baseline (m)', min_value=0.005, max_value=20.0, value=0.07, step=0.01, format='%.3f')
    long_m = st.number_input('Long baseline / array span (m)', min_value=0.005, max_value=20.0, value=0.8, step=0.05, format='%.3f')
    st.header('Emitter: true direction')
    az = st.slider('Azimuth (°)', 0, 360, 35)
    off = st.slider('Off-boresight (°)', 0, 90, 40)
    st.header('Receiver')
    noise = st.slider('Per-pair phase noise (° RMS)', 0.0, 30.0, 1.0, 0.5)
    bias = st.slider('Common phase offset on reference pairs (°)', -30.0, 30.0, 0.0, 0.5)
    seed = st.number_input('Random seed (reproducible)', 0, 999999, 42)
    resolution = st.select_slider('Direction-map resolution', options=[4.0, 2.0, 1.0, 0.5], value=2.0,
                                  help='Coarse grid can miss narrow minima. 0.5° may be slower.')

pos = array_positions(layout, short_m, long_m)
ph_true = phase_difference(pos, f_ghz*1e9, az, off)
ph_meas = measured_phases(ph_true, noise, np.full(len(ph_true), bias), np.random.default_rng(seed))
phase_df = pd.DataFrame({
    'Pair': [f'A{i} → A{j}' for i,j in pairs(pos)],
    'Length (m)': np.linalg.norm(pos[1:] - pos[0], axis=1),
    'Length (λ)': np.linalg.norm(pos[1:] - pos[0], axis=1)/lam,
    'True unwrapped (°)': np.rad2deg(ph_true),
    'True wrapped (°)': np.rad2deg(wrap_rad(ph_true)),
    'Measured wrapped (°)': np.rad2deg(ph_meas),
})

m1, m2, m3 = st.columns(3)
m1.metric('Wavelength', f'{lam*100:.2f} cm')
m2.metric('Antennas / independent reference pairs', f'{len(pos)} / {len(pos)-1}')
m3.metric('Longest reference baseline', f'{max(phase_df["Length (λ)"]):.2f} λ')

t1, t2, t3, t4, t_geo = st.tabs([
    '1 · Phase fundamentals', '2 · Array geometry', '3 · Inverse DOA / ambiguities',
    '4 · Noise experiments', 'Satellite Geolocation'])

with t1:
    st.subheader('Forward model: transmitter direction → measured phase')
    st.latex(r'\Delta\phi_{0j}=\frac{2\pi}{\lambda}(\mathbf r_j-\mathbf r_0)\cdot\hat{\mathbf u},\qquad \phi_{meas}=\mathrm{wrap}_{[-\pi,\pi)}(\Delta\phi+\epsilon+b)')
    st.dataframe(phase_df.style.format({'Length (m)': '{:.3f}', 'Length (λ)': '{:.2f}',
        'True unwrapped (°)': '{:.1f}', 'True wrapped (°)': '{:.1f}', 'Measured wrapped (°)': '{:.1f}'}), hide_index=True, use_container_width=True)
    fig = go.Figure()
    xx = np.linspace(-90,90,721)
    for j in range(1,len(pos)):
        # Fixed y=0 source component; this is a 1D principal plane cut, not a full 2D view.
        b = pos[j]-pos[0]
        p = (2*np.pi/lam)*b[0]*np.sin(np.deg2rad(xx))
        fig.add_trace(go.Scatter(x=xx,y=np.rad2deg(wrap_rad(p)),mode='lines', name=f'A0–A{j}'))
    fig.update_layout(title='Phase wrapping vs incident angle (XZ plane, azimuth 0°)',xaxis_title='Angle from +Z toward +X (°)', yaxis_title='Wrapped phase (°)',height=410)
    st.plotly_chart(fig,use_container_width=True)
    st.caption('A longer baseline has more phase cycles: steeper local sensitivity, but more ambiguous possible angles. ' 
               'The 1D sweep is a cut through the 2D direction field; its azimuth is fixed at 0°.')

with t2:
    st.subheader('Array layout and incoming direction')
    left,right=st.columns([1.15,1])
    with left:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=pos[:,0],y=pos[:,1],mode='markers+text',
                                 marker=dict(size=16,color='#2563eb'), text=[f'A{i}' for i in range(len(pos))],
                                 textposition='top center',name='Elements'))
        for j in range(1,len(pos)):
            fig.add_trace(go.Scatter(x=[pos[0,0],pos[j,0]],y=[pos[0,1],pos[j,1]],
                                     mode='lines',line=dict(dash='dot'),showlegend=False))
        u=direction(az,off)
        scale=max(np.ptp(pos[:,0]),np.ptp(pos[:,1]),.1)*0.65
        fig.add_annotation(x=scale*u[0],y=scale*u[1],ax=0,ay=0,axref='x',ayref='y',
                           showarrow=True,arrowhead=3,arrowwidth=3,arrowcolor='orange',text='Projected arrival')
        fig.update_layout(title='Antenna XY deck (meters)',xaxis_title='X (m)',yaxis_title='Y (m)',height=445)
        fig.update_yaxes(scaleanchor='x',scaleratio=1)
        st.plotly_chart(fig,use_container_width=True)
    with right:
        fig=go.Figure()
        fig.add_trace(go.Scatter3d(x=pos[:,0],y=pos[:,1],z=pos[:,2],mode='markers+text',
                                   marker=dict(size=5,color='#2563eb'),text=[f'A{i}' for i in range(len(pos))],name='Antennas'))
        scale=max(np.ptp(pos[:,0]),np.ptp(pos[:,1]),.2)
        fig.add_trace(go.Scatter3d(x=[0,scale*u[0]],y=[0,scale*u[1]],z=[0,scale*u[2]],
                                   mode='lines+markers',line=dict(color='orange',width=8),name='DOA unit direction'))
        fig.update_layout(title='3D arrival direction (relative to array)',scene=dict(
            xaxis_title='X (m)',yaxis_title='Y (m)',zaxis_title='Z (m)',aspectmode='data'),height=445)
        st.plotly_chart(fig,use_container_width=True)
    st.warning('An XY-plane array cannot distinguish arrivals from +Z versus −Z without a front-hemisphere assumption. '
               'Collinear elements cannot independently estimate both direction cosines. '
               'An L or triangle gives two independent geometric directions, not necessarily a unique unwrapped solution.')

with t3:
    st.subheader('Inverse model: measured phases → candidate directions')
    st.write('Search the front hemisphere for directions whose predicted **wrapped** reference-pair phases match the observations. '
             'The heatmap shows the phase-mismatch cost; multiple dark minima mean ambiguous directions.')
    xx,yy,valid,score=ambiguity_surface(pos,f_ghz*1e9,ph_meas,resolution)
    candidates=candidate_solutions(xx,yy,valid,score,max_candidates=12,
                                   separation=max(.035,2*np.sin(np.deg2rad(resolution))))
    fig=go.Figure(go.Heatmap(x=xx[0],y=yy[:,0],z=score,colorscale='Viridis_r',
        colorbar=dict(title=dict(text='Phase mismatch', side='right'), thickness=14, len=0.75, y=0.46,
                       ticklabelposition='inside', xpad=6),
        hovertemplate='uₓ=%{x:.3f}<br>uᵧ=%{y:.3f}<br>cost=%{z:.4f}<extra></extra>'))
    ring=np.linspace(0,2*np.pi,400)
    fig.add_trace(go.Scatter(x=np.cos(ring),y=np.sin(ring),mode='lines',line=dict(color='white',width=2),showlegend=False))
    true_u=direction(az,off)
    fig.add_trace(go.Scatter(x=[true_u[0]],y=[true_u[1]],mode='markers',marker=dict(size=14,color='red',symbol='x'),name='True DOA'))
    if candidates:
        fig.add_trace(go.Scatter(x=[c.ux for c in candidates],y=[c.uy for c in candidates],mode='markers+text',
            marker=dict(size=9,color='white',line=dict(color='black',width=1)),
            text=[str(i+1) for i in range(len(candidates))],textposition='top center',name='Grid minima'))
    fig.update_layout(title='Direction-cosine ambiguity map (uₓ, uᵧ)',xaxis_title='uₓ = sin θ cos az',
        yaxis_title='uᵧ = sin θ sin az',height=570,
        margin=dict(t=78, r=108, l=64, b=48),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0, xanchor='left'))
    fig.update_yaxes(scaleanchor='x',scaleratio=1)
    st.plotly_chart(fig,use_container_width=True)
    if candidates:
        st.dataframe(pd.DataFrame([{'Candidate':i+1,'Azimuth (°)':round(c.azimuth_deg,2),
            'Off-boresight (°)':round(c.off_boresight_deg,2),'RMS phase mismatch (°)':round(c.phase_rms_deg,2)}
            for i,c in enumerate(candidates)]),hide_index=True,use_container_width=True)
    st.caption('Candidates are local minima on a finite grid, not a guarantee of all mathematically equivalent solutions. '
               'The score is a circular phase-chord mismatch; listed phase RMS values are approximate scores, '
               'not formal statistical confidence intervals. Noise and array geometry may shift the lowest minimum '
               'away from the true location. At boresight, azimuth is undefined.')

with t4:
    st.subheader('Repeated phase-noise experiment')
    st.write('Hold the true arrival fixed, draw independent Gaussian phase noise on each reference pair, '
             'and observe how the measured wrapped phases vary.')
    n=st.slider('Number of trials',100,5000,500,100)
    rng=np.random.default_rng(seed)
    trials=measured_phases(np.broadcast_to(ph_true,(n,len(ph_true))),noise,
                            np.full(len(ph_true),bias),rng)
    fig=go.Figure()
    for j in range(trials.shape[1]):
        fig.add_trace(go.Histogram(x=np.rad2deg(trials[:,j]),name=f'A0–A{j+1}',opacity=.65,nbinsx=55))
    fig.update_layout(barmode='overlay',xaxis_title='Wrapped measured phase (°)',yaxis_title='Count',height=420)
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Noise is applied per reference-pair measurement for teaching purposes; a realistic shared-reference '
               'receiver has correlated pair errors. No SNR-to-phase-estimator model is implied.')

with t_geo:
    render_satellite_tab(pos, f_ghz * 1e9, noise, bias, int(seed), float(resolution), layout, lam, long_m)

st.divider()
st.caption('Far-field, narrowband, simultaneous phase observations and ideal isotropic elements. '
           'No multipath, polarization, mutual coupling, or orbit propagation. '
           'Version 3 uses a spherical circular-orbit VVLH frame (+Z geocentric nadir) and a WGS84 ray intersection. '
           'Wrapped phase is what a receiver would observe. Frame definitions and equations are in the README.')
