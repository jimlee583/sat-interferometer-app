"""Streamlit satellite RF interferometer: versions 1 and 2 (no Earth geolocation)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from interferometer import (wavelength, array_positions, pairs, phase_difference,
                            measured_phases, wrap_rad, ambiguity_surface,
                            candidate_solutions, direction)

st.set_page_config(page_title='Satellite RF Interferometer Lab', layout='wide', page_icon='📡')
st.title('📡 Satellite RF Interferometer Lab')
st.caption('Versions 1–2 · Narrowband far-field phase interferometry · Array geometry, 2D DOA, and ambiguities')
st.info('Coordinate convention: antennas lie in the XY plane; +Z points toward the observed hemisphere. '
        'Azimuth is counterclockwise from +X; off-boresight is measured from +Z. '
        'No satellite orbit or Earth intercept is modeled yet.')

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

t1,t2,t3,t4 = st.tabs(['1 · Phase fundamentals', '2 · Array geometry', '3 · Inverse DOA / ambiguities', '4 · Noise experiments'])

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
                               colorbar=dict(title='Phase mismatch'),hovertemplate='uₓ=%{x:.3f}<br>uᵧ=%{y:.3f}<br>cost=%{z:.4f}<extra></extra>'))
    ring=np.linspace(0,2*np.pi,400)
    fig.add_trace(go.Scatter(x=np.cos(ring),y=np.sin(ring),mode='lines',line=dict(color='white',width=2),showlegend=False))
    true_u=direction(az,off)
    fig.add_trace(go.Scatter(x=[true_u[0]],y=[true_u[1]],mode='markers',marker=dict(size=14,color='red',symbol='x'),name='True DOA'))
    if candidates:
        fig.add_trace(go.Scatter(x=[c.ux for c in candidates],y=[c.uy for c in candidates],mode='markers+text',
            marker=dict(size=9,color='white',line=dict(color='black',width=1)),
            text=[str(i+1) for i in range(len(candidates))],textposition='top center',name='Grid minima'))
    fig.update_layout(title='Direction-cosine ambiguity map (uₓ, uᵧ)',xaxis_title='uₓ = sin θ cos az',
        yaxis_title='uᵧ = sin θ sin az',height=570)
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

st.divider()
st.caption('Scope: far-field, narrowband, simultaneous phase observations; ideal isotropic elements; '
           'no multipath, polarization, coupling, orbital motion, satellite attitude, or Earth geolocation. '
           'True phase is simulated; actual hardware only directly observes wrapped phase.')
