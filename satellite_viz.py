"""Plotly globe for the LEO geolocation scene.

Coastlines are Natural Earth 1:110m, public domain, stored in ``data/coastlines_110m.npz``.
ECEF coordinates are drawn in kilometers with ``aspectmode='data'`` so the
ellipsoid is not stretched.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from geolocation import WGS84_A, WGS84_B, geodesic_arc, geodesic_distance_m, geodetic_to_ecef

_COAST_PATH = Path(__file__).resolve().parent / 'data' / 'coastlines_110m.npz'
_COAST_KM = None


@dataclass
class GlobeLayers:
    earth: bool = True
    coastlines: bool = True
    graticule: bool = True
    satellite: bool = True
    subpoint: bool = True
    true_emitter: bool = True
    estimated: bool = True
    true_los: bool = True
    estimated_los: bool = True
    horizon: bool = False
    error_vector: bool = True


def _km(point) -> np.ndarray:
    return np.asarray(point, dtype=float) / 1000.0


def _coastline_km():
    global _COAST_KM
    if _COAST_KM is not None:
        return _COAST_KM
    if not _COAST_PATH.is_file():
        _COAST_KM = []
        return _COAST_KM
    stored = np.load(_COAST_PATH)
    lat = np.asarray(stored['lat'], dtype=float)
    lon = np.asarray(stored['lon'], dtype=float)
    finite = np.isfinite(lat) & np.isfinite(lon)
    ecef = np.full((lat.size, 3), np.nan)
    if np.any(finite):
        ecef[finite] = geodetic_to_ecef(lat[finite], lon[finite], 20_000.0)
    breaks = np.where(~finite)[0]
    segments = []
    start = 0
    for stop in list(breaks) + [lat.size]:
        chunk = ecef[start:stop]
        if len(chunk) >= 2:
            segments.append(chunk / 1000.0)
        start = stop + 1
    _COAST_KM = segments
    return _COAST_KM


def _graticule_km():
    lines = []
    for lat in range(-60, 90, 30):
        lon = np.linspace(-180.0, 180.0, 181)
        lines.append(geodetic_to_ecef(np.full_like(lon, lat), lon, 20_000.0) / 1000.0)
    for lon in range(-150, 180, 30):
        lat = np.linspace(-90.0, 90.0, 91)
        lines.append(geodetic_to_ecef(lat, np.full_like(lat, lon), 20_000.0) / 1000.0)
    return lines


def _add_polyline(fig, points_km, name, color, width=4, dash=None, showlegend=True):
    pts = np.asarray(points_km, dtype=float)
    if len(pts) < 2:
        return
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='lines',
        line=dict(color=color, width=width, dash=dash),
        name=name,
        showlegend=showlegend,
        hoverinfo='skip',
    ))


def earth_figure(sat_ecef, subpoint_ecef, true_ecef=None, estimates=None, horizon_ecef=None,
                 true_latlon=None, primary_latlon=None, layers: GlobeLayers | None = None) -> go.Figure:
    """Interactive 3D Earth. ``estimates`` is a list of ``{'ecef', 'primary'}``."""
    layers = layers or GlobeLayers()
    estimates = estimates or []
    fig = go.Figure()
    if layers.earth:
        lat = np.linspace(-np.pi / 2, np.pi / 2, 46)
        lon = np.linspace(-np.pi, np.pi, 90)
        x = (WGS84_A / 1000.0) * np.cos(lat)[:, None] * np.cos(lon)[None, :]
        y = (WGS84_A / 1000.0) * np.cos(lat)[:, None] * np.sin(lon)[None, :]
        z = (WGS84_B / 1000.0) * np.sin(lat)[:, None] * np.ones_like(lon)[None, :]
        fig.add_trace(go.Surface(
            x=x, y=y, z=z, colorscale=[[0, '#2f6fed'], [1, '#1d4ed8']],
            showscale=False, opacity=1.0, name='Earth', hoverinfo='skip',
            lighting=dict(ambient=0.72, diffuse=0.55, specular=0.12),
        ))
    if layers.graticule:
        first = True
        for line in _graticule_km():
            _add_polyline(fig, line, 'Graticule', '#dbeafe', width=2, showlegend=first)
            first = False
    if layers.coastlines:
        first = True
        for segment in _coastline_km():
            _add_polyline(fig, segment, 'Coastline', '#f8fafc', width=3, showlegend=first)
            first = False
    sat = _km(sat_ecef)
    sub = _km(subpoint_ecef)
    if layers.satellite:
        _add_polyline(fig, np.vstack((sub, sat)), 'Nadir stem', '#94a3b8', width=3, showlegend=False)
        fig.add_trace(go.Scatter3d(
            x=[sat[0]], y=[sat[1]], z=[sat[2]], mode='markers+text',
            marker=dict(size=6, color='#f97316'),
            text=['Satellite'], textposition='top center', name='Satellite',
        ))
    if layers.subpoint:
        fig.add_trace(go.Scatter3d(
            x=[sub[0]], y=[sub[1]], z=[sub[2]], mode='markers+text',
            marker=dict(size=4, color='#0284c7', symbol='diamond'),
            text=['Subpoint'], textposition='bottom center', name='Geodetic subpoint',
        ))
    if layers.true_emitter and true_ecef is not None:
        true = _km(true_ecef)
        fig.add_trace(go.Scatter3d(
            x=[true[0]], y=[true[1]], z=[true[2]], mode='markers+text',
            marker=dict(size=6, color='#16a34a'),
            text=['True emitter'], textposition='middle left', name='True emitter',
        ))
        if layers.true_los:
            _add_polyline(fig, np.vstack((sat, true)), 'True line of sight', '#16a34a', width=7)
    if layers.estimated:
        for index, item in enumerate(estimates, start=1):
            point = _km(item['ecef'])
            primary = bool(item.get('primary'))
            label = 'Estimated' if primary else f'Alt {index}'
            fig.add_trace(go.Scatter3d(
                x=[point[0]], y=[point[1]], z=[point[2]],
                mode='markers+text' if primary else 'markers',
                marker=dict(size=7 if primary else 4, color='#dc2626' if primary else '#7c3aed', symbol='x'),
                text=[label] if primary else None,
                textposition='middle right',
                name='Estimated emitter' if primary else f'Alternate {index}',
                hovertemplate=f'{label}<extra></extra>',
            ))
            if layers.estimated_los:
                _add_polyline(
                    fig, np.vstack((sat, point)),
                    'Estimated line of sight' if primary else f'Alternate line of sight {index}',
                    '#dc2626' if primary else '#7c3aed',
                    width=6 if primary else 3,
                    dash=None if primary else 'dash',
                )
    if layers.horizon and horizon_ecef is not None and len(horizon_ecef) >= 8:
        ring = np.vstack((horizon_ecef, horizon_ecef[:1])) / 1000.0
        _add_polyline(fig, ring, 'Earth horizon', '#0f766e', width=4)
    if (layers.error_vector and true_latlon is not None and primary_latlon is not None
            and true_ecef is not None and estimates
            and geodesic_distance_m(true_latlon[0], true_latlon[1], primary_latlon[0], primary_latlon[1]) > 100.0):
        arc = geodesic_arc(true_latlon[0], true_latlon[1], primary_latlon[0], primary_latlon[1], n=24, h_m=40_000.0)
        _add_polyline(fig, arc / 1000.0, 'Ground error', '#ca8a04', width=6)
    radial = np.asarray(sat_ecef, dtype=float)
    radial = radial / np.linalg.norm(radial)
    eye = radial * 2.15
    fig.update_layout(
        title='Satellite geolocation scene',
        height=740,
        margin=dict(l=0, r=0, t=48, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        uirevision='globe',
        scene=dict(
            aspectmode='data',
            bgcolor='#f8fafc',
            camera=dict(eye=dict(x=float(eye[0]), y=float(eye[1]), z=float(eye[2])), up=dict(x=0, y=0, z=1)),
            xaxis=dict(title='ECEF X (km)', showbackground=False),
            yaxis=dict(title='ECEF Y (km)', showbackground=False),
            zaxis=dict(title='ECEF Z (km)', showbackground=False),
        ),
    )
    return fig
