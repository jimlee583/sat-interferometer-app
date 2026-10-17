"""Narrowband far-field interferometry in an antenna-plane coordinate frame.

Antennas lie in XY; +Z is boresight. Azimuth is measured counterclockwise
from +X, and theta is off-boresight (0..90 degrees in the front hemisphere).
The complex phase convention is exp(+j*2*pi/lambda * position dot arrival_direction).
All pair measurements use antenna j minus antenna i, with i=0 as reference.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import minimum_filter

C = 299_792_458.0


def wavelength(frequency_hz: float) -> float:
    if frequency_hz <= 0:
        raise ValueError('Frequency must be positive')
    return C / frequency_hz


def wrap_rad(phases):
    """Wrap phases to [-pi, pi)."""
    return (np.asarray(phases) + np.pi) % (2 * np.pi) - np.pi


def direction(az_deg, off_boresight_deg):
    a, t = np.deg2rad(az_deg), np.deg2rad(off_boresight_deg)
    return np.stack((np.sin(t)*np.cos(a), np.sin(t)*np.sin(a), np.cos(t)), axis=-1)


def array_positions(layout: str, short_m: float, long_m: float) -> np.ndarray:
    """Antenna XYZ coordinates, in meters; A0 is the phase reference.

    Three-element L/triangle layouts have two independent baselines only; they
    do not by themselves guarantee unambiguous 2D direction finding.
    """
    if short_m <= 0 or long_m <= 0:
        raise ValueError('Baselines must be positive')
    variants = {
        '2-element': [[0, 0], [long_m, 0]],
        '3-linear': [[0, 0], [short_m, 0], [long_m, 0]],
        '3-L': [[0, 0], [long_m, 0], [0, long_m]],
        '3-triangle': [[0, 0], [long_m, 0], [0.5*long_m, np.sqrt(3)/2*long_m]],
        '4-dual-axis': [[0, 0], [short_m, 0], [0, short_m], [long_m, long_m]],
        '5-cross': [[0, 0], [short_m, 0], [0, short_m], [long_m, 0], [0, long_m]],
    }
    return np.column_stack([np.asarray(variants[layout], dtype=float), np.zeros(len(variants[layout]))])


def pairs(positions):
    return [(0, j) for j in range(1, len(positions))]


def phase_difference(positions, frequency_hz, az_deg, off_boresight_deg):
    """Unwrapped reference-pair phases (radians) for a far-field source."""
    baselines = np.asarray(positions)[1:] - np.asarray(positions)[0]
    return (2*np.pi/wavelength(frequency_hz))*np.einsum('ij,...j->...i', baselines, direction(az_deg, off_boresight_deg))


def measured_phases(true_phases, noise_std_deg=0.0, biases_deg=None, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    true_phases = np.asarray(true_phases)
    if biases_deg is None:
        biases_deg = np.zeros(true_phases.shape[-1])
    return wrap_rad(true_phases + np.deg2rad(biases_deg) + rng.normal(0, np.deg2rad(noise_std_deg), true_phases.shape))


def angular_grid(step_deg=2.0):
    """Regular projected-direction-cosine grid, physically valid front hemisphere."""
    axis = np.linspace(-1, 1, int(round(2 / np.sin(np.deg2rad(step_deg)))) + 1)
    xx, yy = np.meshgrid(axis, axis)
    valid = xx**2 + yy**2 <= 1
    return xx, yy, valid


def ambiguity_surface(positions, frequency_hz, measured_rad, step_deg=2.0):
    """RMS circular phase mismatch in degrees across possible directions.

    Plotted axes are u_x, u_y (direction cosines). The outer disk is the
    front hemisphere. The source-facing +Z sign is explicitly assumed.
    """
    xx, yy, valid = angular_grid(step_deg)
    baselines = (np.asarray(positions)[1:] - np.asarray(positions)[0])[:, :2]
    k = 2*np.pi/wavelength(frequency_hz)
    predicted = k*(baselines[:, 0, None, None]*xx + baselines[:, 1, None, None]*yy)
    residual = wrap_rad(predicted - np.asarray(measured_rad)[:, None, None])
    # Circular chord metric behaves well at wrap boundaries.
    score = np.sqrt(np.mean((2*np.sin(residual/2))**2, axis=0))
    return xx, yy, valid, np.where(valid, score, np.nan)


@dataclass
class Candidate:
    ux: float
    uy: float
    azimuth_deg: float
    off_boresight_deg: float
    phase_rms_deg: float


def candidate_solutions(xx, yy, valid, score, max_candidates=12, separation=0.035):
    """Extract distinct local minima, with no promise of exhaustive resolution."""
    working = np.where(valid & np.isfinite(score), score, np.inf)
    minima = (working == minimum_filter(working, size=3, mode='constant', cval=np.inf)) & np.isfinite(working)
    indices = np.argwhere(minima)
    indices = sorted(indices, key=lambda ij: working[tuple(ij)])
    result = []
    for iy, ix in indices:
        x, y = float(xx[iy, ix]), float(yy[iy, ix])
        if any(np.hypot(x-s.ux, y-s.uy)<separation for s in result):
            continue
        result.append(Candidate(x, y, float(np.rad2deg(np.arctan2(y,x))%360),
                       float(np.rad2deg(np.arcsin(min(1, np.hypot(x,y))))),
                       float(np.rad2deg(2*np.arcsin(np.clip(working[iy, ix]/2, 0, 1))))))
        if len(result)>=max_candidates:
            break
    return result


def candidate_phase_residuals(positions, frequency_hz, measured_rad, candidate):
    return np.rad2deg(wrap_rad(phase_difference(positions, frequency_hz,
                                               candidate.azimuth_deg, candidate.off_boresight_deg) - measured_rad))
