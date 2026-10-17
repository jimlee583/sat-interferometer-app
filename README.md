# Satellite RF Interferometer Lab — Versions 1 & 2

A runnable Python Streamlit app for learning narrowband, far-field RF interferometry, antenna-baseline phase wrapping, two-dimensional direction-of-arrival (DOA) estimation, and angle ambiguities. **No satellite orbit/Earth geolocation yet** (that is a planned Version 3).

## Quick start (Mac / Cursor)

Open the cloned `sat-interferometer-app` repository in Cursor. Copy the project files from the supplied ZIP into that folder, preserving the `tests` directory. From the integrated terminal:

```bash
uv sync
uv run streamlit run app.py
```

Alternatively: `python -m pip install -r requirements.txt && streamlit run app.py`.

Run tests:

```bash
uv run pytest -q
```

## Main features

- Frequency adjustable from 0.1 to 40 GHz; displayed wavelength and baseline in wavelengths.
- Two-, three-, four-, or five-element predefined layouts (including linear, L, and triangular).
- Adjustable emitter azimuth and off-boresight angle.
- Forward model computing true unwrapped, wrapped, and noisy wrapped antenna-reference phases.
- Angle sweep showing phase wrapping, including a 1D principal-plane cut.
- Interactive 2D element layout and 3D arrival-direction plot.
- Inverse 2D direction-cosine grid search over the **front** hemisphere with candidate local minima.
- Repeated noise draws and histograms (fixed random seed for reproducibility).

## Geometry and assumptions

The antennas are in the XY plane; +Z is boresight. Azimuth is counterclockwise from +X. Off-boresight `theta` is in `[0°,90°]`. Arrival unit vector:

`u = (sin(theta) cos(azimuth), sin(theta) sin(azimuth), cos(theta))`.

For a phase-reference pair from antenna 0 to antenna j:

`phase_0j = (2*pi/lambda) * (r_j - r_0) dot u`.

The receiver observes the phase wrapped to `[-pi,pi)`; synthetic independent Gaussian per-reference-pair errors and a common reference-pair phase bias can be applied.

A planar array **cannot distinguish positive vs. negative Z** without hemisphere prior information. A linear array only constrains one angular direction. Three L-shaped elements provide 2D geometric constraints but do **not** guarantee an unambiguous direction when spacings are large. The inverse search displays multiple candidate directions on a finite grid rather than claiming unique solutions. Short baselines (about half a wavelength or smaller) help disambiguate, while longer baselines can increase angular phase sensitivity.

The project simulates ideal isotropic antenna elements in a far-field, narrowband model. Phase noise is chosen directly in degrees, not derived from SNR. Shared-reference noise correlation, channel coherence, multipath, mutual coupling, element patterns, physical attitude, Doppler, antenna placement errors, Earth coordinates, and orbit propagation are deliberately not included.

## Files

- `app.py` — Streamlit user interface and Plotly displays.
- `interferometer.py` — Testable narrowband forward and inverse physics.
- `tests/test_interferometer.py` — Basic numerical tests.
- `pyproject.toml` — Dependency definition for `uv sync`.
- `requirements.txt` — Optional `pip` installation.

## Suggested learning exercises

1. Select `2-element` and set the long baseline to 0.05 m at 2.2 GHz. Observe a slowly varying phase curve.
2. Increase the baseline to 1 m. Count phase wraps and compare multiple inverse candidate directions.
3. Select `3-linear`; notice the inverse uncertainty along a direction-cosine band.
4. Select `3-L` and then `3-triangle`; compare how candidates change with layout.
5. Select `4-dual-axis` or `5-cross` and compare short/long spacing and ambiguity distributions.
6. Increase phase noise and observe the effect on the inverse map; reduce map step size to resolve narrow minima.

## Future Version 3

Add satellite position/attitude, WGS84 Earth intercept, ground truth emitter location, and error ellipse/Monte Carlo geolocation accuracy.
