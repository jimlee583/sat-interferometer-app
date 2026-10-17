# Satellite RF Interferometer Lab

A Python Streamlit lab for narrowband far-field interferometry and LEO emitter geolocation.

Versions 1 and 2 cover antenna baselines, phase wrapping, two-dimensional direction of arrival (DOA), and ambiguities. Version 3 places that array on a nadir-pointing satellite and estimates an emitter’s latitude and longitude. It is an educational simulator, not a navigation system.

## Quick start

```bash
uv sync
uv run streamlit run app.py
```

Alternatively: `python -m pip install -r requirements.txt && streamlit run app.py`.

```bash
uv run pytest -q
```

## Versions 1 and 2

The first four tabs are the array lab. Frequency runs from 0.1 to 40 GHz. Layouts include two-element, linear, L, triangle, dual-axis, and cross arrays. The sidebar sets the emitter azimuth and off-boresight angle used by those tabs.

Antennas lie in the XY plane and +Z is boresight. Azimuth is counterclockwise from +X. Off-boresight `theta` is in `[0°, 90°]` for the front hemisphere:

```text
u = (sin(theta) cos(azimuth), sin(theta) sin(azimuth), cos(theta))
```

For the reference pair from antenna 0 to antenna j:

```text
phase_0j = (2π / λ) (r_j − r_0) · u
```

The receiver sees that phase wrapped to `[-π, π)`. The model can add independent Gaussian phase noise on each reference pair and a common pair bias. A planar array cannot tell +Z from −Z. A linear array constrains only one direction cosine. An L or triangle gives two directions but does not guarantee a unique unwrapped solution when the baselines are long.

## Version 3 — Satellite geolocation

Open the **Satellite Geolocation** tab. Frequency, array, phase noise, common pair bias, seed, and DOA grid resolution still come from the sidebar. Orbit, attitude, and the emitter are set on the tab. The sidebar azimuth and off-boresight sliders do not move the geolocation emitter.

Defaults are a 500 km orbit at 53° inclination, subpoint 20° N, 100° W, nadir pointing, and a 2.2 GHz three-element L array. The emitter starts at 25° N, 95° W, inside the Earth disk. The stock 0.8 m L-array baselines are several wavelengths, so the tab warns when the fix is ambiguous. Shorten the long baseline to about half a wavelength for a unique direction.

### Frames

1. **ECEF.** +X through longitude 0, +Y through 90° E, +Z through the north pole. Distances are meters.
2. **WGS84 geodetic.** Latitude, longitude, and ellipsoidal height. The forward conversion is

```text
N = a / sqrt(1 − e² sin²φ)
X = (N + h) cosφ cosλ
Y = (N + h) cosφ sinλ
Z = (N(1 − e²) + h) sinφ
```

with `a = 6378137 m` and `f = 1/298.257223563`. The inverse uses Bowring’s formula. The geodetic subpoint is the ellipsoid point with the satellite’s latitude and longitude and zero height.

3. **VVLH**, built for a spherical circular orbit at the WGS84 satellite position:

```text
+X = along-track velocity
+Y = cross-track = −orbit normal
+Z = geocentric nadir = −r̂
```

This is not a zenith-positive LVLH frame, and +Z is not the geodetic normal. Heading uses the inclination and whether the pass is ascending or descending:

```text
v_east = cos(i) / cos(δ)
v_north = ±sqrt(1 − v_east²)
```

`δ` is geocentric latitude. The north sign is positive when latitude is increasing. `|δ|` cannot exceed the inclination (or `180° − i` when retrograde). Argument of latitude is reported from the ascending node; it is undefined for an equatorial orbit.

4. **Body.** Coincident with VVLH at zero roll, pitch, and yaw. The attitude sequence is yaw, then pitch, then roll:

```text
v_VVLH = R_z(yaw) R_y(pitch) R_x(roll) v_body
```

Positive yaw rotates the nose toward +Y. Positive pitch rotates the nose toward −Z (up, away from Earth). Positive roll rotates the right wing toward +Z (down).

5. **Antenna.** Aligned with the body. The Version 1–2 phase model is used unchanged, so +Z boresight looks at geocentric nadir when the attitude is zero. Earth is in the front hemisphere for a visible emitter.

The geocentric nadir intercept and the geodetic subpoint differ away from the equator. At 45° and 500 km the separation is about 1.6 km. It grows with altitude.

### Forward model

The emitter is either a geodetic point on the ellipsoid or the ground intercept of a chosen azimuth and off-nadir angle. The line of sight is

```text
û = (r_emitter − r_sat) / ‖r_emitter − r_sat‖
```

That unit vector is rotated into the body frame. Azimuth and off-nadir then drive `phase_difference`. A point is beyond the horizon when the satellite is below the local WGS84 tangent plane. Those geometries are drawn and flagged; they are not inverted, because the satellite would not see the signal.

### Inverse model

Wrapped phases go through the existing direction-cosine search. Each grid minimum is refined by a least-squares fit of the circular phase residual. Refinement does not receive the true direction or the true coordinates.

Each candidate direction is rotated back to ECEF and intersected with the WGS84 ellipsoid. In scaled coordinates `q = (x/a, y/a, z/b)` the ray is a unit-sphere quadratic. The reported fix is the **lowest phase-residual** candidate whose ray meets the Earth in front of the satellite. Other intercepts stay on the globe and in the table. Rays that miss, or that only intersect behind the satellite, are labeled and dropped from the fix.

The solution is marked ambiguous when two or more distinct Earth intercepts fit the phases within 1° of phase RMS, or when the array is collinear. A long baseline can have more aliases than the candidate list retains.

Ground error is the WGS84 geodesic from pyproj. Angular error is the angle between the true and estimated body-frame arrival vectors, computed only for the report.

### What the globe shows

The Plotly scene uses ECEF kilometers and `aspectmode='data'`. Coastlines are Natural Earth 1:110m (public domain). Toggles hide the Earth, coastlines, graticule, satellite, subpoint, true emitter, estimates, both lines of sight, the horizon, and the ground-error arc. Drag to rotate and scroll to zoom.

### Monte Carlo

Channel biases are drawn once and held fixed. Phase noise, attitude-knowledge error, and position-knowledge error are independent on every trial. Position error is one-sigma on each VVLH axis and only moves the assumed ECEF origin. Attitude error is added to roll, pitch, and yaw.

Median, 95th percentile, and RMS are reported twice: once for unambiguous successes, and once for every success using the lowest-cost candidate. The second set includes wrong ambiguity picks. Diamonds on the map are those ambiguous trials.

### Off-nadir sensitivity

The sweep walks from nadir toward the horizon along one body azimuth. At each angle the ground point is compared with the intercept of a ray `0.1°` farther off-nadir (the step is editable). A perturbed ray that misses the Earth leaves a gap. The same angle error maps to a larger ground distance as elevation falls:

```text
ground shift ≈ slant range × δθ / sin(elevation)
```

Optional curves at 400, 800, and 1,200 km show the smaller Earth disk and longer slant range of a higher orbit.

## Files

- `app.py` — Streamlit interface.
- `interferometer.py` — Narrowband forward and inverse interferometry.
- `geolocation.py` — WGS84, VVLH, attitude, ray intersection, geolocation, Monte Carlo, and sensitivity.
- `satellite_viz.py` — Plotly globe.
- `data/coastlines_110m.npz` — Natural Earth coastline.
- `tests/` — Numerical tests for the array lab and geolocation.
- `pyproject.toml`, `uv.lock` — Dependencies, including pyproj.

## Suggested checks

1. On an interferometer tab, shorten and then lengthen a two-element baseline and compare phase wraps.
2. On Satellite Geolocation, set the emitter to the satellite subpoint at the equator. Off-nadir and the ground error should both be about zero with a short baseline and no phase noise.
3. Move the emitter toward the limb and watch the off-nadir sensitivity curve rise.
4. Set the long baseline to 0.8 m on the 3-L layout. The tab should list more than one Earth intercept and say the fix is ambiguous.
5. Run Monte Carlo twice: once with phase noise only, and once with phase noise set to zero and a nonzero fixed channel bias. The bias case should repeat the same error; the noise case should scatter.
