import numpy as np
from interferometer import (C,wavelength,wrap_rad,array_positions,direction,
                            phase_difference,measured_phases,ambiguity_surface,candidate_solutions)


def test_wavelength():
    assert np.isclose(wavelength(2.2e9), C/2.2e9)


def test_wrapping():
    np.testing.assert_allclose(wrap_rad(np.deg2rad([190,-190,360])),np.deg2rad([-170,170,0]))


def test_broadside_zero():
    for layout in ['2-element','3-linear','3-L','3-triangle','4-dual-axis','5-cross']:
        p=array_positions(layout,.07,.8)
        np.testing.assert_allclose(phase_difference(p,2.2e9,15,0),0,atol=1e-14)


def test_single_baseline():
    p=array_positions('2-element',.07,.5)
    got=phase_difference(p,2.2e9,0,30)[0]
    assert np.isclose(got,2*np.pi*.5/wavelength(2.2e9)*.5)


def test_noise_determinism():
    a=measured_phases(np.array([1.,2.]),2,rng=np.random.default_rng(7))
    b=measured_phases(np.array([1.,2.]),2,rng=np.random.default_rng(7))
    np.testing.assert_allclose(a,b)


def test_short_baseline_inverse():
    p=np.array([[0.,0.,0.],[.04,0,0],[0,.04,0]])
    measured=wrap_rad(phase_difference(p,2.2e9,35,40))
    xx,yy,valid,s=ambiguity_surface(p,2.2e9,measured,1)
    candidate=candidate_solutions(xx,yy,valid,s)[0]
    actual=direction(35,40)
    assert np.hypot(candidate.ux-actual[0],candidate.uy-actual[1])<.03


def test_planar_hemisphere_symmetry():
    p=array_positions('3-L',.07,.8)
    u=direction(35,40)
    # Interferometer planar baselines ignore z component.
    assert np.isclose(u[0]**2+u[1]**2+u[2]**2,1)
    b=p[1:]-p[0]
    np.testing.assert_allclose(b@u,b@np.array([u[0],u[1],-u[2]]))
