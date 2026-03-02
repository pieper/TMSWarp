"""Tests for the Heller & van Hulsteyn analytical TMS E-field."""

import numpy as np
import pytest

from tmswarp.analytical import tms_analytical_efield


@pytest.fixture
def standard_dipole():
    """Standard test configuration matching onlinefem.py."""
    return dict(
        dipole_pos=np.array([0.0, 0.0, 0.3]),
        dipole_moment=np.array([1.0, 0.0, 0.0]),
        didt=1e6,
    )


class TestAnalyticalEfield:
    def test_nonzero(self, standard_dipole):
        """E-field should be non-trivially nonzero inside the sphere."""
        pos = np.array([[0.05, 0.03, 0.01]])
        E = tms_analytical_efield(positions=pos, **standard_dipole)
        assert np.linalg.norm(E) > 0

    def test_purely_tangential(self, standard_dipole):
        """E-field should have zero radial component on a sphere."""
        # Points on a sphere of radius 0.08 (inside 0.095)
        r = 0.08
        theta = np.linspace(0.2, np.pi - 0.2, 15)
        phi_ang = np.linspace(0, 2 * np.pi, 15, endpoint=False)
        TH, PH = np.meshgrid(theta, phi_ang)
        positions = r * np.column_stack([
            np.sin(TH.ravel()) * np.cos(PH.ravel()),
            np.sin(TH.ravel()) * np.sin(PH.ravel()),
            np.cos(TH.ravel()),
        ])
        E = tms_analytical_efield(positions=positions, **standard_dipole)
        # Radial component: E . r_hat
        r_hat = positions / np.linalg.norm(positions, axis=1)[:, None]
        radial = np.sum(E * r_hat, axis=1)
        np.testing.assert_allclose(radial, 0.0, atol=1e-6)

    def test_symmetry_under_rotation(self, standard_dipole):
        """Rotating dipole and evaluation points should rotate E consistently."""
        pos = np.array([[0.05, 0.03, 0.01]])
        E1 = tms_analytical_efield(positions=pos, **standard_dipole)

        # 90-degree rotation around z: (x,y,z) -> (-y,x,z)
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        rotated = dict(
            dipole_pos=R @ standard_dipole['dipole_pos'],
            dipole_moment=R @ standard_dipole['dipole_moment'],
            didt=standard_dipole['didt'],
        )
        E2 = tms_analytical_efield(positions=(R @ pos.T).T, **rotated)
        np.testing.assert_allclose(E2, (R @ E1.T).T, rtol=1e-10)

    def test_linear_in_didt(self, standard_dipole):
        """E-field should scale linearly with dI/dt."""
        pos = np.array([[0.05, 0.03, 0.01]])
        E1 = tms_analytical_efield(positions=pos, **standard_dipole)
        cfg2 = {**standard_dipole, 'didt': 2e6}
        E2 = tms_analytical_efield(positions=pos, **cfg2)
        np.testing.assert_allclose(E2, 2.0 * E1, rtol=1e-14)

    def test_decays_with_distance_from_dipole(self, standard_dipole):
        """E-field should be stronger closer to the dipole."""
        # Point close to the dipole (near top of sphere)
        pos_close = np.array([[0.0, 0.0, 0.09]])
        # Point far from dipole (near bottom of sphere)
        pos_far = np.array([[0.0, 0.0, -0.09]])
        E_close = tms_analytical_efield(positions=pos_close, **standard_dipole)
        E_far = tms_analytical_efield(positions=pos_far, **standard_dipole)
        assert np.linalg.norm(E_close) > np.linalg.norm(E_far)
