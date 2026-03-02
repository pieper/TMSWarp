"""Tests for primary field (dA/dt) from magnetic dipole."""

import numpy as np
import pytest

from tmswarp.coil import magnetic_dipole_dadt


class TestMagneticDipoleDadt:
    def test_matches_onlinefem_formula(self):
        """Validate against the exact formula from onlinefem.py."""
        dipole_pos = np.array([0.0, 0.0, 0.3])
        dipole_moment = np.array([1.0, 0.0, 0.0])
        didt = 1e6
        positions = np.array([[0.05, 0.03, 0.01], [-0.02, 0.04, -0.03]])

        dAdt = magnetic_dipole_dadt(dipole_pos, dipole_moment, didt, positions)

        # Manual computation matching onlinefem.py
        r = positions - dipole_pos
        expected = 1e-7 * didt * np.cross(dipole_moment, r) / (
            np.linalg.norm(r, axis=1)[:, None] ** 3
        )
        np.testing.assert_allclose(dAdt, expected, rtol=1e-12)

    def test_inverse_cube_law(self):
        """dA/dt magnitude should fall off as 1/r^2 along displacement axis.

        dA/dt = mu0/(4pi) * didt * cross(m, r) / |r|^3
        |cross(m, r)| ~ |r| when m perp r, so |dA/dt| ~ |r|/|r|^3 = 1/|r|^2
        along directions perpendicular to m.
        """
        dp = np.array([0.0, 0.0, 0.0])
        dm = np.array([1.0, 0.0, 0.0])
        pos1 = np.array([[0.0, 0.1, 0.0]])
        pos2 = np.array([[0.0, 0.2, 0.0]])
        dAdt1 = magnetic_dipole_dadt(dp, dm, 1.0, pos1)
        dAdt2 = magnetic_dipole_dadt(dp, dm, 1.0, pos2)
        ratio = np.linalg.norm(dAdt1) / np.linalg.norm(dAdt2)
        expected = (0.2 / 0.1) ** 2  # 4.0 (cross product adds linear factor)
        np.testing.assert_allclose(ratio, expected, rtol=1e-10)

    def test_perpendicular_to_r_and_m(self):
        """dA/dt = mu0/(4pi) * didt * (m x r) / |r|^3, perpendicular to both."""
        dp = np.array([0.0, 0.0, 0.0])
        dm = np.array([1.0, 0.0, 0.0])
        pos = np.array([[0.0, 0.5, 0.0]])
        dAdt = magnetic_dipole_dadt(dp, dm, 1.0, pos)
        r = pos - dp
        assert abs(np.dot(dAdt[0], dm)) < 1e-15
        assert abs(np.dot(dAdt[0], r[0])) < 1e-15

    def test_linear_in_didt(self):
        """dA/dt should scale linearly with dI/dt."""
        dp = np.array([0.0, 0.0, 0.3])
        dm = np.array([1.0, 0.0, 0.0])
        pos = np.array([[0.05, 0.0, 0.0]])
        dAdt1 = magnetic_dipole_dadt(dp, dm, 1e6, pos)
        dAdt2 = magnetic_dipole_dadt(dp, dm, 2e6, pos)
        np.testing.assert_allclose(dAdt2, 2.0 * dAdt1, rtol=1e-14)
