"""Warp.fem solver tests: validate against analytical solution and NumPy FEM.

Skipped automatically if warp-lang is not installed.
"""

import numpy as np
import pytest

from tmswarp.analytical import tms_analytical_efield
from tmswarp.coil import magnetic_dipole_dadt
from tmswarp.conductor import element_barycenters, make_sphere_mesh
from tmswarp.fields import compute_efield_at_elements, mag, rdm
from tmswarp.solver import assemble_rhs_tms, assemble_stiffness, gradient_operator, solve_fem
from tmswarp.solver_warp import solve_fem_warp, warp_available

pytestmark = pytest.mark.skipif(
    not warp_available(),
    reason="warp-lang not installed",
)


@pytest.fixture(scope="module")
def coarse_mesh():
    return make_sphere_mesh(radius=0.095, n_shells=3, n_surface=80, conductivity=1.0)


@pytest.fixture(scope="module")
def fine_mesh():
    return make_sphere_mesh(
        radius=0.095, n_shells=12, n_surface=1200, conductivity=1.0, min_quality=0.05
    )


@pytest.fixture(scope="module")
def dipole_params():
    return {
        "pos": np.array([0.0, 0.0, 0.3]),
        "moment": np.array([1.0, 0.0, 0.0]),
        "didt": 1e6,
    }


class TestWarpSolverSmoke:
    def test_returns_array(self, coarse_mesh, dipole_params):
        """solve_fem_warp must return a 1-D float64 numpy array."""
        dAdt = magnetic_dipole_dadt(
            dipole_params["pos"], dipole_params["moment"],
            dipole_params["didt"], coarse_mesh.nodes,
        )
        phi = solve_fem_warp(coarse_mesh, dAdt, quiet=True)
        assert isinstance(phi, np.ndarray)
        assert phi.ndim == 1
        assert phi.dtype == np.float64
        assert phi.shape == (len(coarse_mesh.nodes),)

    def test_phi_nonzero(self, coarse_mesh, dipole_params):
        """Potential must be non-trivially non-zero."""
        dAdt = magnetic_dipole_dadt(
            dipole_params["pos"], dipole_params["moment"],
            dipole_params["didt"], coarse_mesh.nodes,
        )
        phi = solve_fem_warp(coarse_mesh, dAdt, quiet=True)
        assert np.linalg.norm(phi) > 0.0

    def test_gauge_node_is_zero(self, coarse_mesh, dipole_params):
        """Node 0 must be pinned to zero."""
        dAdt = magnetic_dipole_dadt(
            dipole_params["pos"], dipole_params["moment"],
            dipole_params["didt"], coarse_mesh.nodes,
        )
        phi = solve_fem_warp(coarse_mesh, dAdt, pin_node=0, quiet=True)
        assert abs(phi[0]) < 1e-6


class TestWarpVsNumpy:
    @pytest.fixture(scope="class")
    def both_solutions(self, coarse_mesh, dipole_params):
        """Solve with both NumPy FEM and Warp.fem on the same coarse mesh."""
        dAdt = magnetic_dipole_dadt(
            dipole_params["pos"], dipole_params["moment"],
            dipole_params["didt"], coarse_mesh.nodes,
        )
        G = gradient_operator(coarse_mesh)
        K = assemble_stiffness(coarse_mesh, G)
        b = assemble_rhs_tms(coarse_mesh, dAdt, G)
        phi_np = solve_fem(K, b, pin_node=0)
        E_np = compute_efield_at_elements(coarse_mesh, phi_np, dAdt, G)

        phi_wp = solve_fem_warp(coarse_mesh, dAdt, quiet=True)
        E_wp = compute_efield_at_elements(coarse_mesh, phi_wp, dAdt, G)

        return E_np, E_wp

    def test_rdm_agreement(self, both_solutions):
        """RDM between Warp.fem and NumPy FEM should be small (< 0.15)."""
        E_np, E_wp = both_solutions
        r = rdm(E_wp, E_np)
        print(f"RDM warp vs numpy = {r:.4f}")
        # Float32 geometry vs float64 causes some discrepancy; 0.15 is generous
        assert r < 0.15

    def test_mag_agreement(self, both_solutions):
        """Magnitude ratio between Warp.fem and NumPy FEM should be small (< 0.1)."""
        E_np, E_wp = both_solutions
        m = mag(E_wp, E_np)
        print(f"MAG warp vs numpy = {m:.4f}")
        assert abs(m) < 0.1


class TestWarpVsAnalytical:
    @pytest.fixture(scope="class")
    def warp_solution(self, fine_mesh, dipole_params):
        """Warp.fem E-field on the fine mesh."""
        dAdt = magnetic_dipole_dadt(
            dipole_params["pos"], dipole_params["moment"],
            dipole_params["didt"], fine_mesh.nodes,
        )
        G = gradient_operator(fine_mesh)
        phi_wp = solve_fem_warp(fine_mesh, dAdt, quiet=True)
        E_wp = compute_efield_at_elements(fine_mesh, phi_wp, dAdt, G)

        bary = element_barycenters(fine_mesh)
        E_ana = tms_analytical_efield(
            dipole_params["pos"], dipole_params["moment"],
            dipole_params["didt"], bary,
        )
        return E_wp, E_ana

    def test_rdm_threshold(self, warp_solution):
        """Warp.fem RDM vs analytical should be < 0.3 on the fine mesh.

        Note: The NumPy solver achieves RDM < 0.2; Warp.fem uses float32
        geometry which slightly degrades accuracy, so we use a relaxed threshold.
        """
        E_wp, E_ana = warp_solution
        r = rdm(E_wp, E_ana)
        print(f"Warp.fem RDM = {r:.4f}")
        assert r < 0.3

    def test_mag_threshold(self, warp_solution):
        """Warp.fem |MAG| vs analytical should be < log(1.2) on the fine mesh."""
        E_wp, E_ana = warp_solution
        m = mag(E_wp, E_ana)
        print(f"Warp.fem MAG = {m:.4f}")
        assert abs(m) < np.log(1.2)
