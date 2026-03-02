"""FEM solver tests: validate against analytical solution on a sphere."""

import numpy as np
import pytest

from tmswarp.analytical import tms_analytical_efield
from tmswarp.coil import magnetic_dipole_dadt
from tmswarp.conductor import TetMesh, element_barycenters, make_sphere_mesh
from tmswarp.fields import compute_efield_at_elements, mag, rdm
from tmswarp.solver import assemble_rhs_tms, assemble_stiffness, gradient_operator, solve_fem


# ---- Mesh fixtures ----

@pytest.fixture(scope="module")
def coarse_mesh():
    """Small mesh for fast unit tests."""
    return make_sphere_mesh(radius=0.095, n_shells=3, n_surface=80,
                            conductivity=1.0)


@pytest.fixture(scope="module")
def fine_mesh():
    """Finer mesh for FEM validation against analytical."""
    return make_sphere_mesh(radius=0.095, n_shells=12, n_surface=1200,
                            conductivity=1.0, min_quality=0.05)


# ---- Gradient operator tests ----

class TestGradientOperator:
    def test_linear_function_exact(self, coarse_mesh):
        """Gradient of a linear function f(x,y,z) = 2x + 3y + 5z should be exact."""
        G = gradient_operator(coarse_mesh)
        f = (2.0 * coarse_mesh.nodes[:, 0]
             + 3.0 * coarse_mesh.nodes[:, 1]
             + 5.0 * coarse_mesh.nodes[:, 2])
        f_elem = f[coarse_mesh.elements]
        grad_f = np.einsum('ei,eid->ed', f_elem, G)
        expected = np.array([2.0, 3.0, 5.0])
        np.testing.assert_allclose(grad_f, np.broadcast_to(expected, grad_f.shape),
                                  atol=1e-10)

    def test_constant_function_zero_gradient(self, coarse_mesh):
        """Gradient of a constant should be zero."""
        G = gradient_operator(coarse_mesh)
        f = np.ones(len(coarse_mesh.nodes)) * 7.0
        f_elem = f[coarse_mesh.elements]
        grad_f = np.einsum('ei,eid->ed', f_elem, G)
        np.testing.assert_allclose(grad_f, 0.0, atol=1e-12)


# ---- Stiffness matrix tests ----

class TestStiffnessMatrix:
    def test_symmetry(self, coarse_mesh):
        """Stiffness matrix should be symmetric."""
        K = assemble_stiffness(coarse_mesh)
        diff = K - K.T
        assert abs(diff).max() < 1e-14

    def test_zero_row_sum(self, coarse_mesh):
        """Row sums should be approximately zero (Neumann consistency)."""
        K = assemble_stiffness(coarse_mesh)
        row_sums = np.abs(np.array(K.sum(axis=1)).ravel())
        assert np.max(row_sums) < 1e-10

    def test_constant_in_null_space(self, coarse_mesh):
        """K @ ones should be approximately zero."""
        K = assemble_stiffness(coarse_mesh)
        ones = np.ones(K.shape[0])
        np.testing.assert_allclose(K @ ones, 0.0, atol=1e-10)


# ---- Full FEM solve vs analytical ----

class TestFEMvsAnalytical:
    @pytest.fixture(scope="class")
    def fem_solution(self, fine_mesh):
        """Full FEM solve on the fine sphere mesh."""
        dipole_pos = np.array([0.0, 0.0, 0.3])
        dipole_moment = np.array([1.0, 0.0, 0.0])
        didt = 1e6

        # Primary field at nodes
        dAdt = magnetic_dipole_dadt(dipole_pos, dipole_moment, didt,
                                    fine_mesh.nodes)

        # FEM solve
        G = gradient_operator(fine_mesh)
        K = assemble_stiffness(fine_mesh, G)
        b = assemble_rhs_tms(fine_mesh, dAdt, G)
        phi = solve_fem(K, b, pin_node=0)

        # E-field at element barycenters
        E_fem = compute_efield_at_elements(fine_mesh, phi, dAdt, G)

        # Analytical E-field at same barycenters
        bary = element_barycenters(fine_mesh)
        E_ana = tms_analytical_efield(dipole_pos, dipole_moment, didt, bary)

        return E_fem, E_ana

    def test_efield_nonzero(self, fem_solution):
        """Both FEM and analytical E-fields should be nonzero."""
        E_fem, E_ana = fem_solution
        assert np.linalg.norm(E_fem) > 0
        assert np.linalg.norm(E_ana) > 0

    def test_rdm_threshold(self, fem_solution):
        """RDM should be < 0.2 (directional accuracy)."""
        E_fem, E_ana = fem_solution
        r = rdm(E_fem, E_ana)
        print(f"RDM = {r:.4f}")
        assert r < 0.2

    def test_mag_threshold(self, fem_solution):
        """Magnitude ratio should be within 10% (|MAG| < log(1.1))."""
        E_fem, E_ana = fem_solution
        m = mag(E_fem, E_ana)
        print(f"MAG = {m:.4f}")
        assert abs(m) < np.log(1.1)
