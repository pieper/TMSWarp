"""Validate TMSWarp solvers against the SimNIBS sphere3 mesh.

Skipped automatically when sphere3_data.npz has not been extracted (requires
SimNIBS installation + running scripts/extract_sphere3.py).

To generate the data file:

    /path/to/SimNIBS-4.5/simnibs_env/bin/python scripts/extract_sphere3.py
"""

from pathlib import Path

import numpy as np
import pytest

from tmswarp.analytical import tms_analytical_efield
from tmswarp.coil import magnetic_dipole_dadt
from tmswarp.conductor import TetMesh, element_barycenters
from tmswarp.fields import compute_efield_at_elements, mag, rdm
from tmswarp.solver import assemble_rhs_tms, assemble_stiffness, gradient_operator, solve_fem
from tmswarp.solver_warp import solve_fem_warp, warp_available

DATA_PATH = Path(__file__).resolve().parents[1] / "sphere3_data.npz"

pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason="sphere3_data.npz not found; run scripts/extract_sphere3.py first",
)

DIPOLE_POS_M = np.array([0.0, 0.0, 0.3])
DIPOLE_MOMENT = np.array([1.0, 0.0, 0.0])
DIDT = 1e6


@pytest.fixture(scope="module")
def sphere3_solution():
    """Load sphere3 mesh and solve with NumPy FEM; return (E_fem, E_ana, G)."""
    data = np.load(DATA_PATH)
    mesh = TetMesh(
        nodes=data["nodes"].astype(np.float64),
        elements=data["elements"].astype(np.int32),
        conductivity=data["conductivity"].astype(np.float64),
    )
    dAdt = magnetic_dipole_dadt(DIPOLE_POS_M, DIPOLE_MOMENT, DIDT, mesh.nodes)
    G = gradient_operator(mesh)
    K = assemble_stiffness(mesh, G)
    b = assemble_rhs_tms(mesh, dAdt, G)
    phi = solve_fem(K, b, pin_node=0)
    E_fem = compute_efield_at_elements(mesh, phi, dAdt, G)
    bary = element_barycenters(mesh)
    E_ana = tms_analytical_efield(DIPOLE_POS_M, DIPOLE_MOMENT, DIDT, bary)
    return mesh, dAdt, G, E_fem, E_ana


class TestNumPyFEMSphere3:
    def test_rdm_below_simnibs_threshold(self, sphere3_solution):
        """NumPy FEM RDM on sphere3 should be < 0.2 (SimNIBS standard)."""
        _, _, _, E_fem, E_ana = sphere3_solution
        r = rdm(E_fem, E_ana)
        print(f"NumPy FEM RDM on sphere3 = {r:.4f}")
        assert r < 0.2

    def test_mag_below_simnibs_threshold(self, sphere3_solution):
        """NumPy FEM |MAG| on sphere3 should be < log(1.1) (SimNIBS standard)."""
        _, _, _, E_fem, E_ana = sphere3_solution
        m = mag(E_fem, E_ana)
        print(f"NumPy FEM |MAG| on sphere3 = {m:.4f}")
        assert abs(m) < np.log(1.1)


class TestWarpFEMSphere3:
    @pytest.fixture(scope="class")
    def warp_solution(self, sphere3_solution):
        if not warp_available():
            pytest.skip("warp-lang not installed")
        mesh, dAdt, G, _, E_ana = sphere3_solution
        phi_wp = solve_fem_warp(mesh, dAdt, quiet=True)
        E_wp = compute_efield_at_elements(mesh, phi_wp, dAdt, G)
        return E_wp, E_ana

    def test_rdm_below_threshold(self, warp_solution):
        """Warp.fem RDM on sphere3 should be < 0.3 (relaxed for float32 geometry)."""
        E_wp, E_ana = warp_solution
        r = rdm(E_wp, E_ana)
        print(f"Warp.fem RDM on sphere3 = {r:.4f}")
        assert r < 0.3

    def test_mag_below_threshold(self, warp_solution):
        """Warp.fem |MAG| on sphere3 should be < log(1.2)."""
        E_wp, E_ana = warp_solution
        m = mag(E_wp, E_ana)
        print(f"Warp.fem |MAG| on sphere3 = {m:.4f}")
        assert abs(m) < np.log(1.2)
