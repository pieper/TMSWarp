"""Validate TMSWarp solvers against SimNIBS sphere3 mesh.

Uses the sphere3.msh test mesh from SimNIBS (extracted to sphere3_data.npz by
scripts/extract_sphere3.py).  Runs the NumPy FEM and Warp.fem solvers and
compares the E-field against the analytical Heller & van Hulsteyn (1992)
solution, using the same dipole parameters as SimNIBS's own TMS validation.

Usage
-----
    # First extract the mesh (requires SimNIBS Python):
    /path/to/SimNIBS-4.5/simnibs_env/bin/python scripts/extract_sphere3.py

    # Then run validation with the pixi environment:
    pixi run python benchmarks/sphere3_validation.py
"""

import time
from pathlib import Path

import numpy as np

from tmswarp.analytical import tms_analytical_efield
from tmswarp.coil import magnetic_dipole_dadt
from tmswarp.conductor import TetMesh, element_barycenters
from tmswarp.fields import compute_efield_at_elements, mag, rdm
from tmswarp.solver import assemble_rhs_tms, assemble_stiffness, gradient_operator, solve_fem
from tmswarp.solver_warp import solve_fem_warp, warp_available

# ---------------------------------------------------------------------------
# Dipole parameters — match SimNIBS TMS validation exactly
# ---------------------------------------------------------------------------
DIPOLE_POS_M = np.array([0.0, 0.0, 0.3])     # 300 mm above sphere, in metres
DIPOLE_MOMENT = np.array([1.0, 0.0, 0.0])    # unit x-direction
DIDT = 1e6                                    # A/s

DATA_PATH = Path(__file__).resolve().parents[1] / "sphere3_data.npz"


def load_sphere3_mesh() -> TetMesh:
    """Load the SimNIBS sphere3 mesh from the pre-extracted .npz file."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Mesh data not found: {DATA_PATH}\n"
            "Run  /path/to/SimNIBS-4.5/simnibs_env/bin/python scripts/extract_sphere3.py  first."
        )
    data = np.load(DATA_PATH)
    nodes = data["nodes"].astype(np.float64)           # (N, 3) in metres
    elements = data["elements"].astype(np.int32)       # (E, 4) 0-based
    conductivity = data["conductivity"].astype(np.float64)  # (E,) S/m
    return TetMesh(nodes=nodes, elements=elements, conductivity=conductivity)


def run_validation():
    print("=" * 68)
    print("TMSWarp vs SimNIBS sphere3 mesh validation")
    print("=" * 68)

    # Load mesh
    mesh = load_sphere3_mesh()
    n_nodes = len(mesh.nodes)
    n_elems = len(mesh.elements)
    print(f"\nMesh: {n_nodes} nodes, {n_elems} elements")
    r_range = np.linalg.norm(mesh.nodes, axis=1)
    print(f"Node radius range: {r_range.min()*1000:.1f} mm – {r_range.max()*1000:.1f} mm")

    # Primary dA/dt at nodes
    dAdt = magnetic_dipole_dadt(DIPOLE_POS_M, DIPOLE_MOMENT, DIDT, mesh.nodes)

    # Analytical E-field at element barycenters
    bary = element_barycenters(mesh)
    E_ana = tms_analytical_efield(DIPOLE_POS_M, DIPOLE_MOMENT, DIDT, bary)
    print(f"\nAnalytical |E| range: {np.linalg.norm(E_ana, axis=1).min():.3f} – "
          f"{np.linalg.norm(E_ana, axis=1).max():.3f} V/m")

    # -----------------------------------------------------------------------
    # NumPy FEM solve
    # -----------------------------------------------------------------------
    print("\n--- NumPy FEM ---")
    t0 = time.perf_counter()
    G = gradient_operator(mesh)
    K = assemble_stiffness(mesh, G)
    b = assemble_rhs_tms(mesh, dAdt, G)
    phi_np = solve_fem(K, b, pin_node=0)
    t_np = time.perf_counter() - t0

    E_np = compute_efield_at_elements(mesh, phi_np, dAdt, G)
    rdm_np = rdm(E_np, E_ana)
    mag_np = mag(E_np, E_ana)

    print(f"  Time:  {t_np:.3f} s")
    print(f"  RDM:   {rdm_np:.4f}  (SimNIBS threshold < 0.2)")
    print(f"  |MAG|: {mag_np:.4f}  (SimNIBS threshold < log(1.1) = {np.log(1.1):.4f})")
    np_pass = rdm_np < 0.2 and abs(mag_np) < np.log(1.1)
    print(f"  PASS:  {np_pass}")

    # -----------------------------------------------------------------------
    # Warp.fem solve (optional)
    # -----------------------------------------------------------------------
    if warp_available():
        print("\n--- Warp.fem ---")
        # Warm-up run (JIT compile)
        _ = solve_fem_warp(mesh, dAdt, quiet=True)

        t0 = time.perf_counter()
        phi_wp = solve_fem_warp(mesh, dAdt, quiet=True)
        t_wp = time.perf_counter() - t0

        E_wp = compute_efield_at_elements(mesh, phi_wp, dAdt, G)
        rdm_wp = rdm(E_wp, E_ana)
        mag_wp = mag(E_wp, E_ana)

        print(f"  Time (cached):  {t_wp:.3f} s")
        print(f"  RDM:   {rdm_wp:.4f}  (threshold < 0.3 due to float32 geometry)")
        print(f"  |MAG|: {mag_wp:.4f}  (threshold < log(1.2) = {np.log(1.2):.4f})")
        wp_pass = rdm_wp < 0.3 and abs(mag_wp) < np.log(1.2)
        print(f"  PASS:  {wp_pass}")
    else:
        print("\n--- Warp.fem ---  (skipped: warp-lang not installed)")
        rdm_wp = mag_wp = t_wp = None
        wp_pass = None

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("Summary")
    print("=" * 68)
    print(f"{'Solver':<14} {'RDM':>8} {'|MAG|':>8} {'Time(s)':>10} {'PASS':>6}")
    print("-" * 50)
    print(f"{'Analytical':<14} {'0.0000':>8} {'0.0000':>8} {'N/A':>10} {'  yes':>6}")
    print(f"{'NumPy FEM':<14} {rdm_np:>8.4f} {mag_np:>8.4f} {t_np:>10.3f} {str(np_pass):>6}")
    if rdm_wp is not None:
        print(f"{'Warp.fem':<14} {rdm_wp:>8.4f} {mag_wp:>8.4f} {t_wp:>10.3f} {str(wp_pass):>6}")
    print()

    return {
        "rdm_np": rdm_np, "mag_np": mag_np, "t_np": t_np, "pass_np": np_pass,
        "rdm_wp": rdm_wp, "mag_wp": mag_wp, "t_wp": t_wp, "pass_wp": wp_pass,
    }


if __name__ == "__main__":
    run_validation()
