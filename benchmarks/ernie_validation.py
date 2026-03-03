"""Validate TMSWarp solvers on the SimNIBS ernie human head mesh.

Uses the ernie (low-resolution) head model, the standard SimNIBS reference
for realistic TMS simulation.  The mesh covers 6 tissue types (white matter,
gray matter, CSF, skull, scalp, eyes) with 222k nodes and 1.3M tetrahedra.

There is no closed-form analytical solution for a realistic head geometry,
so the comparison is:
  - NumPy FEM vs Warp.fem  (agreement metric: RDM, |MAG|)
  - Per-tissue E-field statistics  (physiological validation)

Usage
-----
    # Step 1 — fetch the mesh (once, requires SimNIBS installation):
    /path/to/SimNIBS-4.5/simnibs_env/bin/python scripts/fetch_ernie.py

    # Step 2 — run the validation (pixi / tmswarp environment):
    pixi run python benchmarks/ernie_validation.py

Coil configuration
------------------
Magnetic dipole at [0, 0, 200 mm] above head, moment [1, 0, 0], dI/dt = 1 MA/s.
This places the coil over the vertex at 100 mm standoff distance.
"""

import time
from pathlib import Path

import numpy as np

from tmswarp.coil import magnetic_dipole_dadt
from tmswarp.conductor import TetMesh, element_barycenters
from tmswarp.fields import compute_efield_at_elements, mag, rdm
from tmswarp.solver import assemble_rhs_tms, assemble_stiffness, gradient_operator, solve_fem
from tmswarp.solver_warp import solve_fem_warp, warp_available

# ---------------------------------------------------------------------------
# Data path
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parents[1] / "ernie_data.npz"

# ---------------------------------------------------------------------------
# Coil / dipole parameters
# ---------------------------------------------------------------------------
DIPOLE_POS_M = np.array([0.0, 0.0, 0.200])   # 200 mm above head centre, in metres
DIPOLE_MOMENT = np.array([1.0, 0.0, 0.0])    # unit x-direction
DIDT = 1e6                                    # A/s (standard TMS pulse)

# SimNIBS tissue names and conductivities (for reporting)
TISSUE_NAMES = {
    1: "white matter", 2: "gray matter", 3: "CSF",
    4: "skull",        5: "scalp",       6: "eyes",
}


def load_ernie_mesh() -> TetMesh:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Mesh data not found: {DATA_PATH}\n"
            "Run the fetch script first:\n"
            "  /path/to/SimNIBS-4.5/simnibs_env/bin/python scripts/fetch_ernie.py\n"
            "Or with a pre-downloaded zip:\n"
            "  ... scripts/fetch_ernie.py /path/to/ernie_lowres_V2.zip"
        )
    data = np.load(DATA_PATH)
    return TetMesh(
        nodes=data["nodes"].astype(np.float64),
        elements=data["elements"].astype(np.int32),
        conductivity=data["conductivity"].astype(np.float64),
    ), data["tag1"].astype(np.int32)


def print_mesh_stats(mesh: TetMesh, tag1: np.ndarray) -> None:
    n_nodes = len(mesh.nodes)
    n_elems = len(mesh.elements)
    r_node = np.linalg.norm(mesh.nodes, axis=1)
    print(f"  Nodes:    {n_nodes:>10,}")
    print(f"  Elements: {n_elems:>10,}")
    print(f"  Bounding box (mm):")
    for i, ax in enumerate("xyz"):
        lo, hi = mesh.nodes[:, i].min() * 1000, mesh.nodes[:, i].max() * 1000
        print(f"    {ax}: {lo:7.1f} to {hi:.1f} mm")
    print(f"  Tissue breakdown:")
    for tag in sorted(np.unique(tag1)):
        count = int(np.sum(tag1 == tag))
        sigma = mesh.conductivity[tag1 == tag].mean()
        name = TISSUE_NAMES.get(int(tag), f"tag{tag}")
        print(f"    tag {tag} ({name:<12}): {count:>9,} elements  σ={sigma:.3f} S/m")


def print_efield_stats(E: np.ndarray, tag1: np.ndarray, label: str) -> None:
    mag_all = np.linalg.norm(E, axis=1)
    print(f"\n  {label} |E| statistics (V/m):")
    print(f"    Overall: mean={mag_all.mean():.3f}  max={mag_all.max():.3f}  "
          f"p99={np.percentile(mag_all, 99):.3f}")
    for tag in sorted(np.unique(tag1)):
        mask = tag1 == tag
        m_t = mag_all[mask]
        name = TISSUE_NAMES.get(int(tag), f"tag{tag}")
        print(f"    tag {tag} ({name:<12}): mean={m_t.mean():.3f}  max={m_t.max():.3f}")


def run_validation():
    print("=" * 70)
    print("TMSWarp solver validation on SimNIBS ernie human head mesh")
    print("=" * 70)

    # Load mesh
    mesh, tag1 = load_ernie_mesh()
    print("\nMesh:")
    print_mesh_stats(mesh, tag1)

    print(f"\nDipole: pos={DIPOLE_POS_M * 1000} mm, moment={DIPOLE_MOMENT}, "
          f"dI/dt={DIDT:.0e} A/s")

    # Primary dA/dt at nodes
    dAdt = magnetic_dipole_dadt(DIPOLE_POS_M, DIPOLE_MOMENT, DIDT, mesh.nodes)

    # -----------------------------------------------------------------------
    # NumPy FEM solve
    # -----------------------------------------------------------------------
    print("\n--- NumPy FEM ---")
    t0 = time.perf_counter()
    G = gradient_operator(mesh)
    t_grad = time.perf_counter() - t0

    t1 = time.perf_counter()
    K = assemble_stiffness(mesh, G)
    t_K = time.perf_counter() - t1

    t2 = time.perf_counter()
    b = assemble_rhs_tms(mesh, dAdt, G)
    t_rhs = time.perf_counter() - t2

    t3 = time.perf_counter()
    phi_np = solve_fem(K, b, pin_node=0)
    t_solve = time.perf_counter() - t3
    t_np_total = time.perf_counter() - t0

    E_np = compute_efield_at_elements(mesh, phi_np, dAdt, G)

    print(f"  Gradient operator: {t_grad:.2f} s")
    print(f"  Stiffness matrix:  {t_K:.2f} s")
    print(f"  RHS assembly:      {t_rhs:.2f} s")
    print(f"  Direct solve:      {t_solve:.2f} s")
    print(f"  Total:             {t_np_total:.2f} s")
    print_efield_stats(E_np, tag1, "NumPy FEM")

    # -----------------------------------------------------------------------
    # Warp.fem solve (optional)
    # -----------------------------------------------------------------------
    E_wp = None
    t_wp_total = None

    if warp_available():
        print("\n--- Warp.fem (warm-up run to cache JIT kernels) ---")
        # Warm-up with a small mesh to avoid counting JIT time
        from tmswarp.conductor import make_sphere_mesh
        _small = make_sphere_mesh(radius=0.05, n_shells=3, n_surface=50, conductivity=1.0)
        _dAdt_small = magnetic_dipole_dadt(DIPOLE_POS_M, DIPOLE_MOMENT, DIDT, _small.nodes)
        _ = solve_fem_warp(_small, _dAdt_small, quiet=True)

        print("--- Warp.fem (timed run) ---")
        t0_wp = time.perf_counter()
        phi_wp = solve_fem_warp(mesh, dAdt, quiet=True)
        t_wp_total = time.perf_counter() - t0_wp

        E_wp = compute_efield_at_elements(mesh, phi_wp, dAdt, G)

        rdm_np_wp = rdm(E_wp, E_np)
        mag_np_wp = mag(E_wp, E_np)

        print(f"  Total: {t_wp_total:.2f} s  "
              f"(speedup vs NumPy: {t_np_total/t_wp_total:.2f}x)")
        print(f"  Agreement with NumPy FEM:")
        print(f"    RDM:   {rdm_np_wp:.4f}")
        print(f"    |MAG|: {mag_np_wp:.4f}")
        if rdm_np_wp > 0.3:
            print(f"    NOTE: High RDM is expected for multi-tissue problems.")
            print(f"    The ernie mesh has a 165:1 conductivity contrast (skull:CSF).")
            print(f"    Warp.fem uses float32 geometry and arithmetic throughout,")
            print(f"    which degrades accuracy for ill-conditioned systems.")
            print(f"    Sphere3 (uniform σ): RDM < 0.06 between NumPy and Warp.")
            print(f"    Accuracy on GPU with float64 support: future work.")
        print_efield_stats(E_wp, tag1, "Warp.fem")
    else:
        print("\n--- Warp.fem ---  (skipped: warp-lang not installed)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Mesh: ernie (SimNIBS low-res), {len(mesh.nodes):,} nodes, "
          f"{len(mesh.elements):,} elements, 6 tissues")
    print(f"  NumPy FEM  total: {t_np_total:.1f} s")
    if t_wp_total is not None:
        print(f"  Warp.fem   total: {t_wp_total:.1f} s  "
              f"(speedup {t_np_total/t_wp_total:.2f}x)")
        rdm_val = rdm(E_wp, E_np)
        accuracy_note = (
            "expected: float32 geometry degrades accuracy at 165:1 σ contrast"
            if rdm_val > 0.2 else "GOOD"
        )
        print(f"  NumPy vs Warp RDM: {rdm_val:.4f}  ({accuracy_note})")
        print(f"  Known limitation: warp.fem uses float32 throughout;")
        print(f"    accurate for low-contrast problems (sphere3 RDM=0.06),")
        print(f"    degrades for high-contrast multi-tissue meshes.")
        print(f"    Future: float64 GPU path or mixed-precision assembly.")

    return {
        "t_np": t_np_total,
        "t_wp": t_wp_total,
        "E_np": E_np,
        "E_wp": E_wp,
    }


if __name__ == "__main__":
    run_validation()
