"""Run SimNIBS FEM on the ernie head mesh and save E-field to ernie_simnibs_efield.npz.

Must be run with SimNIBS Python:

    /path/to/SimNIBS/simnibs_env/bin/python scripts/run_simnibs_efield.py

By default the system is solved with SimNIBS's own solver (its default is
PETSc CG with a hypre BoomerAMG preconditioner).  Pass ``--solver scipy`` to
solve SimNIBS's assembled system with SciPy's sparse direct solver instead,
which was the behaviour of earlier versions of this script.

Prerequisite: ernie.msh must exist (extracted when fetch_ernie.py was run, or
available in the ernie_dataset/ directory).  The script searches common locations.

Uses the same dipole configuration as benchmarks/ernie_comparison.py:
    dipole_pos  = [0, 0, 200] mm  (above vertex)
    dipole_moment = [1, 0, 0]
    dI/dt = 1e6 A/s
    conductivities: SimNIBS defaults (WM=0.126, GM=0.275, CSF=1.654,
                    skull=0.010, scalp=0.465, eyes=0.500 S/m)

Output
------
ernie_simnibs_efield.npz  in the TMSWarp directory:
    E       float64 (N_elem, 3)  E-field in V/m at element barycenters
    bary_mm float64 (N_elem, 3)  element barycenters in mm
    tag1    int32   (N_elem,)    tissue tag
"""

import argparse
import os
import sys
import time

import numpy as np
import scipy.sparse.linalg as spalg

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument(
    "--solver", default="simnibs", choices=("simnibs", "scipy"),
    help="simnibs: SimNIBS's own default solver; scipy: spsolve on "
         "SimNIBS's assembled system",
)
parser.add_argument("--mesh", default=None, help="Path to ernie.msh")
args = parser.parse_args()

# Add SimNIBS to path
SIMNIBS_SITE = (
    "/Users/pieper/Applications/SimNIBS-4.5/"
    "simnibs_env/lib/python3.11/site-packages"
)
for candidate in [
    SIMNIBS_SITE,
    "/opt/SimNIBS-4.5/simnibs_env/lib/python3.11/site-packages",
]:
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)
        break

import simnibs
from simnibs.mesh_tools import mesh_io
from simnibs.simulation import fem as simfem

# ---------------------------------------------------------------------------
# Locate ernie.msh
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

ERNIE_CANDIDATES = [
    "/tmp/ernie_lowres/m2m_ernie/ernie.msh",
    os.path.join(ROOT, "ernie_dataset", "m2m_ernie", "ernie.msh"),
]

if args.mesh:
    ERNIE_CANDIDATES.insert(0, args.mesh)

ernie_path = None
for c in ERNIE_CANDIDATES:
    if os.path.isfile(c):
        ernie_path = c
        break

if ernie_path is None:
    raise FileNotFoundError(
        "ernie.msh not found. Run fetch_ernie.py first, or point ERNIE_CANDIDATES "
        "to the extracted ernie_lowres_V2.zip location."
    )

# ---------------------------------------------------------------------------
# Dipole / coil parameters (must match ernie_comparison.py)
# ---------------------------------------------------------------------------
DIPOLE_POS_MM = np.array([0.0, 0.0, 200.0])   # mm
DIPOLE_MOMENT = np.array([1.0, 0.0, 0.0])
DIDT = 1e6

CONDUCTIVITY_MAP = {
    1: 0.126, 2: 0.275, 3: 1.654,
    4: 0.010, 5: 0.465, 6: 0.500,
}

OUTPATH = os.path.join(ROOT, "ernie_simnibs_efield.npz")

# ---------------------------------------------------------------------------
# Load mesh
# ---------------------------------------------------------------------------
print(f"Loading: {ernie_path}")
m = mesh_io.read_msh(ernie_path)
m = m.crop_mesh(elm_type=4)
print(f"  {m.nodes.nr} nodes, {m.elm.nr} elements")

# ---------------------------------------------------------------------------
# dA/dt at nodes (Biot-Savart, same formula as TMSWarp's coil.py)
# ---------------------------------------------------------------------------
r = (m.nodes.node_coord - DIPOLE_POS_MM) * 1e-3     # mm → m
dAdt_vals = (
    1e-7 * DIDT * np.cross(DIPOLE_MOMENT, r)
    / (np.linalg.norm(r, axis=1, keepdims=True) ** 3)
)
dAdt_node = mesh_io.NodeData(dAdt_vals, mesh=m)
dAdt_node.field_name = "dAdt"

# ---------------------------------------------------------------------------
# Per-element conductivity
# ---------------------------------------------------------------------------
cond_vals = np.array(
    [CONDUCTIVITY_MAP.get(int(t), 0.275) for t in m.elm.tag1],
    dtype=np.float64,
)
cond = mesh_io.ElementData(cond_vals)
cond.mesh = m

# ---------------------------------------------------------------------------
# SimNIBS FEM solve
# ---------------------------------------------------------------------------
print("Assembling SimNIBS TMSFEM system ...")
t0 = time.perf_counter()
S = simfem.TMSFEM(m, cond)
b = S.assemble_rhs(dAdt_node)
t_assemble = time.perf_counter() - t0
print(f"  Assembly: {t_assemble:.2f} s")

t1 = time.perf_counter()
if args.solver == "simnibs":
    solver_name = (f"SimNIBS {simnibs.__version__} default solver "
                   f"({getattr(S, '_solver_options', 'default')})")
    print(f"Solving ({solver_name}) ...")
    x = S.solve(b)
else:
    solver_name = f"SimNIBS {simnibs.__version__} assembly, SciPy spsolve"
    print(f"Solving ({solver_name}) ...")
    # S.A is the full matrix, without the grounded node eliminated.  E does
    # not depend on the constant offset this leaves in the potential.
    x = spalg.spsolve(S.A, b)
t_solve = time.perf_counter() - t1
print(f"  Solve:    {t_solve:.2f} s")
print(f"  Total:    {t_assemble + t_solve:.2f} s")

# ---------------------------------------------------------------------------
# Compute E-field at element barycenters
# ---------------------------------------------------------------------------
v = mesh_io.NodeData(x, "v", mesh=m)
# gradient() returns V/mm (nodes in mm); convert to V/m
E_grad = -v.gradient().value * 1e3             # (N_elem, 3) V/m
dAdt_elem = dAdt_node.node_data2elm_data().value  # (N_elem, 3) V/m
E = E_grad - dAdt_elem                          # total E = -∇φ - ∂A/∂t

bary_mm = m.elements_baricenters().value        # (N_elem, 3) mm
tag1 = m.elm.tag1.astype(np.int32)

print(f"\n|E| statistics (V/m):")
mag_E = np.linalg.norm(E, axis=1)
print(f"  mean={mag_E.mean():.3f}  max={mag_E.max():.3f}  p99={np.percentile(mag_E, 99):.3f}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
np.savez_compressed(OUTPATH, E=E, bary_mm=bary_mm, tag1=tag1,
                    solver=np.array(solver_name))
size_mb = os.path.getsize(OUTPATH) / 1e6
print(f"\nSaved: {OUTPATH}  ({size_mb:.1f} MB)")
print("\nNow run the comparison figure:")
print("  pixi run python benchmarks/ernie_comparison.py")
