"""TMSWarp benchmark: timing and accuracy for all solver implementations.

Runs the analytical, NumPy FEM, and Warp.fem solvers across a range of mesh
sizes, saves results to a machine-stamped JSON file in benchmarks/results/.

Usage
-----
    pixi run python benchmarks/run_benchmarks.py              # CPU (default)
    pixi run python benchmarks/run_benchmarks.py --device cuda:0   # GPU
    pixi run python benchmarks/run_benchmarks.py --repeats 5       # more timing repeats

The minimum wall-clock time across repeats is recorded (standard benchmark
practice — eliminates OS scheduling jitter while preserving true floor).

Output JSON schema
------------------
{
  "metadata": {
    "hostname": str,
    "platform": str,        # e.g. "darwin arm64"
    "python": str,
    "numpy_version": str,
    "scipy_version": str,
    "warp_version": str | null,
    "warp_device": str,     # e.g. "cpu (arm)" or "cuda:0 (NVIDIA A100)"
    "timestamp": str,       # ISO-8601
    "repeats": int,
    "notes": str
  },
  "configs": [
    {
      "n_shells": int,
      "n_surface": int,
      "min_quality": float,
      "n_nodes": int,
      "n_elements": int,
      "t_numpy_assembly_s": float,   # gradient_operator + assemble_stiffness + assemble_rhs
      "t_numpy_solve_s":    float,   # solve_fem (direct)
      "t_numpy_total_s":    float,
      "t_warp_assembly_s":  float | null,  # fem.integrate x2 + project
      "t_warp_solve_s":     float | null,  # bsr_cg
      "t_warp_total_s":     float | null,
      "rdm_numpy":  float,
      "mag_numpy":  float,
      "rdm_warp":   float | null,
      "mag_warp":   float | null
    },
    ...
  ]
}
"""

import argparse
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Solver imports
# ---------------------------------------------------------------------------
from tmswarp.analytical import tms_analytical_efield
from tmswarp.coil import magnetic_dipole_dadt
from tmswarp.conductor import element_barycenters, make_sphere_mesh
from tmswarp.fields import compute_efield_at_elements, mag, rdm
from tmswarp.solver import assemble_rhs_tms, assemble_stiffness, gradient_operator, solve_fem
from tmswarp.solver_warp import solve_fem_warp, warp_available

# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------
RADIUS = 0.095
DIPOLE_POS = np.array([0.0, 0.0, 0.3])
DIPOLE_MOMENT = np.array([1.0, 0.0, 0.0])
DIDT = 1e6
CONDUCTIVITY = 1.0

CONFIGS = [
    # (n_shells, n_surface, min_quality)  — matches visualize_convergence.py
    (4,  100,  0.01),
    (6,  300,  0.01),
    (8,  500,  0.01),
    (10, 800,  0.01),
    (12, 1200, 0.05),
]


# ---------------------------------------------------------------------------
# Per-solver timed runners
# ---------------------------------------------------------------------------

def time_numpy(mesh, dAdt_nodes, repeats):
    """Return (assembly_min, solve_min, E_fem) measured over `repeats` runs."""
    t_assemble_min = float("inf")
    t_solve_min = float("inf")

    for _ in range(repeats):
        t0 = time.perf_counter()
        G = gradient_operator(mesh)
        K = assemble_stiffness(mesh, G)
        b = assemble_rhs_tms(mesh, dAdt_nodes, G)
        t_assemble_min = min(t_assemble_min, time.perf_counter() - t0)

        t0 = time.perf_counter()
        phi = solve_fem(K, b, pin_node=0)
        t_solve_min = min(t_solve_min, time.perf_counter() - t0)

    E_fem = compute_efield_at_elements(mesh, phi, dAdt_nodes, G)
    return t_assemble_min, t_solve_min, E_fem


def time_warp(mesh, dAdt_nodes, device, repeats):
    """Return (assembly_min, solve_min, E_fem) or (None, None, None) if unavailable.

    Uses a patched version of solve_fem_warp that times internals separately.
    On first call kernels may be compiled — the warm-up run is excluded from timing.
    """
    if not warp_available():
        return None, None, None

    import warp as wp
    import warp.fem as fem
    from warp.examples.fem.utils import bsr_cg
    from warp.sparse import bsr_set_from_triplets, bsr_zeros

    from tmswarp.solver_warp import _init_warp, _tms_rhs_form, _tms_stiffness_form, _make_gauge_projector

    _init_warp()

    def _build_and_solve(mesh, dAdt_nodes, device):
        positions = wp.array(mesh.nodes.astype(np.float32), dtype=wp.vec3f, device=device)
        tet_indices = wp.array(mesh.elements.astype(np.int32), dtype=int, device=device)
        geo = fem.Tetmesh(tet_indices, positions)

        phi_space = fem.make_polynomial_space(geo, dtype=wp.float32, degree=1)
        dAdt_space = fem.make_polynomial_space(geo, dtype=wp.vec3f, degree=1)

        dAdt_discrete = dAdt_space.make_field()
        dAdt_discrete.dof_values = wp.array(
            dAdt_nodes.astype(np.float32), dtype=wp.vec3f, device=device
        )
        sigma_wp = wp.array(mesh.conductivity.astype(np.float32), dtype=wp.float32, device=device)

        domain = fem.Cells(geometry=geo)
        test = fem.make_test(space=phi_space, domain=domain)
        trial = fem.make_trial(space=phi_space, domain=domain)

        # -- assembly --
        t0 = time.perf_counter()
        K = fem.integrate(
            _tms_stiffness_form,
            fields={"u": trial, "v": test},
            values={"sigma": sigma_wp},
            output_dtype=wp.float32,
        )
        b = fem.integrate(
            _tms_rhs_form,
            fields={"v": test, "dAdt": dAdt_discrete},
            values={"sigma": sigma_wp},
            output_dtype=wp.float32,
        )
        n_nodes = len(mesh.nodes)
        projector = _make_gauge_projector(n_nodes, 0, device=device)
        fixed_val = wp.zeros(n_nodes, dtype=wp.float32, device=device)
        fem.project_linear_system(K, b, projector, fixed_val)
        wp.synchronize()
        t_assemble = time.perf_counter() - t0

        # -- solve --
        x = wp.zeros(n_nodes, dtype=wp.float32, device=device)
        t0 = time.perf_counter()
        bsr_cg(K, b=b, x=x, quiet=True)
        wp.synchronize()
        t_solve = time.perf_counter() - t0

        phi = x.numpy().astype(np.float64)
        return t_assemble, t_solve, phi

    # Warm-up run to trigger any remaining JIT compilation
    _build_and_solve(mesh, dAdt_nodes, device)

    t_assemble_min = float("inf")
    t_solve_min = float("inf")
    phi = None

    for _ in range(repeats):
        ta, ts, phi = _build_and_solve(mesh, dAdt_nodes, device)
        t_assemble_min = min(t_assemble_min, ta)
        t_solve_min = min(t_solve_min, ts)

    from tmswarp.solver import gradient_operator
    G = gradient_operator(mesh)
    E_fem = compute_efield_at_elements(mesh, phi, dAdt_nodes, G)
    return t_assemble_min, t_solve_min, E_fem


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_metadata(device, repeats, notes):
    """Gather machine / environment info."""
    warp_ver = None
    warp_device_str = "unavailable"
    if warp_available():
        import warp as wp
        warp_ver = wp.__version__
        try:
            devs = wp.get_devices()
            if device in devs:
                d = wp.get_device(device)
                warp_device_str = f"{device} ({d.name})"
            else:
                warp_device_str = f"{device} (not found — available: {devs})"
        except Exception:
            warp_device_str = device

    import scipy
    return {
        "hostname": socket.gethostname(),
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "warp_version": warp_ver,
        "warp_device": warp_device_str,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repeats": repeats,
        "notes": notes,
    }


def run_benchmarks(device="cpu", repeats=3, notes=""):
    meta = collect_metadata(device, repeats, notes)
    print(f"\nTMSWarp Benchmark")
    print(f"  Host:     {meta['hostname']}")
    print(f"  Platform: {meta['platform']}")
    print(f"  Warp:     {meta['warp_version']} on {meta['warp_device']}")
    print(f"  Repeats:  {repeats} (minimum time recorded)\n")

    configs_out = []

    header = (
        f"{'Elems':>8}  "
        f"{'NumPy Asm':>10}  {'NumPy Slv':>10}  {'NumPy Tot':>10}  "
        f"{'Warp Asm':>10}  {'Warp Slv':>10}  {'Warp Tot':>10}  "
        f"{'RDM np':>8}  {'RDM wp':>8}"
    )
    print(header)
    print("-" * len(header))

    for n_shells, n_surface, min_q in CONFIGS:
        mesh = make_sphere_mesh(
            radius=RADIUS, n_shells=n_shells, n_surface=n_surface,
            conductivity=CONDUCTIVITY, min_quality=min_q,
        )
        dAdt = magnetic_dipole_dadt(DIPOLE_POS, DIPOLE_MOMENT, DIDT, mesh.nodes)
        bary = element_barycenters(mesh)
        E_ana = tms_analytical_efield(DIPOLE_POS, DIPOLE_MOMENT, DIDT, bary)

        ta_np, ts_np, E_np = time_numpy(mesh, dAdt, repeats)
        ta_wp, ts_wp, E_wp = time_warp(mesh, dAdt, device, repeats)

        rdm_np = rdm(E_np, E_ana)
        mag_np = mag(E_np, E_ana)
        rdm_wp = rdm(E_wp, E_ana) if E_wp is not None else None
        mag_wp = mag(E_wp, E_ana) if E_wp is not None else None

        ne = len(mesh.elements)
        wp_asm = f"{ta_wp:.4f}" if ta_wp is not None else "   N/A"
        wp_slv = f"{ts_wp:.4f}" if ts_wp is not None else "   N/A"
        wp_tot = f"{(ta_wp+ts_wp):.4f}" if ta_wp is not None else "   N/A"
        rdm_wp_s = f"{rdm_wp:.4f}" if rdm_wp is not None else "   N/A"

        print(
            f"{ne:>8}  "
            f"{ta_np:>10.4f}  {ts_np:>10.4f}  {(ta_np+ts_np):>10.4f}  "
            f"{wp_asm:>10}  {wp_slv:>10}  {wp_tot:>10}  "
            f"{rdm_np:>8.4f}  {rdm_wp_s:>8}"
        )

        configs_out.append({
            "n_shells": n_shells,
            "n_surface": n_surface,
            "min_quality": min_q,
            "n_nodes": len(mesh.nodes),
            "n_elements": ne,
            "t_numpy_assembly_s": ta_np,
            "t_numpy_solve_s": ts_np,
            "t_numpy_total_s": ta_np + ts_np,
            "t_warp_assembly_s": ta_wp,
            "t_warp_solve_s": ts_wp,
            "t_warp_total_s": (ta_wp + ts_wp) if ta_wp is not None else None,
            "rdm_numpy": rdm_np,
            "mag_numpy": mag_np,
            "rdm_warp": rdm_wp,
            "mag_warp": mag_wp,
        })

    result = {"metadata": meta, "configs": configs_out}

    # Save to benchmarks/results/<hostname>_<platform>_<device>_<timestamp>.json
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    platform_tag = platform.machine().lower()
    device_tag = device.replace(":", "")
    hostname_tag = socket.gethostname().split(".")[0]  # strip domain
    filename = f"{hostname_tag}_{platform_tag}_{device_tag}_{ts_file}.json"
    outpath = results_dir / filename

    with open(outpath, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to: {outpath}")
    return result, outpath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TMSWarp solver benchmark — saves JSON to benchmarks/results/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Warp device (e.g. 'cpu', 'cuda:0')",
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Number of timing repeats per mesh (minimum is recorded)",
    )
    parser.add_argument(
        "--notes", default="",
        help="Free-text notes appended to the result metadata",
    )
    args = parser.parse_args()

    run_benchmarks(device=args.device, repeats=args.repeats, notes=args.notes)
