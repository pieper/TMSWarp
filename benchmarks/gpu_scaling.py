"""GPU scaling benchmark: find the crossover point where GPU beats CPU.

Generates meshes at increasing sizes and compares three solver configurations:
  1. NumPy FEM (scipy spsolve, direct solver)
  2. Warp.fem on CPU (CG iterative)
  3. Warp.fem on GPU (CG iterative)

Assembly and solve are timed separately. The GPU is confirmed to be in use
by printing the actual device of the assembled matrices.

Usage
-----
    python benchmarks/gpu_scaling.py
    python benchmarks/gpu_scaling.py --max-elements 500000
    python benchmarks/gpu_scaling.py --skip-numpy   # skip NumPy for very large meshes
"""

import argparse
import json
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tmswarp.coil import magnetic_dipole_dadt
from tmswarp.conductor import make_sphere_mesh
from tmswarp.solver import assemble_rhs_tms, assemble_stiffness, gradient_operator, solve_fem
from tmswarp.solver_warp import warp_available

# ---------------------------------------------------------------------------
# Problem setup (same as other benchmarks)
# ---------------------------------------------------------------------------
RADIUS = 0.095
DIPOLE_POS = np.array([0.0, 0.0, 0.3])
DIPOLE_MOMENT = np.array([1.0, 0.0, 0.0])
DIDT = 1e6
CONDUCTIVITY = 1.0

# Mesh configs: (n_shells, n_surface, min_quality)
# Each roughly doubles the element count.
CONFIGS = [
    (4,   100,   0.01),    # ~800 elements
    (6,   300,   0.01),    # ~3,700
    (8,   500,   0.01),    # ~8,200
    (10,  800,   0.01),    # ~16,000
    (12,  1200,  0.05),    # ~29,000
    (16,  2500,  0.05),    # ~80,000
    (20,  4000,  0.05),    # ~150,000
    (25,  6000,  0.05),    # ~300,000
    (30,  9000,  0.05),    # ~500,000+
]


# ---------------------------------------------------------------------------
# Timed solvers
# ---------------------------------------------------------------------------

def time_numpy_solver(mesh, dAdt_nodes):
    """Time NumPy FEM: returns (t_assembly, t_solve) or None if too large."""
    t0 = time.perf_counter()
    G = gradient_operator(mesh)
    K = assemble_stiffness(mesh, G)
    b = assemble_rhs_tms(mesh, dAdt_nodes, G)
    t_asm = time.perf_counter() - t0

    t0 = time.perf_counter()
    phi = solve_fem(K, b, pin_node=0)
    t_slv = time.perf_counter() - t0

    return t_asm, t_slv


def time_warp_solver(mesh, dAdt_nodes, device):
    """Time Warp.fem on a specific device: returns (t_assembly, t_solve, actual_device_str).

    Includes a warm-up call to exclude JIT compilation from timing.
    """
    import warp as wp
    import warp.fem as fem
    from warp.examples.fem.utils import bsr_cg

    from tmswarp.solver_warp import (
        _init_warp, _make_gauge_projector,
        _tms_rhs_form, _tms_stiffness_form,
    )

    _init_warp()

    def _run(mesh, dAdt_nodes, device):
        positions = wp.array(mesh.nodes.astype(np.float32), dtype=wp.vec3f, device=device)
        tet_indices = wp.array(mesh.elements.astype(np.int32), dtype=int, device=device)
        geo = fem.Tetmesh(tet_indices, positions)

        phi_space = fem.make_polynomial_space(geo, dtype=wp.float32, degree=1)
        dAdt_space = fem.make_polynomial_space(geo, dtype=wp.vec3f, degree=1)

        dAdt_discrete = dAdt_space.make_field()
        dAdt_discrete.dof_values = wp.array(
            dAdt_nodes.astype(np.float32), dtype=wp.vec3f, device=device
        )
        sigma_wp = wp.array(
            mesh.conductivity.astype(np.float32), dtype=wp.float32, device=device
        )

        domain = fem.Cells(geometry=geo)
        test = fem.make_test(space=phi_space, domain=domain)
        trial = fem.make_trial(space=phi_space, domain=domain)

        # Assembly
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

        actual_device = b.device
        n_nodes = len(mesh.nodes)
        projector = _make_gauge_projector(n_nodes, 0, device=actual_device)
        fixed_val = wp.zeros(n_nodes, dtype=wp.float32, device=actual_device)
        fem.project_linear_system(K, b, projector, fixed_val)
        wp.synchronize()
        t_asm = time.perf_counter() - t0

        # Solve
        x = wp.zeros(n_nodes, dtype=wp.float32, device=actual_device)
        t0 = time.perf_counter()
        bsr_cg(K, b=b, x=x, quiet=True)
        wp.synchronize()
        t_slv = time.perf_counter() - t0

        return t_asm, t_slv, str(actual_device)

    # Warm-up (JIT compile)
    _run(mesh, dAdt_nodes, device)

    # Timed run
    return _run(mesh, dAdt_nodes, device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPU scaling benchmark — find the CPU/GPU crossover point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--max-elements", type=int, default=600_000,
        help="Stop generating larger meshes after reaching this element count",
    )
    parser.add_argument(
        "--skip-numpy", action="store_true",
        help="Skip NumPy solver (useful for very large meshes where spsolve is slow)",
    )
    parser.add_argument(
        "--notes", default="",
        help="Free-text notes for result metadata",
    )
    args = parser.parse_args()

    if not warp_available():
        print("ERROR: warp-lang is required for this benchmark.")
        return

    import warp as wp
    from tmswarp.solver_warp import _init_warp
    _init_warp()

    has_gpu = any("cuda" in str(d) for d in wp.get_devices())
    if not has_gpu:
        print("WARNING: No CUDA GPU detected. Will only benchmark Warp on CPU.")

    # Print system info
    print(f"\n{'='*70}")
    print(f"TMSWarp GPU Scaling Benchmark")
    print(f"{'='*70}")
    print(f"  Host:     {socket.gethostname()}")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print(f"  Warp:     {wp.__version__}")
    print(f"  Devices:  {[str(d) for d in wp.get_devices()]}")
    if has_gpu:
        for d in wp.get_cuda_devices():
            print(f"  GPU:      {d.name}")
    print(f"  Max elem: {args.max_elements:,}")
    print()

    # Table header
    cols = "Elements     Nodes"
    if not args.skip_numpy:
        cols += "   NP Asm(s)  NP Slv(s)  NP Tot(s)"
    cols += "   WP CPU Asm  WP CPU Slv  WP CPU Tot"
    if has_gpu:
        cols += "   WP GPU Asm  WP GPU Slv  WP GPU Tot  Speedup"
    print(cols)
    print("-" * len(cols))

    results = []

    for n_shells, n_surface, min_q in CONFIGS:
        # Generate mesh
        t0 = time.perf_counter()
        mesh = make_sphere_mesh(
            radius=RADIUS, n_shells=n_shells, n_surface=n_surface,
            conductivity=CONDUCTIVITY, min_quality=min_q,
        )
        t_mesh = time.perf_counter() - t0
        n_elem = len(mesh.elements)
        n_nodes = len(mesh.nodes)

        if n_elem > args.max_elements:
            print(f"  Stopping: {n_elem:,} elements exceeds --max-elements {args.max_elements:,}")
            break

        print(f"  Meshing: {n_elem:,} elements, {n_nodes:,} nodes ({t_mesh:.1f}s)", flush=True)

        dAdt = magnetic_dipole_dadt(DIPOLE_POS, DIPOLE_MOMENT, DIDT, mesh.nodes)

        row = {
            "n_shells": n_shells, "n_surface": n_surface,
            "n_nodes": n_nodes, "n_elements": n_elem,
            "t_mesh_s": t_mesh,
        }

        # NumPy FEM
        if not args.skip_numpy:
            try:
                ta, ts = time_numpy_solver(mesh, dAdt)
                row["t_numpy_asm"] = ta
                row["t_numpy_slv"] = ts
                row["t_numpy_tot"] = ta + ts
            except Exception as e:
                print(f"    NumPy failed: {e}")
                row["t_numpy_asm"] = None
                row["t_numpy_slv"] = None
                row["t_numpy_tot"] = None

        # Warp CPU
        try:
            ta, ts, dev_str = time_warp_solver(mesh, dAdt, "cpu")
            row["t_warp_cpu_asm"] = ta
            row["t_warp_cpu_slv"] = ts
            row["t_warp_cpu_tot"] = ta + ts
            row["warp_cpu_device"] = dev_str
        except Exception as e:
            print(f"    Warp CPU failed: {e}")
            row["t_warp_cpu_asm"] = None
            row["t_warp_cpu_slv"] = None
            row["t_warp_cpu_tot"] = None

        # Warp GPU
        if has_gpu:
            try:
                ta, ts, dev_str = time_warp_solver(mesh, dAdt, "cuda:0")
                row["t_warp_gpu_asm"] = ta
                row["t_warp_gpu_slv"] = ts
                row["t_warp_gpu_tot"] = ta + ts
                row["warp_gpu_device"] = dev_str
            except Exception as e:
                print(f"    Warp GPU failed: {e}")
                row["t_warp_gpu_asm"] = None
                row["t_warp_gpu_slv"] = None
                row["t_warp_gpu_tot"] = None

        results.append(row)

        # Print row
        line = f"{n_elem:>8,}  {n_nodes:>8,}"
        if not args.skip_numpy and row.get("t_numpy_tot") is not None:
            line += f"  {row['t_numpy_asm']:>10.4f} {row['t_numpy_slv']:>10.4f} {row['t_numpy_tot']:>10.4f}"
        elif not args.skip_numpy:
            line += f"  {'N/A':>10} {'N/A':>10} {'N/A':>10}"
        if row.get("t_warp_cpu_tot") is not None:
            line += f"  {row['t_warp_cpu_asm']:>11.4f} {row['t_warp_cpu_slv']:>11.4f} {row['t_warp_cpu_tot']:>11.4f}"
        else:
            line += f"  {'N/A':>11} {'N/A':>11} {'N/A':>11}"
        if has_gpu:
            if row.get("t_warp_gpu_tot") is not None:
                line += f"  {row['t_warp_gpu_asm']:>11.4f} {row['t_warp_gpu_slv']:>11.4f} {row['t_warp_gpu_tot']:>11.4f}"
                # Speedup: CPU total / GPU total
                if row.get("t_warp_cpu_tot"):
                    speedup = row["t_warp_cpu_tot"] / row["t_warp_gpu_tot"]
                    line += f"  {speedup:>7.2f}x"
            else:
                line += f"  {'N/A':>11} {'N/A':>11} {'N/A':>11} {'N/A':>8}"
        print(line)

    # Save JSON
    meta = {
        "hostname": socket.gethostname(),
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "warp_version": wp.__version__,
        "devices": [str(d) for d in wp.get_devices()],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": args.notes,
    }
    cuda_devices = []
    try:
        for dev in wp.get_cuda_devices():
            info = {"name": dev.name, "device_id": str(dev)}
            if hasattr(dev, "total_memory"):
                info["memory_gb"] = round(dev.total_memory / (1024**3), 1)
            if hasattr(dev, "arch"):
                info["arch"] = str(dev.arch)
            cuda_devices.append(info)
    except Exception:
        pass
    if cuda_devices:
        meta["gpu_devices"] = cuda_devices

    output = {"metadata": meta, "results": results}

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    hostname_tag = socket.gethostname().split(".")[0]
    filename = f"{hostname_tag}_gpu_scaling_{ts_file}.json"
    outpath = results_dir / filename

    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {outpath}")

    # Print analysis
    print(f"\n{'='*70}")
    print("Analysis")
    print(f"{'='*70}")
    if has_gpu:
        for r in results:
            if r.get("t_warp_cpu_tot") and r.get("t_warp_gpu_tot"):
                speedup = r["t_warp_cpu_tot"] / r["t_warp_gpu_tot"]
                label = "GPU FASTER" if speedup > 1 else "CPU faster"
                print(f"  {r['n_elements']:>8,} elements: GPU {speedup:.2f}x ({label})")
        print()
        print("  Note: GPU typically dominates at >100k elements for FEM assembly,")
        print("  but CG solve speedup depends on problem conditioning and memory bandwidth.")
        print("  The crossover point varies by GPU model.")
    else:
        print("  No GPU detected — cannot compute CPU/GPU crossover.")


if __name__ == "__main__":
    main()
