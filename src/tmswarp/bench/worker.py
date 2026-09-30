"""Run one benchmark job in its own process.

    python -m tmswarp.bench.worker job.json result.json

Each job is isolated so that an out-of-memory failure or a crash on one
mesh does not stop the rest of the suite, and so that GPU memory numbers
belong to one job only.
"""

import json
import os
import sys
import time
import traceback

import numpy as np

from tmswarp.bench import data
from tmswarp.coil import magnetic_dipole_dadt
from tmswarp.conductor import element_barycenters
from tmswarp.fields import compute_efield_at_elements, mag, rdm


def _log(msg):
    print(msg, flush=True)


def _reference_path(dataset, kind):
    return data.cache_dir() / f"{dataset}.reference-{kind}.npz"


def _stats(times):
    return {"best": float(min(times)), "median": float(np.median(times)),
            "samples": [float(t) for t in times]}


def _accuracy(E, dataset, kind, tag1, mesh, position, moment, didt):
    """Compare a field with the float64 reference and, for spheres, with
    the analytical solution."""
    out = {}
    ref_path = _reference_path(dataset, kind)
    if ref_path.exists():
        E_ref = np.load(ref_path)["E"]
        out["rdm_vs_reference"] = rdm(E, E_ref)
        out["mag_vs_reference"] = mag(E, E_ref)
        if tag1 is not None:
            gm = tag1 == 2
            out["rdm_vs_reference_gm"] = rdm(E[gm], E_ref[gm])
            out["mag_vs_reference_gm"] = mag(E[gm], E_ref[gm])
    if data.is_sphere(dataset):
        from tmswarp.analytical import tms_analytical_efield
        E_ana = tms_analytical_efield(
            position, moment, didt, element_barycenters(mesh)
        )
        out["rdm_vs_analytical"] = rdm(E, E_ana)
        out["mag_vs_analytical"] = mag(E, E_ana)
    norm = np.linalg.norm(E, axis=1)
    out["enorm_mean"] = float(norm.mean())
    out["enorm_max"] = float(norm.max())
    return out


# ---------------------------------------------------------------------------
# CPU, float64: NumPy assembly with SciPy solvers
# ---------------------------------------------------------------------------

def _numpy_system(mesh, dAdt):
    from tmswarp.solver import (assemble_rhs_tms, assemble_stiffness,
                                gradient_operator)
    t0 = time.perf_counter()
    G = gradient_operator(mesh)
    K = assemble_stiffness(mesh, G)
    t_assembly = time.perf_counter() - t0
    t0 = time.perf_counter()
    b = assemble_rhs_tms(mesh, dAdt, G)
    t_rhs = time.perf_counter() - t0
    return G, K, b, t_assembly, t_rhs


def _scipy_cg(K, b, rtol, maxiter=200000):
    from scipy.sparse.linalg import LinearOperator, cg
    Kr = K.tocsr()[1:, 1:].tocsr()
    dinv = 1.0 / Kr.diagonal()
    M = LinearOperator(Kr.shape, matvec=lambda v: dinv * v)
    count = [0]

    def callback(_):
        count[0] += 1

    t0 = time.perf_counter()
    try:
        x, info = cg(Kr, b[1:], rtol=rtol, atol=0.0, maxiter=maxiter, M=M,
                     callback=callback)
    except TypeError:  # SciPy < 1.12 calls the relative tolerance "tol"
        x, info = cg(Kr, b[1:], tol=rtol, atol=0.0, maxiter=maxiter, M=M,
                     callback=callback)
    elapsed = time.perf_counter() - t0
    phi = np.zeros(K.shape[0])
    phi[1:] = x
    return phi, count[0], elapsed, info == 0


def job_reference(job, mesh, tag1, dAdt, dipole):
    """float64 Jacobi-CG to rtol=1e-10; cached as the accuracy reference."""
    path = _reference_path(job["dataset"], job["dipole"])
    G, K, b, t_assembly, t_rhs = _numpy_system(mesh, dAdt)
    phi, iters, t_solve, converged = _scipy_cg(K, b, 1e-10)
    E = compute_efield_at_elements(mesh, phi, dAdt, G)
    np.savez(path, E=E)
    result = {"t_assembly": t_assembly, "t_rhs": t_rhs, "t_solve": t_solve,
              "iterations": iters, "converged": converged, "rtol": 1e-10}
    result.update(_accuracy(E, job["dataset"], job["dipole"], tag1, mesh, *dipole))
    return result


def job_scipy_cg(job, mesh, tag1, dAdt, dipole):
    G, K, b, t_assembly, t_rhs = _numpy_system(mesh, dAdt)
    runs = []
    for rtol in job["rtols"]:
        phi, iters, t_solve, converged = _scipy_cg(K, b, rtol)
        E = compute_efield_at_elements(mesh, phi, dAdt, G)
        run = {"rtol": rtol, "t_solve": _stats([t_solve]), "iterations": iters,
               "converged": converged}
        run.update(_accuracy(E, job["dataset"], job["dipole"], tag1, mesh, *dipole))
        runs.append(run)
    return {"t_assembly": t_assembly, "t_rhs": t_rhs, "runs": runs}


def job_numpy_direct(job, mesh, tag1, dAdt, dipole):
    from scipy.sparse.linalg import factorized
    G, K, b, t_assembly, t_rhs = _numpy_system(mesh, dAdt)
    t0 = time.perf_counter()
    solve = factorized(K.tocsr()[1:, 1:].tocsc())
    t_factor = time.perf_counter() - t0
    times = []
    for _ in range(job.get("repeats", 3)):
        t0 = time.perf_counter()
        x = solve(b[1:])
        times.append(time.perf_counter() - t0)
    phi = np.zeros(K.shape[0])
    phi[1:] = x
    E = compute_efield_at_elements(mesh, phi, dAdt, G)
    result = {"t_assembly": t_assembly, "t_rhs": t_rhs,
              "t_factorization": t_factor, "t_solve": _stats(times)}
    result.update(_accuracy(E, job["dataset"], job["dipole"], tag1, mesh, *dipole))
    return result


# ---------------------------------------------------------------------------
# Warp, float32, CPU or GPU
# ---------------------------------------------------------------------------

def _warp_warmup(device):
    """Compile and load every kernel on a tiny mesh; return the time taken."""
    from tmswarp.conductor import make_sphere_mesh
    from tmswarp.solver_warp import WarpFEMContext
    t0 = time.perf_counter()
    small = make_sphere_mesh(radius=0.05, n_shells=3, n_surface=50,
                             conductivity=1.0)
    dAdt = magnetic_dipole_dadt(np.array([0.0, 0.0, 0.1]),
                                np.array([1.0, 0.0, 0.0]), 1e6, small.nodes)
    ctx = WarpFEMContext(small, device=device)
    ctx.solve(dAdt, rtol=1e-3)
    ctx.compute_enorm()
    ctx.objective_and_gradient([0.0, 0.0, 0.1], [1.0, 0.0, 0.0], 0)
    return time.perf_counter() - t0


def _gpu_memory_gb(wp, device):
    dev = wp.get_device(device)
    if not dev.is_cuda:
        return None
    try:
        return round(wp.get_mempool_used_mem_high(dev) / 2**30, 3)
    except Exception:
        return round((dev.total_memory - dev.free_memory) / 2**30, 3)


def _target_element(dataset, mesh, tag1, position):
    """A deterministic target element some 30 mm below the near dipole."""
    bary = element_barycenters(mesh)
    point = position - np.array([0.0, 0.0, 0.040])
    candidates = np.arange(len(bary))
    if tag1 is not None and np.any(tag1 == 2):
        candidates = np.nonzero(tag1 == 2)[0]
    return int(candidates[np.argmin(
        np.linalg.norm(bary[candidates] - point, axis=1))])


def job_warp(job, mesh, tag1, dAdt, dipole):
    import warp as wp
    wp.config.quiet = True
    from tmswarp.solver import gradient_operator
    from tmswarp.solver_warp import WarpFEMContext

    device = job["device"]
    repeats = job.get("repeats", 3)
    max_iters = job.get("max_iters", 100000)
    position, moment, didt = dipole

    result = {"device": device, "t_kernel_warmup": _warp_warmup(device)}

    t0 = time.perf_counter()
    ctx = WarpFEMContext(mesh, device=device)
    wp.synchronize_device(device)
    result["t_assembly"] = time.perf_counter() - t0
    result["actual_device"] = str(ctx._device)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        ctx.set_rhs(dAdt)
        wp.synchronize_device(device)
        times.append(time.perf_counter() - t0)
    result["t_rhs"] = _stats(times)

    G = gradient_operator(mesh)
    runs = []
    for rtol in job["rtols"]:
        ctx._ensure_diff_state()
        times = []
        for _ in range(repeats):
            ctx.x.zero_()
            t0 = time.perf_counter()
            err, iters = ctx._solve_relative(ctx.b, ctx.x, rtol, max_iters)
            wp.synchronize_device(device)
            times.append(time.perf_counter() - t0)
        E = compute_efield_at_elements(mesh, ctx.get_phi(), dAdt, G)
        run = {"rtol": rtol, "t_solve": _stats(times), "iterations": iters,
               "residual": err, "converged": bool(err <= rtol),
               "ms_per_iteration": 1e3 * min(times) / max(iters, 1)}
        run.update(_accuracy(E, job["dataset"], job["dipole"], tag1, mesh, *dipole))
        runs.append(run)
        _log(f"    rtol={rtol:g}: {iters} iterations, {min(times):.3f} s, "
             f"RDM {run.get('rdm_vs_reference', float('nan')):.4f}")
    result["runs"] = runs

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        ctx.compute_enorm()
        times.append(time.perf_counter() - t0)
    result["t_enorm_readback"] = _stats(times)

    # Gradient of |E| at a target: one cold evaluation, then warm-started
    # evaluations as the coil moves in 1 mm steps
    if job.get("gradient", True):
        near_pos, near_mom, _ = data.dipole(job["dataset"], mesh, "near")
        target = _target_element(job["dataset"], mesh, tag1, near_pos)
        rtol = job.get("gradient_rtol", 1e-3)
        ctx.x.zero_()
        t0 = time.perf_counter()
        loss, _, _ = ctx.objective_and_gradient(
            near_pos, near_mom, target, didt=didt, rtol=rtol, max_iters=max_iters)
        t_cold = time.perf_counter() - t0
        warm = []
        for k in range(1, 6):
            t0 = time.perf_counter()
            ctx.objective_and_gradient(
                near_pos + np.array([0.001 * k, 0.0, 0.0]), near_mom, target,
                didt=didt, rtol=rtol, max_iters=max_iters)
            warm.append(time.perf_counter() - t0)
        result["gradient"] = {"rtol": rtol, "target_element": target,
                              "enorm_at_target": -loss, "t_cold": t_cold,
                              "t_warm": _stats(warm)}

    result["gpu_memory_gb"] = _gpu_memory_gb(wp, device)
    return result


# ---------------------------------------------------------------------------
# Coil optimization as run by SlicerTMS (needs SlicerTMS's TMSService.py)
# ---------------------------------------------------------------------------

def job_optimization(job, mesh, tag1, dAdt, dipole):
    import contextlib
    import importlib.util
    import io

    import warp as wp
    wp.config.quiet = True
    from tmswarp.solver_warp import WarpFEMContext

    spec = importlib.util.spec_from_file_location(
        "tmsservice_bench", job["tmsservice"])
    service_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(service_module)

    device = job["device"]
    _warp_warmup(device)
    service = service_module.TMSService()
    service._drain_stdin_nonblocking = lambda: None
    service.mesh = mesh
    service._tag1 = tag1
    service._solver = "warp_gpu"
    service._warp_ctx = WarpFEMContext(mesh, device=device)
    service._build_surface_data()

    # Reference field at the final position, for the accuracy of the result
    G, K, _, _, _ = _numpy_system(mesh, dAdt)

    def reference_enorm(pos_m, normal, elem):
        from tmswarp.solver import assemble_rhs_tms
        d = magnetic_dipole_dadt(pos_m, normal, 1e6, mesh.nodes)
        phi, _, _, _ = _scipy_cg(K, assemble_rhs_tms(mesh, d, G), 1e-10)
        return float(np.linalg.norm(
            compute_efield_at_elements(mesh, phi, d, G)[elem]))

    runs = []
    for target_mm in job["targets_mm"]:
        service._warp_ctx.x.zero_()
        service._last_probe_mat = None
        out = io.StringIO()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(out):
            service._optimize_coil(target_mm)
        elapsed = time.perf_counter() - t0
        lines = out.getvalue().splitlines()
        iterations = None
        for line in lines:
            if line.startswith("OPTIMIZE_DONE"):
                iterations = int(line.split("iter=")[1].split()[0]) + 1
        mat = service._last_probe_mat
        elem = service._find_nearest_element(np.array(target_mm) * 1e-3)
        found = float(service._warp_ctx.compute_enorm()[elem])
        runs.append({
            "target_mm": target_mm, "t_total": elapsed,
            "adam_iterations": iterations,
            "coil_position_mm": [float(v) for v in mat[:3, 3]],
            "enorm_at_target": found,
            "enorm_at_target_reference": reference_enorm(
                mat[:3, 3] * 1e-3, mat[:3, 2], elem),
        })
        _log(f"    target {target_mm}: {elapsed:.2f} s, |E| = {found:.2f} V/m")
    return {"device": device, "runs": runs,
            "opt_rtol": getattr(service_module, "OPT_RTOL", None)}


# ---------------------------------------------------------------------------

JOBS = {
    "reference": job_reference,
    "scipy-cg": job_scipy_cg,
    "numpy-direct": job_numpy_direct,
    "warp": job_warp,
    "optimization": job_optimization,
}


def run(job):
    mesh, tag1, _ = data.load(job["dataset"], log=_log)
    dipole = data.dipole(job["dataset"], mesh, job["dipole"])
    dAdt = magnetic_dipole_dadt(dipole[0], dipole[1], dipole[2], mesh.nodes)
    result = JOBS[job["kind"]](job, mesh, tag1, dAdt, dipole)
    result["n_nodes"] = int(len(mesh.nodes))
    result["n_elements"] = int(len(mesh.elements))
    return result


def main(argv):
    job_path, result_path = argv[1], argv[2]
    with open(job_path) as f:
        job = json.load(f)
    t0 = time.perf_counter()
    try:
        result = run(job)
        result["status"] = "ok"
    except MemoryError:
        result = {"status": "out-of-memory", "error": traceback.format_exc()}
    except Exception as exc:
        status = "out-of-memory" if "out of memory" in str(exc).lower() else "error"
        result = {"status": status, "error": traceback.format_exc()}
    result["job"] = job
    result["t_job_total"] = time.perf_counter() - t0
    tmp = result_path + ".part"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=1)
    os.replace(tmp, result_path)


if __name__ == "__main__":
    main(sys.argv)
