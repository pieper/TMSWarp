"""Solve one TMS problem with SimNIBS and record timings.

Run with SimNIBS's own Python; this file imports nothing from TMSWarp so
that it works in the SimNIBS environment:

    /path/to/SimNIBS/simnibs_env/bin/python simnibs_runner.py job.json result.json

The mesh comes from a TMSWarp ``.npz`` file and the dipole field is computed
with the same formula TMSWarp uses, so the comparison is between the FEM
solves and not between coil models.
"""

import json
import os
import sys
import time
import traceback

import numpy as np


def peak_rss_gb():
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is kilobytes on Linux, bytes on macOS
        return round(rss / (2**20 if sys.platform != "darwin" else 2**30), 2)
    except Exception:
        return None


def write(path, result):
    with open(path + ".part", "w") as f:
        json.dump(result, f, indent=1)
    os.replace(path + ".part", path)


def run(job, result_path):
    import simnibs
    from simnibs.mesh_tools import mesh_io
    from simnibs.simulation import fem

    d = np.load(job["mesh"])
    nodes_mm = d["nodes"] * 1e3
    msh = mesh_io.Msh(mesh_io.Nodes(nodes_mm),
                      mesh_io.Elements(tetrahedra=d["elements"] + 1))
    if "tag1" in d:
        msh.elm.tag1 = d["tag1"].astype(np.int32)
        msh.elm.tag2 = d["tag1"].astype(np.int32)
    cond = mesh_io.ElementData(d["conductivity"].astype(np.float64))
    cond.mesh = msh

    position = np.array(job["position"])
    moment = np.array(job["moment"])
    r = d["nodes"] - position
    dAdt = (1e-7 * job["didt"] * np.cross(moment, r)
            / np.linalg.norm(r, axis=1, keepdims=True) ** 3)
    dAdt = mesh_io.NodeData(dAdt, mesh=msh)
    dAdt.field_name = "dAdt"

    option = job.get("solver", "hypre")
    t0 = time.perf_counter()
    system = fem.TMSFEM(msh, cond, solver_options=option)
    t_assembly = time.perf_counter() - t0

    t0 = time.perf_counter()
    b = system.assemble_rhs(dAdt)
    t_rhs = time.perf_counter() - t0

    # Preconditioner setup (hypre) or factorization (pardiso, mumps)
    t0 = time.perf_counter()
    system.prepare_solver()
    t_setup = time.perf_counter() - t0

    times = []
    for _ in range(job.get("repeats", 3)):
        t0 = time.perf_counter()
        x = system.solve(b)
        times.append(time.perf_counter() - t0)

    result = {
        "simnibs": simnibs.__version__,
        "solver": option,
        "cpu_threads": os.cpu_count(),
        "t_assembly": t_assembly,
        "t_rhs": t_rhs,
        "t_setup": t_setup,
        "t_solve": {"best": float(min(times)),
                    "median": float(np.median(times)),
                    "samples": [float(t) for t in times]},
        "n_nodes": int(msh.nodes.nr),
        "n_elements": int(msh.elm.nr),
        "peak_rss_gb": peak_rss_gb(),
    }
    # The timings are safe now; computing the field needs more memory and
    # may fail on the largest meshes, leaving the job "partial".
    write(result_path, dict(result, status="partial", job=job))

    v = mesh_io.NodeData(x, "v", mesh=msh)
    E = -v.gradient().value * 1e3 - dAdt.node_data2elm_data().value
    np.savez(job["efield"], E=E)
    result["peak_rss_gb"] = peak_rss_gb()
    return result


def limit_memory(fraction=0.75):
    """Fail cleanly instead of exhausting the machine's memory."""
    try:
        import resource
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        limit = int(fraction * total)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception:
        pass


def main(argv):
    with open(argv[1]) as f:
        job = json.load(f)
    limit_memory()
    t0 = time.perf_counter()
    try:
        result = run(job, argv[2])
        result["status"] = "ok"
    except MemoryError:
        result = {"status": "out-of-memory", "error": traceback.format_exc()}
    except Exception:
        result = {"status": "error", "error": traceback.format_exc()}
    if result["status"] != "ok" and os.path.exists(argv[2]):
        # Keep the timings written before the failure
        with open(argv[2]) as f:
            partial = json.load(f)
        partial["error"] = result.get("error")
        result = partial
    result["job"] = job
    result["t_job_total"] = time.perf_counter() - t0
    write(argv[2], result)


if __name__ == "__main__":
    main(sys.argv)
