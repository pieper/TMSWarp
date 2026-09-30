"""Turn benchmark result files into Markdown tables."""


def _fmt_time(seconds):
    if seconds is None:
        return "—"
    if seconds < 0.1:
        return f"{1e3 * seconds:.0f} ms"
    if seconds < 100:
        return f"{seconds:.2f} s"
    return f"{seconds:.0f} s"


def _fmt(value, spec=".4f"):
    return "—" if value is None else format(value, spec)


def _fmt_tol(rtol):
    mantissa, exponent = f"{rtol:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _count(n):
    if n >= 1e6:
        return f"{n / 1e6:.2f} M"
    if n >= 1e3:
        return f"{n / 1e3:.0f} k"
    return str(n)


def _table(header, rows):
    if not rows:
        return "_No results._\n"
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def machine_name(run):
    m = run["machine"]
    gpu = m["gpus"][0]["name"] if m.get("gpus") else "no GPU"
    label = m.get("label") or m.get("hostname")
    provider = f", {m['provider']}" if m.get("provider") else ""
    return f"{label} ({gpu}{provider})"


def _jobs(run, kind, dataset=None, **match):
    out = []
    for r in run["jobs"]:
        job = r.get("job", {})
        if job.get("kind") != kind or r.get("status") != "ok":
            continue
        if dataset and job.get("dataset") != dataset:
            continue
        if all(job.get(k) == v for k, v in match.items()):
            out.append(r)
    return out


def _first(items):
    return items[0] if items else None


def _warp_run(result, rtol):
    if not result:
        return None
    for r in result["runs"]:
        if abs(r["rtol"] - rtol) < 1e-12 * max(1.0, rtol):
            return r
    return None


def _gpu_device(run):
    for r in run["jobs"]:
        device = r.get("job", {}).get("device", "")
        if device.startswith("cuda"):
            return device
    return None


def render_machine(run):
    m = run["machine"]
    gpus = ", ".join(f"{g['name']} ({g['memory_gb']} GB)" for g in m.get("gpus", []))
    rows = [
        ["Label", m.get("label") or "—"],
        ["Provider", m.get("provider") or "—"],
        ["CPU", f"{m.get('cpu')}, {m.get('cpu_threads')} threads"],
        ["Memory", f"{m.get('memory_gb')} GB"],
        ["GPU", gpus or "none"],
        ["OS", m.get("os")],
        ["Warp", m.get("warp") or "not installed"],
        ["TMSWarp", f"{m.get('tmswarp')} ({m.get('tmswarp_commit') or 'no commit'})"],
        ["Suite", m.get("suite")],
        ["Date (UTC)", m.get("timestamp")],
    ]
    return _table(["", ""], rows)


def render_solves(run):
    """One row per dataset and solver."""
    rows = []
    for name, info in run["datasets"].items():
        if "n_nodes" not in info:
            continue
        size = f"{_count(info['n_nodes'])} nodes, {_count(info['n_elements'])} tets"
        for r in _jobs(run, "simnibs", name):
            setup = "factorization" if r["solver"] in ("pardiso", "mumps") else "setup"
            rows.append([name, size, f"SimNIBS {r['simnibs']} {r['solver']}", "CPU",
                         "—", f"{_fmt_time(r['t_assembly'])} + {_fmt_time(r['t_setup'])} {setup}",
                         _fmt_time(r["t_solve"]["best"]), "—",
                         _fmt(r.get("rdm_vs_reference"))])
        for r in _jobs(run, "numpy-direct", name):
            rows.append([name, size, "SciPy direct, float64", "CPU", "—",
                         f"{_fmt_time(r['t_assembly'])} + "
                         f"{_fmt_time(r['t_factorization'])} factorization",
                         _fmt_time(r["t_solve"]["best"]), "—",
                         _fmt(r.get("rdm_vs_reference"))])
        for r in _jobs(run, "scipy-cg", name):
            for sub in r["runs"]:
                rows.append([name, size, "SciPy CG, float64", "CPU",
                             _fmt_tol(sub["rtol"]), _fmt_time(r["t_assembly"]),
                             _fmt_time(sub["t_solve"]["best"]), sub["iterations"],
                             _fmt(sub.get("rdm_vs_reference"))])
        for r in _jobs(run, "warp", name):
            device = "GPU" if r["actual_device"].startswith("cuda") else "CPU"
            for sub in r["runs"]:
                rows.append([name, size, "Warp CG, float32", device,
                             _fmt_tol(sub["rtol"]), _fmt_time(r["t_assembly"]),
                             _fmt_time(sub["t_solve"]["best"]), sub["iterations"],
                             _fmt(sub.get("rdm_vs_reference"))])
    return _table(["Mesh", "Size", "Solver", "Device", "Tolerance", "Setup",
                   "Solve", "Iterations", "RDM vs reference"], rows)


def render_resolution(run):
    """Accuracy against mesh resolution."""
    rows = []
    device = _gpu_device(run) or "cpu"
    for name, info in run["datasets"].items():
        if "n_nodes" not in info:
            continue
        ref = _first(_jobs(run, "reference", name))
        if not ref or "rdm_vs_analytical" not in ref:
            continue
        warp = _warp_run(_first(_jobs(run, "warp", name, device=device)), 1e-6)
        rows.append([name, _count(info["n_elements"]),
                     _fmt(ref["rdm_vs_analytical"]), _fmt(ref["mag_vs_analytical"]),
                     _fmt(warp["rdm_vs_analytical"]) if warp else "—",
                     _fmt_time(warp["t_solve"]["best"]) if warp else "—"])
    text = _table(["Sphere mesh", "Tets", "RDM vs analytical (float64)",
                   "MAG vs analytical", "RDM vs analytical (Warp, 1e-6)",
                   "Warp solve"], rows)

    rows = []
    for d in run.get("discretization", []):
        has_gm = d.get("rdm_gm") is not None
        rows.append([d["dataset"], d["refined"], _fmt(d["rdm"]),
                     _fmt(d.get("rdm_gm")),
                     f"{100 * d['gm_enorm_change_median']:.1f}%" if has_gm else "—",
                     f"{100 * d['gm_enorm_change_p95']:.1f}%" if has_gm else "—"])
    if rows:
        text += ("\nChange in the float64 reference solution when the mesh is "
                 "refined once more:\n\n")
        text += _table(["Mesh", "Refined mesh", "RDM", "RDM in grey matter",
                        "Median change of E magnitude in grey matter",
                        "95th percentile"], rows)
    return text


def render_gradient(run):
    rows = []
    for name, info in run["datasets"].items():
        if "n_nodes" not in info:
            continue
        for r in _jobs(run, "warp", name):
            g = r.get("gradient")
            if not g:
                continue
            device = "GPU" if r["actual_device"].startswith("cuda") else "CPU"
            rows.append([name, _count(info["n_elements"]), device, _fmt_tol(g["rtol"]),
                         _fmt_time(r["t_rhs"]["best"]), _fmt_time(g["t_cold"]),
                         _fmt_time(g["t_warm"]["median"]),
                         _fmt(r.get("gpu_memory_gb"), ".2f")])
    return _table(["Mesh", "Tets", "Device", "Tolerance", "Right-hand side",
                   "Gradient, cold start", "Gradient, warm start",
                   "GPU memory (GB)"], rows)


def render_optimization(run):
    rows = []
    for r in _jobs(run, "optimization"):
        name = r["job"]["dataset"]
        for sub in r["runs"]:
            rows.append([name, str(sub["target_mm"]), _fmt_time(sub["t_total"]),
                         sub["adam_iterations"],
                         _fmt(sub["enorm_at_target"], ".2f"),
                         _fmt(sub["enorm_at_target_reference"], ".2f")])
    return _table(["Mesh", "Target (mm)", "Time", "Adam iterations",
                   "E magnitude at target (V/m)", "Reference (V/m)"], rows)


def render_failures(run):
    rows = []
    for r in run["jobs"]:
        if r.get("status") == "ok":
            continue
        job = r.get("job", {})
        what = " ".join(str(job.get(k)) for k in ("kind", "dataset", "device", "solver")
                        if job.get(k))
        last = str(r.get("error", "")).strip().splitlines()
        rows.append([what, r.get("status"), (last[-1] if last else "")[:120]])
    return _table(["Job", "Status", "Last line of error"], rows) if rows else ""


def render_summary(runs):
    """Across machines: the few numbers a paper table needs."""
    rows = []
    for run in runs:
        device = _gpu_device(run)
        for name, info in run["datasets"].items():
            if "n_nodes" not in info or not name.startswith("ernie"):
                continue
            warp = _first(_jobs(run, "warp", name, device=device)) if device else None
            w6 = _warp_run(warp, 1e-6)
            w4 = _warp_run(warp, 1e-4)
            hypre = _first(_jobs(run, "simnibs", name, solver="hypre"))
            pardiso = _first(_jobs(run, "simnibs", name, solver="pardiso"))
            opt = _first(_jobs(run, "optimization", name))
            opt_time = None
            if opt:
                times = [s["t_total"] for s in opt["runs"]]
                opt_time = sum(times) / len(times)
            grad = warp["gradient"]["t_warm"]["median"] if warp and warp.get("gradient") else None
            rows.append([
                machine_name(run), name, _count(info["n_elements"]),
                _fmt_time(w4["t_solve"]["best"]) if w4 else "—",
                _fmt_time(w6["t_solve"]["best"]) if w6 else "—",
                _fmt(w6.get("rdm_vs_reference")) if w6 else "—",
                _fmt_time(hypre["t_solve"]["best"]) if hypre else "—",
                _fmt_time(pardiso["t_solve"]["best"]) if pardiso else "—",
                _fmt_time(grad), _fmt_time(opt_time),
            ])
    return _table(["Machine", "Mesh", "Tets", "Warp GPU solve, 1e-4",
                   "Warp GPU solve, 1e-6", "RDM at 1e-6", "SimNIBS hypre solve",
                   "SimNIBS PARDISO solve", "Gradient, warm",
                   "Optimization, mean"], rows)


def render(runs):
    parts = ["# TMSWarp benchmark results\n"]
    if len(runs) > 1:
        parts.append("## Summary across machines\n")
        parts.append(render_summary(runs))
    for run in runs:
        parts.append(f"## {machine_name(run)}\n")
        parts.append(render_machine(run))
        if len(runs) == 1:
            parts.append("### Head meshes at a glance\n")
            parts.append(render_summary([run]))
        parts.append("### Solve times\n")
        parts.append("Cold-start solves, best of the repeats. Setup is matrix "
                     "assembly, plus preconditioner setup or factorization "
                     "where noted. The reference is a float64 conjugate-"
                     "gradient solve to a relative residual of 1e-10.\n")
        parts.append(render_solves(run))
        parts.append("### Accuracy against mesh resolution\n")
        parts.append(render_resolution(run))
        parts.append("### Gradient of the E magnitude at a target\n")
        parts.append(render_gradient(run))
        parts.append("### Coil position optimization\n")
        parts.append(render_optimization(run))
        failures = render_failures(run)
        if failures:
            parts.append("### Jobs that did not finish\n")
            parts.append(failures)
    return "\n".join(parts)
