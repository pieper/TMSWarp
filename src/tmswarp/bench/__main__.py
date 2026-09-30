"""Command line for the TMSWarp benchmarks.

    python -m tmswarp.bench run --suite quick --label my-laptop
    python -m tmswarp.bench report results/*.json
    python -m tmswarp.bench fetch --suite standard
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from tmswarp.bench import data, machine, report
from tmswarp.bench.worker import sanitize

# Problems larger than this are not run with the direct solver unless the
# suite asks for it: they take minutes to hours.
SLOW_CPU_NODES = 60000

SUITES = {
    "quick": {
        "datasets": ["sphere3", "sphere3-r1", "ernie-lowres"],
        "rtols": [1e-3, 1e-4, 1e-6],
        "simnibs": ["hypre"],
        "slow_cpu": [],
        "warp_cpu_nodes": 10000,
        "optimization": ["ernie-lowres"],
        "timeout": 1800,
    },
    "standard": {
        "datasets": ["sphere3", "sphere3-r1", "sphere3-r2", "ernie-lowres",
                     "ernie-lowres-r1", "ernie-full"],
        "rtols": [1e-3, 1e-4, 1e-6],
        "simnibs": ["hypre", "pardiso"],
        "slow_cpu": [],
        "warp_cpu_nodes": 60000,
        "optimization": ["ernie-lowres", "ernie-full"],
        "timeout": 3600,
    },
    "full": {
        "datasets": ["sphere3", "sphere3-r1", "sphere3-r2", "sphere3-r3",
                     "ernie-lowres", "ernie-lowres-r1", "ernie-full",
                     "ernie-full-r1"],
        "rtols": [1e-3, 1e-4, 1e-5, 1e-6],
        "simnibs": ["hypre", "pardiso"],
        "slow_cpu": ["ernie-lowres"],
        "warp_cpu_nodes": 60000,
        "optimization": ["ernie-lowres", "ernie-lowres-r1", "ernie-full"],
        "timeout": 4 * 3600,
    },
}

OPTIMIZATION_TARGETS_MM = [
    [-40.0, -10.0, 60.0], [-40.0, 40.0, 40.0], [30.0, -80.0, 10.0],
]


def log(msg=""):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Discovery of optional components
# ---------------------------------------------------------------------------

def find_simnibs_python(explicit=None):
    candidates = [explicit, os.environ.get("SIMNIBS_PYTHON")]
    home = Path.home()
    for pattern in ("tmswarp-bench/SimNIBS*", "SimNIBS*", "Applications/SimNIBS*"):
        for root in sorted(home.glob(pattern), reverse=True):
            candidates.append(root / "simnibs_env" / "bin" / "python")
            candidates.append(root / "simnibs_env" / "python.exe")
    candidates.append(shutil.which("simnibs_python"))
    for c in candidates:
        if c and Path(c).is_file():
            return str(c)
    return None


def find_tmsservice(explicit=None):
    here = Path(__file__).resolve()
    candidates = [explicit, os.environ.get("TMSWARP_TMSSERVICE")]
    if len(here.parents) > 4:
        candidates.append(here.parents[4] / "Experiments" / "TMSService.py")
    candidates.append(Path.home() / "tmswarp-bench" / "SlicerTMS"
                      / "Experiments" / "TMSService.py")
    for c in candidates:
        if c and Path(c).is_file():
            return str(c)
    return None


def warp_devices(explicit=None):
    """Return (devices to benchmark, has_cuda)."""
    try:
        import warp as wp
        wp.config.quiet = True
        wp.init()
    except ImportError:
        return [], False
    cuda = [str(d) for d in wp.get_devices() if d.is_cuda]
    if explicit:
        return [d.strip() for d in explicit.split(",")], bool(cuda)
    return ([cuda[0], "cpu"] if cuda else ["cpu"]), bool(cuda)


# ---------------------------------------------------------------------------
# Running jobs
# ---------------------------------------------------------------------------

def _clean_env():
    """Environment for child processes that use a different Python."""
    env = dict(os.environ)
    for key in ("LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME", "DYLD_LIBRARY_PATH"):
        env.pop(key, None)
    return env


def run_job(job, timeout, python=None, script=None, env=None):
    """Run one job in a child process and return its result dict."""
    work = Path(tempfile.mkdtemp(prefix="tmswarp_job_", dir=data.cache_dir()))
    job_path, result_path = work / "job.json", work / "result.json"
    with open(job_path, "w") as f:
        json.dump(job, f)
    if script:
        cmd = [python, str(script), str(job_path), str(result_path)]
    else:
        cmd = [sys.executable, "-m", "tmswarp.bench.worker",
               str(job_path), str(result_path)]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, timeout=timeout, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True)
        output = proc.stdout or ""
        for line in output.splitlines():
            if line.startswith("    "):
                log(line)
        tail = "\n".join(output.splitlines()[-15:])
        if result_path.exists():
            with open(result_path) as f:
                result = json.load(f)
        else:
            killed = proc.returncode in (-9, 137)
            result = {"status": "out-of-memory" if killed else "crashed",
                      "error": f"exit code {proc.returncode}\n{tail}", "job": job}
        if result.get("status") == "partial":
            # Timings were recorded but the job did not finish
            killed = proc.returncode in (-9, 137)
            result["status"] = "out-of-memory" if killed else "error"
            result["error"] = result.get("error") or f"exit code {proc.returncode}\n{tail}"
            result["partial"] = True
    except subprocess.TimeoutExpired:
        result = {"status": "timeout", "error": f"exceeded {timeout} s", "job": job}
    result["t_wall"] = time.perf_counter() - t0
    shutil.rmtree(work, ignore_errors=True)
    return result


def _describe(job):
    parts = [job["kind"], job["dataset"]]
    for key in ("device", "solver"):
        if key in job:
            parts.append(str(job[key]))
    return " ".join(parts)


def build_jobs(suite, devices, has_cuda, simnibs_python, tmsservice, datasets):
    jobs = []
    for name in datasets:
        n_nodes = len(np.load(data.path_for(name))["nodes"])
        small = n_nodes <= SLOW_CPU_NODES
        slow_ok = small or name in suite["slow_cpu"]
        base = {"dataset": name, "dipole": "far"}

        jobs.append(dict(base, kind="reference"))
        jobs.append(dict(base, kind="scipy-cg", rtols=[1e-6]))
        if slow_ok:
            jobs.append(dict(base, kind="numpy-direct"))
        for device in devices:
            rtols = suite["rtols"]
            quick_on_cpu = n_nodes <= suite["warp_cpu_nodes"]
            if device == "cpu" and not quick_on_cpu:
                # Warp's CPU path is slow: only where the suite asks for it,
                # or where there is no GPU and the mesh is still manageable
                if name in suite["slow_cpu"] or (not has_cuda and small):
                    rtols = [1e-4]
                else:
                    continue
            jobs.append(dict(base, kind="warp", device=device, rtols=rtols,
                             repeats=3 if device != "cpu" or quick_on_cpu else 1,
                             gradient=device != "cpu" or quick_on_cpu))
        if simnibs_python:
            for option in suite["simnibs"]:
                jobs.append(dict(base, kind="simnibs", solver=option))
        if tmsservice and name in suite["optimization"] and devices:
            if has_cuda or small:
                jobs.append(dict(base, kind="optimization", device=devices[0],
                                 tmsservice=tmsservice,
                                 targets_mm=OPTIMIZATION_TARGETS_MM))
    return jobs


def run_simnibs(job, simnibs_python, timeout):
    """Run SimNIBS on a dataset and score its field against the reference."""
    from tmswarp.fields import mag, rdm

    mesh, tag1, _ = data.load(job["dataset"], log=log)
    position, moment, didt = data.dipole(job["dataset"], mesh, job["dipole"])
    efield = data.cache_dir() / f"{job['dataset']}.simnibs-{job['solver']}.npz"
    spec = dict(job, mesh=str(data.path_for(job["dataset"])),
                position=position.tolist(), moment=moment.tolist(), didt=didt,
                efield=str(efield), repeats=3)
    script = Path(__file__).with_name("simnibs_runner.py")
    result = run_job(spec, timeout, python=simnibs_python, script=script,
                     env=_clean_env())
    if result.get("status") == "ok" and efield.exists():
        E = np.load(efield)["E"]
        ref = data.cache_dir() / f"{job['dataset']}.reference-{job['dipole']}.npz"
        if ref.exists():
            E_ref = np.load(ref)["E"]
            result["rdm_vs_reference"] = rdm(E, E_ref)
            result["mag_vs_reference"] = mag(E, E_ref)
        if data.is_sphere(job["dataset"]):
            from tmswarp.analytical import tms_analytical_efield
            from tmswarp.conductor import element_barycenters
            E_ana = tms_analytical_efield(position, moment, didt,
                                          element_barycenters(mesh))
            result["rdm_vs_analytical"] = rdm(E, E_ana)
            result["mag_vs_analytical"] = mag(E, E_ana)
    if efield.exists():
        efield.unlink()
    result["job"] = job
    return result


def discretization(datasets, dipole="far"):
    """Change in the reference solution each time a mesh is refined."""
    from tmswarp.conductor import element_volumes
    from tmswarp.fields import mag, rdm
    from tmswarp.refine import restrict_to_parents

    rows = []
    for name in datasets:
        base, levels = data.DATASETS[name]
        if levels == 0:
            continue
        coarser = base if levels == 1 else f"{base}-r{levels - 1}"
        fine_ref = data.cache_dir() / f"{name}.reference-{dipole}.npz"
        coarse_ref = data.cache_dir() / f"{coarser}.reference-{dipole}.npz"
        if not (fine_ref.exists() and coarse_ref.exists()):
            continue
        mesh, tag1, _ = data.load(name, log=log)
        E_fine = np.load(fine_ref)["E"]
        E_coarse = np.load(coarse_ref)["E"]
        parent = np.arange(len(E_fine)) // 8
        restricted = restrict_to_parents(
            E_fine, element_volumes(mesh), parent, len(E_coarse))
        row = {"dataset": coarser, "refined": name,
               "rdm": rdm(E_coarse, restricted),
               "mag": mag(E_coarse, restricted)}
        if tag1 is not None and np.any(tag1 == 2):
            gm = tag1[::8] == 2
            row["rdm_gm"] = rdm(E_coarse[gm], restricted[gm])
            row["mag_gm"] = mag(E_coarse[gm], restricted[gm])
            norm_c = np.linalg.norm(E_coarse[gm], axis=1)
            norm_f = np.linalg.norm(restricted[gm], axis=1)
            rel = np.abs(norm_c - norm_f) / np.maximum(norm_f, 1e-12)
            row["gm_enorm_change_median"] = float(np.median(rel))
            row["gm_enorm_change_p95"] = float(np.percentile(rel, 95))
        rows.append(row)
    return rows


def _slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "-", str(text)).strip("-").lower()


def cmd_run(args):
    suite = dict(SUITES[args.suite])
    if args.simnibs_solvers:
        suite["simnibs"] = args.simnibs_solvers.split(",")
    datasets = (args.datasets.split(",") if args.datasets else suite["datasets"])
    devices, has_cuda = warp_devices(args.devices)
    simnibs_python = None if args.no_simnibs else find_simnibs_python(args.simnibs_python)
    tmsservice = None if args.no_optimization else find_tmsservice(args.tmsservice)

    info = machine.describe(label=args.label, provider=args.provider)
    info["suite"] = args.suite
    info["simnibs_python"] = simnibs_python
    info["tmsservice"] = tmsservice

    gpu = info["gpus"][0]["name"] if info["gpus"] else "cpu"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = "_".join([_slug(args.label or info["hostname"]), _slug(gpu), stamp])
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{name}.json"
    out_md = out_dir / f"{name}.md"

    log("=" * 70)
    log(f"TMSWarp benchmark, suite '{args.suite}'")
    log(f"  machine:  {info['cpu']}, {info['cpu_threads']} threads, "
        f"{info['memory_gb']} GB")
    log(f"  GPU:      {', '.join(g['name'] for g in info['gpus']) or 'none'}")
    log(f"  devices:  {devices or 'Warp not installed'}")
    log(f"  SimNIBS:  {simnibs_python or 'not found (skipped)'}")
    log(f"  optimizer: {tmsservice or 'TMSService.py not found (skipped)'}")
    log(f"  results:  {out_json}")
    log("=" * 70)

    results = {"machine": info, "datasets": {}, "jobs": [], "discretization": []}

    def save():
        tmp = str(out_json) + ".part"
        with open(tmp, "w") as f:
            json.dump(sanitize(results), f, indent=1)
        os.replace(tmp, out_json)

    available = []
    for ds in datasets:
        t0 = time.perf_counter()
        try:
            path = data.ensure(ds, log=log)
            d = np.load(path)
            results["datasets"][ds] = {
                "n_nodes": int(len(d["nodes"])),
                "n_elements": int(len(d["elements"])),
                "t_prepare": time.perf_counter() - t0,
            }
            available.append(ds)
        except Exception as exc:
            log(f"  could not prepare {ds}: {exc}")
            results["datasets"][ds] = {"error": str(exc)}
        save()

    jobs = build_jobs(suite, devices, has_cuda, simnibs_python, tmsservice,
                      available)
    if args.only:
        kinds = args.only.split(",")
        jobs = [j for j in jobs if j["kind"] in kinds]
    t_start = time.perf_counter()
    for i, job in enumerate(jobs, 1):
        log(f"[{i}/{len(jobs)}] {_describe(job)}")
        if job["kind"] == "simnibs":
            result = run_simnibs(job, simnibs_python, suite["timeout"])
        else:
            result = run_job(job, suite["timeout"])
        status = result.get("status")
        log(f"    -> {status} in {result['t_wall']:.1f} s")
        if status != "ok":
            log("    " + str(result.get("error", "")).strip().splitlines()[-1])
        results["jobs"].append(result)
        save()

    try:
        results["discretization"] = discretization(available)
    except Exception as exc:
        log(f"discretization comparison failed: {exc}")
    results["machine"]["t_total"] = time.perf_counter() - t_start
    save()

    text = report.render([results])
    with open(out_md, "w") as f:
        f.write(text)
    log()
    log(text)
    log(f"Results: {out_json}")
    log(f"Report:  {out_md}")


def cmd_fetch(args):
    for ds in SUITES[args.suite]["datasets"]:
        data.ensure(ds, log=log)
    log(f"Datasets are in {data.cache_dir()}")


def cmd_report(args):
    paths = []
    for pattern in args.results:
        paths.extend(sorted(glob.glob(pattern)))
    runs = []
    for path in paths:
        with open(path) as f:
            runs.append(json.load(f))
    text = report.render(runs)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
    print(text)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tmswarp.bench",
                                     description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a benchmark suite")
    run.add_argument("--suite", default="quick", choices=sorted(SUITES))
    run.add_argument("--label", help="short name for this machine")
    run.add_argument("--provider", help="e.g. vast.ai, jetstream2, laptop")
    run.add_argument("--output", default="benchmark-results",
                     help="directory for the result files")
    run.add_argument("--datasets", help="comma-separated; overrides the suite")
    run.add_argument("--only", help="comma-separated job kinds to run, e.g. warp,simnibs")
    run.add_argument("--simnibs-solvers", help="comma-separated, e.g. hypre,pardiso")
    run.add_argument("--devices", help="comma-separated Warp devices")
    run.add_argument("--simnibs-python", help="path to SimNIBS's python")
    run.add_argument("--no-simnibs", action="store_true")
    run.add_argument("--tmsservice", help="path to SlicerTMS's TMSService.py")
    run.add_argument("--no-optimization", action="store_true")
    run.set_defaults(func=cmd_run)

    fetch = sub.add_parser("fetch", help="download and prepare the datasets")
    fetch.add_argument("--suite", default="quick", choices=sorted(SUITES))
    fetch.set_defaults(func=cmd_fetch)

    rep = sub.add_parser("report", help="summarize result files")
    rep.add_argument("results", nargs="+", help="result .json files")
    rep.add_argument("--output", help="write the report to this file")
    rep.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
