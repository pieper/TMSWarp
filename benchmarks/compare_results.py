"""Compare TMSWarp benchmark results across machines and devices.

Loads all JSON files from benchmarks/results/ and prints a leaderboard
sorted by best speedup, plus optional detailed comparison tables and plots.

Usage
-----
    python benchmarks/compare_results.py                       # leaderboard (default)
    python benchmarks/compare_results.py --detail              # full comparison table
    python benchmarks/compare_results.py --metric warp_total   # specific metric table
    python benchmarks/compare_results.py --plot                # comparison plot
"""

import argparse
import json
from pathlib import Path

import numpy as np


RESULTS_DIR = Path(__file__).parent / "results"


def load_results():
    """Load all JSON files in results/, return list of dicts sorted by timestamp."""
    files = sorted(RESULTS_DIR.glob("*.json"))
    runs = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        data["_filename"] = f.name
        runs.append(data)
    return runs


def _get_configs(run):
    """Get the config list from either run_benchmarks or gpu_scaling format."""
    return run.get("configs") or run.get("results") or []


def _run_description(run):
    """One-line description of a benchmark run."""
    m = run["metadata"]
    host = m["hostname"].split(".")[0]
    plat = m["platform"]
    device = m.get("warp_device", "")
    gpus = m.get("gpu_devices", [])
    gpu_str = gpus[0]["name"] if gpus else ""
    return host, plat, device, gpu_str


def _compute_speedups(run):
    """Compute speedup ratios for each config in a run.

    Returns list of dicts with keys: n_elements, numpy_vs_warp, cpu_vs_gpu.
    Handles both run_benchmarks format (configs) and gpu_scaling format (results).
    """
    configs = _get_configs(run)
    rows = []
    for c in configs:
        ne = c.get("n_elements", 0)
        row = {"n_elements": ne}

        # run_benchmarks format: NumPy vs Warp (single device)
        np_t = c.get("t_numpy_total_s") or c.get("t_numpy_tot")
        wp_t = c.get("t_warp_total_s") or c.get("t_warp_cpu_tot")
        if np_t and wp_t and wp_t > 0:
            row["numpy_vs_warp"] = np_t / wp_t

        # gpu_scaling format: Warp CPU vs Warp GPU
        cpu_t = c.get("t_warp_cpu_tot")
        gpu_t = c.get("t_warp_gpu_tot")
        if cpu_t and gpu_t and gpu_t > 0:
            row["cpu_vs_gpu"] = cpu_t / gpu_t

        # NumPy vs Warp GPU (overall best speedup)
        if np_t and gpu_t and gpu_t > 0:
            row["numpy_vs_gpu"] = np_t / gpu_t

        # Best solver time at this mesh size
        times = [t for t in [np_t, wp_t, gpu_t] if t is not None and t > 0]
        if times:
            row["best_time"] = min(times)

        rows.append(row)
    return rows


def print_leaderboard(runs):
    """Print a leaderboard: one row per result file, sorted by best speedup."""
    if not runs:
        print("No benchmark results found in", RESULTS_DIR)
        return

    entries = []
    for run in runs:
        host, plat, device, gpu_str = _run_description(run)
        configs = _get_configs(run)
        speedups = _compute_speedups(run)

        if not speedups:
            continue

        # Largest mesh
        largest = max(speedups, key=lambda r: r["n_elements"])
        n_elem = largest["n_elements"]

        # Best time at largest mesh
        best_time = largest.get("best_time")

        # Pick the most relevant speedup
        # Prefer GPU speedup if available, else NumPy vs Warp
        gpu_speedup = largest.get("cpu_vs_gpu")
        np_speedup = largest.get("numpy_vs_warp")

        # For sorting: use GPU speedup if available, else numpy/warp ratio
        sort_key = gpu_speedup or np_speedup or 0.0

        entries.append({
            "host": host,
            "platform": plat,
            "device": device,
            "gpu": gpu_str,
            "n_elements": n_elem,
            "best_time": best_time,
            "gpu_speedup": gpu_speedup,
            "np_speedup": np_speedup,
            "sort_key": sort_key,
            "filename": run["_filename"],
        })

    # Sort by speedup (highest first)
    entries.sort(key=lambda e: e["sort_key"], reverse=True)

    # Print
    print(f"\n{'='*80}")
    print("TMSWarp Benchmark Leaderboard")
    print(f"{'='*80}")
    print(f"{'#':>3}  {'Host':<16} {'Platform':<16} {'Device/GPU':<28} "
          f"{'Elements':>10} {'Best(s)':>8} {'Speedup':>8}")
    print("-" * 96)

    for i, e in enumerate(entries, 1):
        device_col = e["gpu"] if e["gpu"] else e["device"]
        if len(device_col) > 27:
            device_col = device_col[:24] + "..."

        best = f"{e['best_time']:.3f}" if e["best_time"] is not None else "N/A"

        if e["gpu_speedup"] is not None:
            speedup_str = f"{e['gpu_speedup']:.2f}x GPU"
        elif e["np_speedup"] is not None:
            ratio = e["np_speedup"]  # numpy_time / warp_time
            if ratio > 1:
                # NumPy takes longer → Warp is faster
                speedup_str = f"{ratio:.2f}x Warp"
            else:
                # NumPy takes less time → NumPy is faster
                speedup_str = f"{1/ratio:.2f}x NP"
        else:
            speedup_str = "N/A"

        print(f"{i:>3}  {e['host']:<16} {e['platform']:<16} {device_col:<28} "
              f"{e['n_elements']:>10,} {best:>8} {speedup_str:>8}")

    print()
    print("Speedup column:")
    print("  'X.XXx GPU'  = Warp GPU is Xx faster than Warp CPU (gpu_scaling results)")
    print("  'X.XXx Warp' = Warp is Xx faster than NumPy")
    print("  'X.XXx NP'   = NumPy is Xx faster than Warp (small meshes, expected)")
    print(f"\nFiles: {RESULTS_DIR}")


# ---------------------------------------------------------------------------
# Detailed comparison (original functionality)
# ---------------------------------------------------------------------------

def run_label(run):
    """Short human-readable label for a benchmark run."""
    m = run["metadata"]
    host = m["hostname"].split(".")[0]
    plat = m["platform"]
    dev = m.get("warp_device", "?").split("(")[0].strip()
    ts = m["timestamp"][:10]
    return f"{host} | {plat} | warp:{dev} | {ts}"


def print_comparison(runs, metric="numpy_total"):
    """Print a table of timing/accuracy values across runs, one column per run."""
    if not runs:
        print("No benchmark results found in", RESULTS_DIR)
        return

    # Only use runs that have 'configs' (run_benchmarks format)
    compat_runs = [r for r in runs if "configs" in r]
    if not compat_runs:
        print("No standard benchmark results found (only gpu_scaling results).")
        return

    ref_elems = [c["n_elements"] for c in compat_runs[0]["configs"]]

    labels = [run_label(r) for r in compat_runs]
    col_w = max(14, max(len(l) for l in labels))
    elem_w = 8

    metric_map = {
        "numpy_total":    ("t_numpy_total_s",    "NumPy total (s)"),
        "numpy_assembly": ("t_numpy_assembly_s",  "NumPy assembly (s)"),
        "numpy_solve":    ("t_numpy_solve_s",     "NumPy solve (s)"),
        "warp_total":     ("t_warp_total_s",      "Warp total (s)"),
        "warp_assembly":  ("t_warp_assembly_s",   "Warp assembly (s)"),
        "warp_solve":     ("t_warp_solve_s",      "Warp solve (s)"),
        "rdm_numpy":      ("rdm_numpy",           "RDM (NumPy)"),
        "rdm_warp":       ("rdm_warp",            "RDM (Warp)"),
    }

    key, title = metric_map.get(metric, ("t_numpy_total_s", metric))

    print(f"\n{'':>{elem_w}}  " + "  ".join(f"{l:>{col_w}}" for l in labels))
    print(f"{'Elements':>{elem_w}}  " + "  ".join("-" * col_w for _ in compat_runs))

    for elem_count in ref_elems:
        row = f"{elem_count:>{elem_w}}"
        for run in compat_runs:
            configs = run["configs"]
            match = min(configs, key=lambda c: abs(c["n_elements"] - elem_count))
            val = match.get(key)
            if val is None:
                cell = "N/A"
            else:
                cell = f"{val:.4f}"
            row += f"  {cell:>{col_w}}"
        print(row)

    print(f"\nMetric: {title}")
    print(f"Files:  {', '.join(r['_filename'] for r in compat_runs)}")


def plot_comparison(runs, metric="numpy_total", output=None):
    """Generate a timing comparison plot across runs."""
    import matplotlib
    matplotlib.use("Agg" if output else "TkAgg")
    import matplotlib.pyplot as plt

    metric_map = {
        "numpy_total":   ("t_numpy_total_s",   "NumPy total (s)"),
        "warp_total":    ("t_warp_total_s",     "Warp total (s)"),
        "warp_assembly": ("t_warp_assembly_s",  "Warp assembly (s)"),
        "warp_solve":    ("t_warp_solve_s",     "Warp solve (s)"),
        "rdm_numpy":     ("rdm_numpy",          "RDM (NumPy)"),
        "rdm_warp":      ("rdm_warp",           "RDM (Warp)"),
    }
    key, ylabel = metric_map.get(metric, ("t_numpy_total_s", metric))

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10.colors

    for i, run in enumerate(runs):
        configs = _get_configs(run)
        elems = [c["n_elements"] for c in configs]
        vals = [c.get(key) for c in configs]
        valid = [(e, v) for e, v in zip(elems, vals) if v is not None]
        if not valid:
            continue
        ex, vx = zip(*valid)
        label = run_label(run)
        ax.semilogy(ex, vx, "o-", color=colors[i % len(colors)], lw=2, ms=6, label=label)

    ax.set_xlabel("Number of Elements")
    ax.set_ylabel(ylabel)
    ax.set_title(f"TMSWarp Benchmark: {ylabel}")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare TMSWarp benchmark results across machines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--detail", action="store_true",
        help="Show detailed per-element comparison table instead of leaderboard",
    )
    parser.add_argument(
        "--metric",
        default="numpy_total",
        choices=[
            "numpy_total", "numpy_assembly", "numpy_solve",
            "warp_total", "warp_assembly", "warp_solve",
            "rdm_numpy", "rdm_warp",
        ],
        help="Which metric to display in --detail mode",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate a comparison plot",
    )
    parser.add_argument(
        "--save-plot", metavar="PATH",
        help="Save plot to this path instead of showing interactively",
    )
    args = parser.parse_args()

    runs = load_results()

    if args.detail:
        print_comparison(runs, metric=args.metric)
    else:
        print_leaderboard(runs)

    if args.plot or args.save_plot:
        plot_comparison(runs, metric=args.metric, output=args.save_plot)
