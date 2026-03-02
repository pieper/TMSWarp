"""Compare TMSWarp benchmark results across machines and devices.

Loads all JSON files from benchmarks/results/ and prints a comparison table.
Optionally generates a comparison plot.

Usage
-----
    pixi run python benchmarks/compare_results.py
    pixi run python benchmarks/compare_results.py --plot
    pixi run python benchmarks/compare_results.py --metric warp_total
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


def run_label(run):
    """Short human-readable label for a benchmark run."""
    m = run["metadata"]
    host = m["hostname"].split(".")[0]
    plat = m["platform"]
    dev = m["warp_device"].split("(")[0].strip()
    ts = m["timestamp"][:10]
    return f"{host} | {plat} | warp:{dev} | {ts}"


def print_comparison(runs, metric="numpy_total"):
    """Print a table of timing/accuracy values across runs, one column per run."""
    if not runs:
        print("No benchmark results found in", RESULTS_DIR)
        return

    # Collect element counts (from first run as reference)
    ref_elems = [c["n_elements"] for c in runs[0]["configs"]]

    # Header
    labels = [run_label(r) for r in runs]
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
    print(f"{'Elements':>{elem_w}}  " + "  ".join("-" * col_w for _ in runs))

    for ref_ne, elem_count in enumerate(ref_elems):
        row = f"{elem_count:>{elem_w}}"
        for run in runs:
            # Find matching config by element count (closest)
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
    print(f"Files:  {', '.join(r['_filename'] for r in runs)}")


def print_summary_table(runs):
    """Print metadata summary for all runs."""
    print(f"\n{'File':<50}  {'Host':<20}  {'Platform':<15}  {'Warp device':<25}  {'Date':<12}")
    print("-" * 130)
    for r in runs:
        m = r["metadata"]
        print(
            f"{r['_filename']:<50}  "
            f"{m['hostname'].split('.')[0]:<20}  "
            f"{m['platform']:<15}  "
            f"{m['warp_device']:<25}  "
            f"{m['timestamp'][:10]:<12}"
        )


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
        elems = [c["n_elements"] for c in run["configs"]]
        vals = [c.get(key) for c in run["configs"]]
        # Drop None
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
        "--metric",
        default="numpy_total",
        choices=[
            "numpy_total", "numpy_assembly", "numpy_solve",
            "warp_total", "warp_assembly", "warp_solve",
            "rdm_numpy", "rdm_warp",
        ],
        help="Which metric to display/plot",
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
    print_summary_table(runs)
    print_comparison(runs, metric=args.metric)

    if args.plot or args.save_plot:
        plot_comparison(runs, metric=args.metric, output=args.save_plot)
