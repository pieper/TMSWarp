"""Timing comparison: SimNIBS vs TMSWarp NumPy FEM vs TMSWarp Warp.fem.

Runs all three solvers on the ernie head mesh and produces ernie_timing.png.

Usage
-----
    # Full run (re-times everything; ~10 min on Apple Silicon CPU):
    /path/to/SimNIBS-4.5/.../python benchmarks/ernie_timing.py --simnibs
    pixi run python benchmarks/ernie_timing.py

    # Quick plot only (uses cached results from previous runs):
    pixi run python benchmarks/ernie_timing.py --plot-only

Prerequisites: ernie_data.npz + ernie_simnibs_efield.npz
(run scripts/fetch_ernie.py and scripts/run_simnibs_efield.py first)
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.interpolate import griddata

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
ERNIE_MESH_PATH   = ROOT / "ernie_data.npz"
SIMNIBS_PATH      = ROOT / "ernie_simnibs_efield.npz"
TIMING_CACHE      = ROOT / "ernie_timing_cache.json"
OUTPATH           = ROOT / "ernie_timing.png"
COMPARISON_OUTPATH = ROOT / "ernie_solver_comparison.png"

# Slice planes (mm) — same as ernie_comparison.py
AXIAL_Z    =  40.0
CORONAL_Y  =  50.0
SAGITTAL_X =   0.0
SLAB_HALF  =   4.0
GRID_N     = 200

DIPOLE_POS_M  = np.array([0.0, 0.0, 0.200])
DIPOLE_MOMENT = np.array([1.0, 0.0, 0.0])
DIDT          = 1e6


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_tmswarp_numpy():
    """Run TMSWarp NumPy FEM and return (timing_dict, phi, mesh, dAdt, G)."""
    from tmswarp.coil import magnetic_dipole_dadt
    from tmswarp.conductor import TetMesh
    from tmswarp.solver import assemble_rhs_tms, assemble_stiffness, gradient_operator, solve_fem

    data = np.load(ERNIE_MESH_PATH)
    mesh = TetMesh(
        nodes=data["nodes"].astype(np.float64),
        elements=data["elements"].astype(np.int32),
        conductivity=data["conductivity"].astype(np.float64),
    )
    dAdt = magnetic_dipole_dadt(DIPOLE_POS_M, DIPOLE_MOMENT, DIDT, mesh.nodes)

    t0 = time.perf_counter()
    G   = gradient_operator(mesh)
    t1  = time.perf_counter()
    K   = assemble_stiffness(mesh, G)
    t2  = time.perf_counter()
    b   = assemble_rhs_tms(mesh, dAdt, G)
    t3  = time.perf_counter()
    phi = solve_fem(K, b, pin_node=0)
    t4  = time.perf_counter()

    timing = {
        "assembly": t3 - t0,
        "solve":    t4 - t3,
        "total":    t4 - t0,
    }
    return timing, phi, mesh, dAdt, G


def time_tmswarp_warp(mesh=None, dAdt=None, device=None):
    """Run TMSWarp Warp.fem and return (timing_dict, phi).

    Parameters
    ----------
    mesh, dAdt : optional
        Reuse mesh/dAdt from a prior NumPy run to avoid a second file load.
    device : str or None
        Warp device string (e.g. ``"cpu"`` or ``"cuda:0"``).
        None means warp's preferred device (GPU if available, else CPU).

    Notes
    -----
    Runs with quiet=False so CG iteration count is visible in output,
    confirming the solver is genuinely working.  solve_fem_warp performs
    an explicit wp.synchronize_device() before returning, so the timer
    captures the true wall-clock time including GPU completion.
    """
    from tmswarp.coil import magnetic_dipole_dadt
    from tmswarp.conductor import TetMesh, make_sphere_mesh
    from tmswarp.solver_warp import solve_fem_warp, warp_available

    if not warp_available():
        return None, None

    if mesh is None or dAdt is None:
        data = np.load(ERNIE_MESH_PATH)
        mesh = TetMesh(
            nodes=data["nodes"].astype(np.float64),
            elements=data["elements"].astype(np.int32),
            conductivity=data["conductivity"].astype(np.float64),
        )
        dAdt = magnetic_dipole_dadt(DIPOLE_POS_M, DIPOLE_MOMENT, DIDT, mesh.nodes)

    # Warm up JIT on a small mesh so JIT compilation is excluded from timing
    small = make_sphere_mesh(radius=0.05, n_shells=3, n_surface=50, conductivity=1.0)
    dAdt_s = magnetic_dipole_dadt(DIPOLE_POS_M, DIPOLE_MOMENT, DIDT, small.nodes)
    _ = solve_fem_warp(small, dAdt_s, device=device, quiet=True)

    t0  = time.perf_counter()
    phi = solve_fem_warp(mesh, dAdt, device=device, quiet=False)
    t1  = time.perf_counter()

    timing = {"assembly": None, "solve": None, "total": t1 - t0}
    return timing, phi


def time_simnibs():
    """Run SimNIBS TMSFEM and return timing dict. Must be called from SimNIBS Python."""
    import os
    sys.path.insert(0, (
        "/Users/pieper/Applications/SimNIBS-4.5/"
        "simnibs_env/lib/python3.11/site-packages"
    ))
    import scipy.sparse.linalg as spalg
    from simnibs.mesh_tools import mesh_io
    from simnibs.simulation import fem as simfem

    ernie_path = "/tmp/ernie_lowres/m2m_ernie/ernie.msh"
    if not os.path.isfile(ernie_path):
        ernie_path = str(ROOT / "ernie_dataset" / "m2m_ernie" / "ernie.msh")
    m = mesh_io.read_msh(ernie_path)
    m = m.crop_mesh(elm_type=4)

    CONDUCTIVITY_MAP = {1: 0.126, 2: 0.275, 3: 1.654, 4: 0.010, 5: 0.465, 6: 0.500}
    r = (m.nodes.node_coord - np.array([0., 0., 200.])) * 1e-3
    dAdt_vals = 1e-7 * DIDT * np.cross(DIPOLE_MOMENT, r) / (np.linalg.norm(r, axis=1, keepdims=True)**3)
    dAdt_node = mesh_io.NodeData(dAdt_vals, mesh=m)
    cond_vals = np.array([CONDUCTIVITY_MAP.get(int(t), 0.275) for t in m.elm.tag1])
    cond = mesh_io.ElementData(cond_vals)
    cond.mesh = m

    t0 = time.perf_counter()
    S = simfem.TMSFEM(m, cond)
    b = S.assemble_rhs(dAdt_node)
    t1 = time.perf_counter()
    _ = spalg.spsolve(S.A, b)
    t2 = time.perf_counter()
    return {"assembly": t1 - t0, "solve": t2 - t1, "total": t2 - t0}


# ---------------------------------------------------------------------------
# Slice-comparison figure helpers
# ---------------------------------------------------------------------------

def _slab_mask(bary, axis, plane_val):
    return np.abs(bary[:, axis] - plane_val) < SLAB_HALF


def _slice_axes(axis):
    return [a for a in range(3) if a != axis]


def _interp_slice(bary, values, mask, axis):
    ax0, ax1 = _slice_axes(axis)
    pts = bary[mask][:, [ax0, ax1]]
    if len(pts) == 0:
        xi = np.linspace(-1, 1, GRID_N)
        yi = np.linspace(-1, 1, GRID_N)
        return xi, yi, np.full((GRID_N, GRID_N), np.nan)
    vals = values[mask]
    lo0, hi0 = pts[:, 0].min(), pts[:, 0].max()
    lo1, hi1 = pts[:, 1].min(), pts[:, 1].max()
    xi = np.linspace(lo0, hi0, GRID_N)
    yi = np.linspace(lo1, hi1, GRID_N)
    Xg, Yg = np.meshgrid(xi, yi)
    Zi = griddata(pts, vals, (Xg, Yg), method="linear")
    return xi, yi, Zi


def make_solver_comparison_figure(efields: dict, mesh):
    """Generate a slice comparison figure for all available solvers.

    Parameters
    ----------
    efields : dict
        Mapping of label → (N_elem, 3) E-field array.
        The first entry is treated as the reference (NumPy FEM).
    mesh : TetMesh
        The mesh (nodes in metres).
    """
    from tmswarp.conductor import element_barycenters

    labels = list(efields.keys())
    ref_label = labels[0]
    E_ref = efields[ref_label]
    mag_ref = np.linalg.norm(E_ref, axis=1)

    bary = element_barycenters(mesh) * 1000  # m → mm

    # Colour limits: 99th percentile of all solvers combined
    all_mags = np.concatenate([np.linalg.norm(E, axis=1) for E in efields.values()])
    vmax = np.percentile(all_mags, 99)
    e_norm = Normalize(vmin=0, vmax=vmax)

    # Difference colour limit: 98th percentile of absolute differences
    all_diffs = np.concatenate([
        np.abs(np.linalg.norm(E, axis=1) - mag_ref)
        for E in list(efields.values())[1:]
    ])
    dlim = max(np.percentile(all_diffs, 98), 1e-9)
    diff_norm = TwoSlopeNorm(vcenter=0, vmin=-dlim, vmax=dlim)

    # Slice planes
    slice_specs = [
        (2, AXIAL_Z,    f"Axial z={AXIAL_Z:.0f} mm"),
        (1, CORONAL_Y,  f"Coronal y={CORONAL_Y:.0f} mm"),
        (0, SAGITTAL_X, f"Sagittal x={SAGITTAL_X:.0f} mm"),
    ]
    axis_labels = [["x (mm)", "y (mm)"], ["x (mm)", "z (mm)"], ["y (mm)", "z (mm)"]]

    # Layout: reference column + (|E| + diff) per non-reference solver
    n_warp = len(labels) - 1
    n_cols = 1 + 2 * n_warp
    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 4.5 * n_rows),
                             squeeze=False)

    for row, (axis, plane_val, row_label) in enumerate(slice_specs):
        mask = _slab_mask(bary, axis, plane_val)
        xl, yl = axis_labels[axis][0], axis_labels[axis][1]

        def _show(ax, xi, yi, Zi, norm, cmap, title):
            extent = [xi[0], xi[-1], yi[0], yi[-1]]
            ax.imshow(Zi, origin="lower", extent=extent,
                      norm=norm, cmap=cmap, aspect="equal",
                      interpolation="bilinear")
            ax.set_title(title, fontsize=8)
            ax.set_xlabel(xl, fontsize=7)
            ax.set_ylabel(yl, fontsize=7)
            ax.tick_params(labelsize=6)

        # Reference column
        xi, yi, Zi_ref = _interp_slice(bary, mag_ref, mask, axis)
        _show(axes[row, 0], xi, yi, Zi_ref, e_norm, "hot",
              f"{ref_label}\n|E|")
        if row == 0:
            axes[row, 0].set_ylabel(f"{row_label}\n{yl}", fontsize=7)

        # One pair of columns per non-reference solver
        for k, label in enumerate(labels[1:]):
            mag_k = np.linalg.norm(efields[label], axis=1)
            diff_k = mag_k - mag_ref

            _, _, Zi_k    = _interp_slice(bary, mag_k,  mask, axis)
            _, _, Zi_diff = _interp_slice(bary, diff_k, mask, axis)

            col_e    = 1 + 2 * k
            col_diff = 2 + 2 * k

            _show(axes[row, col_e], xi, yi, Zi_k, e_norm, "hot",
                  f"{label}\n|E|")
            _show(axes[row, col_diff], xi, yi, Zi_diff, diff_norm, "RdBu_r",
                  f"{label} − {ref_label}")

    # Shared colour bars
    fig.subplots_adjust(left=0.06, right=0.95, top=0.91,
                        bottom=0.10, hspace=0.55, wspace=0.40)

    cb_e = fig.colorbar(
        cm.ScalarMappable(norm=e_norm, cmap="hot"),
        ax=axes[:, 0], shrink=0.6, pad=0.02,
        orientation="horizontal", fraction=0.03,
    )
    cb_e.set_label("|E| (V/m)", fontsize=9)

    for k, label in enumerate(labels[1:]):
        col_diff = 2 + 2 * k
        cb_d = fig.colorbar(
            cm.ScalarMappable(norm=diff_norm, cmap="RdBu_r"),
            ax=axes[:, col_diff], shrink=0.6, pad=0.02,
            orientation="horizontal", fraction=0.03,
        )
        cb_d.set_label(f"Δ|E| (V/m)  [{label}]", fontsize=8)

    # Global stats in title
    stats_lines = []
    for label in labels[1:]:
        from tmswarp.fields import rdm as _rdm, mag as _mag
        rv = _rdm(efields[label], E_ref)
        mv = _mag(efields[label], E_ref)
        stats_lines.append(f"{label}: RDM={rv:.3f}  |MAG|={mv:.3f}")

    machine = f"{platform.system()} {platform.machine()}"
    fig.suptitle(
        f"Solver E-field comparison — ernie head mesh  |  {machine}\n"
        + "  |  ".join(stats_lines),
        fontsize=10, y=0.97,
    )

    fig.savefig(str(COMPARISON_OUTPATH), dpi=150, bbox_inches="tight")
    print(f"Saved: {COMPARISON_OUTPATH}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_timing_plot(results: dict):
    """Create a bar chart comparing solver timings."""
    solvers = list(results.keys())
    n = len(solvers)
    # Enough palette entries for up to 6 bars: blue, green, red, purple, orange, teal
    _asm_palette = ["#4878cf", "#6acc65", "#d65f5f", "#9b59b6", "#e67e22", "#1abc9c"]
    _slv_palette = ["#2c5282", "#3d7a3f", "#922b21", "#6c3483", "#9a5c0a", "#0e6655"]
    _tot_palette = ["#1a3a5c", "#234d27", "#5c1a1a", "#4a235a", "#6e4606", "#0a4136"]
    colors_asm = [_asm_palette[i % len(_asm_palette)] for i in range(n)]
    colors_slv = [_slv_palette[i % len(_slv_palette)] for i in range(n)]
    colors_tot = [_tot_palette[i % len(_tot_palette)] for i in range(n)]

    fig_w = max(13, 4 * n)  # widen automatically for more bars
    fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(fig_w, 6))

    def _bar(ax, ylim=None):
        x = np.arange(len(solvers))
        w = 0.55

        for i, (solver, d) in enumerate(results.items()):
            t_asm   = d.get("assembly") or 0.0
            t_slv   = d.get("solve")    or 0.0
            t_other = d["total"] - t_asm - t_slv  # CG overhead etc.

            if t_asm > 0:
                b1 = ax.bar(x[i], t_asm,   w, label="Assembly"  if i == 0 else "",
                            color=colors_asm[i],  alpha=0.85)
                b2 = ax.bar(x[i], t_slv,   w, bottom=t_asm,
                            label="Direct solve" if i == 0 else "",
                            color=colors_slv[i],  alpha=0.85)
                if t_other > 0.5:
                    ax.bar(x[i], t_other, w, bottom=t_asm + t_slv,
                           label="CG / other" if i == 0 else "",
                           color=colors_tot[i],  alpha=0.85)
            else:
                ax.bar(x[i], d["total"], w, color=colors_asm[i], alpha=0.85)

            text_y = d["total"] + (ylim or d["total"]) * 0.02
            if ylim is None or d["total"] < ylim:
                ax.text(x[i], text_y, f"{d['total']:.1f}s",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(solvers, fontsize=11)
        ax.set_ylabel("Wall-clock time (s)", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        if ylim:
            ax.set_ylim(0, ylim)
        return ax

    # Full-scale view
    _bar(ax_main)
    ax_main.set_title("Full timing (ernie head mesh, 1.3M elements)", fontsize=12)

    # Zoomed view to show assembly clearly
    _bar(ax_zoom, ylim=20)
    ax_zoom.set_title("Assembly detail (zoomed, first 20 s)", fontsize=12)
    ax_zoom.set_ylim(0, 20)

    # Legend
    patches = [
        mpatches.Patch(color="#4878cf", alpha=0.85, label="Assembly (stiffness + RHS)"),
        mpatches.Patch(color="#2c5282", alpha=0.85, label="Direct solve (spsolve)"),
        mpatches.Patch(color="#1a3a5c", alpha=0.85, label="CG iterations (Warp.fem)"),
    ]
    ax_main.legend(handles=patches, fontsize=9, loc="upper right")

    # Speedup annotation
    baseline = results[solvers[0]]["total"]
    for i, (solver, d) in enumerate(results.items()):
        if i == 0:
            continue
        speedup = baseline / d["total"]
        ax_main.annotate(
            f"{speedup:.2f}× faster",
            xy=(i, d["total"] / 2),
            ha="center", va="center", fontsize=9,
            color="white", fontweight="bold",
        )

    machine = f"{platform.system()} {platform.machine()}"
    fig.suptitle(
        f"Solver Timing — ernie head mesh (222k nodes, 1.3M elements, 6 tissues)\n"
        f"Dipole at (0, 0, 200 mm), dI/dt=1 MA/s  |  {machine}",
        fontsize=12, y=1.01,
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.12, wspace=0.3)
    fig.savefig(str(OUTPATH), dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPATH}")
    plt.close()


def print_table(results):
    print(f"\n{'Solver':<22} {'Assembly':>10} {'Solve':>10} {'Total':>10} {'Speedup':>10}")
    print("-" * 66)
    baseline = None
    for solver, d in results.items():
        if baseline is None:
            baseline = d["total"]
        speedup = baseline / d["total"]
        asm_s = f"{d['assembly']:.2f}s" if d.get("assembly") is not None else "   —"
        slv_s = f"{d['solve']:.2f}s"    if d.get("solve")    is not None else "   —"
        print(f"{solver:<22} {asm_s:>10} {slv_s:>10} {d['total']:>9.2f}s {speedup:>9.2f}×")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simnibs",   action="store_true",
                        help="Include SimNIBS timing (run with SimNIBS Python)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip solves; use cached timing JSON if available")
    args = parser.parse_args()

    if args.plot_only and TIMING_CACHE.exists():
        with open(TIMING_CACHE) as f:
            results = json.load(f)
        print("Loaded cached timings:")
    else:
        results = {}

        if args.simnibs:
            print("Timing SimNIBS TMSFEM ...")
            sn = time_simnibs()
            results["SimNIBS\nTMSFEM"] = sn
            print(f"  total: {sn['total']:.2f}s")

        if not ERNIE_MESH_PATH.exists():
            print(f"ERROR: {ERNIE_MESH_PATH} not found. Run scripts/fetch_ernie.py first.")
            sys.exit(1)

        print("Timing TMSWarp NumPy FEM ...")
        np_r, phi_np, _mesh, _dAdt, _G = time_tmswarp_numpy()
        results["TMSWarp\nNumPy FEM"] = np_r
        print(f"  assembly: {np_r['assembly']:.2f}s  "
              f"solve: {np_r['solve']:.2f}s  "
              f"total: {np_r['total']:.2f}s")
        phi_np_stats = (float(np.min(phi_np)), float(np.max(phi_np)),
                        float(np.std(phi_np)))
        print(f"  phi: min={phi_np_stats[0]:.4f}  max={phi_np_stats[1]:.4f}"
              f"  std={phi_np_stats[2]:.4f} V  (sanity: must be non-trivial)")
        if phi_np_stats[2] < 1e-10:
            print("  WARNING: NumPy phi is essentially zero — something is wrong.")

        from tmswarp.fields import compute_efield_at_elements, rdm, mag
        from tmswarp.solver_warp import warp_available

        # E-field from NumPy reference — used for validation and comparison figure
        E_np_ref = compute_efield_at_elements(_mesh, phi_np, _dAdt, _G)

        # Collect E-fields for the comparison figure: label → E array
        efields_for_plot = {"NumPy FEM": E_np_ref}

        def _run_warp(label, device):
            """Time Warp.fem on one device, validate result, add to results."""
            print(f"Timing TMSWarp Warp.fem ({label}) ...")
            wp_r, phi_wp = time_tmswarp_warp(mesh=_mesh, dAdt=_dAdt, device=device)
            if wp_r is None:
                print("  warp-lang not available, skipping.")
                return
            results[f"TMSWarp\nWarp.fem\n{label}"] = wp_r
            print(f"  total: {wp_r['total']:.2f}s")

            phi_std = float(np.std(phi_wp))
            print(f"  phi std={phi_std:.4f} V  (must be non-trivial)")
            if phi_std < 1e-10:
                print("  ERROR: Warp phi is essentially zero — CG may not have run.")
                return

            E_wp = compute_efield_at_elements(_mesh, phi_wp, _dAdt, _G)
            rdm_val = rdm(E_wp, E_np_ref)
            mag_val = mag(E_wp, E_np_ref)
            print(f"  Agreement with NumPy FEM:  RDM={rdm_val:.4f}  |MAG|={mag_val:.4f}")
            print(f"  (expected on ernie float32: RDM~0.5 due to 165:1 conductivity contrast)")
            if rdm_val > 1.0:
                print("  WARNING: RDM > 1.0 — result is likely wrong (check CG convergence).")

            efields_for_plot[f"Warp {label}"] = E_wp

        if warp_available():
            import warp as _wp
            _wp.init()
            # Always run on CPU explicitly, regardless of GPU availability
            _run_warp("CPU", device="cpu")
            # Run on GPU if present
            gpu_devices = [d for d in _wp.get_devices() if d.is_cuda]
            for gdev in gpu_devices:
                _run_warp(f"GPU:{gdev.ordinal}", device=str(gdev))
        else:
            print("Timing TMSWarp Warp.fem ... warp-lang not available, skipping.")

        with open(TIMING_CACHE, "w") as f:
            json.dump(results, f, indent=2)

        # Slice comparison figure (only when we actually computed E-fields)
        if len(efields_for_plot) > 1:
            print("\nGenerating solver comparison figure ...")
            make_solver_comparison_figure(efields_for_plot, _mesh)

    print_table(results)
    make_timing_plot(results)
