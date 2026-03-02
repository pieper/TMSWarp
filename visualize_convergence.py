"""Generate convergence visualization comparing analytical, NumPy FEM, and Warp.fem solvers.

Produces:
  convergence_visualization.png — z=0 cross-sections at each mesh resolution.
    Columns: Analytical |E|, NumPy FEM |E|, Warp.fem |E|, Relative Error (Warp vs Ana).
  convergence_plot.png — RDM and |MAG| vs element count for both FEM solvers.
  timing_plot.png — Wall-clock times for all three methods at each resolution.

All E-field colorbars are shared so magnitudes are directly comparable.
"""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.tri import Triangulation

from tmswarp.analytical import tms_analytical_efield
from tmswarp.coil import magnetic_dipole_dadt
from tmswarp.conductor import element_barycenters, element_volumes, make_sphere_mesh  # noqa: F401
from tmswarp.fields import compute_efield_at_elements, mag, rdm
from tmswarp.solver import assemble_rhs_tms, assemble_stiffness, gradient_operator, solve_fem
from tmswarp.solver_warp import solve_fem_warp, warp_available

# ---------------------------------------------------------------------------
# Problem parameters (SI units throughout)
# ---------------------------------------------------------------------------
RADIUS = 0.095          # sphere radius, m
DIPOLE_POS = np.array([0.0, 0.0, 0.3])    # m
DIPOLE_MOMENT = np.array([1.0, 0.0, 0.0])  # unit vector
DIDT = 1e6              # A/s
CONDUCTIVITY = 1.0      # S/m

OUTDIR = "/Users/pieper/slicer/latest/SlicerTMS/TMSWarp"


# ---------------------------------------------------------------------------
# Solver helpers
# ---------------------------------------------------------------------------

def solve_numpy(mesh, dAdt_nodes):
    """NumPy FEM solve; returns (phi, timing_s)."""
    t0 = time.perf_counter()
    G = gradient_operator(mesh)
    K = assemble_stiffness(mesh, G)
    b = assemble_rhs_tms(mesh, dAdt_nodes, G)
    phi = solve_fem(K, b, pin_node=0)
    return phi, G, time.perf_counter() - t0


def solve_warp(mesh, dAdt_nodes):
    """Warp.fem solve; returns (phi, timing_s) or (None, None) if unavailable."""
    if not warp_available():
        return None, None
    t0 = time.perf_counter()
    phi = solve_fem_warp(mesh, dAdt_nodes, quiet=True)
    return phi, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Per-resolution solver
# ---------------------------------------------------------------------------

def solve_at_resolution(n_shells, n_surface, min_quality=0.05):
    """Generate mesh, solve with all methods, return dict of results."""
    mesh = make_sphere_mesh(
        radius=RADIUS, n_shells=n_shells, n_surface=n_surface,
        conductivity=CONDUCTIVITY, min_quality=min_quality,
    )

    dAdt = magnetic_dipole_dadt(DIPOLE_POS, DIPOLE_MOMENT, DIDT, mesh.nodes)

    # Analytical E-field at barycenters
    bary = element_barycenters(mesh)
    E_ana = tms_analytical_efield(DIPOLE_POS, DIPOLE_MOMENT, DIDT, bary)

    # NumPy FEM
    phi_np, G, t_np = solve_numpy(mesh, dAdt)
    E_fem_np = compute_efield_at_elements(mesh, phi_np, dAdt, G)

    # Warp.fem
    phi_wp, t_wp = solve_warp(mesh, dAdt)
    if phi_wp is not None:
        E_fem_wp = compute_efield_at_elements(mesh, phi_wp, dAdt, G)
    else:
        E_fem_wp = None

    return {
        "mesh": mesh,
        "bary": bary,
        "E_ana": E_ana,
        "E_fem_np": E_fem_np,
        "E_fem_wp": E_fem_wp,
        "rdm_np": rdm(E_fem_np, E_ana),
        "mag_np": mag(E_fem_np, E_ana),
        "rdm_wp": rdm(E_fem_wp, E_ana) if E_fem_wp is not None else None,
        "mag_wp": mag(E_fem_wp, E_ana) if E_fem_wp is not None else None,
        "t_np": t_np,
        "t_wp": t_wp,
        "n_shells": n_shells,
        "n_surface": n_surface,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def get_cross_section_mask(bary, thickness=0.008):
    return np.abs(bary[:, 2]) < thickness


def build_2d_triangulation(mesh, mask):
    """Build a 2D Delaunay triangulation of element barycenters in the slice."""
    from scipy.spatial import Delaunay

    bary = mesh.nodes[mesh.elements].mean(axis=1)
    pts_2d = bary[mask, :2]

    if len(pts_2d) < 3:
        return pts_2d, np.array([], dtype=int).reshape(0, 3)

    return pts_2d, Delaunay(pts_2d).simplices


def plot_slice(ax, pts_2d, triangles, values, title, norm, cmap):
    """Filled tripcolor plot of a z=0 cross-section."""
    if len(triangles) == 0:
        ax.set_title(title)
        return None

    tri = Triangulation(pts_2d[:, 0] * 1000, pts_2d[:, 1] * 1000, triangles)
    tc = ax.tripcolor(tri, values, cmap=cmap, norm=norm, shading="flat")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(95 * np.cos(theta), 95 * np.sin(theta), "k-", linewidth=0.8, alpha=0.5)

    return tc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    configs = [
        (4, 100,  0.01),
        (6, 300,  0.01),
        (8, 500,  0.01),
        (10, 800, 0.01),
        (12, 1200, 0.05),
    ]

    print("Solving at each resolution (analytical + NumPy FEM + Warp.fem)...")
    results = []
    for n_shells, n_surface, min_q in configs:
        print(f"  shells={n_shells}, surface={n_surface} ...")
        res = solve_at_resolution(n_shells, n_surface, min_q)
        results.append(res)
        t_wp_str = f"{res['t_wp']:.2f}s" if res['t_wp'] is not None else "N/A"
        print(
            f"    {len(res['mesh'].nodes)} nodes, {len(res['mesh'].elements)} elems | "
            f"numpy: RDM={res['rdm_np']:.4f} MAG={res['mag_np']:.4f} t={res['t_np']:.2f}s | "
            f"warp:  RDM={res['rdm_wp']:.4f} MAG={res['mag_wp']:.4f} t={t_wp_str}"
            if res['rdm_wp'] is not None else
            f"    {len(res['mesh'].nodes)} nodes, {len(res['mesh'].elements)} elems | "
            f"numpy: RDM={res['rdm_np']:.4f} MAG={res['mag_np']:.4f} t={res['t_np']:.2f}s | "
            f"warp: N/A"
        )

    # -----------------------------------------------------------------------
    # Figure 1: Cross-section visualization
    # -----------------------------------------------------------------------
    # Global colour scale from all E-field magnitudes in cross-section
    all_norms = []
    for res in results:
        sl = get_cross_section_mask(res["bary"])
        all_norms.append(np.linalg.norm(res["E_ana"][sl], axis=1))
        all_norms.append(np.linalg.norm(res["E_fem_np"][sl], axis=1))
        if res["E_fem_wp"] is not None:
            all_norms.append(np.linalg.norm(res["E_fem_wp"][sl], axis=1))
    global_vmax = np.concatenate(all_norms).max()
    e_norm = Normalize(vmin=0.0, vmax=global_vmax)
    err_norm = Normalize(vmin=0, vmax=1.0)

    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for i, res in enumerate(results):
        mesh = res["mesh"]
        bary = res["bary"]
        sl = get_cross_section_mask(bary)
        pts_2d, triangles = build_2d_triangulation(mesh, sl)

        norms_ana = np.linalg.norm(res["E_ana"][sl], axis=1)
        norms_np = np.linalg.norm(res["E_fem_np"][sl], axis=1)
        norms_wp = (np.linalg.norm(res["E_fem_wp"][sl], axis=1)
                    if res["E_fem_wp"] is not None else norms_np * np.nan)

        rel_err_wp = np.clip(
            np.abs(norms_wp - norms_ana) / (norms_ana + 1e-30), 0, 1.0
        )

        n_elem = len(mesh.elements)
        rdm_np_str = f"RDM={res['rdm_np']:.3f}"
        rdm_wp_str = f"RDM={res['rdm_wp']:.3f}" if res["rdm_wp"] is not None else "N/A"

        plot_slice(axes[i, 0], pts_2d, triangles, norms_ana,
                   f"Analytical |E| — {n_elem} elem", e_norm, "viridis")
        plot_slice(axes[i, 1], pts_2d, triangles, norms_np,
                   f"NumPy FEM |E| — {rdm_np_str}", e_norm, "viridis")
        plot_slice(axes[i, 2], pts_2d, triangles, norms_wp,
                   f"Warp.fem |E| — {rdm_wp_str}", e_norm, "viridis")
        plot_slice(axes[i, 3], pts_2d, triangles, rel_err_wp,
                   f"Warp Relative Error vs Ana", err_norm, "hot")

    # Shared colorbars
    cbar_ax1 = fig.add_axes([0.04, 0.02, 0.65, 0.012])
    sm1 = cm.ScalarMappable(norm=e_norm, cmap="viridis")
    cb1 = fig.colorbar(sm1, cax=cbar_ax1, orientation="horizontal")
    cb1.set_label("|E| (V/m)")

    cbar_ax2 = fig.add_axes([0.74, 0.02, 0.22, 0.012])
    sm2 = cm.ScalarMappable(norm=err_norm, cmap="hot")
    cb2 = fig.colorbar(sm2, cax=cbar_ax2, orientation="horizontal")
    cb2.set_label("Relative Error |E_warp - E_ana| / |E_ana|")

    fig.suptitle(
        "TMS FEM Convergence: z=0 Cross-Section\n"
        "Dipole at (0,0,300mm), moment (1,0,0), dI/dt=1MA/s",
        fontsize=13, y=0.998,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    out1 = f"{OUTDIR}/convergence_visualization.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out1}")
    plt.close()

    # -----------------------------------------------------------------------
    # Figure 2: RDM / MAG convergence curves
    # -----------------------------------------------------------------------
    n_elements = [len(r["mesh"].elements) for r in results]
    rdms_np = [r["rdm_np"] for r in results]
    mags_np = [abs(r["mag_np"]) for r in results]
    rdms_wp = [r["rdm_wp"] for r in results if r["rdm_wp"] is not None]
    mags_wp = [abs(r["mag_wp"]) for r in results if r["mag_wp"] is not None]
    nelems_wp = [len(r["mesh"].elements) for r in results if r["rdm_wp"] is not None]

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.semilogy(n_elements, rdms_np, "o-", color="tab:blue", lw=2, ms=7, label="NumPy FEM")
    if rdms_wp:
        ax1.semilogy(nelems_wp, rdms_wp, "s--", color="tab:orange", lw=2, ms=7, label="Warp.fem")
    ax1.axhline(y=0.2, color="red", linestyle=":", label="RDM threshold (0.2)")
    ax1.set_xlabel("Number of Elements")
    ax1.set_ylabel("RDM")
    ax1.set_title("Directional Accuracy (RDM)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(n_elements, mags_np, "o-", color="tab:blue", lw=2, ms=7, label="NumPy FEM")
    if mags_wp:
        ax2.semilogy(nelems_wp, mags_wp, "s--", color="tab:orange", lw=2, ms=7, label="Warp.fem")
    ax2.axhline(y=np.log(1.1), color="red", linestyle=":", label="|MAG| threshold (log 1.1)")
    ax2.set_xlabel("Number of Elements")
    ax2.set_ylabel("|MAG|")
    ax2.set_title("Magnitude Accuracy (|MAG|)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig2.suptitle("FEM Convergence vs Analytical Solution", fontsize=13)
    fig2.tight_layout()
    out2 = f"{OUTDIR}/convergence_plot.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close()

    # -----------------------------------------------------------------------
    # Figure 3: Timing comparison
    # -----------------------------------------------------------------------
    t_np_vals = [r["t_np"] for r in results]
    t_wp_vals = [r["t_wp"] for r in results]
    has_warp_timing = any(t is not None for t in t_wp_vals)

    fig3, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(n_elements, t_np_vals, "o-", color="tab:blue", lw=2, ms=7, label="NumPy FEM (scipy direct)")
    if has_warp_timing:
        t_wp_plot = [t for t in t_wp_vals if t is not None]
        ne_wp_plot = [n for n, t in zip(n_elements, t_wp_vals) if t is not None]
        ax.semilogy(ne_wp_plot, t_wp_plot, "s--", color="tab:orange", lw=2, ms=7, label="Warp.fem (CG, CPU)")
    ax.set_xlabel("Number of Elements")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Solver Timing vs Mesh Size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate with actual element counts
    for ne, tnp in zip(n_elements, t_np_vals):
        ax.annotate(f"{ne}", (ne, tnp), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="tab:blue")

    import platform
    machine_info = f"{platform.system()} {platform.machine()}"
    fig3.suptitle(f"Solver Timing Comparison — {machine_info}", fontsize=13)
    fig3.tight_layout()
    out3 = f"{OUTDIR}/timing_plot.png"
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"Saved: {out3}")
    plt.close()

    # Summary table
    print("\n--- Summary ---")
    print(f"{'Elements':>10}  {'NumPy RDM':>10}  {'Warp RDM':>10}  {'NumPy t(s)':>11}  {'Warp t(s)':>10}")
    for r in results:
        ne = len(r["mesh"].elements)
        wrdm = f"{r['rdm_wp']:.4f}" if r["rdm_wp"] is not None else "   N/A"
        wt = f"{r['t_wp']:.3f}" if r["t_wp"] is not None else "   N/A"
        print(f"{ne:>10}  {r['rdm_np']:>10.4f}  {wrdm:>10}  {r['t_np']:>11.3f}  {wt:>10}")


if __name__ == "__main__":
    main()
