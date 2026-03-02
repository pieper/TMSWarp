"""Generate convergence visualization: analytical, FEM, and error maps at each mesh resolution.

Produces a grid of images showing cross-sections through the sphere at z=0:
- Row per mesh resolution
- Columns: mesh + analytical E-field, FEM E-field, relative error
- Shared colorbars so errors are directly comparable across resolutions.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize, LogNorm
import matplotlib.cm as cm

from tmswarp.conductor import make_sphere_mesh, element_barycenters, element_volumes
from tmswarp.coil import magnetic_dipole_dadt
from tmswarp.analytical import tms_analytical_efield
from tmswarp.solver import gradient_operator, assemble_stiffness, assemble_rhs_tms, solve_fem
from tmswarp.fields import compute_efield_at_elements, rdm, mag


def solve_at_resolution(n_shells, n_surface, min_quality=0.05):
    """Generate mesh, solve FEM, compute analytical, return everything needed for plotting."""
    radius = 0.095
    dipole_pos = np.array([0.0, 0.0, 0.3])
    dipole_moment = np.array([1.0, 0.0, 0.0])
    didt = 1e6

    mesh = make_sphere_mesh(radius=radius, n_shells=n_shells, n_surface=n_surface,
                            conductivity=1.0, min_quality=min_quality)

    dAdt = magnetic_dipole_dadt(dipole_pos, dipole_moment, didt, mesh.nodes)
    G = gradient_operator(mesh)
    K = assemble_stiffness(mesh, G)
    b = assemble_rhs_tms(mesh, dAdt, G)
    phi = solve_fem(K, b, pin_node=0)
    E_fem = compute_efield_at_elements(mesh, phi, dAdt, G)

    bary = element_barycenters(mesh)
    E_ana = tms_analytical_efield(dipole_pos, dipole_moment, didt, bary)

    r = rdm(E_fem, E_ana)
    m = mag(E_fem, E_ana)

    return {
        "mesh": mesh,
        "bary": bary,
        "E_fem": E_fem,
        "E_ana": E_ana,
        "rdm": r,
        "mag": m,
        "n_shells": n_shells,
        "n_surface": n_surface,
    }


def get_cross_section_elements(bary, thickness=0.008):
    """Find elements whose barycenters are near the z=0 plane."""
    return np.abs(bary[:, 2]) < thickness


def plot_cross_section(ax, bary_2d, triangles, values, title, norm, cmap, show_mesh=False):
    """Plot a filled triangulation of a cross-section."""
    if len(triangles) == 0:
        ax.set_title(title)
        return None

    tri = Triangulation(bary_2d[:, 0] * 1000, bary_2d[:, 1] * 1000, triangles)
    tc = ax.tripcolor(tri, values, cmap=cmap, norm=norm, shading="flat")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    # Draw sphere outline
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(95 * np.cos(theta), 95 * np.sin(theta), "k-", linewidth=0.8, alpha=0.5)

    return tc


def build_triangulation_for_slice(mesh, slice_mask):
    """Build a 2D triangulation from element barycenters in the slice.

    For each sliced element, project its barycenter to (x, y). Then build
    a Delaunay triangulation of those 2D points for visualization.
    """
    from scipy.spatial import Delaunay

    bary = mesh.nodes[mesh.elements].mean(axis=1)
    bary_slice = bary[slice_mask]
    pts_2d = bary_slice[:, :2]  # project to x-y

    if len(pts_2d) < 3:
        return pts_2d, np.array([], dtype=int).reshape(0, 3)

    tri = Delaunay(pts_2d)
    return pts_2d, tri.simplices


def main():
    # Mesh configurations to test
    configs = [
        (4, 100, 0.01),
        (6, 300, 0.01),
        (8, 500, 0.01),
        (10, 800, 0.01),
        (12, 1200, 0.05),
    ]

    print("Solving at each resolution...")
    results = []
    for n_shells, n_surface, min_q in configs:
        print("  shells=%d, surface=%d ..." % (n_shells, n_surface))
        res = solve_at_resolution(n_shells, n_surface, min_q)
        results.append(res)
        print("    %d nodes, %d elements, RDM=%.4f, MAG=%.4f" % (
            len(res["mesh"].nodes), len(res["mesh"].elements), res["rdm"], res["mag"]))

    # Find global color scale for E-field magnitude
    all_ana_norms = []
    all_fem_norms = []
    for res in results:
        sl = get_cross_section_elements(res["bary"])
        all_ana_norms.append(np.linalg.norm(res["E_ana"][sl], axis=1))
        all_fem_norms.append(np.linalg.norm(res["E_fem"][sl], axis=1))
    global_vmin = 0.0
    global_vmax = max(np.concatenate(all_ana_norms).max(),
                      np.concatenate(all_fem_norms).max())
    e_norm = Normalize(vmin=global_vmin, vmax=global_vmax)

    # Fixed error scale
    err_norm = Normalize(vmin=0, vmax=1.0)

    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for i, res in enumerate(results):
        mesh = res["mesh"]
        bary = res["bary"]
        E_fem = res["E_fem"]
        E_ana = res["E_ana"]

        # Cross section mask
        sl = get_cross_section_elements(bary)

        # Build 2D triangulation for plotting
        pts_2d, triangles = build_triangulation_for_slice(mesh, sl)

        # E-field magnitudes in slice
        norms_ana = np.linalg.norm(E_ana[sl], axis=1)
        norms_fem = np.linalg.norm(E_fem[sl], axis=1)

        # Per-element relative error
        rel_err = np.abs(norms_fem - norms_ana) / (norms_ana + 1e-30)
        # Clip for display
        rel_err = np.clip(rel_err, 0, 1.0)

        label = "%d elem (RDM=%.3f)" % (len(mesh.elements), res["rdm"])

        # Column 0: Analytical
        tc0 = plot_cross_section(
            axes[i, 0], pts_2d, triangles, norms_ana,
            "Analytical |E| — " + label, e_norm, "viridis")

        # Column 1: FEM
        tc1 = plot_cross_section(
            axes[i, 1], pts_2d, triangles, norms_fem,
            "FEM |E| — " + label, e_norm, "viridis")

        # Column 2: Relative error
        tc2 = plot_cross_section(
            axes[i, 2], pts_2d, triangles, rel_err,
            "Relative Error — " + label, err_norm, "hot")

    # Add colorbars
    # E-field colorbar (shared for columns 0-1)
    cbar_ax1 = fig.add_axes([0.05, 0.02, 0.55, 0.012])
    sm1 = cm.ScalarMappable(norm=e_norm, cmap="viridis")
    cb1 = fig.colorbar(sm1, cax=cbar_ax1, orientation="horizontal")
    cb1.set_label("|E| (V/m)")

    # Error colorbar (column 2)
    cbar_ax2 = fig.add_axes([0.68, 0.02, 0.25, 0.012])
    sm2 = cm.ScalarMappable(norm=err_norm, cmap="hot")
    cb2 = fig.colorbar(sm2, cax=cbar_ax2, orientation="horizontal")
    cb2.set_label("Relative Error |E_fem - E_ana| / |E_ana|")

    fig.suptitle(
        "TMS FEM Convergence: z=0 Cross-Section of Unit Sphere\n"
        "Dipole at (0, 0, 300mm), moment (1, 0, 0), dI/dt = 1 MA/s",
        fontsize=13, y=0.995)

    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    outpath = "/Users/pieper/slicer/latest/SlicerTMS/TMSWarp/convergence_visualization.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print("\nSaved to %s" % outpath)
    plt.close()

    # Also generate a convergence plot
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    n_elements = [len(r["mesh"].elements) for r in results]
    rdms = [r["rdm"] for r in results]
    mags = [abs(r["mag"]) for r in results]

    ax1.semilogy(n_elements, rdms, "o-", color="tab:blue", linewidth=2, markersize=8)
    ax1.axhline(y=0.2, color="red", linestyle="--", label="RDM threshold (0.2)")
    ax1.set_xlabel("Number of Elements")
    ax1.set_ylabel("RDM")
    ax1.set_title("Directional Accuracy (RDM)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(n_elements, mags, "o-", color="tab:orange", linewidth=2, markersize=8)
    ax2.axhline(y=np.log(1.1), color="red", linestyle="--", label="|MAG| threshold (log 1.1)")
    ax2.set_xlabel("Number of Elements")
    ax2.set_ylabel("|MAG|")
    ax2.set_title("Magnitude Accuracy (|MAG|)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig2.suptitle("FEM Convergence vs Analytical Solution", fontsize=13)
    fig2.tight_layout()
    outpath2 = "/Users/pieper/slicer/latest/SlicerTMS/TMSWarp/convergence_plot.png"
    fig2.savefig(outpath2, dpi=150, bbox_inches="tight")
    print("Saved to %s" % outpath2)
    plt.close()


if __name__ == "__main__":
    main()
