"""Compare TMSWarp vs SimNIBS E-field on the ernie human head mesh.

Generates ernie_comparison.png — a 3-row × 4-column figure showing axial,
coronal, and sagittal slices of |E| for both solvers and the difference map.

Prerequistes
-----------
1. ernie_data.npz   (run scripts/fetch_ernie.py with SimNIBS Python)
2. ernie_simnibs_efield.npz  (run scripts/run_simnibs_efield.py with SimNIBS Python)

Usage
-----
    pixi run python benchmarks/ernie_comparison.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.interpolate import griddata

from tmswarp.coil import magnetic_dipole_dadt
from tmswarp.conductor import TetMesh, element_barycenters
from tmswarp.fields import compute_efield_at_elements
from tmswarp.solver import assemble_rhs_tms, assemble_stiffness, gradient_operator, solve_fem

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
ERNIE_MESH_PATH   = ROOT / "ernie_data.npz"
SIMNIBS_PATH      = ROOT / "ernie_simnibs_efield.npz"
TMSWARP_CACHE     = ROOT / "ernie_tmswarp_efield.npz"
OUTPATH           = ROOT / "ernie_comparison.png"

# ---------------------------------------------------------------------------
# Dipole parameters (must match scripts/run_simnibs_efield.py)
# ---------------------------------------------------------------------------
DIPOLE_POS_M  = np.array([0.0, 0.0, 0.200])
DIPOLE_MOMENT = np.array([1.0, 0.0, 0.0])
DIDT          = 1e6

# Slice planes in mm (head coordinates)
AXIAL_Z    =  40.0   # mm — through upper cortex
CORONAL_Y  =  50.0   # mm — through precentral gyrus
SAGITTAL_X =   0.0   # mm — midline

SLAB_HALF  =  4.0    # mm — half-thickness of each slice slab
GRID_N     = 200     # pixels per axis for interpolated slice image

TISSUE_NAMES = {1: "WM", 2: "GM", 3: "CSF", 4: "skull", 5: "scalp", 6: "eyes"}

# ---------------------------------------------------------------------------
# Load TMSWarp mesh and compute E-field
# ---------------------------------------------------------------------------

def load_tmswarp_efield():
    """Load TMSWarp NumPy FEM result from cache, or compute and cache it."""
    if TMSWARP_CACHE.exists():
        print(f"Loading cached TMSWarp result from {TMSWARP_CACHE.name} ...")
        d = np.load(TMSWARP_CACHE)
        E, bary, tag1 = d["E"], d["bary_mm"], d["tag1"]
        print(f"  TMSWarp |E| mean={np.linalg.norm(E,axis=1).mean():.3f} max={np.linalg.norm(E,axis=1).max():.3f} V/m")
        return E, bary, tag1

    if not ERNIE_MESH_PATH.exists():
        raise FileNotFoundError(
            f"Missing {ERNIE_MESH_PATH}\nRun: /path/to/SimNIBS-4.5/.../python scripts/fetch_ernie.py"
        )
    print("Running TMSWarp NumPy FEM (first time; will be cached) ...")
    data = np.load(ERNIE_MESH_PATH)
    mesh = TetMesh(
        nodes=data["nodes"].astype(np.float64),
        elements=data["elements"].astype(np.int32),
        conductivity=data["conductivity"].astype(np.float64),
    )
    tag1 = data["tag1"].astype(np.int32)

    dAdt = magnetic_dipole_dadt(DIPOLE_POS_M, DIPOLE_MOMENT, DIDT, mesh.nodes)
    G  = gradient_operator(mesh)
    K  = assemble_stiffness(mesh, G)
    b  = assemble_rhs_tms(mesh, dAdt, G)
    phi = solve_fem(K, b, pin_node=0)
    E  = compute_efield_at_elements(mesh, phi, dAdt, G)

    bary = element_barycenters(mesh) * 1000  # m → mm
    np.savez_compressed(str(TMSWARP_CACHE), E=E, bary_mm=bary, tag1=tag1)
    print(f"  Cached to {TMSWARP_CACHE.name}")
    print(f"  TMSWarp |E| mean={np.linalg.norm(E,axis=1).mean():.3f} max={np.linalg.norm(E,axis=1).max():.3f} V/m")
    return E, bary, tag1


def load_simnibs_efield():
    if not SIMNIBS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SIMNIBS_PATH}\n"
            "Run: /path/to/SimNIBS-4.5/.../python scripts/run_simnibs_efield.py"
        )
    data = np.load(SIMNIBS_PATH)
    E      = data["E"]
    bary   = data["bary_mm"]
    tag1   = data["tag1"]
    print(f"  SimNIBS  |E| mean={np.linalg.norm(E,axis=1).mean():.3f} max={np.linalg.norm(E,axis=1).max():.3f} V/m")
    return E, bary, tag1


# ---------------------------------------------------------------------------
# Slice helpers
# ---------------------------------------------------------------------------

def _slab_mask(bary, axis: int, plane_val: float, half: float):
    """Boolean mask selecting elements within a slab around a plane."""
    return np.abs(bary[:, axis] - plane_val) < half


def _slice_axes(axis: int):
    """Return the two axes parallel to the slice plane."""
    return [a for a in range(3) if a != axis]


def _make_grid(bary, mask, axis: int):
    """Build a regular 2D grid covering the slice footprint."""
    ax0, ax1 = _slice_axes(axis)
    pts = bary[mask]
    lo0, hi0 = pts[:, ax0].min(), pts[:, ax0].max()
    lo1, hi1 = pts[:, ax1].min(), pts[:, ax1].max()
    xi = np.linspace(lo0, hi0, GRID_N)
    yi = np.linspace(lo1, hi1, GRID_N)
    return xi, yi, np.meshgrid(xi, yi)


def _interp_slice(bary, values, mask, axis: int):
    """Interpolate scalar values to a 2D regular grid for a cross-section."""
    ax0, ax1 = _slice_axes(axis)
    pts   = bary[mask][:, [ax0, ax1]]
    vals  = values[mask]
    xi, yi, (Xg, Yg) = _make_grid(bary, mask, axis)
    Zi = griddata(pts, vals, (Xg, Yg), method="linear")
    return xi, yi, Zi


def _axis_labels(axis: int):
    labels = ["x (mm)", "y (mm)", "z (mm)"]
    ax0, ax1 = _slice_axes(axis)
    return labels[ax0], labels[ax1]


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_comparison_figure(E_tw, bary_tw, E_sn, bary_sn):
    """Create a 3-row × 4-column figure comparing TMSWarp vs SimNIBS slices."""

    mag_tw = np.linalg.norm(E_tw, axis=1)
    mag_sn = np.linalg.norm(E_sn, axis=1)

    # Global |E| colour scale (99th percentile to suppress outliers)
    vmax = np.percentile(np.concatenate([mag_tw, mag_sn]), 99)
    e_norm = Normalize(vmin=0, vmax=vmax)

    # Difference  (TMSWarp - SimNIBS)
    diff = mag_tw - mag_sn
    dlim = np.percentile(np.abs(diff), 98)
    diff_norm = TwoSlopeNorm(vcenter=0, vmin=-dlim, vmax=dlim)

    # Relative difference (percent)
    ref = np.maximum(mag_sn, 1e-9)
    rel_diff = (mag_tw - mag_sn) / ref * 100   # percent
    rlim = np.percentile(np.abs(rel_diff), 97)
    rel_norm = TwoSlopeNorm(vcenter=0, vmin=-rlim, vmax=rlim)

    slices = [
        # (axis_int, plane_value_mm, row_label)
        (2, AXIAL_Z,    f"Axial  z={AXIAL_Z:.0f} mm"),
        (1, CORONAL_Y,  f"Coronal  y={CORONAL_Y:.0f} mm"),
        (0, SAGITTAL_X, f"Sagittal  x={SAGITTAL_X:.0f} mm"),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(22, 16))

    for row, (axis, plane_val, row_label) in enumerate(slices):
        mask_tw = _slab_mask(bary_tw, axis, plane_val, SLAB_HALF)
        mask_sn = _slab_mask(bary_sn, axis, plane_val, SLAB_HALF)

        xlabel, ylabel = _axis_labels(axis)

        # Interpolate
        xi_tw, yi_tw, Zi_tw = _interp_slice(bary_tw, mag_tw, mask_tw, axis)
        xi_sn, yi_sn, Zi_sn = _interp_slice(bary_sn, mag_sn, mask_sn, axis)
        xi_di, yi_di, Zi_di = _interp_slice(bary_tw, diff, mask_tw, axis)
        xi_ri, yi_ri, Zi_ri = _interp_slice(bary_tw, rel_diff, mask_tw, axis)

        def _show(ax, xi, yi, Zi, norm, cmap, title):
            extent = [xi[0], xi[-1], yi[0], yi[-1]]
            im = ax.imshow(
                Zi, origin="lower", extent=extent,
                norm=norm, cmap=cmap, aspect="equal", interpolation="bilinear",
            )
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)
            return im

        im_tw = _show(axes[row, 0], xi_tw, yi_tw, Zi_tw, e_norm, "hot", "TMSWarp |E|")
        im_sn = _show(axes[row, 1], xi_sn, yi_sn, Zi_sn, e_norm, "hot", "SimNIBS |E|")
        im_di = _show(axes[row, 2], xi_di, yi_di, Zi_di, diff_norm, "RdBu_r",
                      "Diff: TMSWarp − SimNIBS (V/m)")
        im_ri = _show(axes[row, 3], xi_ri, yi_ri, Zi_ri, rel_norm, "RdBu_r",
                      "Rel diff (%)")

        # Row label
        axes[row, 0].set_ylabel(f"{row_label}\n{ylabel}", fontsize=8)

    # Colorbars
    fig.subplots_adjust(left=0.06, right=0.95, top=0.93, bottom=0.10, hspace=0.45, wspace=0.35)

    # |E| colorbar spans columns 0-1
    cb_e = fig.colorbar(
        cm.ScalarMappable(norm=e_norm, cmap="hot"),
        ax=axes[:, :2], shrink=0.6, pad=0.02, orientation="horizontal",
        fraction=0.03,
    )
    cb_e.set_label("|E| (V/m)", fontsize=10)

    # Abs diff colorbar for column 2
    cb_d = fig.colorbar(
        cm.ScalarMappable(norm=diff_norm, cmap="RdBu_r"),
        ax=axes[:, 2], shrink=0.6, pad=0.02, orientation="horizontal",
        fraction=0.03,
    )
    cb_d.set_label("Δ|E| (V/m)", fontsize=10)

    # Rel diff colorbar for column 3
    cb_r = fig.colorbar(
        cm.ScalarMappable(norm=rel_norm, cmap="RdBu_r"),
        ax=axes[:, 3], shrink=0.6, pad=0.02, orientation="horizontal",
        fraction=0.03,
    )
    cb_r.set_label("Rel diff (%)", fontsize=10)

    # Global title
    rdm_val = float(np.linalg.norm(
        E_tw / np.linalg.norm(E_tw) - E_sn / np.linalg.norm(E_sn)
    ))
    mag_val = float(abs(np.log(np.linalg.norm(E_tw) / np.linalg.norm(E_sn))))
    fig.suptitle(
        f"TMSWarp vs SimNIBS — ernie head mesh\n"
        f"Dipole at (0, 0, 200 mm), dI/dt=1 MA/s  |  "
        f"Global RDM={rdm_val:.3f}  |MAG|={mag_val:.3f}",
        fontsize=13, y=0.98,
    )

    fig.savefig(str(OUTPATH), dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUTPATH}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=" * 66)
    print("TMSWarp vs SimNIBS ernie comparison")
    print("=" * 66)

    print("\nLoading SimNIBS result ...")
    E_sn, bary_sn, tag1_sn = load_simnibs_efield()

    print("\nComputing TMSWarp result ...")
    t0 = time.perf_counter()
    E_tw, bary_tw, tag1_tw = load_tmswarp_efield()
    t_tw = time.perf_counter() - t0
    print(f"  TMSWarp solve: {t_tw:.1f} s")

    print("\nGenerating comparison figure ...")
    make_comparison_figure(E_tw, bary_tw, E_sn, bary_sn)
