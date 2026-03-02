"""Warp.fem GPU-accelerated FEM solver for TMS.

Solves the quasistatic Poisson equation:

    div(sigma * grad(phi)) = -div(sigma * dA/dt)

using P1 (linear) tetrahedral finite elements via Nvidia Warp's FEM framework.

The weak form (natural Neumann BC, after integration by parts):

    integral sigma grad(phi) . grad(v) dV = -integral sigma dA/dt . grad(v) dV

Gauge condition (phi[pin_node]=0) is enforced via a Dirichlet projector.
The linear system is solved with Conjugate Gradient (bsr_cg).

Notes
-----
- warp.fem.Tetmesh requires float32 node positions; all FEM quantities are
  therefore computed in float32.  The resulting phi array is converted to
  float64 before return so it is compatible with the existing numpy pipeline.
- Integrands must live at module scope (warp uses inspect.getsource for JIT).
- Call ``warp_available()`` to check whether warp-lang is installed before
  using the solver.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Lazy warp initialisation
# ---------------------------------------------------------------------------
_warp_initialized = False


def _init_warp():
    """Initialize Warp runtime (safe to call multiple times)."""
    global _warp_initialized
    if not _warp_initialized:
        import warp as wp
        wp.init()
        _warp_initialized = True


def warp_available() -> bool:
    """Return True if warp-lang is installed and importable."""
    try:
        import warp  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Warp.fem integrands  (module-level so inspect.getsource can find them)
# ---------------------------------------------------------------------------
# These are defined unconditionally at import time only when warp is present.
# If warp is absent the module still imports fine; the integrands are None.

try:
    import warp as wp
    import warp.fem as fem

    @fem.integrand
    def _tms_stiffness_form(
        s: fem.Sample,
        u: fem.Field,
        v: fem.Field,
        sigma: wp.array(dtype=wp.float32),
    ):
        """Bilinear stiffness form:  sigma_e * grad(u) · grad(v)"""
        cell_sigma = sigma[s.element_index]
        return cell_sigma * wp.dot(fem.grad(u, s), fem.grad(v, s))

    @fem.integrand
    def _tms_rhs_form(
        s: fem.Sample,
        v: fem.Field,
        dAdt: fem.Field,
        sigma: wp.array(dtype=wp.float32),
    ):
        """Linear RHS form:  -sigma_e * dAdt · grad(v)

        ``dAdt`` is a discrete P1 vec3f field interpolated at sample point.
        """
        cell_sigma = sigma[s.element_index]
        return -cell_sigma * wp.dot(dAdt(s), fem.grad(v, s))

    _WARP_INTEGRANDS_DEFINED = True

except ImportError:
    _WARP_INTEGRANDS_DEFINED = False


# ---------------------------------------------------------------------------
# Gauge projector helper
# ---------------------------------------------------------------------------

def _make_gauge_projector(n_nodes: int, pin_node: int, device: str):
    """Build a single-entry BSR projector P with P[pin_node, pin_node] = 1.

    ``fem.project_linear_system(K, b, P, x0)`` enforces phi[pin_node] = 0.
    """
    import warp as wp
    from warp.sparse import bsr_zeros, bsr_set_from_triplets

    projector = bsr_zeros(n_nodes, n_nodes, block_type=wp.float32, device=device)
    rows = wp.array([pin_node], dtype=int, device=device)
    cols = wp.array([pin_node], dtype=int, device=device)
    vals = wp.array([1.0], dtype=wp.float32, device=device)
    bsr_set_from_triplets(projector, rows, cols, vals)
    return projector


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------

def solve_fem_warp(
    mesh,
    dAdt_nodes: np.ndarray,
    pin_node: int = 0,
    device: str = "cpu",
    quiet: bool = True,
) -> np.ndarray:
    """Warp.fem solver for the TMS Poisson equation.

    Uses the same physics as ``tmswarp.solver.solve_fem`` but assembles the
    stiffness matrix and RHS using Warp's FEM kernels (GPU-ready).

    Parameters
    ----------
    mesh : TetMesh
        Tetrahedral mesh (nodes in metres, per-element conductivity in S/m).
    dAdt_nodes : (n_nodes, 3) array
        Primary field dA/dt at mesh nodes (V/m).
    pin_node : int
        Node index to pin to phi = 0 (gauge condition).
    device : str
        Warp device string, e.g. ``"cpu"`` or ``"cuda:0"``.
    quiet : bool
        Suppress CG iteration residual output.

    Returns
    -------
    (n_nodes,) float64 array
        Scalar potential phi at each node.

    Raises
    ------
    ImportError
        If warp-lang is not installed.
    """
    if not _WARP_INTEGRANDS_DEFINED:
        raise ImportError(
            "warp-lang is required for solve_fem_warp. "
            "Install it with: pip install warp-lang"
        )

    _init_warp()

    import warp as wp
    import warp.fem as fem
    from warp.examples.fem.utils import bsr_cg

    # ------------------------------------------------------------------
    # Build warp.fem geometry  (Tetmesh requires float32 positions)
    # ------------------------------------------------------------------
    positions = wp.array(
        mesh.nodes.astype(np.float32), dtype=wp.vec3f, device=device
    )
    tet_indices = wp.array(
        mesh.elements.astype(np.int32), dtype=int, device=device
    )
    geo = fem.Tetmesh(tet_indices, positions)

    # ------------------------------------------------------------------
    # P1 function spaces  (float32 matches geometry scalar type)
    # ------------------------------------------------------------------
    phi_space = fem.make_polynomial_space(geo, dtype=wp.float32, degree=1)
    dAdt_space = fem.make_polynomial_space(geo, dtype=wp.vec3f, degree=1)

    # Populate dA/dt discrete field from nodal values
    dAdt_discrete = dAdt_space.make_field()
    dAdt_discrete.dof_values = wp.array(
        dAdt_nodes.astype(np.float32), dtype=wp.vec3f, device=device
    )

    # Per-element conductivity (float32)
    sigma_wp = wp.array(
        mesh.conductivity.astype(np.float32), dtype=wp.float32, device=device
    )

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------
    domain = fem.Cells(geometry=geo)
    test = fem.make_test(space=phi_space, domain=domain)
    trial = fem.make_trial(space=phi_space, domain=domain)

    K = fem.integrate(
        _tms_stiffness_form,
        fields={"u": trial, "v": test},
        values={"sigma": sigma_wp},
        output_dtype=wp.float32,
    )

    b = fem.integrate(
        _tms_rhs_form,
        fields={"v": test, "dAdt": dAdt_discrete},
        values={"sigma": sigma_wp},
        output_dtype=wp.float32,
    )

    # ------------------------------------------------------------------
    # Gauge condition: pin phi[pin_node] = 0
    # ------------------------------------------------------------------
    n_nodes = len(mesh.nodes)
    projector = _make_gauge_projector(n_nodes, pin_node, device=device)
    fixed_val = wp.zeros(n_nodes, dtype=wp.float32, device=device)
    fem.project_linear_system(K, b, projector, fixed_val)

    # ------------------------------------------------------------------
    # Conjugate Gradient solve
    # ------------------------------------------------------------------
    x = wp.zeros(n_nodes, dtype=wp.float32, device=device)
    bsr_cg(K, b=b, x=x, quiet=quiet)

    # Return as float64 for compatibility with the rest of the pipeline
    return x.numpy().astype(np.float64)
