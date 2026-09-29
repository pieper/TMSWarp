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

    @wp.func
    def _element_efield(
        phi: wp.array(dtype=wp.float32),
        elements: wp.array2d(dtype=wp.int32),
        G: wp.array2d(dtype=wp.float32),
        dAdt_nodes: wp.array(dtype=wp.vec3f),
        e: int,
    ):
        """E = -grad(phi) - dA/dt_bary in element e (see _compute_enorm_kernel)."""
        n0 = elements[e, 0]
        n1 = elements[e, 1]
        n2 = elements[e, 2]
        n3 = elements[e, 3]

        g0 = wp.vec3f(G[e, 0], G[e, 1], G[e, 2])
        g1 = wp.vec3f(G[e, 3], G[e, 4], G[e, 5])
        g2 = wp.vec3f(G[e, 6], G[e, 7], G[e, 8])
        g3 = wp.vec3f(G[e, 9], G[e, 10], G[e, 11])

        grad_phi = phi[n0] * g0 + phi[n1] * g1 + phi[n2] * g2 + phi[n3] * g3
        dAdt_bary = (
            dAdt_nodes[n0] + dAdt_nodes[n1] + dAdt_nodes[n2] + dAdt_nodes[n3]
        ) * 0.25
        return -grad_phi - dAdt_bary

    @wp.kernel
    def _dipole_dadt_kernel(
        nodes: wp.array(dtype=wp.vec3f),
        dipole_pos: wp.array(dtype=wp.vec3f),
        dipole_moment: wp.array(dtype=wp.vec3f),
        scale: wp.float32,
        dAdt: wp.array(dtype=wp.vec3f),
    ):
        """Differentiable version of ``tmswarp.coil.magnetic_dipole_dadt``.

        ``dipole_pos`` and ``dipole_moment`` are length-1 arrays so that
        gradients with respect to them can be recorded on a ``wp.Tape``.
        ``scale`` is (mu0/4pi) * dI/dt.
        """
        i = wp.tid()
        r = nodes[i] - dipole_pos[0]
        d = wp.length(r)
        m = wp.normalize(dipole_moment[0])
        dAdt[i] = scale * wp.cross(m, r) / (d * d * d)

    @wp.kernel
    def _target_loss_kernel(
        phi: wp.array(dtype=wp.float32),
        elements: wp.array2d(dtype=wp.int32),
        G: wp.array2d(dtype=wp.float32),
        dAdt_nodes: wp.array(dtype=wp.vec3f),
        target_elem: int,
        loss: wp.array(dtype=wp.float32),
    ):
        """loss = -|E| in the target element (minimizing maximizes |E|)."""
        E = _element_efield(phi, elements, G, dAdt_nodes, target_elem)
        loss[0] = -wp.length(E)

    @wp.kernel
    def _scale_kernel(
        src: wp.array(dtype=wp.float32),
        s: wp.float32,
        dst: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        dst[i] = s * src[i]

    @wp.kernel
    def _add_kernel(
        src: wp.array(dtype=wp.float32),
        dst: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        dst[i] = dst[i] + src[i]

    @wp.kernel
    def _zero_entry_kernel(
        a: wp.array(dtype=wp.float32),
        index: int,
    ):
        a[index] = 0.0

    @wp.kernel
    def _compute_enorm_kernel(
        phi: wp.array(dtype=wp.float32),
        elements: wp.array2d(dtype=wp.int32),
        G: wp.array2d(dtype=wp.float32),
        dAdt_nodes: wp.array(dtype=wp.vec3f),
        enorm: wp.array(dtype=wp.float32),
    ):
        """Compute |E| per element on GPU.

        E = -grad(phi) - dA/dt_bary
        grad(phi)|_e = sum_i phi[elements[e,i]] * G[e, i*3+d]
        dA/dt_bary   = mean of 4 nodal dAdt values
        """
        e = wp.tid()
        n0 = elements[e, 0]
        n1 = elements[e, 1]
        n2 = elements[e, 2]
        n3 = elements[e, 3]

        p0 = phi[n0]
        p1 = phi[n1]
        p2 = phi[n2]
        p3 = phi[n3]

        # grad(phi) = sum_i phi_i * G[e, i, :]  (G stored as n_elem x 12)
        gx = p0 * G[e, 0] + p1 * G[e, 3] + p2 * G[e, 6] + p3 * G[e, 9]
        gy = p0 * G[e, 1] + p1 * G[e, 4] + p2 * G[e, 7] + p3 * G[e, 10]
        gz = p0 * G[e, 2] + p1 * G[e, 5] + p2 * G[e, 8] + p3 * G[e, 11]

        # dA/dt at barycenter = mean of 4 nodal values
        d0 = dAdt_nodes[n0]
        d1 = dAdt_nodes[n1]
        d2 = dAdt_nodes[n2]
        d3 = dAdt_nodes[n3]
        dAdt_bary = wp.vec3f(
            (d0[0] + d1[0] + d2[0] + d3[0]) * 0.25,
            (d0[1] + d1[1] + d2[1] + d3[1]) * 0.25,
            (d0[2] + d1[2] + d2[2] + d3[2]) * 0.25,
        )

        E = wp.vec3f(-gx - dAdt_bary[0], -gy - dAdt_bary[1], -gz - dAdt_bary[2])
        enorm[e] = wp.length(E)

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
    device=None,
    quiet: bool = True,
    tol: float = 1e-4,
    max_iters: int = 0,
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
    device : str or None
        Warp device string, e.g. ``"cpu"`` or ``"cuda:0"``.
        If None (default), uses warp's preferred device (GPU if available).
    quiet : bool
        Suppress CG iteration residual output.
    tol : float
        Relative residual tolerance for CG convergence (default 1e-4).
    max_iters : int
        Maximum CG iterations; 0 means up to the system size.

    Returns
    -------
    (n_nodes,) float64 array
        Scalar potential phi at each node.

    Raises
    ------
    ImportError
        If warp-lang is not installed.
    RuntimeError
        If CG fails to converge within ``max_iters`` iterations.
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

    # Default to warp's preferred device (cuda:0 if GPU available, else cpu)
    if device is None:
        device = wp.get_preferred_device()

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
    # fem.integrate may place output on the geometry's device, which can
    # differ from the requested device (e.g. cuda:0 when GPU is present).
    # Detect the actual device from the assembled RHS to stay consistent.
    actual_device = b.device
    n_nodes = len(mesh.nodes)
    projector = _make_gauge_projector(n_nodes, pin_node, device=actual_device)
    fixed_val = wp.zeros(n_nodes, dtype=wp.float32, device=actual_device)
    fem.project_linear_system(K, b, projector, fixed_val)

    # ------------------------------------------------------------------
    # Conjugate Gradient solve
    # ------------------------------------------------------------------
    x = wp.zeros(n_nodes, dtype=wp.float32, device=actual_device)
    err, iters = bsr_cg(K, b=b, x=x, tol=tol, max_iters=max_iters, quiet=quiet)

    # Explicit synchronization before reading back results.
    # On CUDA, bsr_cg submits work via CUDA graphs which may be asynchronous.
    # This guarantees the GPU has finished before the timer stops or data is read.
    wp.synchronize_device(actual_device)

    if not quiet:
        print(f"  bsr_cg: {iters} iterations, final residual {err:.3e} (tol={tol:.1e})")

    if err > tol:
        raise RuntimeError(
            f"bsr_cg did not converge: residual={err:.3e} > tol={tol:.1e} "
            f"after {iters} iterations. "
            "The solution is likely inaccurate. "
            "Try increasing max_iters or loosening tol."
        )

    # Return as float64 for compatibility with the rest of the pipeline
    return x.numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Incremental CG context for streaming solves
# ---------------------------------------------------------------------------

class WarpFEMContext:
    """Persistent warp.fem state for incremental CG solves.

    One-time cost: geometry construction, K assembly, gauge projector.
    Then call ``set_rhs()`` for each new coil position, and ``step()``
    repeatedly to advance CG.  ``x`` is kept across calls as a warm start.

    Example
    -------
    >>> ctx = WarpFEMContext(mesh, device="cpu")
    >>> ctx.set_rhs(dAdt_nodes)
    >>> while True:
    ...     err, iters, converged = ctx.step(n_iters=50)
    ...     phi = ctx.get_phi()
    ...     # compute E-field from phi, update visualization
    ...     if converged:
    ...         break
    """

    def __init__(self, mesh, pin_node=0, device=None, tol=1e-4):
        if not _WARP_INTEGRANDS_DEFINED:
            raise ImportError(
                "warp-lang is required for WarpFEMContext. "
                "Install it with: pip install warp-lang"
            )
        _init_warp()

        import warp as wp
        import warp.fem as fem

        if device is None:
            device = wp.get_preferred_device()

        # Build geometry
        positions = wp.array(
            mesh.nodes.astype(np.float32), dtype=wp.vec3f, device=device
        )
        tet_indices = wp.array(
            mesh.elements.astype(np.int32), dtype=int, device=device
        )
        geo = fem.Tetmesh(tet_indices, positions)

        # Function spaces
        self._phi_space = fem.make_polynomial_space(
            geo, dtype=wp.float32, degree=1
        )
        self._dAdt_space = fem.make_polynomial_space(
            geo, dtype=wp.vec3f, degree=1
        )
        self._sigma_wp = wp.array(
            mesh.conductivity.astype(np.float32),
            dtype=wp.float32, device=device,
        )

        # Assemble stiffness matrix K (one-time)
        domain = fem.Cells(geometry=geo)
        test = fem.make_test(space=self._phi_space, domain=domain)
        trial = fem.make_trial(space=self._phi_space, domain=domain)
        self._test = test
        self._domain = domain
        self.K = fem.integrate(
            _tms_stiffness_form,
            fields={"u": trial, "v": test},
            values={"sigma": self._sigma_wp},
            output_dtype=wp.float32,
        )

        # Gauge projector
        n_nodes = len(mesh.nodes)
        self._device = self.K.values.device
        self._n_nodes = n_nodes
        self._pin_node = pin_node
        self._nodes_np = mesh.nodes
        self._diff_ready = False  # differentiable state built on first use
        self._projector = _make_gauge_projector(
            n_nodes, pin_node, device=self._device
        )
        self._fixed_val = wp.zeros(
            n_nodes, dtype=wp.float32, device=self._device
        )

        # Apply gauge to K (idempotent — safe to re-apply with new b)
        # We need a dummy b for the first projection of K
        dummy_b = wp.zeros(n_nodes, dtype=wp.float32, device=self._device)
        fem.project_linear_system(
            self.K, dummy_b, self._projector, self._fixed_val
        )

        # Persistent solution vector (warm start across set_rhs calls)
        self.x = wp.zeros(n_nodes, dtype=wp.float32, device=self._device)
        self.b = None
        self.tol = tol
        self._total_iters = 0
        self._converged = False

        # GPU arrays for E-field / Enorm computation
        from tmswarp.solver import gradient_operator
        n_elements = len(mesh.elements)
        self._n_elements = n_elements
        G_cpu = gradient_operator(mesh).astype(np.float32)  # (n_elem, 4, 3)
        # Flatten to (n_elem, 12) for simple 2D indexing in kernel
        self._G_wp = wp.array(
            G_cpu.reshape(n_elements, 12),
            dtype=wp.float32, device=self._device,
        )
        self._elements_wp = wp.array(
            mesh.elements.astype(np.int32),
            dtype=wp.int32, device=self._device,
        ).reshape((n_elements, 4))
        self._enorm_wp = wp.zeros(
            n_elements, dtype=wp.float32, device=self._device
        )
        self._dAdt_wp = None  # set by set_rhs()

    def set_rhs(self, dAdt_nodes):
        """Assemble new RHS b for updated dA/dt.  Keep x for warm start."""
        import warp as wp
        import warp.fem as fem

        # Populate dAdt discrete field
        dAdt_wp = wp.array(
            dAdt_nodes.astype(np.float32), dtype=wp.vec3f, device=self._device
        )
        self._dAdt_wp = dAdt_wp  # keep for compute_enorm()
        dAdt_discrete = self._dAdt_space.make_field()
        dAdt_discrete.dof_values = dAdt_wp

        # Assemble RHS
        self.b = fem.integrate(
            _tms_rhs_form,
            fields={"v": self._test, "dAdt": dAdt_discrete},
            values={"sigma": self._sigma_wp},
            output_dtype=wp.float32,
        )

        # Apply gauge to b (K already has gauge applied — idempotent)
        fem.project_linear_system(
            self.K, self.b, self._projector, self._fixed_val
        )

        self._total_iters = 0
        self._converged = False

    def step(self, n_iters=50):
        """Run n_iters CG iterations from current x.

        Returns (residual, total_iters, converged).
        """
        import warp as wp
        from warp.examples.fem.utils import bsr_cg

        if self.b is None:
            raise RuntimeError("Call set_rhs() before step()")

        err, iters = bsr_cg(
            self.K, b=self.b, x=self.x,
            max_iters=n_iters, tol=self.tol, quiet=True,
        )
        wp.synchronize_device(self._device)
        self._total_iters += iters
        self._converged = (err <= self.tol)
        return err, self._total_iters, self._converged

    def get_phi(self):
        """Return current phi estimate as float64 numpy array."""
        import warp as wp
        wp.synchronize_device(self._device)
        return self.x.numpy().astype(np.float64)

    def compute_enorm(self):
        """Compute |E| per element entirely on GPU.

        Returns (n_elements,) float64 numpy array.
        Requires set_rhs() to have been called (for dAdt).
        """
        import warp as wp
        if self._dAdt_wp is None:
            raise RuntimeError("Call set_rhs() before compute_enorm()")
        wp.launch(
            _compute_enorm_kernel,
            dim=self._n_elements,
            inputs=[self.x, self._elements_wp, self._G_wp,
                    self._dAdt_wp, self._enorm_wp],
            device=self._device,
        )
        wp.synchronize_device(self._device)
        return self._enorm_wp.numpy().astype(np.float64)

    @property
    def converged(self):
        return self._converged

    # ------------------------------------------------------------------
    # Differentiable objective (wp.Tape + adjoint linear solve)
    # ------------------------------------------------------------------

    def _ensure_diff_state(self):
        """Allocate the arrays recorded on the tape (one-time)."""
        if self._diff_ready:
            return
        import warp as wp

        n = self._n_nodes
        dev = self._device
        self._nodes_wp = wp.array(
            self._nodes_np.astype(np.float32), dtype=wp.vec3f, device=dev
        )
        self._pos_wp = wp.zeros(1, dtype=wp.vec3f, device=dev, requires_grad=True)
        self._mom_wp = wp.zeros(1, dtype=wp.vec3f, device=dev, requires_grad=True)
        self._dAdt_field = self._dAdt_space.make_field()
        self._dAdt_field.dof_values.requires_grad = True
        self._b_diff = wp.zeros(n, dtype=wp.float32, device=dev, requires_grad=True)
        self._loss_wp = wp.zeros(1, dtype=wp.float32, device=dev, requires_grad=True)
        self.x.requires_grad = True

        # Work arrays for the solves (never recorded on the tape)
        self._b_scaled = wp.zeros(n, dtype=wp.float32, device=dev)
        self._x_scaled = wp.zeros(n, dtype=wp.float32, device=dev)
        self._adj_rhs = wp.zeros(n, dtype=wp.float32, device=dev)
        self._lam = wp.zeros(n, dtype=wp.float32, device=dev)
        self._lam_target = None
        self._diff_ready = True

    def _solve_relative(self, b, x, rtol, max_iters):
        """Solve K x = b to a residual of ``rtol * |b|``, warm-started from x.

        bsr_cg stops at ``max(tol * |b|, tol)``, which is an absolute
        threshold whenever |b| < 1.  Solving for x / |b| against b / |b|
        makes the tolerance relative for any right-hand-side scale.

        Returns (relative_residual, iterations).
        """
        import warp as wp
        from warp.examples.fem.utils import bsr_cg

        n = self._n_nodes
        dev = self._device
        b_norm = float(np.sqrt(wp.utils.array_inner(b, b)))
        if b_norm == 0.0:
            x.zero_()
            return 0.0, 0

        wp.launch(_scale_kernel, dim=n, inputs=[b, 1.0 / b_norm],
                  outputs=[self._b_scaled], device=dev)
        wp.launch(_scale_kernel, dim=n, inputs=[x, 1.0 / b_norm],
                  outputs=[self._x_scaled], device=dev)
        err, iters = bsr_cg(
            self.K, b=self._b_scaled, x=self._x_scaled,
            max_iters=max_iters, tol=rtol, quiet=True,
        )
        wp.launch(_scale_kernel, dim=n, inputs=[self._x_scaled, b_norm],
                  outputs=[x], device=dev)
        wp.synchronize_device(dev)
        return float(err), int(iters)

    def solve(self, dAdt_nodes, rtol=1e-4, max_iters=2000):
        """set_rhs() then solve to a relative residual, warm-started from x.

        Returns (relative_residual, iterations, converged).
        """
        self._ensure_diff_state()
        self.set_rhs(dAdt_nodes)
        err, iters = self._solve_relative(self.b, self.x, rtol, max_iters)
        self._total_iters = iters
        self._converged = err <= rtol
        return err, iters, self._converged

    def objective_and_gradient(self, dipole_pos, dipole_moment, target_elem,
                               didt=1e6, rtol=1e-4, max_iters=2000):
        """Evaluate loss = -|E[target_elem]| and its gradient by autodiff.

        The dipole field, right-hand-side assembly and loss are recorded on
        a ``wp.Tape``.  The linear solve is differentiated implicitly: its
        backward step solves the adjoint system K lam = dL/dphi (K is
        symmetric), so one gradient costs one extra solve regardless of
        the number of parameters.

        Parameters
        ----------
        dipole_pos : (3,) array
            Dipole position in metres.
        dipole_moment : (3,) array
            Dipole moment direction (normalized internally).
        target_elem : int
            Index of the element whose |E| is maximized.
        didt : float
            Rate of current change dI/dt in A/s.
        rtol : float
            Relative residual tolerance for the forward and adjoint solves.
        max_iters : int
            Maximum CG iterations per solve.

        Returns
        -------
        loss : float
            -|E[target_elem]| in V/m.
        grad_pos : (3,) float64 array
            d(loss)/d(dipole_pos), per metre.
        grad_moment : (3,) float64 array
            d(loss)/d(dipole_moment).

        After the call ``get_phi()`` and ``compute_enorm()`` return the
        field for this dipole.
        """
        import warp as wp
        import warp.fem as fem

        self._ensure_diff_state()
        n = self._n_nodes
        dev = self._device
        target_elem = int(target_elem)

        self._pos_wp.assign(np.asarray(dipole_pos, dtype=np.float32).reshape(1, 3))
        self._mom_wp.assign(np.asarray(dipole_moment, dtype=np.float32).reshape(1, 3))
        dAdt = self._dAdt_field.dof_values
        b = self._b_diff
        x = self.x

        tape = wp.Tape()
        with tape:
            wp.launch(
                _dipole_dadt_kernel, dim=n,
                inputs=[self._nodes_wp, self._pos_wp, self._mom_wp,
                        float(1e-7 * didt)],
                outputs=[dAdt], device=dev,
            )
            fem.integrate(
                _tms_rhs_form,
                fields={"v": self._test, "dAdt": self._dAdt_field},
                values={"sigma": self._sigma_wp},
                output=b,
            )

        # Gauge: K already has the pin row/column eliminated
        wp.launch(_zero_entry_kernel, dim=1, inputs=[b, self._pin_node], device=dev)

        err, iters = self._solve_relative(b, x, rtol, max_iters)

        def adjoint_solve():
            # dL/db = K^-1 dL/dphi
            wp.copy(self._adj_rhs, x.grad)
            wp.launch(_zero_entry_kernel, dim=1,
                      inputs=[self._adj_rhs, self._pin_node], device=dev)
            self._solve_relative(self._adj_rhs, self._lam, rtol, max_iters)
            wp.launch(_add_kernel, dim=n, inputs=[self._lam],
                      outputs=[b.grad], device=dev)

        tape.record_func(adjoint_solve, arrays=(b, x))

        if self._lam_target != target_elem:
            self._lam.zero_()
            self._lam_target = target_elem

        with tape:
            wp.launch(
                _target_loss_kernel, dim=1,
                inputs=[x, self._elements_wp, self._G_wp, dAdt, target_elem],
                outputs=[self._loss_wp], device=dev,
            )

        tape.backward(loss=self._loss_wp)
        wp.synchronize_device(dev)

        loss = float(self._loss_wp.numpy()[0])
        grad_pos = self._pos_wp.grad.numpy()[0].astype(np.float64)
        grad_moment = self._mom_wp.grad.numpy()[0].astype(np.float64)
        tape.zero()

        # Keep the context consistent for step() / compute_enorm()
        self.b = b
        self._dAdt_wp = dAdt
        self._total_iters = iters
        self._converged = err <= rtol
        return loss, grad_pos, grad_moment
