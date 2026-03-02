# TMSWarp — Project Context for Claude Code

## What This Project Is

TMSWarp is a Python package for TMS (Transcranial Magnetic Stimulation) electric field simulation using the Finite Element Method. It is part of the SlicerTMS ecosystem but has **no dependencies on 3D Slicer or SimNIBS** — it is a standalone PyPI-publishable package.

The ultimate goal is a GPU-accelerated FEM solver using Nvidia's `warp.fem` framework, with an API designed for easy integration with RPyC-based interprocess communication (matching the patterns in the parent SlicerTMS project).

## Current State (as of March 2026)

### What's Done: Pure-NumPy Reference Implementation

A complete, working FEM solver using only numpy + scipy:

- **`src/tmswarp/analytical.py`** — Heller & van Hulsteyn (1992) analytical E-field for spherical conductors. Uses Sarvas (1987) F/grad_F quantities via MEG-TMS reciprocity. Key property: E-field is purely tangential and conductivity-independent for spheres.

- **`src/tmswarp/coil.py`** — Biot-Savart primary field (dA/dt) from a magnetic dipole source. Formula: `dAdt = 1e-7 * didt * cross(m, r) / |r|^3`.

- **`src/tmswarp/conductor.py`** — `TetMesh` dataclass (nodes, elements, conductivity arrays) + `make_sphere_mesh()` that generates test meshes via Fibonacci sphere point distribution + `scipy.spatial.Delaunay`.

- **`src/tmswarp/solver.py`** — Core FEM: `gradient_operator()` (P1 basis gradients), `assemble_stiffness()` (sparse matrix via scipy.sparse), `assemble_rhs_tms()` (source term from dA/dt), `solve_fem()` (direct solve via scipy.sparse.linalg.spsolve with gauge pin).

- **`src/tmswarp/fields.py`** — Post-processing: `compute_efield_at_elements()` (E = -grad(phi) - dA/dt), plus `rdm()` and `mag()` validation metrics.

### Test Results: 20 passed, 1 skipped

All tests pass. The FEM validates against the analytical solution:
- **RDM = 0.192** (threshold < 0.2) — directional accuracy
- **MAG = 0.019** (threshold < log(1.1) = 0.095) — magnitude accuracy

These thresholds match SimNIBS's own validation criteria.

### Convergence (Delaunay meshes, P1 elements)

| Elements | RDM   | MAG   |
|----------|-------|-------|
| 847      | 0.594 | 0.181 |
| 3,704    | 0.395 | 0.073 |
| 8,202    | 0.313 | 0.045 |
| 16,337   | 0.258 | 0.029 |
| 29,000   | 0.192 | 0.019 |

The slow RDM convergence is due to Delaunay mesh quality (slivers, irregular elements). With proper meshes (e.g., from SimNIBS or gmsh), convergence will be much faster.

### Visualization

- `convergence_visualization.png` — z=0 cross-sections showing analytical |E|, FEM |E|, and relative error at each mesh resolution
- `convergence_plot.png` — RDM and |MAG| vs element count with threshold lines
- `visualize_convergence.py` — script that generates both images

## What's Next: Warp.fem GPU Implementation

### The Physics (Quick Reference)

TMS simulation solves the quasistatic Poisson equation:
```
div(σ ∇φ) = -div(σ ∂A/∂t)
```
Total E-field: `E = -∇φ - ∂A/∂t`

- `φ` = unknown scalar potential (solved by FEM)
- `σ` = tissue conductivity (known, per-element)
- `∂A/∂t` = primary field from TMS coil (computed analytically)
- BC: zero normal current on outer surface (natural Neumann, free in FEM)
- Gauge: pin one node to φ=0 for uniqueness

### Warp.fem Implementation Plan

The `warp.fem` module has everything needed:

1. **`fem.Tetmesh`** — accepts numpy arrays of nodes/elements directly
2. **`fem.make_polynomial_space(geo, degree=1, element_basis=LAGRANGE)`** — P1 scalar space for φ
3. **Integrands** via `@fem.integrand` decorator:
   - Stiffness: `σ * wp.dot(fem.grad(u, s), fem.grad(v, s))`
   - RHS: `σ * wp.dot(dAdt_field(s), fem.grad(v, s))`
4. **Assembly**: `fem.integrate(integrand, fields={"u": trial, "v": test})`
5. **BCs**: `fem.BoundarySides(geo)` + `fem.project_linear_system()`
6. **Solve**: `bsr_cg()` from `warp.optim.linear` (CG for SPD system)

Key reference: the `warp/examples/fem/example_magnetostatics.py` demonstrates the full pattern (curl-curl with Nedelec elements — our scalar Poisson is simpler). The `example_diffusion_3d.py` is the closest analogue.

**Critical: Nedelec elements are NOT needed.** TMS uses scalar potential φ with standard Lagrange elements. The magnetostatics example uses Nedelec because it solves for the vector potential A directly — a different formulation.

### RPyC Service Interface

The package should expose a clean API that maps to the RPyC pattern in the parent SlicerTMS project (see `SlicerTMS/Experiments/SimNIBSService.py`):

```python
# TMSWarp has no RPyC dependency itself — just returns numpy arrays.
# The service wrapper lives in SlicerTMS integration code.
class TMSWarpSolver:
    def initialize(self, nodes, elements, conductivity): ...
    def update_e_field(self, coil_matrix, didt) -> np.ndarray: ...
```

The existing SimNIBS RPyC service uses `multiprocessing.shared_memory` to efficiently transfer E-field arrays between processes. The new solver should support the same pattern.

## Key Files in the Parent SlicerTMS Project

For context on the integration target:

- `SlicerTMS/Experiments/SimNIBSService.py` — RPyC service wrapping SimNIBS OnlineFEM solver (the pattern to replicate)
- `SlicerTMS/Experiments/SlicerSimNIBSClient.py` — Client in Slicer that connects via RPyC + shared memory
- `SlicerTMS/Experiments/onlinefem.py` — Standalone SimNIBS FEM script with analytical validation
- `SlicerTMS/server/server.py` — CNN-based E-field prediction server (alternate approach)

## Development Environment

- Use `pixi` for environment management (pixi.toml is in the repo)
- `pixi add python=3.12 numpy scipy pytest matplotlib` for development
- `pixi run pip install -e "." --no-deps` to install tmswarp in editable mode
- `pixi run pytest -v` to run tests
- GPU work requires a Linux machine with Nvidia GPU + CUDA for warp-lang

## Package Structure

```
TMSWarp/
├── pyproject.toml          # hatchling build, numpy+scipy deps, warp-lang optional under [gpu]
├── pixi.toml               # pixi environment config
├── src/tmswarp/
│   ├── __init__.py          # Public API re-exports
│   ├── analytical.py        # Heller & van Hulsteyn analytical solution
│   ├── coil.py              # Biot-Savart dA/dt
│   ├── conductor.py         # TetMesh dataclass + sphere mesh generator
│   ├── solver.py            # FEM assembly + solve (numpy/scipy)
│   ├── fields.py            # E-field post-processing + RDM/MAG metrics
│   └── py.typed             # PEP 561 marker
├── tests/
│   ├── test_install.py      # Import smoke tests
│   ├── test_analytical.py   # Analytical solution properties
│   ├── test_coil.py         # dA/dt correctness
│   └── test_solver.py       # FEM vs analytical validation
├── .github/workflows/
│   ├── test.yml             # CI: pytest on Python 3.9-3.13
│   └── publish.yml          # PyPI trusted publishing on GitHub release
└── visualize_convergence.py # Generates convergence comparison images
```

## Conventions

- All physics in **SI units** (meters, S/m, A/s, V/m) — no millimeter conversions
- Test configuration matches SimNIBS: sphere radius 95mm, dipole at (0,0,300mm), moment (1,0,0), dI/dt = 1e6 A/s, conductivity = 1.0 S/m
- Validation thresholds: RDM < 0.2, |MAG| < log(1.1) (same as SimNIBS)
- Apache 2.0 license
