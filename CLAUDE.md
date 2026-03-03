# TMSWarp — Project Context for Claude Code

## What This Project Is

TMSWarp is a Python package for TMS (Transcranial Magnetic Stimulation) electric field simulation using the Finite Element Method. It is part of the SlicerTMS ecosystem but has **no dependencies on 3D Slicer or SimNIBS** — it is a standalone PyPI-publishable package.

The ultimate goal is a GPU-accelerated FEM solver using Nvidia's `warp.fem` framework, with an API designed for easy integration with RPyC-based interprocess communication (matching the patterns in the parent SlicerTMS project).

## Current State (as of March 2026)

### What's Done: Warp.fem GPU Implementation + Pure-NumPy Reference

A complete, working FEM solver using only numpy + scipy:

- **`src/tmswarp/analytical.py`** — Heller & van Hulsteyn (1992) analytical E-field for spherical conductors. Uses Sarvas (1987) F/grad_F quantities via MEG-TMS reciprocity. Key property: E-field is purely tangential and conductivity-independent for spheres.

- **`src/tmswarp/coil.py`** — Biot-Savart primary field (dA/dt) from a magnetic dipole source. Formula: `dAdt = 1e-7 * didt * cross(m, r) / |r|^3`.

- **`src/tmswarp/conductor.py`** — `TetMesh` dataclass (nodes, elements, conductivity arrays) + `make_sphere_mesh()` that generates test meshes via Fibonacci sphere point distribution + `scipy.spatial.Delaunay`.

- **`src/tmswarp/solver.py`** — Core FEM: `gradient_operator()` (P1 basis gradients), `assemble_stiffness()` (sparse matrix via scipy.sparse), `assemble_rhs_tms()` (source term from dA/dt), `solve_fem()` (direct solve via scipy.sparse.linalg.spsolve with gauge pin).

- **`src/tmswarp/fields.py`** — Post-processing: `compute_efield_at_elements()` (E = -grad(phi) - dA/dt), plus `rdm()` and `mag()` validation metrics.

### Test Results: 32 passed

All tests pass. Both FEM solvers validate against the analytical solution:
- **NumPy FEM**: RDM = 0.192 (threshold < 0.2), MAG = 0.019 (threshold < log(1.1))
- **Warp.fem**: RDM = 0.230 (threshold < 0.3 due to float32 geometry), MAG = 0.027

These thresholds match SimNIBS's own validation criteria.

### SimNIBS sphere3 Validation

The `sphere3_data.npz` file (22 704 tets, 4 556 nodes) is extracted from SimNIBS's
`sphere3.msh` reference mesh by running:

    /path/to/SimNIBS-4.5/simnibs_env/bin/python scripts/extract_sphere3.py

Results on the SimNIBS mesh (same dipole config as SimNIBS's own test_fem.py):
- **NumPy FEM**: RDM = 0.169, MAG = 0.015 — **passes SimNIBS < 0.2 / < log(1.1)**
- **Warp.fem**: RDM = 0.198, MAG = 0.020 — passes relaxed thresholds

The Warp.fem result nearly meets SimNIBS's strict RDM < 0.2 threshold despite float32 geometry.

### Convergence (Delaunay meshes, P1 elements)

| Elements | RDM   | MAG   |
|----------|-------|-------|
| 847      | 0.594 | 0.181 |
| 3,704    | 0.395 | 0.073 |
| 8,202    | 0.313 | 0.045 |
| 16,337   | 0.258 | 0.029 |
| 29,000   | 0.192 | 0.019 |

The slow RDM convergence is due to Delaunay mesh quality (slivers, irregular elements). With proper meshes (e.g., from SimNIBS or gmsh), convergence will be much faster.

### Warp.fem Implementation

- **`src/tmswarp/solver_warp.py`** — Warp.fem solver. Key design:
  - `warp.fem.Tetmesh` requires float32 node positions; all FEM is float32
  - Per-element sigma passed as `wp.array(dtype=wp.float32)`, accessed via `s.element_index`
  - dA/dt as a P1 discrete vector field (`dtype=wp.vec3f`)
  - Gauge (phi[0]=0) via `fem.project_linear_system` with a single-entry BSR projector
  - Solve: `bsr_cg` from `warp.examples.fem.utils`
  - Returns float64 numpy array for compatibility with existing post-processing
  - **Integrands must be module-level** (warp uses `inspect.getsource` for JIT)
  - `warp_available()` guards import so module loads without warp installed

### Timing (Apple Silicon CPU, cached kernels)

| Elements | NumPy FEM (s) | Warp.fem CPU (s) | Note |
|----------|---------------|------------------|------|
| 847      | 0.57          | 7.25             | First call: JIT compile |
| 3,704    | 0.008         | 0.09             |      |
| 8,202    | 0.023         | 0.23             |      |
| 16,337   | 0.07          | 0.49             |      |
| 29,000   | 0.22          | 0.76             | GPU expected to dominate |

NumPy uses `scipy.sparse.linalg.spsolve` (direct). Warp uses CG, which converges
slower but scales better to GPU and very large systems.

### Visualization

- `convergence_visualization.png` — z=0 cross-sections: Analytical, NumPy FEM, Warp.fem, Error
- `convergence_plot.png` — RDM and |MAG| vs element count for both solvers
- `timing_plot.png` — Wall-clock time vs element count for all methods
- `visualize_convergence.py` — generates all three images

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

### Warp.fem Implementation Notes (DONE — see solver_warp.py)

Key lessons learned during implementation:
- `fem.Tetmesh` requires `wp.vec3f` (float32) positions — float64 fails at kernel launch
- All FEM quantities must be float32 to match; mixed precision fails with "scalar type mismatch"
- Per-element conductivity: use `wp.array(dtype=wp.float32)`, indexed via `s.element_index`
- dA/dt: create a separate `make_polynomial_space(geo, dtype=wp.vec3f)` space, then `make_field()`
- Gauge: `bsr_zeros(n,n,wp.float32)` + `bsr_set_from_triplets` + `fem.project_linear_system`
- CG solver: `from warp.examples.fem.utils import bsr_cg`; not from `warp.optim.linear`
- `@fem.integrand` decorators must be at module scope (not inside functions or `if` blocks)
- `example_diffusion_3d.py` in warp examples is the closest analogue to our problem

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
- `pixi install` sets up the environment (includes warp-lang via [pypi-dependencies])
- `pixi run pytest -v` to run tests (28 tests, all pass)
- `pixi run python visualize_convergence.py` to regenerate comparison plots
- pixi.toml has `osx-64`, `osx-arm64`, and `linux-64` platforms
- warp-lang ships a universal2 macOS binary that works on both Intel and Apple Silicon
- GPU work: test on Linux+CUDA; just change `device="cpu"` to `device="cuda:0"`

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
│   ├── solver.py            # FEM assembly + solve (numpy/scipy, float64)
│   ├── solver_warp.py       # FEM assembly + solve (warp.fem, float32, GPU-ready)
│   ├── fields.py            # E-field post-processing + RDM/MAG metrics
│   └── py.typed             # PEP 561 marker
├── tests/
│   ├── test_install.py           # Import smoke tests
│   ├── test_analytical.py        # Analytical solution properties
│   ├── test_coil.py              # dA/dt correctness
│   ├── test_solver.py            # NumPy FEM vs analytical validation
│   ├── test_solver_warp.py       # Warp.fem vs NumPy FEM + analytical (skipped if no warp)
│   └── test_sphere3_validation.py  # Both solvers vs SimNIBS sphere3 mesh (skipped if no data)
├── scripts/
│   └── extract_sphere3.py    # Run with SimNIBS Python to create sphere3_data.npz
├── benchmarks/
│   └── sphere3_validation.py # Standalone validation + timing against sphere3 mesh
├── .github/workflows/
│   ├── test.yml             # CI: pytest on Python 3.9-3.13
│   └── publish.yml          # PyPI trusted publishing on GitHub release
├── visualize_convergence.py # Generates all comparison images (3 output PNGs)
├── convergence_visualization.png  # z=0 slices: Ana / NumPy / Warp / Error
├── convergence_plot.png           # RDM+MAG convergence curves
└── timing_plot.png                # Wall-clock timing comparison
```

## Conventions

- All physics in **SI units** (meters, S/m, A/s, V/m) — no millimeter conversions
- Test configuration matches SimNIBS: sphere radius 95mm, dipole at (0,0,300mm), moment (1,0,0), dI/dt = 1e6 A/s, conductivity = 1.0 S/m
- Validation thresholds: RDM < 0.2, |MAG| < log(1.1) (same as SimNIBS)
- Apache 2.0 license
