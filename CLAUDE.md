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

### Test Results: 42 passed

All tests pass. Both FEM solvers validate against the analytical solution:
- **NumPy FEM**: RDM = 0.192 (threshold < 0.2), MAG = 0.019 (threshold < log(1.1))
- **Warp.fem**: RDM = 0.230, MAG = 0.027, with `solve_fem_warp()` at its default
  `tol=1e-4` (test threshold relaxed to < 0.3)

The NumPy thresholds match SimNIBS's own validation criteria.  The Warp numbers
are worse because of the default CG stopping tolerance, not because of float32;
see "CG tolerance semantics" below.

### SimNIBS sphere3 Validation

The `sphere3_data.npz` file (22 704 tets, 4 556 nodes) is extracted from SimNIBS's
`sphere3.msh` reference mesh by running:

    /path/to/SimNIBS-4.5/simnibs_env/bin/python scripts/extract_sphere3.py

Results on the SimNIBS mesh (same dipole config as SimNIBS's own test_fem.py):
- **NumPy FEM**: RDM = 0.169, MAG = 0.015 — **passes SimNIBS < 0.2 / < log(1.1)**
- **Warp.fem**: RDM = 0.198, MAG = 0.020 at the default `tol=1e-4`; RDM = 0.169
  (the same as NumPy) at `tol=1e-7`

### Ernie Human Head Mesh Validation

The `ernie_data.npz` file (1.3M tets, 222k nodes, 6 tissues) is fetched from the SimNIBS
example dataset (not committed to git — it's 19 MB and separately licensed):

    /path/to/SimNIBS-4.5/simnibs_env/bin/python scripts/fetch_ernie.py [/optional/path/to/zip]

The script downloads from `https://github.com/simnibs/example-dataset/releases/download/v4.0-lowres/ernie_lowres_V2.zip`.

**Comparison figures**: axial/coronal/sagittal slices of |E|, with TMSWarp vs
SimNIBS side-by-side and absolute + relative difference maps.

| Figure | TMSWarp solver | RDM | \|MAG\| |
|--------|----------------|-----|---------|
| `ernie_comparison.png` | Warp float32 CG, rtol=1e-6, RTX 3070 | 0.0003 | 0.0000 |
| `ernie_comparison_numpy.png` | NumPy float64, SciPy direct solve | 0.0000 | 0.0000 |

Both compare against SimNIBS 4.6.0 solving with its own default solver (PETSc
CG with hypre BoomerAMG).  Before September 2026 `ernie_comparison.png` showed
the NumPy solver against SimNIBS's assembled system solved with SciPy.

Generate via `scripts/run_simnibs_efield.py` (SimNIBS Python; `--solver scipy`
gives the old behaviour), then `benchmarks/ernie_comparison.py` (`--solver
warp` is the default, `--solver numpy` writes the second figure).  The dipole
field is computed with the same formula on both sides, so the figures compare
the FEM solves, not the coil models.

Results on the ernie mesh (dipole at z=200mm, 6 tissues, Apple Silicon CPU):

| Solver    | Time   | vs NumPy | E-field (mean/max V/m) | Note |
|-----------|--------|----------|----------------------|------|
| NumPy FEM | 246 s  | —        | 0.484 / 41.2         | float64, direct solve |
| Warp.fem  | 111 s  | 2.21×    | 0.569 / 49.4         | float32, CG, default `tol=1e-4` |

**The Warp row is an under-converged solve, not a float32 limit.**  The
difference from NumPy (RDM = 0.53, MAG = 0.21) comes from the default CG
stopping tolerance.  Measured September 2026 on the same mesh and dipole,
cold start, on an RTX 3070 (Warp 1.17.0), against a float64 CG reference
(Jacobi, rtol=1e-10):

| Warp float32 solve                     | CG iterations | RDM    | MAG    | mean / max V/m |
|----------------------------------------|---------------|--------|--------|----------------|
| `solve_fem_warp()`, default `tol=1e-4` | 170           | 0.5304 | 0.2073 | 0.569 / 49.46  |
| `WarpFEMContext.solve(rtol=1e-3)`      | 325           | 0.1208 | 0.0070 | 0.486 / 40.96  |
| `WarpFEMContext.solve(rtol=1e-4)`      | 700           | 0.0019 | 0.0000 | 0.484 / 41.08  |
| `WarpFEMContext.solve(rtol=1e-5)`      | 784           | 0.0004 | 0.0000 | 0.484 / 41.17  |
| `WarpFEMContext.solve(rtol=1e-6)`      | 857           | 0.0003 | 0.0000 | 0.484 / 41.16  |

So float32 agrees with the float64 reference to RDM = 0.0003 on this mesh,
despite the 165:1 conductivity contrast (skull 0.01 vs CSF 1.654 S/m).  The
111 s timing above is for the under-converged solve; a converged one takes
about four times as many iterations.

Timing on the same machine (12 cores, RTX 3070), ernie, cold start:

| Solver | Setup | Solve | RDM vs SimNIBS |
|--------|-------|-------|----------------|
| SimNIBS 4.6.0, hypre, CPU | 2.7 s assembly + 0.6 s solver setup | 1.7 s | — |
| Warp GPU, rtol=1e-3 | 0.7 s assembly | 0.12 s (325 iterations) | 0.1208 |
| Warp GPU, rtol=1e-4 | 0.7 s | 0.22 s (700) | 0.0019 |
| Warp GPU, rtol=1e-6 | 0.7 s | 0.26 s (857) | 0.0003 |
| Warp CPU, rtol=1e-3 | 12 s | 194 s (330) | 0.1209 |
| Warp CPU, rtol=1e-4 | 12 s | 410 s (700) | 0.0020 |
| Warp CPU, rtol=1e-6 | 12 s | 500 s (860) | 0.0003 |
| NumPy, SciPy direct | — | 1098 s in total | 0.0000 |

The NumPy run shared the machine with other jobs.

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
  - All FEM is float32.  Warp 1.9.1 required it (`fem.Tetmesh` accepted only
    float32 positions); Warp >= 1.13 supports float64 in `warp.fem`, not yet
    used or tested here
  - Per-element sigma passed as `wp.array(dtype=wp.float32)`, accessed via `s.element_index`
  - dA/dt as a P1 discrete vector field (`dtype=wp.vec3f`)
  - Gauge (phi[0]=0) via `fem.project_linear_system` with a single-entry BSR projector
  - Solve: `bsr_cg` from `warp.examples.fem.utils`
  - Returns float64 numpy array for compatibility with existing post-processing
  - **Integrands must be module-level** (warp uses `inspect.getsource` for JIT)
  - `warp_available()` guards import so module loads without warp installed

### Differentiable objective (added September 2026)

`WarpFEMContext.objective_and_gradient(dipole_pos, dipole_moment, target_elem)`
returns loss = -|E[target_elem]| and its gradient with respect to dipole
position and moment, computed with Warp autodiff:

- The dipole field (`_dipole_dadt_kernel`), the RHS assembly (`fem.integrate`
  of `_tms_rhs_form`) and the loss (`_target_loss_kernel`) are recorded on a
  `wp.Tape`.
- The CG solve is not taped.  Its backward step is registered with
  `tape.record_func` and solves the adjoint system K lam = dL/dphi, following
  `warp/examples/fem/example_darcy_ls_optimization.py`.  One gradient costs one
  extra solve.
- `SlicerTMS/Experiments/TMSService.py` uses this for the Warp optimization
  path (it previously used finite differences, four solves per iteration).
- Validated in `tests/test_solver_warp.py::TestWarpGradient` against float64
  central finite differences of the NumPy direct solver.  On sphere3 the
  gradient agrees to about 1e-5 (rtol=1e-6) and 3e-4 (rtol=1e-4).
- Tested on CPU with Warp 1.9.1 (the last release with Intel Mac wheels) and
  on an RTX 3070 with Warp 1.17.0.

### Surface constraint for the optimizer

`src/tmswarp/surface.py` has `closest_point_on_triangles()` and
`vertex_normals()`.  TMSService uses them to keep the coil on the scalp:
`_project_to_surface()` returns the closest point on the boundary triangles
plus the offset along a normal interpolated from vertex normals, so position
and orientation vary continuously.  (Projecting to face centres made the
refinement stick on one face.)

Because the moment follows the surface normal, moving the coil also rotates
it.  `_surface_gradient()` combines the autodiff gradients with respect to
position and moment through central differences of the projection, and
returns a tangent vector.  The Adam step size decays by `OPT_LR_DECAY` per
iteration, and the best position found is the one returned.

On ernie (RTX 3070, targets at (-40,-10,60), (-40,40,40), (30,-80,10) mm) the
final |E| at the target went from 32.68 / 16.22 / 67.86 V/m with face-centre
projection to 35.99 / 18.22 / 71.93 V/m.

### Device handling

Until September 2026 `solve_fem_warp()` and `WarpFEMContext` ignored a
requested device of `"cpu"` on machines with CUDA: `warp.fem` allocates on
Warp's default device, so everything ran on the GPU.  They now do all their
work inside `wp.ScopedDevice(device)`.  Any "Warp CPU" number measured on a
CUDA machine before that fix was a GPU run.  The CPU-only benchmark results
in `benchmarks/results/` are not affected.

### CG tolerance semantics

`bsr_cg(tol=...)` stops at `max(tol * |b|, tol)`.  For this problem |b| is about
1e-3, so `tol=1e-4` in `solve_fem_warp()` and `WarpFEMContext.step()` is an
absolute threshold, roughly a 3-7% relative residual.  Measured on sphere3 with
layered conductivity (0.275/0.010/0.465), cold start, against the float64
direct solve: RDM 1.10 at `tol=1e-4`, 0.011 at `tol=1e-7`.  The ernie
measurements are in the table above.  The Warp-vs-NumPy differences in this
file come from the stopping tolerance, not from float32.

The defaults of `solve_fem_warp()` and `WarpFEMContext.step()` have not been
changed, so the tests, benchmarks and figures that call them still produce
under-converged Warp results.  `benchmarks/ernie_validation.py` still prints
the float32 explanation.

`WarpFEMContext.solve()` and `objective_and_gradient()` take `rtol`, a true
relative tolerance (they solve the system scaled by 1/|b|).  TMSService uses
`--opt-rtol` (default 1e-3) for all solves during optimization.

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
- With Warp 1.9.1, `fem.Tetmesh` requires `wp.vec3f` (float32) positions — float64 fails at kernel launch
- All FEM quantities must match the geometry's precision; mixing fails with "scalar type mismatch"
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
- `pixi run pytest -v` to run tests (42 tests, all pass)
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
│   ├── extract_sphere3.py       # Run with SimNIBS Python to create sphere3_data.npz
│   ├── fetch_ernie.py           # Download + extract ernie head mesh to ernie_data.npz
│   └── run_simnibs_efield.py    # Run SimNIBS FEM on ernie → ernie_simnibs_efield.npz
├── benchmarks/
│   ├── sphere3_validation.py    # Validation against SimNIBS sphere3 mesh
│   ├── ernie_validation.py      # Timing + comparison on realistic human head mesh
│   └── ernie_comparison.py      # Load both results; generate ernie_comparison.png
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
