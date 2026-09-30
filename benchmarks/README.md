# TMSWarp benchmarks

`python -m tmswarp.bench` times the TMSWarp solvers on meshes of several
sizes and scores every result against a float64 reference solve. Where
SimNIBS is installed its own solvers are timed on the same machine, and where
SlicerTMS's `TMSService.py` is available the coil position optimization is
timed too. Results are written as a machine-stamped JSON file and a Markdown
report.

## Running on a fresh Linux machine

One command sets up everything under `~/tmswarp-bench` and runs the standard
suite. It needs `git`, `python3` and network access; the datasets and SimNIBS
are downloaded on first use.

```sh
curl -sSL https://raw.githubusercontent.com/pieper/TMSWarp/main/benchmarks/bootstrap.sh \
  | bash -s -- --label vast-4090 --install-simnibs
```

The label names the machine in the results. On vast.ai this line can go in
the instance's on-start script; the run takes roughly half an hour on a
mid-range GPU (see the timings below) and finishes by creating
`~/tmswarp-bench/DONE` (or `FAILED`). Progress is in
`~/tmswarp-bench/bootstrap.log`.

Options: `--suite quick|standard|full`, `--provider jetstream2`,
`--branch NAME`, `--no-optimization`. Re-running reuses the environment,
datasets and SimNIBS install already there.

## Running on a laptop or an existing environment

```sh
pip install "tmswarp[bench] @ git+https://github.com/pieper/TMSWarp"
python -m tmswarp.bench run --suite quick --label my-laptop
```

This works on Linux, macOS and Windows wherever `warp-lang` has a wheel.
Without a CUDA GPU the Warp solver runs on the CPU, which is slow on large
meshes, so only the small meshes are timed that way. SimNIBS is used if it is
found (`~/SimNIBS*`, `~/Applications/SimNIBS*`, `--simnibs-python PATH` or
the `SIMNIBS_PYTHON` variable), the optimization if `TMSService.py` is found
(`--tmsservice PATH` or `TMSWARP_TMSSERVICE`).

`python -m tmswarp.bench fetch --suite standard` downloads the datasets
ahead of time. They live in `~/.cache/tmswarp` (`TMSWARP_CACHE` overrides).

## What is measured

| Mesh | Nodes | Tets | Origin |
|---|---|---|---|
| sphere3 | 4.6 k | 23 k | SimNIBS test mesh, uniform conductivity |
| sphere3-r1, -r2, -r3 | 33 k, 254 k, 2.0 M | 182 k, 1.45 M, 11.6 M | sphere3 refined 1, 2, 3 times |
| ernie-lowres | 223 k | 1.3 M | SimNIBS example head, low resolution |
| ernie-lowres-r1 | 1.8 M | 10.5 M | ernie-lowres refined once |
| ernie-full | 797 k | 4.5 M | SimNIBS example head |
| ernie-full-r1 | 6.1 M | 35.7 M | ernie-full refined once |

Refinement splits every tetrahedron into eight. It does not move tissue
boundaries, so comparing solutions across levels measures the numerical
discretization error only. For the spheres the outer surface is projected
back onto the sphere and the analytical solution is available as well.

The suites are `quick` (sphere3, sphere3-r1, ernie-lowres), `standard`
(adds sphere3-r2, ernie-lowres-r1, ernie-full and SimNIBS's PARDISO direct
solver) and `full` (adds sphere3-r3, ernie-full-r1, and the slow CPU runs).

For each mesh, with the dipole configuration of the validation figures:

- **Reference:** float64 conjugate gradients to a relative residual of 1e-10,
  cached and used to score everything else (RDM and MAG, overall and in grey
  matter).
- **SciPy CG, float64, CPU:** the same algorithm as the Warp solver, on the
  CPU, to a relative residual of 1e-6.
- **SciPy direct, float64, CPU:** LU factorization once, then back
  substitution; small meshes only unless the suite asks.
- **Warp CG, float32:** on the GPU (and on the CPU for small meshes), at
  relative residuals of 1e-3, 1e-4 and 1e-6. Assembly, right-hand side,
  solve, iterations, and the E magnitude readback are timed separately, plus
  the gradient of the E magnitude at a target element (one cold evaluation,
  then warm-started evaluations as the coil moves 1 mm at a time).
- **SimNIBS:** its default solver (PETSc CG with hypre BoomerAMG) and its
  PARDISO direct solver, on the same mesh and dipole field.
- **Optimization:** the SlicerTMS coil optimizer for three fixed targets on
  the head meshes, with the field at the returned position checked against
  the reference.

Every job runs in its own process with a timeout, so a job that runs out of
memory or crashes is recorded as such and the run continues. Times are
wall-clock, cold start unless stated, best of three repeats. Kernel
compilation is done beforehand on a tiny mesh and not included.

## Reports

Each run writes `results/<label>_<gpu>_<timestamp>.json` and `.md`. To
combine runs from several machines into one report:

```sh
python -m tmswarp.bench report results/*.json --output comparison.md
```

Results from the machines used for the paper are kept in
`benchmarks/results/bench/` (the older `benchmarks/results/*.json` files are
from `run_benchmarks.py`, which has a different format). Timings across machines are only comparable with
care: the CPU rows depend on the host's CPU, and rented instances of the same
GPU model differ in host CPU, memory and PCIe generation.
