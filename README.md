# TMSWarp

TMS simulation utilities using [Nvidia Warp](https://github.com/NVIDIA/warp).

Includes a pure-NumPy/SciPy FEM solver and a GPU-accelerated solver via `warp.fem`, both validated against the Heller & van Hulsteyn (1992) analytical solution on a spherical conductor.



https://github.com/user-attachments/assets/1da92c75-caf3-43ce-b754-12fca91ae1b4

Example interactions using an optimizer to maximize the electric field at a target point in the brain.  This leverages the diffential Warp-based simulation.

## Installation

```bash
pip install tmswarp
```

For GPU support (requires CUDA):

```bash
pip install "tmswarp[gpu]"
```

## Development

This project uses [pixi](https://pixi.sh) for environment management:

```bash
cd TMSWarp
pixi install
pixi run pip install -e "." --no-deps
pixi run pytest -v
```

Or with pip directly:

```bash
pip install -e ".[test]"
pytest -v
```

## Benchmarks

The benchmarking system measures timing and accuracy of all solver implementations across mesh sizes and saves machine-stamped results as JSON for cross-machine comparison.

### Running benchmarks

```bash
cd TMSWarp
pixi run python benchmarks/run_benchmarks.py
```

The script auto-detects the best available device — it will use `cuda:0` on a machine with an Nvidia GPU, or `cpu` otherwise. You can override this:

```bash
pixi run python benchmarks/run_benchmarks.py --device cpu       # force CPU
pixi run python benchmarks/run_benchmarks.py --device cuda:0    # force specific GPU
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | auto-detect | Warp device (`cpu`, `cuda:0`, etc.). Uses GPU if available. |
| `--repeats` | `3` | Timing repeats per mesh size (minimum is recorded) |
| `--notes` | `""` | Free-text note stored in the result metadata |

The script runs both the NumPy and Warp.fem solvers at five mesh resolutions, validates against the analytical solution, and saves a JSON file to `benchmarks/results/`. The JSON includes full machine metadata (hostname, platform, Python/NumPy/SciPy/Warp versions, GPU device names and memory). The filename encodes hostname, architecture, device, and timestamp, e.g.:

```
benchmarks/results/slate_arm64_cpu_20260302_164440.json
benchmarks/results/workstation_x86_64_cuda0_20260302_201530.json
```

### Submitting results to the repo

After running benchmarks on a new machine, commit the JSON file:

```bash
git add benchmarks/results/*.json
git commit -m "Add benchmark results from <machine description>"
git push
```

Include a short machine description in the commit message (e.g. "M2 MacBook Pro, CPU-only" or "RTX 5060 Ti, CUDA 12.8").

### GPU scaling benchmark

The standard benchmark uses small meshes (up to 29k elements) optimized for accuracy validation, which is too small for GPU to show speedups. The GPU scaling benchmark goes up to 500k+ elements to find the CPU/GPU crossover point:

```bash
python benchmarks/gpu_scaling.py
python benchmarks/gpu_scaling.py --max-elements 500000
```

This compares three solver configurations side-by-side at each mesh size:
- **NumPy FEM** — scipy `spsolve` (direct solver, CPU-only)
- **Warp.fem CPU** — CG iterative solver on CPU
- **Warp.fem GPU** — CG iterative solver on CUDA GPU

The output prints the GPU/CPU speedup for each mesh size, showing where GPU starts to dominate (typically around 100k+ elements). The script confirms which device is actually being used and saves results to `benchmarks/results/`.

### Comparing results across machines

Once multiple result files are checked in, compare them:

```bash
pixi run python benchmarks/compare_results.py
pixi run python benchmarks/compare_results.py --metric warp_total
pixi run python benchmarks/compare_results.py --metric warp_total --plot
```

Available metrics: `numpy_total`, `numpy_assembly`, `numpy_solve`, `warp_total`, `warp_assembly`, `warp_solve`, `rdm_numpy`, `rdm_warp`.

### Convergence visualization

The `visualize_convergence.py` script generates publication-quality images comparing all three methods (Analytical, NumPy FEM, Warp.fem) at each mesh resolution:

| Output file | Contents |
|---|---|
| `convergence_visualization.png` | z=0 cross-sections: Analytical \|E\|, NumPy FEM \|E\|, Warp.fem \|E\|, Relative Error |
| `convergence_plot.png` | RDM and \|MAG\| convergence vs element count |
| `timing_plot.png` | Wall-clock solver time vs element count |

```bash
pixi run python visualize_convergence.py
```
