# TMSWarp

TMS simulation utilities using [Nvidia Warp](https://github.com/NVIDIA/warp).

Includes a pure-NumPy/SciPy FEM solver and a GPU-accelerated solver via `warp.fem`, both validated against the Heller & van Hulsteyn (1992) analytical solution on a spherical conductor.

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

On a machine with an Nvidia GPU, pass `--device cuda:0`:

```bash
pixi run python benchmarks/run_benchmarks.py --device cuda:0
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `cpu` | Warp device (`cpu`, `cuda:0`, etc.) |
| `--repeats` | `3` | Timing repeats per mesh size (minimum is recorded) |
| `--notes` | `""` | Free-text note stored in the result metadata |

The script runs both the NumPy and Warp.fem solvers at five mesh resolutions, validates against the analytical solution, and saves a JSON file to `benchmarks/results/`. The filename encodes hostname, architecture, device, and timestamp, e.g.:

```
benchmarks/results/slate_arm64_cpu_20260302_164440.json
```

### Submitting results to the repo

After running benchmarks on a new machine, commit the JSON file:

```bash
git add benchmarks/results/*.json
git commit -m "Add benchmark results from <machine description>"
git push
```

Include a short machine description in the commit message (e.g. "M2 MacBook Pro, CPU-only" or "RTX 5060 Ti, CUDA 12.8").

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
