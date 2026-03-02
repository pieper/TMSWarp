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

## Running Benchmarks

The `visualize_convergence.py` script solves the TMS sphere benchmark at five mesh resolutions using all available solvers (Analytical, NumPy FEM, and Warp.fem if installed). It produces three images:

| Output file | Contents |
|---|---|
| `convergence_visualization.png` | z=0 cross-sections: Analytical \|E\|, NumPy FEM \|E\|, Warp.fem \|E\|, Relative Error |
| `convergence_plot.png` | RDM and \|MAG\| convergence vs element count |
| `timing_plot.png` | Wall-clock solver time vs element count |

To run:

```bash
cd TMSWarp
pixi run python visualize_convergence.py
```

The script auto-detects whether Warp is available and which device to use (CPU or CUDA GPU). On a machine with an Nvidia GPU the Warp solver will run on `cuda:0` automatically.

### Submitting benchmark results

After running the benchmarks on a new machine, commit the updated images and push:

```bash
git add convergence_visualization.png convergence_plot.png timing_plot.png
git commit -m "Update benchmark results from <machine description>"
git push
```

This keeps the repository's benchmark images current with the latest hardware/software results. Include a short description of the machine in the commit message (e.g. "M2 MacBook Pro, CPU-only" or "RTX 5060 Ti, CUDA 12.8").
