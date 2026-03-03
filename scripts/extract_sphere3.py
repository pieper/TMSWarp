"""Extract sphere3.msh mesh data from SimNIBS installation to a .npz file.

Run this script with SimNIBS Python:

    /path/to/SimNIBS-4.5/simnibs_env/bin/python scripts/extract_sphere3.py

This creates sphere3_data.npz in the current directory (TMSWarp/), which can
then be used by benchmarks/sphere3_validation.py with the pixi/tmswarp env.
"""

import os
import sys
import numpy as np

# Locate SimNIBS (try common install paths)
SIMNIBS_CANDIDATES = [
    "/Users/pieper/Applications/SimNIBS-4.5/simnibs_env/lib/python3.11/site-packages",
    "/opt/SimNIBS-4.5/simnibs_env/lib/python3.11/site-packages",
    "/usr/local/SimNIBS-4.5/simnibs_env/lib/python3.11/site-packages",
]

for candidate in SIMNIBS_CANDIDATES:
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)
        break

from simnibs import SIMNIBSDIR
from simnibs.mesh_tools import mesh_io

fn = os.path.join(SIMNIBSDIR, "_internal_resources", "testing_files", "sphere3.msh")
print(f"Loading: {fn}")
m = mesh_io.read_msh(fn)

# Keep only tetrahedra (elm_type=4)
m = m.crop_mesh(elm_type=4)

# Nodes in mm — convert to metres (SI units throughout)
nodes_m = m.nodes.node_coord * 1e-3  # shape (N, 3), float64

# Element connectivity: node_number_list is 1-based, 4 nodes per tet
elements = m.elm.node_number_list[:, :4] - 1  # shape (E, 4), 0-based, int32

# Region tags (3, 4, 5 for sphere3's three shells)
tag1 = m.elm.tag1.astype(np.int32)  # shape (E,)

# Uniform conductivity (same as SimNIBS TMS validation)
conductivity = np.ones(len(elements), dtype=np.float64)

outpath = os.path.join(os.path.dirname(__file__), "..", "sphere3_data.npz")
outpath = os.path.abspath(outpath)
np.savez_compressed(
    outpath,
    nodes=nodes_m,
    elements=elements,
    conductivity=conductivity,
    tag1=tag1,
)
print(f"Saved: {outpath}")
print(f"  nodes:    {nodes_m.shape}")
print(f"  elements: {elements.shape}")
print(f"  tags:     {np.unique(tag1)}")
print(f"  radius range: {np.linalg.norm(nodes_m, axis=1).min()*1000:.1f} mm "
      f"to {np.linalg.norm(nodes_m, axis=1).max()*1000:.1f} mm")
