"""Benchmark datasets: fetch, convert, refine and cache.

Nothing here is redistributed with TMSWarp.  The head meshes are downloaded
from the SimNIBS example dataset on first use and cached under
``~/.cache/tmswarp`` (override with the ``TMSWARP_CACHE`` environment
variable).
"""

import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from tmswarp.conductor import TetMesh
from tmswarp.refine import refine_uniform

# SimNIBS standard conductivities (S/m) by tissue tag
CONDUCTIVITY = {
    1: 0.126,    # white matter
    2: 0.275,    # grey matter
    3: 1.654,    # CSF
    4: 0.010,    # bone
    5: 0.465,    # scalp
    6: 0.500,    # eye balls
    7: 0.008,    # compact bone
    8: 0.025,    # spongy bone
    9: 0.600,    # blood
    10: 0.160,   # muscle
    11: 0.880,   # cartilage
    12: 0.078,   # fat
}

_SPHERE3_URL = (
    "https://raw.githubusercontent.com/pieper/TMSWarp/main/sphere3_data.npz"
)
_ERNIE_URLS = {
    "ernie-lowres": (
        "https://github.com/simnibs/example-dataset/releases/"
        "download/v4.0-lowres/ernie_lowres_V2.zip"
    ),
    "ernie-full": (
        "https://github.com/simnibs/example-dataset/releases/"
        "download/v4.1/simnibs4_examples.zip"
    ),
}

# name -> (base dataset, refinement levels)
DATASETS = {
    "sphere3": ("sphere3", 0),
    "sphere3-r1": ("sphere3", 1),
    "sphere3-r2": ("sphere3", 2),
    "sphere3-r3": ("sphere3", 3),
    "ernie-lowres": ("ernie-lowres", 0),
    "ernie-lowres-r1": ("ernie-lowres", 1),
    "ernie-full": ("ernie-full", 0),
    "ernie-full-r1": ("ernie-full", 1),
}


def cache_dir():
    path = Path(os.environ.get("TMSWARP_CACHE", "~/.cache/tmswarp")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_sphere(name):
    return DATASETS[name][0] == "sphere3"


def _download(url, dest, log=print):
    log(f"  downloading {url}")
    tmp = str(dest) + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)


def _fetch_sphere3(path, log):
    local = Path(__file__).resolve().parents[3] / "sphere3_data.npz"
    if local.exists():
        shutil.copy2(local, path)
    else:
        _download(_SPHERE3_URL, path, log)


def _fetch_ernie(name, path, log):
    try:
        import meshio
    except ImportError as exc:
        raise ImportError(
            "meshio is needed to convert the SimNIBS example mesh: "
            "pip install meshio"
        ) from exc

    tmp_dir = Path(tempfile.mkdtemp(prefix="tmswarp_ernie_", dir=cache_dir()))
    try:
        zip_path = tmp_dir / "ernie.zip"
        _download(_ERNIE_URLS[name], zip_path, log)
        with zipfile.ZipFile(zip_path) as z:
            names = [n for n in z.namelist() if n.endswith("m2m_ernie/ernie.msh")]
            if not names:
                raise FileNotFoundError("ernie.msh not found in the download")
            z.extract(names[0], tmp_dir)
        log("  converting ernie.msh")
        m = meshio.read(tmp_dir / names[0])
        tets = m.cells_dict["tetra"].astype(np.int32)
        tags = m.cell_data_dict["gmsh:physical"]["tetra"].astype(np.int32)
        unknown = sorted(set(np.unique(tags)) - set(CONDUCTIVITY))
        if unknown:
            raise ValueError(f"No conductivity for tissue tags {unknown}")
        sigma = np.array([CONDUCTIVITY[int(t)] for t in tags])

        # Keep only the nodes that tetrahedra use (the file also has surfaces)
        used = np.unique(tets)
        remap = np.full(len(m.points), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        np.savez(path, nodes=m.points[used] * 1e-3,
                 elements=remap[tets].astype(np.int32),
                 conductivity=sigma, tag1=tags)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def path_for(name):
    return cache_dir() / f"{name}.npz"


def ensure(name, log=print):
    """Make sure dataset ``name`` is in the cache; return its path."""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    path = path_for(name)
    if path.exists():
        return path

    base, levels = DATASETS[name]
    if levels == 0:
        log(f"Preparing {name}")
        if base == "sphere3":
            _fetch_sphere3(path, log)
        else:
            _fetch_ernie(base, path, log)
        return path

    base_path = ensure(base, log)
    log(f"Refining {base} by {levels} level(s) -> {name}")
    data = np.load(base_path)
    mesh = TetMesh(data["nodes"].astype(np.float64),
                   data["elements"].astype(np.int32),
                   data["conductivity"].astype(np.float64))
    refined, parent = refine_uniform(
        mesh, levels=levels,
        sphere_center=np.zeros(3) if base == "sphere3" else None,
    )
    extra = {}
    if "tag1" in data:
        extra["tag1"] = data["tag1"][parent]
    np.savez(path, nodes=refined.nodes, elements=refined.elements,
             conductivity=refined.conductivity, parent=parent, **extra)
    return path


def load(name, log=print):
    """Return (mesh, tag1 or None, parent or None) for dataset ``name``."""
    data = np.load(ensure(name, log))
    mesh = TetMesh(
        nodes=data["nodes"].astype(np.float64),
        elements=data["elements"].astype(np.int32),
        conductivity=data["conductivity"].astype(np.float64),
    )
    tag1 = data["tag1"].astype(np.int32) if "tag1" in data else None
    parent = data["parent"] if "parent" in data else None
    return mesh, tag1, parent


def dipole(name, mesh, kind):
    """Dipole position (m), moment and dI/dt (A/s) for a dataset.

    ``far`` matches the validation figures (a distant dipole with a
    tangential moment).  ``near`` is 10 mm above the top of the scalp with
    the moment normal to it, which is how the coil is used interactively.
    """
    if kind == "far":
        z = 0.3 if is_sphere(name) else 0.2
        return np.array([0.0, 0.0, z]), np.array([1.0, 0.0, 0.0]), 1e6
    if kind == "near":
        top = mesh.nodes[np.argmax(mesh.nodes[:, 2])]
        return top + np.array([0.0, 0.0, 0.010]), np.array([0.0, 0.0, 1.0]), 1e6
    raise ValueError(f"Unknown dipole kind {kind!r}")
