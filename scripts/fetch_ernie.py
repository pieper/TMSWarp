"""Fetch the SimNIBS ernie example dataset and extract it to ernie_data.npz.

The ernie (low-resolution) head mesh is the standard SimNIBS human head model
used to validate realistic TMS simulations.  It has 6 tissue types and
~1.3 million tetrahedra (222k nodes).

Requirements
------------
- SimNIBS must be installed; this script must be run with SimNIBS Python:
    /path/to/SimNIBS-4.5/simnibs_env/bin/python scripts/fetch_ernie.py

- `curl` must be available on PATH (used for download to avoid SSL cert
  issues with the SimNIBS bundled Python).

- Alternatively, pre-download the zip and pass it as an argument:
    curl -L https://github.com/simnibs/example-dataset/releases/download/v4.0-lowres/ernie_lowres_V2.zip \\
         -o /tmp/ernie_lowres_V2.zip
    /path/to/SimNIBS-4.5/simnibs_env/bin/python scripts/fetch_ernie.py /tmp/ernie_lowres_V2.zip

Output
------
Creates  ernie_data.npz  in the TMSWarp directory (same level as this scripts/
folder).  The file contains:
    nodes           float64 (N, 3)  — node coordinates in metres
    elements        int32   (E, 4)  — tet connectivity, 0-based
    conductivity    float64 (E,)    — tissue conductivity in S/m
    tag1            int32   (E,)    — SimNIBS tissue tag (1-6)

The ernie_data.npz file is gitignored (it is ~25 MB and must be fetched from
the SimNIBS example dataset which is separately licensed).

Tissue conductivities (SimNIBS defaults, Saturnino et al. 2019)
--------------------------------------------------------------
    1  White matter   0.126 S/m
    2  Gray matter    0.275 S/m
    3  CSF            1.654 S/m
    4  Skull/bone     0.010 S/m
    5  Scalp          0.465 S/m
    6  Eye balls      0.500 S/m
"""

import os
import subprocess
import sys
import tempfile
import zipfile

import numpy as np

# ---------------------------------------------------------------------------
# SimNIBS path
# ---------------------------------------------------------------------------
SIMNIBS_SITE = (
    "/Users/pieper/Applications/SimNIBS-4.5/"
    "simnibs_env/lib/python3.11/site-packages"
)
for candidate in [
    SIMNIBS_SITE,
    "/opt/SimNIBS-4.5/simnibs_env/lib/python3.11/site-packages",
    "/usr/local/SimNIBS-4.5/simnibs_env/lib/python3.11/site-packages",
]:
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, candidate)
        break

from simnibs.mesh_tools import mesh_io  # noqa: E402

# ---------------------------------------------------------------------------
# Default conductivities (Saturnino et al. 2019, SimNIBS 4)
# ---------------------------------------------------------------------------
CONDUCTIVITY_MAP = {
    1: 0.126,   # white matter
    2: 0.275,   # gray matter
    3: 1.654,   # CSF
    4: 0.010,   # skull
    5: 0.465,   # scalp
    6: 0.500,   # eye balls
}

DOWNLOAD_URL = (
    "https://github.com/simnibs/example-dataset/releases/"
    "download/v4.0-lowres/ernie_lowres_V2.zip"
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.abspath(os.path.join(HERE, ".."))
OUTPATH = os.path.join(OUTDIR, "ernie_data.npz")


def download_zip(dest_path: str) -> None:
    """Download the ernie dataset zip using curl."""
    print(f"Downloading {DOWNLOAD_URL}")
    print("(This is ~394 MB; may take a few minutes on slow connections.)")
    ret = subprocess.run(
        ["curl", "-L", "--progress-bar", DOWNLOAD_URL, "-o", dest_path],
        check=True,
    )


def extract_msh(zip_path: str, extract_dir: str) -> str:
    """Extract ernie.msh from the zip; return the path to the .msh file."""
    print(f"Extracting from {zip_path} ...")
    target = "m2m_ernie/ernie.msh"
    with zipfile.ZipFile(zip_path) as z:
        # Verify the expected file is present
        names = z.namelist()
        matches = [n for n in names if n.endswith("ernie.msh")]
        if not matches:
            raise FileNotFoundError(
                f"ernie.msh not found in zip. Available .msh files: "
                f"{[n for n in names if n.endswith('.msh')]}"
            )
        msh_name = matches[0]
        print(f"  Extracting: {msh_name}")
        z.extract(msh_name, extract_dir)
    return os.path.join(extract_dir, msh_name)


def convert_to_npz(msh_path: str, out_path: str) -> None:
    """Read ernie.msh with SimNIBS and save relevant arrays to .npz."""
    print(f"Reading mesh: {msh_path}")
    m = mesh_io.read_msh(msh_path)
    m = m.crop_mesh(elm_type=4)  # tetrahedra only

    nodes_m = m.nodes.node_coord * 1e-3              # mm → metres, (N, 3)
    elements = m.elm.node_number_list[:, :4] - 1     # 1-based → 0-based, (E, 4)
    tag1 = m.elm.tag1.astype(np.int32)               # tissue tag, (E,)

    conductivity = np.array(
        [CONDUCTIVITY_MAP.get(int(t), 0.275) for t in tag1],
        dtype=np.float64,
    )

    print(f"  nodes:      {nodes_m.shape}")
    print(f"  elements:   {elements.shape}")
    tag_counts = {int(t): int(np.sum(tag1 == t)) for t in np.unique(tag1)}
    tissue_names = {
        1: "white matter", 2: "gray matter", 3: "CSF",
        4: "skull", 5: "scalp", 6: "eyes",
    }
    for tag, count in sorted(tag_counts.items()):
        name = tissue_names.get(tag, f"tag{tag}")
        sigma = CONDUCTIVITY_MAP.get(tag, 0.0)
        print(f"  tag {tag} ({name:<12}): {count:>8,} elements  σ={sigma} S/m")

    np.savez_compressed(
        out_path,
        nodes=nodes_m,
        elements=elements,
        conductivity=conductivity,
        tag1=tag1,
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nSaved: {out_path}  ({size_mb:.1f} MB)")


def main():
    zip_arg = sys.argv[1] if len(sys.argv) > 1 else None

    if zip_arg and os.path.isfile(zip_arg):
        zip_path = os.path.abspath(zip_arg)
        own_zip = False
    else:
        # Auto-download
        tmp_zip = tempfile.mktemp(suffix=".zip")
        download_zip(tmp_zip)
        zip_path = tmp_zip
        own_zip = True

    tmp_extract = tempfile.mkdtemp(prefix="ernie_extract_")
    try:
        msh_path = extract_msh(zip_path, tmp_extract)
        convert_to_npz(msh_path, OUTPATH)
    finally:
        import shutil
        shutil.rmtree(tmp_extract, ignore_errors=True)
        if own_zip:
            os.remove(zip_path)

    print("\nDone. Run the validation with:")
    print("  pixi run python benchmarks/ernie_validation.py")


if __name__ == "__main__":
    main()
