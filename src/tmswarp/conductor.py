"""Volume conductor model: tetrahedral mesh with conductivity."""

import dataclasses

import numpy as np


@dataclasses.dataclass
class TetMesh:
    """Container for a tetrahedral mesh with per-element conductivity.

    Attributes
    ----------
    nodes : (n_nodes, 3) array
        Node coordinates in meters.
    elements : (n_elements, 4) int array
        Node indices per tetrahedron.
    conductivity : (n_elements,) array
        Scalar conductivity in S/m per element.
    """

    nodes: np.ndarray
    elements: np.ndarray
    conductivity: np.ndarray


def element_barycenters(mesh: TetMesh) -> np.ndarray:
    """Compute element barycenters.

    Returns
    -------
    (n_elements, 3) array
    """
    return mesh.nodes[mesh.elements].mean(axis=1)


def element_volumes(mesh: TetMesh) -> np.ndarray:
    """Compute volume of each tetrahedron.

    Returns
    -------
    (n_elements,) array
    """
    coords = mesh.nodes[mesh.elements]  # (n_elem, 4, 3)
    edges = coords[:, 1:] - coords[:, 0:1]  # (n_elem, 3, 3)
    return np.abs(np.linalg.det(edges)) / 6.0


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Generate n approximately uniformly distributed points on a unit sphere.

    Uses the golden spiral method for quasi-uniform coverage.

    Returns
    -------
    (n, 3) array of points on the unit sphere.
    """
    golden_ratio = (1 + np.sqrt(5)) / 2
    i = np.arange(n)
    theta = 2 * np.pi * i / golden_ratio
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])


def make_sphere_mesh(
    radius: float = 0.095,
    n_shells: int = 5,
    n_surface: int = 200,
    conductivity: float = 1.0,
    min_quality: float = 0.01,
) -> TetMesh:
    """Generate a tetrahedral mesh of a sphere using scipy Delaunay.

    Points are distributed on concentric spherical shells using the
    Fibonacci spiral method, then tetrahedralized with scipy.spatial.Delaunay.
    Elements with centroids outside the sphere or with very poor quality
    (near-degenerate slivers) are removed.

    Parameters
    ----------
    radius : float
        Sphere radius in meters. Default 0.095 (95 mm).
    n_shells : int
        Number of concentric shells (not counting center point).
    n_surface : int
        Approximate number of points on the outermost shell.
    conductivity : float
        Uniform conductivity in S/m.
    min_quality : float
        Minimum element quality (volume / ideal volume ratio) to keep.

    Returns
    -------
    TetMesh
    """
    from scipy.spatial import Delaunay

    # Generate points: center + concentric shells
    points = [np.array([[0.0, 0.0, 0.0]])]
    for i in range(1, n_shells + 1):
        r_shell = radius * i / n_shells
        # Scale point count with shell surface area
        n_pts = max(12, int(n_surface * (i / n_shells) ** 2))
        pts = _fibonacci_sphere(n_pts) * r_shell
        points.append(pts)

    all_points = np.vstack(points)

    # Tetrahedralize
    tri = Delaunay(all_points)
    elements = tri.simplices

    # Filter: keep only tets with centroid inside sphere
    centroids = all_points[elements].mean(axis=1)
    inside = np.linalg.norm(centroids, axis=1) <= radius * 1.01
    elements = elements[inside]

    # Filter: remove degenerate elements (near-zero volume)
    coords = all_points[elements]
    edges = coords[:, 1:] - coords[:, 0:1]
    volumes = np.abs(np.linalg.det(edges)) / 6.0

    # Quality: compare volume to cube of max edge length
    max_edge = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            edge_len = np.linalg.norm(coords[:, i] - coords[:, j], axis=1)
            max_edge = np.maximum(max_edge, edge_len)
    # Ideal tet volume for given max edge: (max_edge^3) / (6*sqrt(2))
    ideal_vol = max_edge**3 / (6.0 * np.sqrt(2.0))
    quality = np.where(ideal_vol > 0, volumes / ideal_vol, 0.0)

    good = quality >= min_quality
    elements = elements[good]

    cond = np.full(len(elements), conductivity, dtype=np.float64)
    return TetMesh(nodes=all_points, elements=elements, conductivity=cond)
