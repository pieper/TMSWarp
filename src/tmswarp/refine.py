"""Uniform refinement of tetrahedral meshes.

Each tetrahedron is split into eight (four corner tetrahedra and four from
the inner octahedron, Bey 1995), so every level multiplies the element count
by eight and halves the edge length.  The refined mesh is nested in the
original: each child lies inside its parent and inherits its conductivity.

Refinement does not improve the geometry.  Tissue boundaries stay where the
original mesh put them, so comparing solutions across levels measures the
discretization error of the FEM solve, not the segmentation error.  For
spheres the outer boundary can be projected back onto the sphere.
"""

import numpy as np

from tmswarp.conductor import TetMesh

# Local vertex pairs of the six edges, in the order used below
_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])

# Children in terms of local indices 0-3 (parent vertices) and 4-9 (midpoints
# of the edges in _EDGES order: m01, m02, m03, m12, m13, m23)
_CHILDREN = np.array([
    [0, 4, 5, 6],
    [4, 1, 7, 8],
    [5, 7, 2, 9],
    [6, 8, 9, 3],
    [4, 5, 6, 8],
    [4, 5, 7, 8],
    [5, 6, 8, 9],
    [5, 7, 8, 9],
])


def _signed_volumes(nodes, elements):
    coords = nodes[elements]
    return np.linalg.det(coords[:, 1:] - coords[:, 0:1]) / 6.0


def _boundary_edge_keys(elements, n_nodes):
    """Keys (i * n_nodes + j, i < j) of the edges of the boundary faces."""
    face_idx = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    faces = np.sort(elements[:, face_idx].reshape(-1, 3), axis=1).astype(np.int64)
    keys = (faces[:, 0] * n_nodes + faces[:, 1]) * n_nodes + faces[:, 2]
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    boundary = faces[counts[inverse] == 1]
    edges = np.concatenate(
        [boundary[:, [0, 1]], boundary[:, [0, 2]], boundary[:, [1, 2]]]
    )
    return np.unique(edges[:, 0] * n_nodes + edges[:, 1])


def refine_uniform(mesh, levels=1, sphere_center=None):
    """Refine a mesh uniformly.

    Parameters
    ----------
    mesh : TetMesh
    levels : int
        Number of refinement levels; each multiplies the element count by 8.
    sphere_center : (3,) array or None
        If given, the mesh is taken to be a sphere about this point, and new
        nodes on the outer boundary are moved radially onto the sphere.

    Returns
    -------
    refined : TetMesh
    parent : (n_refined_elements,) int array
        Index, in the input mesh, of the element each refined element lies in.
    """
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    elements = np.asarray(mesh.elements, dtype=np.int64)
    conductivity = np.asarray(mesh.conductivity)
    parent = np.arange(len(elements))

    for _ in range(levels):
        n_nodes = len(nodes)
        pairs = np.sort(elements[:, _EDGES].reshape(-1, 2), axis=1)
        keys = pairs[:, 0] * n_nodes + pairs[:, 1]
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        a = unique_keys // n_nodes
        b = unique_keys % n_nodes
        midpoints = 0.5 * (nodes[a] + nodes[b])

        if sphere_center is not None:
            on_boundary = np.isin(
                unique_keys, _boundary_edge_keys(elements, n_nodes)
            )
            center = np.asarray(sphere_center, dtype=np.float64)
            radius = 0.5 * (
                np.linalg.norm(nodes[a[on_boundary]] - center, axis=1)
                + np.linalg.norm(nodes[b[on_boundary]] - center, axis=1)
            )
            offset = midpoints[on_boundary] - center
            offset *= (radius / np.linalg.norm(offset, axis=1))[:, None]
            midpoints[on_boundary] = center + offset

        # Local index table per element: 4 parent vertices + 6 midpoints
        local = np.concatenate(
            [elements, n_nodes + inverse.reshape(-1, 6)], axis=1
        )
        children = local[:, _CHILDREN].reshape(-1, 4)
        nodes = np.concatenate([nodes, midpoints])

        # Give every child the orientation of its parent
        parent_sign = np.sign(_signed_volumes(nodes, elements))
        child_sign = np.sign(_signed_volumes(nodes, children))
        flip = child_sign != np.repeat(parent_sign, 8)
        children[flip] = children[flip][:, [1, 0, 2, 3]]

        elements = children
        conductivity = np.repeat(conductivity, 8)
        parent = np.repeat(parent, 8)

    refined = TetMesh(
        nodes=nodes,
        elements=elements.astype(np.int32),
        conductivity=conductivity,
    )
    return refined, parent


def restrict_to_parents(values, volumes, parent, n_parents):
    """Volume-weighted average of per-element values over each parent.

    Parameters
    ----------
    values : (n_refined, k) or (n_refined,) array
        Per-element quantity on the refined mesh, e.g. the E-field.
    volumes : (n_refined,) array
        Volumes of the refined elements.
    parent : (n_refined,) int array
        From ``refine_uniform``.
    n_parents : int

    Returns
    -------
    (n_parents, k) or (n_parents,) array
    """
    values = np.asarray(values)
    total = np.bincount(parent, weights=volumes, minlength=n_parents)
    if values.ndim == 1:
        return np.bincount(
            parent, weights=values * volumes, minlength=n_parents
        ) / total
    out = np.empty((n_parents, values.shape[1]))
    for k in range(values.shape[1]):
        out[:, k] = np.bincount(
            parent, weights=values[:, k] * volumes, minlength=n_parents
        ) / total
    return out
