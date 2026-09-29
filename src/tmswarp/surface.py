"""Boundary-surface helpers for constraining a coil to the scalp."""

import numpy as np


def closest_point_on_triangles(p, a, b, c):
    """Closest point to ``p`` on each of a set of triangles.

    Vectorized form of the region test in Ericson, "Real-Time Collision
    Detection" (2005), section 5.1.5.

    Parameters
    ----------
    p : (3,) array
        Query point.
    a, b, c : (n, 3) arrays
        Triangle vertices.

    Returns
    -------
    points : (n, 3) array
        Closest point on each triangle.
    bary : (n, 3) array
        Barycentric coordinates of each closest point with respect to
        (a, b, c); non-negative and summing to one.
    """
    p = np.asarray(p, dtype=np.float64)
    ab = b - a
    ac = c - a
    ap = p - a
    bp = p - b
    cp = p - c
    d1 = np.einsum('ij,ij->i', ab, ap)
    d2 = np.einsum('ij,ij->i', ac, ap)
    d3 = np.einsum('ij,ij->i', ab, bp)
    d4 = np.einsum('ij,ij->i', ac, bp)
    d5 = np.einsum('ij,ij->i', ab, cp)
    d6 = np.einsum('ij,ij->i', ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    def ratio(num, den):
        return num / np.where(den != 0.0, den, 1.0)

    zero = np.zeros_like(d1)
    one = np.ones_like(d1)

    # Interior of the face
    v = ratio(vb, va + vb + vc)
    w = ratio(vc, va + vb + vc)
    bary = np.stack([1.0 - v - w, v, w], axis=1)

    # The remaining regions are assigned in reverse order of Ericson's
    # early returns, so that the earlier tests take precedence.
    def assign(mask, values):
        bary[mask] = np.stack(values, axis=1)[mask]

    t = ratio(d4 - d3, (d4 - d3) + (d5 - d6))
    assign((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), [zero, 1.0 - t, t])
    t = ratio(d2, d2 - d6)
    assign((vb <= 0) & (d2 >= 0) & (d6 <= 0), [1.0 - t, zero, t])
    assign((d6 >= 0) & (d5 <= d6), [zero, zero, one])
    t = ratio(d1, d1 - d3)
    assign((vc <= 0) & (d1 >= 0) & (d3 <= 0), [1.0 - t, t, zero])
    assign((d3 >= 0) & (d4 <= d3), [zero, one, zero])
    assign((d1 <= 0) & (d2 <= 0), [one, zero, zero])

    points = bary[:, 0:1] * a + bary[:, 1:2] * b + bary[:, 2:3] * c
    return points, bary


def vertex_normals(n_nodes, faces, face_normals, face_areas):
    """Area-weighted average of the adjacent face normals at each vertex.

    Parameters
    ----------
    n_nodes : int
    faces : (n_faces, 3) int array
    face_normals : (n_faces, 3) array
        Consistently oriented unit normals.
    face_areas : (n_faces,) array

    Returns
    -------
    (n_nodes, 3) array
        Unit normals; zero for vertices that belong to no face.
    """
    normals = np.zeros((n_nodes, 3), dtype=np.float64)
    weighted = face_normals * face_areas[:, None]
    for i in range(3):
        np.add.at(normals, faces[:, i], weighted)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-30)
