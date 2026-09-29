"""Tests for the boundary-surface helpers."""

import numpy as np

from tmswarp.surface import closest_point_on_triangles, vertex_normals


def _brute_force_closest(p, a, b, c, n=400):
    """Closest point by dense sampling of barycentric coordinates."""
    u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    keep = u + v <= 1.0
    u, v = u[keep], v[keep]
    pts = (1 - u - v)[:, None] * a + u[:, None] * b + v[:, None] * c
    return pts[np.argmin(np.linalg.norm(pts - p, axis=1))]


class TestClosestPointOnTriangles:
    def test_matches_brute_force(self):
        """Random triangles and points, covering face, edge and vertex regions."""
        rng = np.random.default_rng(0)
        a, b, c = rng.normal(size=(3, 60, 3))
        for p in rng.normal(scale=2.0, size=(5, 3)):
            points, bary = closest_point_on_triangles(p, a, b, c)
            for i in range(len(a)):
                ref = _brute_force_closest(p, a[i], b[i], c[i])
                d = np.linalg.norm(points[i] - p)
                d_ref = np.linalg.norm(ref - p)
                # The analytic answer can only be closer than the sampled one
                assert d <= d_ref + 1e-9
                assert d_ref - d < 1e-2

    def test_barycentric_coordinates_valid(self):
        rng = np.random.default_rng(1)
        a, b, c = rng.normal(size=(3, 200, 3))
        points, bary = closest_point_on_triangles(rng.normal(size=3), a, b, c)
        assert np.all(bary >= -1e-12)
        assert np.allclose(bary.sum(axis=1), 1.0)
        rebuilt = bary[:, 0:1] * a + bary[:, 1:2] * b + bary[:, 2:3] * c
        assert np.allclose(points, rebuilt)

    def test_point_above_face_projects_orthogonally(self):
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[1.0, 0.0, 0.0]])
        c = np.array([[0.0, 1.0, 0.0]])
        points, _ = closest_point_on_triangles([0.2, 0.3, 5.0], a, b, c)
        assert np.allclose(points[0], [0.2, 0.3, 0.0])

    def test_continuous_across_shared_edge(self):
        """Moving the query across a convex edge moves the result continuously.

        (Across a concave fold the closest point can jump; the scalp seen
        from outside is convex almost everywhere.)
        """
        a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        b = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, -0.2]])
        c = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        previous = None
        for s in np.linspace(0.3, 0.7, 81):
            p = np.array([s, s, 0.5])
            points, _ = closest_point_on_triangles(p, a, b, c)
            q = points[np.argmin(np.linalg.norm(points - p, axis=1))]
            if previous is not None:
                assert np.linalg.norm(q - previous) < 0.02
            previous = q


class TestVertexNormals:
    def test_flat_patch(self):
        faces = np.array([[0, 1, 2], [1, 3, 2]])
        face_normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        normals = vertex_normals(4, faces, face_normals, np.array([0.5, 0.5]))
        assert np.allclose(normals, [[0.0, 0.0, 1.0]] * 4)

    def test_area_weighting(self):
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        face_normals = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        normals = vertex_normals(4, faces, face_normals, np.array([3.0, 1.0]))
        expected = np.array([1.0, 0.0, 3.0]) / np.sqrt(10.0)
        assert np.allclose(normals[0], expected)
        assert np.allclose(normals[1], [0.0, 0.0, 1.0])
