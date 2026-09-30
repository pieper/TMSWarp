"""Tests for uniform mesh refinement."""

from pathlib import Path

import numpy as np
import pytest

from tmswarp.analytical import tms_analytical_efield
from tmswarp.coil import magnetic_dipole_dadt
from tmswarp.conductor import (TetMesh, element_barycenters, element_volumes,
                               make_sphere_mesh)
from tmswarp.fields import compute_efield_at_elements, rdm
from tmswarp.refine import refine_uniform, restrict_to_parents
from tmswarp.solver import (assemble_rhs_tms, assemble_stiffness,
                            gradient_operator, solve_fem)

SPHERE3 = Path(__file__).resolve().parents[1] / "sphere3_data.npz"


@pytest.fixture(scope="module")
def coarse():
    return make_sphere_mesh(radius=0.095, n_shells=3, n_surface=80, conductivity=1.0)


class TestRefineUniform:
    def test_counts(self, coarse):
        refined, parent = refine_uniform(coarse, levels=2)
        assert len(refined.elements) == 64 * len(coarse.elements)
        assert len(parent) == len(refined.elements)
        assert np.array_equal(np.bincount(parent), np.full(len(coarse.elements), 64))

    def test_volume_is_preserved_per_parent(self, coarse):
        refined, parent = refine_uniform(coarse, levels=1)
        child_volume = np.bincount(
            parent, weights=element_volumes(refined), minlength=len(coarse.elements)
        )
        assert np.allclose(child_volume, element_volumes(coarse), rtol=1e-10)

    def test_mesh_is_conforming(self, coarse):
        """Every interior face must be shared by exactly two elements."""
        refined, _ = refine_uniform(coarse, levels=1)
        face_idx = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        faces = np.sort(refined.elements[:, face_idx].reshape(-1, 3), axis=1)
        _, counts = np.unique(faces, axis=0, return_counts=True)
        assert set(np.unique(counts)) <= {1, 2}
        # The boundary of the refined mesh has 4 faces per coarse boundary face
        coarse_faces = np.sort(coarse.elements[:, face_idx].reshape(-1, 3), axis=1)
        _, coarse_counts = np.unique(coarse_faces, axis=0, return_counts=True)
        assert (counts == 1).sum() == 4 * (coarse_counts == 1).sum()

    def test_orientation_follows_parent(self, coarse):
        refined, parent = refine_uniform(coarse, levels=1)

        def sign(mesh):
            c = mesh.nodes[mesh.elements]
            return np.sign(np.linalg.det(c[:, 1:] - c[:, 0:1]))

        assert np.array_equal(sign(refined), sign(coarse)[parent])

    def test_conductivity_is_inherited(self, coarse):
        sigma = np.linspace(0.1, 1.0, len(coarse.elements))
        mesh = TetMesh(coarse.nodes, coarse.elements, sigma)
        refined, parent = refine_uniform(mesh, levels=1)
        assert np.array_equal(refined.conductivity, sigma[parent])

    def test_sphere_projection(self, coarse):
        """With projection, all boundary nodes stay on the sphere."""
        refined, _ = refine_uniform(coarse, levels=1, sphere_center=np.zeros(3))
        face_idx = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        faces = np.sort(refined.elements[:, face_idx].reshape(-1, 3), axis=1)
        unique, counts = np.unique(faces, axis=0, return_counts=True)
        boundary_nodes = np.unique(unique[counts == 1])
        radius = np.linalg.norm(refined.nodes[boundary_nodes], axis=1)
        coarse_radius = np.linalg.norm(coarse.nodes, axis=1).max()
        assert np.allclose(radius, coarse_radius, rtol=1e-6)
        assert np.all(element_volumes(refined) > 0)


class TestRestrictToParents:
    def test_constant_field(self, coarse):
        refined, parent = refine_uniform(coarse, levels=1)
        values = np.tile([1.0, -2.0, 3.0], (len(refined.elements), 1))
        out = restrict_to_parents(
            values, element_volumes(refined), parent, len(coarse.elements)
        )
        assert np.allclose(out, [1.0, -2.0, 3.0])


@pytest.mark.skipif(not SPHERE3.exists(), reason="sphere3_data.npz not found")
def test_refinement_reduces_error_against_analytical():
    data = np.load(SPHERE3)
    mesh = TetMesh(
        nodes=data["nodes"].astype(np.float64),
        elements=data["elements"].astype(np.int32),
        conductivity=data["conductivity"].astype(np.float64),
    )
    pos = np.array([0.0, 0.0, 0.3])
    moment = np.array([1.0, 0.0, 0.0])

    def error(m):
        dAdt = magnetic_dipole_dadt(pos, moment, 1e6, m.nodes)
        G = gradient_operator(m)
        phi = solve_fem(assemble_stiffness(m, G), assemble_rhs_tms(m, dAdt, G))
        E = compute_efield_at_elements(m, phi, dAdt, G)
        E_ana = tms_analytical_efield(pos, moment, 1e6, element_barycenters(m))
        return rdm(E, E_ana)

    refined, _ = refine_uniform(mesh, levels=1, sphere_center=np.zeros(3))
    e0, e1 = error(mesh), error(refined)
    print(f"RDM vs analytical: {e0:.4f} -> {e1:.4f}")
    assert e1 < 0.7 * e0
