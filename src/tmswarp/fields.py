"""E-field post-processing and validation metrics."""

import numpy as np


def compute_efield_at_elements(mesh, phi: np.ndarray, dAdt_nodes: np.ndarray,
                               G=None) -> np.ndarray:
    """Compute total E-field at element barycenters.

    E = -grad(phi) - dA/dt

    For P1 elements, grad(phi) is constant within each element:
        grad(phi)|_e = sum_i phi_i * G[e, i, :]

    dA/dt at the barycenter is the average of the 4 nodal values.

    Parameters
    ----------
    mesh : TetMesh
    phi : (n_nodes,) array
        Scalar potential from FEM solve.
    dAdt_nodes : (n_nodes, 3) array
        Primary field at nodes.
    G : (n_elements, 4, 3) array, optional
        Precomputed gradient operator.

    Returns
    -------
    (n_elements, 3) array
        Total E-field at element barycenters.
    """
    if G is None:
        from tmswarp.solver import gradient_operator
        G = gradient_operator(mesh)

    # grad(phi) per element
    phi_elem = phi[mesh.elements]  # (n_elem, 4)
    grad_phi = np.einsum('ei,eid->ed', phi_elem, G)  # (n_elem, 3)

    # dA/dt at barycenters
    dAdt_bary = dAdt_nodes[mesh.elements].mean(axis=1)  # (n_elem, 3)

    return -grad_phi - dAdt_bary


def rdm(a: np.ndarray, b: np.ndarray) -> float:
    """Relative Difference Measure between two vector fields.

    RDM = ||a/||a|| - b/||b||||

    A value of 0 means perfect directional agreement.
    """
    return float(np.linalg.norm(
        a / np.linalg.norm(a) - b / np.linalg.norm(b)
    ))


def mag(a: np.ndarray, b: np.ndarray) -> float:
    """Log magnitude ratio between two vector fields.

    MAG = log(||a|| / ||b||)

    A value of 0 means perfect magnitude agreement.
    """
    return float(np.log(np.linalg.norm(a) / np.linalg.norm(b)))
