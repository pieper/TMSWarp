"""Pure-numpy FEM solver for the TMS forward problem.

Solves the quasistatic Poisson equation:

    div(sigma * grad(phi)) = -div(sigma * dA/dt)

using P1 (linear) tetrahedral finite elements.

The weak form (after integration by parts, with natural Neumann BC):

    integral sigma grad(phi) . grad(v) dV = -integral sigma dA/dt . grad(v) dV

which yields the linear system K @ phi = b.
"""

import numpy as np
import scipy.sparse as sp


def gradient_operator(mesh) -> np.ndarray:
    """Compute gradients of P1 basis functions for each tetrahedron.

    For a linear tetrahedron with vertices x0, x1, x2, x3:
        T = [x1-x0, x2-x0, x3-x0]^T   (3x3 Jacobian)
        Reference gradients (in reference tet coordinates):
            phi_0: [-1, -1, -1]
            phi_1: [ 1,  0,  0]
            phi_2: [ 0,  1,  0]
            phi_3: [ 0,  0,  1]
        Physical gradients: G = T^{-T} @ ref_grads

    Parameters
    ----------
    mesh : TetMesh

    Returns
    -------
    (n_elements, 4, 3) array
        G[e, i, :] = gradient of basis function i in element e.
    """
    coords = mesh.nodes[mesh.elements]  # (n_elem, 4, 3)
    # Jacobian: T[e] = [x1-x0, x2-x0, x3-x0] as rows
    T = coords[:, 1:4] - coords[:, 0:1]  # (n_elem, 3, 3)

    # Reference gradient matrix A (3x4): columns are ref grads of phi_0..phi_3
    # phi_0 ref grad = [-1,-1,-1], phi_1 = [1,0,0], phi_2 = [0,1,0], phi_3 = [0,0,1]
    # A = [[-1, 1, 0, 0],
    #      [-1, 0, 1, 0],
    #      [-1, 0, 0, 1]]
    A = np.array([[-1, 1, 0, 0],
                  [-1, 0, 1, 0],
                  [-1, 0, 0, 1]], dtype=np.float64)

    # Solve T @ X = A for each element => X = T^{-1} @ A
    # X has shape (n_elem, 3, 4)
    # Then physical gradient of phi_i = X[:, :, i], i.e. column i of X
    # So G[e, i, :] = X[e, :, i] => G = X.transpose(0, 2, 1)
    X = np.linalg.solve(T, np.broadcast_to(A, (len(T), 3, 4)))
    G = X.transpose(0, 2, 1)  # (n_elem, 4, 3)
    return G


def assemble_stiffness(mesh, G=None):
    """Assemble the global stiffness matrix.

    K_ij = sum_e sigma_e * V_e * (grad_phi_i . grad_phi_j)

    Parameters
    ----------
    mesh : TetMesh
    G : (n_elements, 4, 3) array, optional
        Precomputed gradient operator.

    Returns
    -------
    scipy.sparse.csc_matrix of shape (n_nodes, n_nodes)
    """
    from tmswarp.conductor import element_volumes

    if G is None:
        G = gradient_operator(mesh)

    vols = element_volumes(mesh)
    sigma = mesh.conductivity
    n_nodes = len(mesh.nodes)

    # Weight: w_e = sigma_e * V_e
    w = vols * sigma  # (n_elem,)

    # Build COO data for all 4x4 local entries
    rows_list = []
    cols_list = []
    data_list = []

    for i in range(4):
        for j in range(4):
            # K_ij contribution from each element
            Ke = w * np.sum(G[:, i, :] * G[:, j, :], axis=1)
            rows_list.append(mesh.elements[:, i])
            cols_list.append(mesh.elements[:, j])
            data_list.append(Ke)

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    data = np.concatenate(data_list)

    K = sp.csc_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    K.eliminate_zeros()
    return K


def assemble_rhs_tms(mesh, dAdt_nodes: np.ndarray, G=None) -> np.ndarray:
    """Assemble the right-hand side for the TMS FEM problem.

    From the weak form:
        b_i = -integral sigma dA/dt . grad(phi_i) dV

    For P1 elements with dA/dt linearly interpolated from nodes:
        b_i = -sum_e sigma_e * V_e * (mean dAdt_e) . grad(phi_i)_e

    where mean dAdt_e = average of dA/dt at the 4 element nodes.

    Parameters
    ----------
    mesh : TetMesh
    dAdt_nodes : (n_nodes, 3) array
        Primary field dA/dt at mesh nodes.
    G : (n_elements, 4, 3) array, optional
        Precomputed gradient operator.

    Returns
    -------
    (n_nodes,) array
    """
    from tmswarp.conductor import element_volumes

    if G is None:
        G = gradient_operator(mesh)

    vols = element_volumes(mesh)
    sigma = mesh.conductivity

    # Average dA/dt per element
    dAdt_elem = dAdt_nodes[mesh.elements].mean(axis=1)  # (n_elem, 3)

    # Weighted: w_e * dAdt_e
    w_dAdt = (vols * sigma)[:, None] * dAdt_elem  # (n_elem, 3)

    # For each local node i: contribution = -w_dAdt . G[:, i, :]
    contributions = np.zeros((len(mesh.elements), 4), dtype=np.float64)
    for i in range(4):
        contributions[:, i] = -np.sum(w_dAdt * G[:, i, :], axis=1)

    # Assemble into global vector
    b = np.bincount(
        mesh.elements.ravel(),
        weights=contributions.ravel(),
        minlength=len(mesh.nodes),
    )
    return b


def solve_fem(K, b: np.ndarray, pin_node: int = 0) -> np.ndarray:
    """Solve K @ phi = b with a gauge condition (pin one node to zero).

    The pure Neumann system is singular (constant functions are in the null
    space). Pinning one node to phi=0 makes it non-singular. This does not
    affect the E-field since E = -grad(phi) - dA/dt, and adding a constant
    to phi does not change grad(phi).

    Parameters
    ----------
    K : sparse matrix, (n_nodes, n_nodes)
    b : (n_nodes,) array
    pin_node : int
        Node index to pin to zero potential.

    Returns
    -------
    (n_nodes,) array
        Scalar potential phi at each node.
    """
    from scipy.sparse.linalg import spsolve

    n = K.shape[0]
    keep = np.ones(n, dtype=bool)
    keep[pin_node] = False

    K_csr = K.tocsr()
    K_reduced = K_csr[keep][:, keep]
    b_reduced = b[keep]

    phi_reduced = spsolve(K_reduced.tocsc(), b_reduced)

    phi = np.zeros(n, dtype=np.float64)
    phi[keep] = phi_reduced
    return phi
