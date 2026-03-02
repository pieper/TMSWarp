"""Analytical TMS E-field for spherical conductors (Heller & van Hulsteyn 1992).

Uses the Sarvas (1987) auxiliary quantities F and grad_F via the reciprocity
theorem connecting TMS to MEG forward modeling.

Key properties of this solution:
- E-field is independent of the radial conductivity profile
- The radial component of E is identically zero (purely tangential)
- Works for any number of concentric spherical shells

References
----------
Heller, L. & van Hulsteyn, D.B. (1992). Brain stimulation using
electromagnetic sources: theoretical aspects. Biophysical Journal, 63(1),
129-138.

Sarvas, J. (1987). Basic mathematical and electromagnetic concepts of the
biomagnetic inverse problem. Physics in Medicine and Biology, 32(1), 11-22.
"""

import numpy as np


def tms_analytical_efield(
    dipole_pos: np.ndarray,
    dipole_moment: np.ndarray,
    didt: float,
    positions: np.ndarray,
) -> np.ndarray:
    """Compute the analytical TMS E-field inside a spherical conductor.

    Parameters
    ----------
    dipole_pos : (3,) or (m, 3) array
        Magnetic dipole position(s) in meters, outside the sphere.
    dipole_moment : (3,) or (m, 3) array
        Unit magnetic moment direction(s).
    didt : float
        Rate of current change dI/dt in A/s.
    positions : (n, 3) array
        Evaluation points in meters, inside the sphere.

    Returns
    -------
    (n, 3) array
        E-field in V/m at each evaluation point.
    """
    mu0_4pi = 1e-7

    dp = np.atleast_2d(dipole_pos)
    dm = np.atleast_2d(dipole_moment)
    r1 = np.asarray(positions, dtype=np.float64)

    E = np.zeros_like(r1, dtype=np.float64)

    for m, r2 in zip(dm, dp):
        a = r2 - r1  # (n, 3)
        norm_a = np.linalg.norm(a, axis=1, keepdims=True)  # (n, 1)
        norm_r1 = np.linalg.norm(r1, axis=1, keepdims=True)  # (n, 1)
        norm_r2 = np.linalg.norm(r2)  # scalar

        r2_dot_a = np.sum(r2 * a, axis=1, keepdims=True)  # (n, 1)

        # Sarvas F scalar
        F = norm_a * (norm_r2 * norm_a + r2_dot_a)  # (n, 1)

        # Sarvas grad_F vector
        grad_F = (
            (norm_a**2 / norm_r2 + 2.0 * norm_a + 2.0 * norm_r2
             + r2_dot_a / norm_a) * r2
            - (norm_a + 2.0 * norm_r2 + r2_dot_a / norm_a) * r1
        )  # (n, 3)

        m_dot_grad_F = np.sum(m * grad_F, axis=1, keepdims=True)  # (n, 1)

        E += (
            -didt * mu0_4pi / F**2
            * (F * np.cross(r1, m) - np.cross(m_dot_grad_F * r1, r2))
        )

    return E
