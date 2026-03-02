"""TMS coil primary field computation via Biot-Savart law."""

import numpy as np


def magnetic_dipole_dadt(
    dipole_pos: np.ndarray,
    dipole_moment: np.ndarray,
    didt: float,
    positions: np.ndarray,
) -> np.ndarray:
    """Compute dA/dt from a magnetic dipole source via Biot-Savart.

    For a magnetic dipole with moment m at position r0, the time derivative
    of the vector potential at position r is:

        dA/dt = (mu0/4pi) * dI/dt * (m x (r - r0)) / |r - r0|^3

    Parameters
    ----------
    dipole_pos : (3,) array
        Dipole location in meters.
    dipole_moment : (3,) array
        Unit magnetic moment direction.
    didt : float
        Rate of current change dI/dt in A/s.
    positions : (n, 3) array
        Evaluation points in meters.

    Returns
    -------
    (n, 3) array
        dA/dt in V/m at each evaluation point.
    """
    mu0_4pi = 1e-7
    r = positions - dipole_pos
    r_norm = np.linalg.norm(r, axis=1)[:, None]
    return mu0_4pi * didt * np.cross(dipole_moment, r) / r_norm**3
