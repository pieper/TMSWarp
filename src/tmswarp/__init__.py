"""TMSWarp: TMS simulation utilities using Nvidia Warp."""

__version__ = "0.1.0"

from tmswarp.analytical import tms_analytical_efield
from tmswarp.coil import magnetic_dipole_dadt
from tmswarp.conductor import TetMesh, make_sphere_mesh
from tmswarp.fields import compute_efield_at_elements, mag, rdm
from tmswarp.solver import assemble_rhs_tms, assemble_stiffness, solve_fem
from tmswarp.solver_warp import solve_fem_warp, warp_available
