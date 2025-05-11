import os
import sys
import numpy as np

"""
HEU-MET-FAST-001
Utilizes very coarse mesh for the sake of a test
k_eff = 1.017786
"""

if "opensn_console" not in globals():
    from mpi4py import MPI
    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.mesh import FromFileMeshGenerator
    from pyopensn.xs import MultiGroupXS
    from pyopensn.aquad import GLCProductQuadrature3DXYZ
    from pyopensn.solver import DiscreteOrdinatesProblem, PowerIterationKEigenSolver

if __name__ == "__main__":
    num_procs = 4
    if size != num_procs:
        sys.exit(f"Incorrect number of processors. Expected {num_procs} but got {size}.")

    # Setup mesh
    meshgen = FromFileMeshGenerator(
        filename='../../../../assets/mesh/Godiva_SN',
    )

    grid=meshgen.Execute()

    # Need to scale volume(s) to real volume(s).
    vol_per_blk = grid.ComputeVolumePerBlockID()

    # radius of real system in m:
    radius = {0: 0.087407}

    block_ids = sorted(vol_per_blk.keys())

    # Check for empty entries:
    missing = [blk for blk in block_ids if blk not in radius]
    if missing:
        raise KeyError(f"No radius provided for block ID: {missing}")

    # compute real volume(s)
    exact_vols_per_blk = {}
    prev_R = 0.0
    for blk in block_ids:
        R = radius[blk]
        exact_vols_per_blk[blk] = 4.0 * np.pi / 3.0 * (R**3 - prev_R**3)
        prev_R = R

    # Build scaling factor(s)
    ratios = np.array([
        exact_vols_per_blk[blk]/vol_per_blk[blk]
    ])

    # Define MGXS
    num_groups=23
    xs_uranium = MultiGroupXS()
    xs_uranium.LoadFromOpenSn("xs_godiva_g23.xs")
    xs_uranium.SetScalingFactor(ratios[0])

    # Setup Physics for Solver
    pquad = GLCProductQuadrature3DXYZ(8, 16)

    phys = DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=num_groups,
        groupsets=[
            {
                "groups_from_to" : [0, 22],
                "angular_quadrature": pquad,
                "angle_aggregation_type": "single",
                "inner_linear_method": "petsc_richardson",
                "l_abs_tol": 1.0e-6,
                "l_max_its": 300,
            },
        ],
        xs_map =[
            {"block_ids":[0], "xs": xs_uranium},
        ],
        options={
            "scattering_order": 3,
            "use_precursors": False,
            "verbose_inner_iterations": False,
            "verbose_outer_iterations": True,

        }
    )
    k_solver = PowerIterationKEigenSolver(
        lbs_problem=phys, 
        k_tol=1.0e-6
    )
    k_solver.Initialize()
    k_solver.Execute()

    # Export vtu
    # fflist = phys.GetScalarFieldFunctionList(only_scalar_flux=True)
    # vtk_basename = "HEU_MET_FAST_001"
    # FieldFunctionGridBased.ExportMultipleToVTK(
    #     [fflist[g] for g in range(num_groups)],
    #     vtk_basename
    # )


