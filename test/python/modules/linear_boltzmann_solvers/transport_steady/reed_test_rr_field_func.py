#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard Reed 1D 1-group problem testing the reaction rate density field functions.

import os
import sys
import numpy as np

if "opensn_console" not in globals():
    from mpi4py import MPI
    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.mesh import OrthogonalMeshGenerator
    from pyopensn.xs import MultiGroupXS
    from pyopensn.source import VolumetricSource
    from pyopensn.aquad import GLProductQuadrature1DSlab
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSourceSolver
    from pyopensn.logvol import RPPLogicalVolume
    from pyopensn.fieldfuncinterp import FieldFunctionInterpolationVolume

if __name__ == "__main__":

    # Create Mesh
    widths = [2., 1., 2., 1., 2.]
    nrefs = [200, 200, 200, 200, 200]
    Nmat = len(widths)
    nodes = [0.]
    for imat in range(Nmat):
        dx = widths[imat] / nrefs[imat]
        for i in range(nrefs[imat]):
            nodes.append(nodes[-1] + dx)
    meshgen = OrthogonalMeshGenerator(node_sets=[nodes])
    grid = meshgen.Execute()

    # Set block IDs
    z_min = 0.0
    for imat in range(Nmat):
        z_max = z_min + widths[imat]
        if rank == 0:
            print("imat=", imat, ", zmin=", z_min, ", zmax=", z_max)
        lv = RPPLogicalVolume(infx=True, infy=True, zmin=z_min, zmax=z_max)
        grid.SetBlockIDFromLogicalVolume(lv, imat, True)
        z_min = z_max

    # Add cross sections to materials
    sigt = [50., 5., 0., 1., 1.]
    c = [0., 0., 0., 0.9, 0.9]
    xs_map = len(sigt) * [None]
    for imat in range(Nmat):
        xs_ = MultiGroupXS()
        xs_.CreateSimpleOneGroup(sigt[imat], c[imat])
        xs_map[imat] = {
            "block_ids": [imat],
            "xs": xs_,
        }

    # Create sources in 1st and 4th materials
    src0 = VolumetricSource(block_ids=[0], group_strength=[50.])
    src1 = VolumetricSource(block_ids=[3], group_strength=[1.])

    # Angular Quadrature
    gl_quad = GLProductQuadrature1DSlab(n_polar=128, scattering_order=0)

    # Whole-domain logical volume
    logvol_whole_domain = RPPLogicalVolume(
        infx=True,
        infy=True,
        zmin=0.0,
        zmax=sum(widths),
    )

    # LBS problem
    num_groups = 1
    phys = DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=num_groups,
        groupsets=[
            {
                "groups_from_to": (0, num_groups - 1),
                "angular_quadrature": gl_quad,
                "inner_linear_method": "petsc_gmres",
                "l_abs_tol": 1.0e-9,
                "l_max_its": 300,
                "gmres_restart_interval": 30,
            },
        ],
        xs_map=xs_map,
        volumetric_sources=[src0, src1],
        boundary_conditions=[
            {"name": "zmin", "type": "reflecting"},
            {"name": "zmax", "type": "reflecting"},
        ],
        options={
            "reaction_rate_density_field_functions": [
                {
                    "reaction": "total",
                    "total": True,
                },
                {
                    "reaction": "absorption",
                    "total": True,
                }

            ]
        },
    )

    # Initialize and execute solver
    ss_solver = SteadyStateSourceSolver(problem=phys)
    ss_solver.Initialize()
    ss_solver.Execute()

    # Get RR field functions
    rr_ffs = phys.GetReactionRateDensityFieldFunctions()

    # Get scalar flux FF
    flux_ffs = phys.GetScalarFluxFieldFunction(True)
    phi_ff = flux_ffs[0]

    # Integrate RR over whole domain
    rr_vals = np.zeros(len(rr_ffs))
    for i, ff in enumerate(rr_ffs):
        ffi = FieldFunctionInterpolationVolume()
        ffi.SetOperationType("sum")
        ffi.SetLogicalVolume(logvol_whole_domain)
        ffi.AddFieldFunction(ff)
        ffi.Initialize()
        ffi.Execute()
        rr_vals[i] = ffi.GetValue()

    manual_total = 0.0
    manual_abs = 0.0

    for mat_id in range(Nmat):
        zmin = float(sum(widths[:mat_id]))
        zmax = float(zmin + widths[mat_id])

        lv = RPPLogicalVolume(infx=True, infy=True, zmin=zmin, zmax=zmax)

        ffi = FieldFunctionInterpolationVolume()
        ffi.SetOperationType("sum")
        ffi.SetLogicalVolume(lv)
        ffi.AddFieldFunction(phi_ff)
        ffi.Initialize()
        ffi.Execute()

        phi_integral = ffi.GetValue()
        manual_total += sigt[mat_id] * phi_integral
        manual_abs += sigt[mat_id] * (1-c[mat_id]) * phi_integral
    

    if rank == 0:
        rel_err_total = abs((rr_vals[0] - manual_total) / manual_total)
        rel_err_abs = abs((rr_vals[1] - manual_abs) / manual_abs)
        print("rr_total=", rr_vals[0])
        print("manual_total=", manual_total)
        print("Relative-error-total=", rel_err_total)
        print("rr_abs=", rr_vals[1])
        print("manual_abs=", manual_abs)
        print("Relative-error-abs=", rel_err_abs)