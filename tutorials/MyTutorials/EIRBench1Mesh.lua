-- Check num_procs
num_procs = 4
if check_num_procs == nil and number_of_processes ~= num_procs then
  Log(
    LOG_0ERROR,
    "Incorrect amount of processors. "
      .. "Expected "
      .. tostring(num_procs)
      .. ". Pass check_num_procs=false to override if possible."
  )
  os.exit(false)
end

-- Setup the mesh
nodes = {}
N = 89
L = 8.9
xmin = 0
dx = L / N
for i = 1, (N + 1) do
  k = i - 1
  nodes[i] = xmin + k * dx
end

meshgen1 = mesh.OrthogonalMeshGenerator.Create({ node_sets = { nodes, nodes } })
mesh.MeshGenerator.Execute(meshgen1)

-- Set Material IDs
vol0 = logvol.RPPLogicalVolume.Create({ xmin = 0., xmax = 8.9, ymin = 0., ymax = 8.9, infz = true })
mesh.SetMaterialIDFromLogicalVolume(vol0, 0)
vol1 = logvol.RPPLogicalVolume.Create({ xmin = 1., xmax = 7.4, ymin = 1., ymax = 7.4, infz = true })
mesh.SetMaterialIDFromLogicalVolume(vol1, 1)

-- Add Materials
materials = {}
materials[1] = mat.AddMaterial("Water")
materials[2] = mat.AddMaterial("Fuel")
mat.SetProperty(materials[1], TRANSPORT_XSECTIONS, OPENSN_XSFILE, 'water.mgxs')
mat.SetProperty(materials[2], TRANSPORT_XSECTIONS, OPENSN_XSFILE, 'fuel.mgxs')

-- Set up physics
num_g, num_m = 2, 1
pquad = aquad.CreateProductQuadrature(GAUSS_LEGENDRE_CHEBYSHEV, 4, 16)
aquad.OptimizeForPolarSymmetry(pquad, 4.0 * math.pi)

lbs_block = {
  num_groups = num_g,
  groupsets = {
    {
      groups_from_to = { 0, num_groups - 1 },
      angular_quadrature_handle = pquad,
      inner_linear_method = "gmres",
      l_max_its = 50,
      gmres_restart_interval = 50,
      l_abs_tol = 1.0e-10,
      groupset_num_subsets = num_g,
    },
  },
}

lbs_options = {
  boundary_conditions = {
    { name = "xmin", type = "reflecting" },
    { name = "ymin", type = "reflecting" },
    { name = "xmax", type = "reflecting" },
    { name = "ymax", type = "reflecting" }
  },
  scattering_order = num_m,

  use_precursors = false,

  verbose_inner_iterations = false,
  verbose_outer_iterations = true,
}

phys1 = lbs.DiscreteOrdinatesSolver.Create(lbs_block)
lbs.SetOptions(phys1, lbs_options)

k_solver0 = lbs.NonLinearKEigen.Create({ lbs_solver_handle = phys1 })
solver.Initialize(k_solver0)
solver.Execute(k_solver0)

fflist, count = lbs.GetScalarFieldFunctionList(phys1)

-- Exporting the mesh
-- mesh.ExportToPVTU("EIRBenchmark1")