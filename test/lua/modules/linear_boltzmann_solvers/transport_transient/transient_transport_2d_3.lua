-- 1D Transient Transport test with Vacuum BC.
-- SDM: PWLD
-- Test:
num_procs = 2

-- Check num_procs
if check_num_procs == nil and number_of_processes ~= num_procs then
  log.Log(
    LOG_0ERROR,
    "Incorrect amount of processors. "
      .. "Expected "
      .. tostring(num_procs)
      .. ". Pass check_num_procs=false to override if possible."
  )
  os.exit(false)
end

-- Setup mesh
nodes = {}
N = 160
L = 80.96897163
xmin = -L / 2
dx = L / N
for i = 1, (N + 1) do
  k = i - 1
  nodes[i] = xmin + k * dx
end

meshgen1 = mesh.OrthogonalMeshGenerator.Create({ node_sets = { nodes, nodes } })
grid = meshgen1:Execute()

-- Set block IDs
grid:SetUniformBlockID(0)

vol0 = logvol.RPPLogicalVolume.Create({
  xmin = -L / 16,
  xmax = L / 16,
  ymin = -L / 16,
  ymax = L / 16,
  zmin = -L / 16,
  zmax = L / 16,
})
grid:SetBlockIDFromLogicalVolume(vol0, 1, true)

mesh.ExportToPVTU("TheMesh")

-- Add materials
materials = {}
materials[1] = mat.AddMaterial("Strong fuel")
materials[2] = mat.AddMaterial("Weak fuel")

-- Define microscopic cross sections
xs_strong_fuel_micro = xs.Create()
xs.Set(xs_strong_fuel_micro, OPENSN_XSFILE, "tests/transport_transient/xs_inf_k1_6_1g.xs")
xs_weak_fuelA_micro = xs.Create()
xs.Set(xs_weak_fuelA_micro, OPENSN_XSFILE, "tests/transport_transient/xs_inf_critical_1g.xs")
xs_weak_fuelB_micro = xs.Create()
xs.Set(xs_weak_fuelB_micro, OPENSN_XSFILE, "tests/transport_transient/xs_inf_weak2_1g.xs")

atom_density = 0.056559
xs_strong_fuel = xs.MakeScaled(xs_strong_fuel_micro, atom_density) --critical
xs_weak_fuelA = xs.MakeScaled(xs_weak_fuelA_micro, atom_density) --critical
xs_weak_fuelB = xs.MakeScaled(xs_weak_fuelB_micro, atom_density) --critical

num_groups = 1
mat.SetProperty(materials[1], TRANSPORT_XSECTIONS, EXISTING, xs_strong_fuel)
mat.SetProperty(materials[2], TRANSPORT_XSECTIONS, EXISTING, xs_weak_fuelA)

mat.SetProperty(materials[1], ISOTROPIC_MG_SOURCE, FROM_ARRAY, { 0.0 })
mat.SetProperty(materials[2], ISOTROPIC_MG_SOURCE, FROM_ARRAY, { 0.0 })

function SwapXS(solver_handle, new_xs)
  mat.SetProperty(materials[2], TRANSPORT_XSECTIONS, EXISTING, new_xs)
  lbs.InitializeMaterials(solver_handle)
end

-- Setup Physics
phys1 = LBSCreateTransientSolver()

--========== Groups
grp = {}
for g = 1, num_groups do
  grp[g] = LBSCreateGroup(phys1)
end

--========== ProdQuad
fac = 3
pquad = aquad.CreateProductQuadrature(GAUSS_LEGENDRE_CHEBYSHEV, 2 * fac, 2 * fac)
aquad.OptimizeForPolarSymmetry(pquad, 4.0 * math.pi)

--========== Groupset def
gs0 = LBSCreateGroupset(phys1)
cur_gs = gs0
LBSGroupsetAddGroups(phys1, cur_gs, 0, num_groups - 1)
LBSGroupsetSetQuadrature(phys1, cur_gs, pquad)
LBSGroupsetSetAngleAggDiv(phys1, cur_gs, 1)
LBSGroupsetSetGroupSubsets(phys1, cur_gs, 1)
LBSGroupsetSetIterativeMethod(phys1, cur_gs, KRYLOV_GMRES_CYCLES)
LBSGroupsetSetResidualTolerance(phys1, cur_gs, 1.0e-6)
LBSGroupsetSetMaxIterations(phys1, cur_gs, 1000)
LBSGroupsetSetGMRESRestartIntvl(phys1, cur_gs, 100)
--LBSGroupsetSetWGDSA(phys1,cur_gs,30,1.0e-4,false," ")
--LBSGroupsetSetTGDSA(phys1,cur_gs,30,1.0e-4,false," ")

--
---- Set boundary conditions
--bsrc={}
--for g=1,num_groups do
--    bsrc[g] = 0.0
--end
--bsrc[1] = 1.0/2
--LBSSetProperty(phys1,BOUNDARY_CONDITION,ZMIN,LBSBoundaryTypes.REFLECTING);
--LBSSetProperty(phys1,BOUNDARY_CONDITION,ZMAX,LBSBoundaryTypes.REFLECTING);
--
LBSSetProperty(phys1, DISCRETIZATION_METHOD, PWLD)
LBSSetProperty(phys1, SCATTERING_ORDER, 0)

LBKESSetProperty(phys1, "MAX_ITERATIONS", 1000)
LBKESSetProperty(phys1, "TOLERANCE", 1.0e-8)

LBSSetProperty(phys1, USE_PRECURSORS, true)

--LBSSetProperty(phys1, VERBOSE_INNER_ITERATIONS, false)
LBSSetProperty(phys1, VERBOSE_INNER_ITERATIONS, false)
LBSSetProperty(phys1, VERBOSE_OUTER_ITERATIONS, true)

-- Initialize and Execute Solver
solver.Initialize(phys1)

LBTSSetProperty(phys1, "TIMESTEP", 1e-3)
LBTSSetProperty(phys1, "VERBOSITY_LEVEL", 0)
LBTSSetProperty(phys1, "TIMESTEP_METHOD", "CRANK_NICHOLSON")

phys1name = solver.GetName(phys1)
initial_FR = lbs.ComputeFissionRate(phys1, "OLD")

--time = 0.0
--for k=1,2 do
--    --LBTSSetProperty(phys1, "INHIBIT_ADVANCE", true)
--    solver.Step(phys1)
--    FRf = lbs.ComputeFissionRate(phys1,"NEW")
--    FRi = lbs.ComputeFissionRate(phys1,"OLD")
--    dt = LBTSGetProperty(phys1, "TIMESTEP")
--    time = LBTSGetProperty(phys1, "TIME")
--    period = dt/math.log(FRf/FRi)
--    log.Log(LOG_0, string.format("%s %4d time=%10.3g dt=%10.4g period=%10.3g FR=%10.3e",
--            phys1name,k,time,dt,period,FRf/initial_FR))
--end

time = 0.0
time_stop = 1.0
k = 0
swapped = false
while time < time_stop do
  k = k + 1
  solver.Step(phys1)
  FRf = lbs.ComputeFissionRate(phys1, "NEW")
  FRi = lbs.ComputeFissionRate(phys1, "OLD")
  dt = LBTSGetProperty(phys1, "TIMESTEP")
  time = LBTSGetProperty(phys1, "TIME")
  period = dt / math.log(FRf / FRi)
  log.Log(
    LOG_0,
    string.format(
      "%s %4d time=%10.3g dt=%10.4g period=%10.3g FR=%10.3e",
      phys1name,
      k,
      time,
      dt,
      period,
      FRf / initial_FR
    )
  )
  if time >= 0.2 and not swapped then
    SwapXS(phys1, xs_weak_fuelB)
    swapped = true
  end

  LBTSAdvanceTimeData(phys1)
end
