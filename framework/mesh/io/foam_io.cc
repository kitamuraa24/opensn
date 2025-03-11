// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT
// #define OPENSN_WITH_OPENFOAM true
#ifdef OPENSN_WITH_OPENFOAM // compile this file
#include "framework/mesh/io/mesh_io.h"
#include "framework/mesh/mesh_continuum/mesh_continuum.h"
#include "framework/runtime.h"
#include "framework/logging/log.h"
#include "framework/utils/utils.h"
#include <fstream>
#include "argList.H"
#include "fvMesh.H"
#include "IOobject.H"
#include "Time.H"
#include "cellShape.H"
#include "cellModel.H"
#include "UPstream.H"
#include "face.H"
#include "cell.H"
#undef Log

namespace Foam
{
// Create an override to prevent OpneFOAM from finalizing MPI.
// When destructor is called for fvMesh, this function is called.
void
UPstream::shutdown(int)
{
  opensn::log.Log0Verbose1() << "Intercepted OpenFOAM shutdown: Preventing MPI_Finalize()";
  return; // Does nothing—prevents OpenFOAM from finalizing MPI
}

} // namespace Foam

namespace opensn
{

namespace
{

std::shared_ptr<UnpartitionedMesh::LightWeightCell>
CreateCellFromOpenFOAMCell(const Foam::fvMesh& foam_mesh, Foam::label cell)
{
  const std::string fname = "CreateCellFromOpenFOAMCell";

  // Retrieve cell shape and determine type
  const Foam::cellShape& c_type = foam_mesh.cellShapes()[cell];
  const Foam::cellModel& cell_model = c_type.model();

  Foam::label model_index = cell_model.index();
  CellType sub_type;
  switch (model_index)
  {
    case Foam::cellModel::TET:
      sub_type = CellType::TETRAHEDRON;
      break;
    case Foam::cellModel::HEX:
      sub_type = CellType::HEXAHEDRON;
      break;
    case Foam::cellModel::PYR:
      sub_type = CellType::PYRAMID;
      break;
    case Foam::cellModel::WEDGE: // Wedge and Prism are the same in OpenFOAM
    case Foam::cellModel::PRISM:
      sub_type = CellType::WEDGE;
      break;
    case Foam::cellModel::UNKNOWN: // Unknown acts as polyhedral assignment in OpenFOAM
      sub_type = CellType::POLYHEDRON;
      break;
    default:
      throw std::logic_error(fname + ": Unsupported 3D cell type.");
  }
  auto polyh_cell =
    std::make_shared<UnpartitionedMesh::LightWeightCell>(CellType::POLYHEDRON, sub_type);

  // Grab current face of current cell.
  // Note: OpenFOAM defines a cell by face idx, not vertex idx.
  const Foam::cell& cell_faces = foam_mesh.cells()[cell];
  auto num_cfaces = cell_faces.size();
  // Grab vertex idx of current cell.
  auto cell_points = foam_mesh.cellShapes()[cell];
  auto num_cpoints = cell_points.nPoints();

  polyh_cell->vertex_ids.reserve(num_cpoints);

  for (int p = 0; p < num_cpoints; ++p)
  {
    uint64_t point_id = cell_points[p];
    polyh_cell->vertex_ids.push_back(point_id);
  }

  // Handle Cell Type-Specific Face Mapping
  switch (sub_type)
  {
    // The cell vertex ids in OpenFOAM are not the same as OpenSn
    // so we must remap the vertices and faces of the cell.
    case CellType::HEXAHEDRON:
    {
      std::vector<uint64_t> remapping = {4, 5, 1, 0, 7, 6, 2, 3};
      std::vector<uint64_t> remapped(8, 0);
      for (int i = 0; i < 8; ++i)
        remapped[i] = polyh_cell->vertex_ids[remapping[i]];
      polyh_cell->vertex_ids = remapped;
      // 2D array of (faces, node idx of face)
      std::vector<std::vector<uint64_t>> face_vids = {
        {1, 2, 6, 5}, {3, 0, 4, 7}, {2, 3, 7, 6}, {0, 1, 5, 4}, {4, 5, 6, 7}, {3, 2, 1, 0}};
      for (int f = 0; f < 6; ++f)
      {
        UnpartitionedMesh::LightWeightFace face;
        face.vertex_ids.reserve(4);
        for (int p = 0; p < 4; ++p)
          face.vertex_ids.push_back(polyh_cell->vertex_ids[face_vids[f][p]]);

        polyh_cell->faces.push_back(face);
      }
      break;
    }
    // Vertex ids do not need to be remapped, but faces do.
    case CellType::TETRAHEDRON:
    {
      // 2D array of (faces, node idx of face)
      std::vector<std::vector<uint64_t>> face_vids = {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {3, 1, 2}};
      for (int f = 0; f < 4; ++f)
      {
        UnpartitionedMesh::LightWeightFace face;
        face.vertex_ids.reserve(3);
        for (int p = 0; p < 3; ++p)
          face.vertex_ids.push_back(polyh_cell->vertex_ids[face_vids[f][p]]);

        polyh_cell->faces.push_back(face);
      }
      break;
    }
    // Vertex ids do not need to be remapped, but the faces do.
    case CellType::WEDGE:
    {
      std::vector<std::vector<uint64_t>> face_vids = {
        {0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}, {3, 4, 5}, {0, 2, 1}};
      for (int f = 0; f < 5; ++f)
      {
        UnpartitionedMesh::LightWeightFace face;
        face.vertex_ids.reserve(4);
        for (int p = 0; p < face_vids[f].size(); ++p)
          face.vertex_ids.push_back(polyh_cell->vertex_ids[face_vids[f][p]]);

        polyh_cell->faces.push_back(face);
      }
      break;
    }
    default:
    {
      polyh_cell->faces.reserve(num_cfaces);
      for (int f = 0; f < num_cfaces; ++f)
      {
        UnpartitionedMesh::LightWeightFace face;
        auto foam_face = cell_faces[f];
        auto foam_face_points = foam_mesh.faces()[foam_face];
        auto num_fpoints = foam_face_points.size();
        face.vertex_ids.reserve(num_fpoints);
        for (int p = 0; p < num_fpoints; ++p)
        {
          uint64_t point_id = foam_face_points[p];
          face.vertex_ids.push_back(point_id);
        }

        polyh_cell->faces.push_back(face);
      }
      break;
    }
  }

  return polyh_cell;
}

void
CopyFoamMesh(std::shared_ptr<UnpartitionedMesh> mesh,
             const Foam::fvMesh& foam_mesh,
             const double scale)
{
  const std::string fname = "CopyFoamMesh";

  const Foam::label total_cell_count = foam_mesh.cells().size();
  const Foam::label total_point_count = foam_mesh.points().size();

  std::vector<std::shared_ptr<UnpartitionedMesh::LightWeightCell>> cells;
  std::vector<std::shared_ptr<Vector3>> vertices(total_point_count);

  // Process Vertices
  for (Foam::label p = 0; p < total_point_count; ++p)
  {
    const Foam::point& pt = foam_mesh.points()[p];
    auto vertex = std::make_shared<Vector3>(pt.x(), pt.y(), pt.z());
    *vertex *= scale; // Apply scaling
    vertices[p] = vertex;
  }

  // Process Cells with Vertex ID Validation
  for (Foam::label c = 0; c < total_cell_count; ++c)
  {
    std::shared_ptr<UnpartitionedMesh::LightWeightCell> raw_cell =
      CreateCellFromOpenFOAMCell(foam_mesh, c);
    cells.push_back(raw_cell);
  }

  // Store Data into `UnpartitionedMesh`
  mesh->GetRawCells() = cells;
  mesh->GetVertices().reserve(total_point_count);
  for (const auto& vertex_ptr : vertices)
    mesh->GetVertices().push_back(*vertex_ptr);

  mesh->ComputeBoundingBox();

  log.Log() << fname + ": Mesh conversion completed successfully.";
}

// Read in 'materials' from OpenFOAM
// This is typically done by cellZones in polyMesh
std::vector<int>
BuildMaterialIDsFromFoamCellZones(const Foam::fvMesh& foam_mesh)
{
  const size_t total_cell_count = foam_mesh.nCells();
  std::vector<int> block_ids(total_cell_count, 0);

  const Foam::cellZoneMesh& cellZones = foam_mesh.cellZones();

  for (Foam::label zoneID = 0; zoneID < cellZones.size(); ++zoneID)
  {
    const Foam::cellZone& cz = cellZones[zoneID];

    for (Foam::label cell : cz)
    {
        block_ids[cell] = zoneID;
    }
  }
  return block_ids;
}

void
SetMaterialsFromMaterialsList(std::shared_ptr<UnpartitionedMesh> mesh,
                         const std::vector<int>& block_ids)
{
  auto& raw_cells = mesh->GetRawCells();
  const size_t total_cell_count = raw_cells.size();
  for (size_t c = 0; c < total_cell_count; ++c)
    raw_cells[c]->block_id = block_ids[c];
}

} // namespace

std::shared_ptr<UnpartitionedMesh>
MeshIO::FromOpenFOAM(const UnpartitionedMesh::Options& options)
{
  const std::string fname = "MeshIO::FromOpenFOAMFile"; // variable for error debugging
  const std::filesystem::path case_dir(
    options.file_name); // set case_dir to be passed arg from options

  setenv("FOAM_SIGFPE", "false", 1); // Disable floating-point exceptions for debugging
  // Check if valid case directory
  if (!std::filesystem::exists(case_dir))
  {
    throw std::runtime_error(
      fname + ": The specified OpenFOAM case directory does not exist: " + case_dir.string());
  }

  // Load args in Foam format
  int argc = 3;
  const char* temp_args[] = {"Foam2Sn_Mesh", "-case", case_dir.c_str()};
  char** argv = const_cast<char**>(temp_args);
  Foam::argList args(argc, argv, false);
  Foam::Time run_time(Foam::Time::controlDictName, args);
  log.Log() << fname << ": OpenFOAM runtime initialized: " << run_time.timeName(); // Debug

  Foam::fvMesh foam_mesh(Foam::IOobject(
    Foam::fvMesh::defaultRegion, run_time.timeName(), run_time, Foam::IOobject::MUST_READ));

  log.Log() << fname << ": OpenFOAM fvMESH loaded successfully."; // Debug

  // Create and populate UnpartitionedMesh
  auto mesh = std::make_shared<UnpartitionedMesh>();
  CopyFoamMesh(mesh, foam_mesh, options.scale);

  // Set material ids
  const auto block_ids = BuildMaterialIDsFromFoamCellZones(foam_mesh);
  SetMaterialsFromMaterialsList(mesh, block_ids);

  mesh->SetDimension(3);
  mesh->SetType(UNSTRUCTURED);
  mesh->ComputeCentroids();
  mesh->CheckQuality();
  mesh->BuildMeshConnectivity();

  log.Log() << "Done processing " << options.file_name << ".\n"
            << "Number of nodes read: " << mesh->GetVertices().size() << "\n"
            << "Number of cells read: " << mesh->GetRawCells().size();

  return mesh;
}

} // namespace opensn

#endif // OPENSN_WITH_OPENFOAM