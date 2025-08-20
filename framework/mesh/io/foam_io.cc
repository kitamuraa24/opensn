// SPDX-FileCopyrightText: 2024 The OpenSn Authors
// SPDX-License-Identifier: MIT

#include "framework/mesh/io/mesh_io.h"
#include "framework/runtime.h"
#include "framework/logging/log.h"

// system includes
#include <algorithm>
#include <fstream>

namespace opensn {
namespace {
// Utility functions

// trim off the OpenFOAM header
void TrimInPlace(std::string& str)
{
  auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };

  auto first = std::find_if_not(str.begin(), str.end(),
                                [&](char ch){ return is_space(static_cast<unsigned char>(ch)); });
  auto last = std::find_if_not(str.rbegin(), str.rend(),
                                [&](char ch){ return is_space(static_cast<unsigned char>(ch)); }).base(); 

  if (first >= last) { str.clear(); return; }

  str.erase(last, str.end());
  str.erase(str.begin(), first);
}

// verify file section identifiers
inline bool StartsWithKey(std::string_view str, std::string_view key)
{
  return str.size() >= key.size() && std::equal(key.begin(), key.end(), str.begin());
}

inline bool IsSpaceOnly(std::string_view str)
{
  for (char ch : str) if (!std::isspace(static_cast<unsigned char>(ch))) return false;
  return true;
}

// points: formatted like "(x y z)" or "x y z"
inline bool ReadPoint(std::ifstream& in, double& x, double& y, double& z)
{
  std::string line;
  if (!std::getline(in, line)) return false;
  TrimInPlace(line);
  if (!line.empty() && line.front() == '(' && line.back() == ')')
  {
    line.front() = ' '; line.back() = ' ';
  }
  std::istringstream iss(line);
  return static_cast<bool>(iss >> x >> y >> z);
}

// face line: "n(v0 v1 ...)" or multi-line form
inline bool ReadFace(std::ifstream& in, std::vector<int>& out)
{
  std::string line;
  if (!std::getline(in, line)) return false;
  TrimInPlace(line);
  if (line.empty()) return false;

  int n = 0;
  size_t p = line.find('(');
  if (p == std::string::npos)
  {
    std::istringstream iss(line);
    if (!(iss >> n) || n < 0) return false;
    if (!std::getline(in, line)) return false;
    TrimInPlace(line);
    p = line.find('(');
    if (p == std::string::npos) return false;
  }
  else
  {
    std::istringstream iss(line.substr(0, p));
    if (!(iss >> n) || n < 0) return false;
  }

  size_t q = line.find(')', p + 1);
  std::string inside = (q != std::string::npos) ? line.substr(p + 1, q - (p + 1)) : "";
  while (q == std::string::npos)
  {
    std::string more;
    if (!std::getline(in, more)) return false;
    TrimInPlace(more);
    size_t qq = more.find(')');
    if (inside.empty()) inside = more;
    else { inside.push_back(' '); inside += more; }
    if (qq != std::string::npos) break;
  }

  std::istringstream iss(inside);
  out.resize(n);
  for (int i = 0; i < n; ++i)
    if (!(iss >> out[i])) return false;

  return true;
}

// 'owner'/'neighbour' files: single int per line
inline bool ReadLabel(std::ifstream& in, int& v)
{
  std::string line;
  if (!std::getline(in, line)) return false;
  TrimInPlace(line);
  std::istringstream iss(line);
  return static_cast<bool>(iss >> v);
}

// struct BoundaryPatch
// {
//   std::string name;
//   std::string type;
//   int start_face = 0;
//   int nfaces = 0;
// };

// Skip OpenFOAM header block until data is parsed
void SkipFoamHeader(std::ifstream& in)
{
  std::string line;
  while (std::getline(in, line))
  {
    std::string tmp = line;
    TrimInPlace(tmp);
    if (tmp.empty() || StartsWithKey(tmp, "//")) continue;
    if (!tmp.empty() && (std::isdigit(static_cast<unsigned char>(tmp[0])) || tmp[0] == '-'))
    {
      // rewind to start of this line
      in.seekg(-static_cast<std::streamoff>(line.size()) - 1, std::ios_base::cur);
      return;
    }
  }
  in.clear();
}

// Generalized function to read OpenFOAM Lists ( data structure in file )"
template <class ItemReader>
void ReadFoamList(std::ifstream& in, int& count_out, ItemReader item_fn,
                  const std::string& fname, const char* name)
{
  std::string line;

  if (!std::getline(in, line))
    throw std::logic_error(fname + ": EOF reading " + std::string(name) + " count.");
  TrimInPlace(line);
  {
    std::istringstream iss(line);
    if (!(iss >> count_out) || count_out < 0)
      throw std::logic_error(fname + ": Bad " + std::string(name) + " count line: '" + line + "'");
  }

  if (!std::getline(in, line))
    throw std::logic_error(fname + ": EOF before '(' for " + std::string(name));
  TrimInPlace(line);
  if (line != "(" && line.find('(') == std::string::npos)
    throw std::logic_error(fname + ": Expected '(' after " + std::string(name) + " count.");

  for (int i = 0; i < count_out; ++i)
    if (!item_fn(in, i))
      throw std::logic_error(fname + ": Failed reading " + std::string(name) + " item " + std::to_string(i));

  // read ')'
  if (!std::getline(in, line))
    throw std::logic_error(fname + ": EOF reading ')' for " + std::string(name));
  TrimInPlace(line);
  if (line != ")")
  {
    // permit stray comments/whitespace before ')'
    bool ok = false;
    for (int g = 0; g < 32 && std::getline(in, line); ++g)
    {
      TrimInPlace(line);
      if (line == ")") { ok = true; break; }
      if (line.empty() || StartsWithKey(line, "//")) continue;
    }
    if (!ok) throw std::logic_error(fname + ": Missing ')' for " + std::string(name));
  }
}

// std::vector<BoundaryPatch> ReadBoundary(const std::filesystem::path& path,
//                                         const std::string& fname)
// {
//   std::ifstream in(path);
//   if (!in.is_open()) return {};
//   SkipFoamHeader(in);

//   int npatches = 0;
//   std::vector<BoundaryPatch> patches;

//   auto read_item = [&](std::ifstream& in2, int)->bool
//   {
//     std::string line;
//     if (!std::getline(in2, line)) return false;
//     TrimInPlace(line);
//     if (line.empty()) return false;

//     BoundaryPatch p;
//     p.name = line;

//     if (!std::getline(in2, line)) return false;
//     TrimInPlace(line);
//     if (line != "{") return false;

//     std::unordered_map<std::string, std::string> dict;
//     while (std::getline(in2, line))
//     {
//       TrimInPlace(line);
//       if (line == "}") break;
//       if (line.empty() || StartsWithKey(line, "//")) continue;
//       auto semi = line.find(';'); if (semi != std::string::npos) line.erase(semi);
//       TrimInPlace(line);
//       auto sp = line.find_first_of(" \t");
//       if (sp == std::string::npos) continue;
//       std::string key = line.substr(0, sp);
//       std::string val = line.substr(sp + 1);
//       TrimInPlace(val);
//       dict[key] = val;
//     }
//     if (dict.count("type")) p.type = dict["type"];
//     if (dict.count("nFaces")) p.nfaces = std::stoi(dict["nFaces"]);
//     if (dict.count("startFace")) p.start_face = std::stoi(dict["startFace"]);
//     patches.push_back(std::move(p));
//     return true;
//   };

//   ReadFoamList(in, npatches, read_item, fname, "boundary patches");
//   return patches;
// }

} // namespace

std::shared_ptr<UnpartitionedMesh>
MeshIO::FromOpenFOAM(const UnpartitionedMesh::Options& options)
{
  const std::string fname = "MeshIO::FromOpenFOAM";

  std::filesystem::path base = options.file_name;
  const auto poly_mesh = base / "constant" / "polyMesh";
  if (std::filesystem::is_directory(poly_mesh)) base = poly_mesh;
  else if (std::filesystem::is_directory(base) && base.filename() == "polyMesh") { /* ok */ }
  else
    throw std::logic_error(fname + ": polyMesh not found under '" + options.file_name +
                                   "'. Point to <case>/constant/polyMesh or the polyMesh folder.");

  if (base.string().find("processor") != std::string::npos)
    throw std::logic_error(fname + ": Decomposed meshes (processor*/polyMesh) are not supported.");

  const auto points_p   = base / "points";
  const auto faces_p    = base / "faces";
  const auto owner_p    = base / "owner";
  const auto neigh_p    = base / "neighbour";
  const auto boundary_p = base / "boundary";

  if (!std::filesystem::exists(neigh_p))
    throw std::logic_error(fname + ": Missing 'neighbour'. Even with zero internal faces it "
                                   "should exist with count 0.");

  auto mesh = std::make_shared<UnpartitionedMesh>();
  log.Log() << "Reading OpenFOAM polyMesh from " << base.string();

  // read " points "
  {
    std::ifstream in(points_p);
    if (!in.is_open()) throw std::runtime_error(fname + ": Failed to open " + points_p.string());
    SkipFoamHeader(in);

    int npoints = 0;
    std::vector<Vector3> verts;
    verts.reserve(1024);
    auto read_pt = [&](std::ifstream& in2, int)->bool{
      double x,y,z; if (!ReadPoint(in2,x,y,z)) return false;
      verts.push_back({x,y,z}); return true;
    };
    ReadFoamList(in, npoints, read_pt, fname, "points");
    if (verts.size() != npoints) {
      throw std::logic_error(fname + ": number of points mismatch (expected " + std::to_string(npoints) +
        ", but got " + std::to_string(verts.size()) + ")."
      );
    }
    mesh->GetVertices() = std::move(verts);
  }

  // read " faces "
  std::vector<std::vector<int>> face_verts;
  {
    std::ifstream in(faces_p);
    if (!in.is_open()) throw std::runtime_error(fname + ": Failed to open " + faces_p.string());
    SkipFoamHeader(in);

    int nfaces = 0;
    auto read_face = [&](std::ifstream& in2, int)->bool{
      std::vector<int> fv;
      if (!ReadFace(in2, fv)) {
        return false;
      }
      else {
        face_verts.emplace_back(std::move(fv));
        return true;
      }
    };
    ReadFoamList(in, nfaces, read_face, fname, "faces");
    if (face_verts.size() != nfaces) {
      throw std::logic_error(
        fname + " : number of faces mismatch (expected " + std::to_string(nfaces) + ", but got " +
        std::to_string(face_verts.size()) + ")."
      );
    }
  }

  // read " owner "
  std::vector<int> owner;
  {
    std::ifstream in(owner_p);
    if (!in.is_open()) throw std::runtime_error(fname + ": Failed to open " + owner_p.string());
    SkipFoamHeader(in);

    int nfaces = 0;
    auto read_lab = [&](std::ifstream& in2, int)->bool{
      int v; if (!ReadLabel(in2, v)) return false; owner.push_back(v); return true;
    };
    ReadFoamList(in, nfaces, read_lab, fname, "owner");
    if (owner.size() != nfaces) {
      throw std::logic_error(fname + ": owner faces count mismatch (expected " + std::to_string(nfaces) +
      ", but got " + std::to_string(owner.size()) + ")."); 
    }
  }

  // read " neighbour " (must exist, but count can be 0)
  std::vector<int> neigh;
  {
    std::ifstream in(neigh_p);
    if (!in.is_open()) throw std::runtime_error(fname + ": Failed to open " + neigh_p.string());
    SkipFoamHeader(in);

    int nfaces = 0;
    auto read_lab = [&](std::ifstream& in2, int)->bool{
      int v; if (!ReadLabel(in2, v)) return false; neigh.push_back(v); return true;
    };
    ReadFoamList(in, nfaces, read_lab, fname, "neighbour");
    if (neigh.size() != nfaces) {
      throw std::logic_error(fname + ": neigh.size() != nfaces ( expected " + std::to_string(nfaces) + 
    ", but got " + std::to_string(neigh.size()) + ").");
    }
  }

  /**
   *  @TODO: Implementation for boundary treatment once OpenSn can handle this.
   */
  // const std::vector<BoundaryPatch> patches = ReadBoundary(boundary_p, fname);

  // construct cells 
  const int nfaces = static_cast<int>(face_verts.size());
  const int ninternal_faces = static_cast<int>(neigh.size());
  if (owner.size() != nfaces) {
    throw std::logic_error(fname + ": owner.size() != nfaces ( expected " + std::to_string(nfaces) + 
    ", but got " + std::to_string(owner.size()) + ").");
  }

  int max_cell = -1;
  for (int v : owner)  max_cell = std::max(max_cell, v);
  for (int v : neigh)  max_cell = std::max(max_cell, v);
  const int ncells = max_cell + 1;
  if (ncells <= 0) throw std::logic_error(fname + ": Non-positive number of cells computed.");

  // // build patch index per face
  // std::vector<int> face_patch(nfaces, -1);
  // for (int patch_idx = 0; patch_idx < (int)patches.size(); ++patch_idx)
  // {
  //   const auto& p = patches[patch_idx];
  //   for (int f = 0; f < p.nfaces; ++f)
  //   {
  //     const int f_id = p.start_face + f;
  //     if (f_id < 0 || f_id >= nfaces)
  //       throw std::logic_error(fname + ": boundary face id out of range.");
  //     face_patch[f_id] = patch_idx;
  //   }
  // }

  auto& raw_cells  = mesh->GetRawCells();
  raw_cells.reserve(ncells);

  // For each cell, collect faces according to OpenFOAM orientation
  std::vector<std::vector<int>> cell_faces(ncells);
  for (int f = 0; f < ninternal_faces; ++f)
  {
    const int c_o = owner[f];
    const int c_n = neigh[f];
    if (c_o < 0 || c_o >= ncells || c_n < 0 || c_n >= ncells)
      throw std::logic_error(fname + ": owner/neighbour id out of range (face " + std::to_string(f) + ").");
    cell_faces[c_o].push_back(+f);          // owner side: outward as stored
    cell_faces[c_n].push_back(-f - 1);      // neighbour side: reversed for outward
  }
  for (int f = ninternal_faces; f < nfaces; ++f)
  {
    const int c_o = owner[f];
    if (c_o < 0 || c_o >= ncells)
      throw std::logic_error(fname + ": owner id out of range (boundary face " + std::to_string(f) + ").");
    cell_faces[c_o].push_back(+f);          // boundary face outward w.r.t owner
  }

  // Volume cells (POLYHEDRON)
  for (int c = 0; c < ncells; ++c)
  {
    auto cell = std::make_shared<UnpartitionedMesh::LightWeightCell>(CellType::POLYHEDRON, CellType::POLYHEDRON);
    cell->block_id = 0;
    /**
     * @TODO: Need to implement material assignment via cellZones 
     */ 

    for (int code : cell_faces[c])
    {
      const int f = (code >= 0) ? code : (-code - 1);
      const auto& f_v = face_verts[f];

      UnpartitionedMesh::LightWeightFace lwf;
      lwf.vertex_ids.reserve(f_v.size());
      if (code >= 0) {
        for (int v : f_v) lwf.vertex_ids.push_back(static_cast<uint64_t>(v));
      } else {
        for (auto it = f_v.rbegin(); it != f_v.rend(); ++it)
          lwf.vertex_ids.push_back(static_cast<uint64_t>(*it));
      }
      cell->faces.push_back(std::move(lwf));
    }

    // per cell vertex ids
    std::vector<uint64_t> v_set;
    for (const auto& f : cell->faces)
      v_set.insert(v_set.end(), f.vertex_ids.begin(), f.vertex_ids.end());
    std::sort(v_set.begin(), v_set.end());
    v_set.erase(std::unique(v_set.begin(), v_set.end()), v_set.end());
    cell->vertex_ids = std::move(v_set);

    raw_cells.push_back(std::move(cell));
  }

  mesh->SetDimension(3);
  mesh->SetType(UNSTRUCTURED);
  mesh->ComputeCentroids();
  mesh->CheckQuality();
  mesh->BuildMeshConnectivity();

  log.Log() << "OpenFOAM polyMesh processed.\n"
            << "  Number of nodes read     : " << mesh->GetVertices().size() << "\n"
            << "  Number of cells read  : " << mesh->GetRawCells().size() << "\n";

  return mesh;
}

} // opensn
