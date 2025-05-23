// SPDX-FileCopyrightText: 2024 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/discrete_ordinates_problem/sweep/fluds/fluds_common_data.h"
#include <set>
#include <cstddef>
#include <cstdint>

namespace opensn
{

class GridFaceHistogram;
class Cell;
class CellFace;

// face_slot index, vertex ids
using CompactFaceView = std::pair<int, std::vector<uint64_t>>;

// cell_global_id, faces
using CompactCellView = std::pair<int, std::vector<CompactFaceView>>;

class SPDS;

class AAH_FLUDSCommonData : public FLUDSCommonData
{
public:
  struct INCOMING_FACE_INFO
  {
    int slot_address = 0;
    std::vector<short> upwind_dof_mapping;

    void Setup(const std::pair<int, std::vector<short>>& input)
    {
      slot_address = input.first;
      upwind_dof_mapping.resize(input.second.size());
      std::copy(input.second.begin(), input.second.end(), upwind_dof_mapping.begin());
    }

    ~INCOMING_FACE_INFO() {}
  }; // TODO: Make common
public:
  explicit AAH_FLUDSCommonData(const std::vector<CellFaceNodalMapping>& grid_nodal_mappings,
                               const SPDS& spds,
                               const GridFaceHistogram& grid_face_histogram);

protected:
  friend class AAH_FLUDS;
  int largest_face_ = 0;
  /// Number of face categories
  size_t num_face_categories_ = 0;
  /// Group-angle-faceDOF stride per cat
  std::vector<size_t> local_psi_stride_;
  /// Number of faces in each cat
  std::vector<size_t> local_psi_max_elements_;
  /// Group-angle-faceDOF stride delayed cat
  size_t delayed_local_psi_stride_ = 0;
  /// Number of faces in delayed cat
  size_t delayed_local_psi_max_elements_ = 0;

  /**
   * local_psi_n_block_stride[fc]. Given face category fc, the value is
   * total number of faces that store information in this category's buffer
   * per angle
   */
  std::vector<size_t> local_psi_n_block_stride_;
  std::vector<size_t> local_psi_Gn_block_strideG_;
  size_t delayed_local_psi_Gn_block_stride_ = 0;
  size_t delayed_local_psi_Gn_block_strideG_ = 0;

  /**
   * This is a small vector [deplocI] that holds the number of face dofs for each dependent
   * location.
   */
  std::vector<int> deplocI_face_dof_count_;

  /**
   * This is a vector [dependent_location][unordered_cell_index]
   * that holds an AlphaPair. AlphaPair-first is the cell's global_id
   * and AlphaPair-second holds a number of BetaPairs. Each BetaPair
   * comprises BetaPair-first = face_slot_index (the location of this
   * faces data in the psi vector, and then a vector of vertex indexes
   * that can be used for dof_mapping.
   * Filled during slot-dynamics.
   * Cleared after beta-pass.
   */
  std::vector<std::vector<CompactCellView>> deplocI_cell_views_;

  /**
   * This is a vector [cell_sweep_order_index][outgoing_face_count] which holds the slot address in
   * the local psi vector where the first face dof will store its data
   */
  std::vector<std::vector<int>> so_cell_outb_face_slot_indices_;

  /**
   * This is a vector [cell_sweep_order_index][outgoing_face_count]  which holds the face
   * categorization for the face. i.e. the local psi vector that hold faces of the same category.
   */
  std::vector<std::vector<short>> so_cell_outb_face_face_category_;

  /**
   * This is a vector [cell_sweep_order_index][incoming_face_count] which holds the face
   * categorization for the face. i.e. the local  psi vector that hold faces of the same category.
   */
  std::vector<std::vector<short>> so_cell_inco_face_face_category_;

  /**
   * This is a vector [cell_sweep_order_index][incoming_face_count] that will hold a structure.
   * struct.slot_address holds the slot address where this face's upwind data is stored.
   * struct.upwind_dof_mapping is a mapping of each of this face's dofs to the upwinded face's dofs
   */
private:
  std::vector<std::vector<INCOMING_FACE_INFO>> so_cell_inco_face_dof_indices_;

  /**
   * This is a vector [non_local_outgoing_face_count]  that maps a face to a dependent location and
   * associated slot index
   */
  std::vector<std::pair<int, int>> nonlocal_outb_face_deplocI_slot_;

private:
  /**
   * This is a vector [predecessor_location][unordered_cell_index]
   * that holds an AlphaPair. AlphaPair-first is the cell's global_id
   * and AlphaPair-second holds a number of BetaPairs. Each BetaPair
   * comprises BetaPair-first = face_slot_index (the location of this
   * faces data in the psi vector, and then a vector of vertex indexes
   * that can be used for dof_mapping.
   * Filled in beta-pass
   * Cleared after beta-pass.
   */
  std::vector<std::vector<CompactCellView>> prelocI_cell_views_;
  std::vector<std::vector<CompactCellView>> delayed_prelocI_cell_views_;

  /**
   * This is a small vector [prelocI] that holds the number of  face dofs for each predecessor
   * location.
   */
  std::vector<int> prelocI_face_dof_count_;
  std::vector<int> delayed_prelocI_face_dof_count_;

  /**
   * This is a vector [nonlocal_inc_face_counter] containing AlphaPairs. AlphaPair-first is the
   * prelocI index and AlphaPair-second is a BetaPair. The BetaPair-first is the slot where
   * the face storage begins and BetaPair-second is a dof mapping
   */
  std::vector<std::pair<int, std::pair<int, std::vector<int>>>> nonlocal_inc_face_prelocI_slot_dof_;

  std::vector<std::pair<int, std::pair<int, std::vector<int>>>>
    delayed_nonlocal_inc_face_prelocI_slot_dof_;

  void InitializeAlphaElements(const SPDS& spds, const GridFaceHistogram& grid_face_histogram);

  void SlotDynamics(const Cell& cell,
                    const SPDS& spds,
                    const GridFaceHistogram& grid_face_histogram,
                    std::vector<std::vector<std::pair<int, short>>>& lock_boxes,
                    std::vector<std::pair<int, short>>& delayed_lock_box);

  /**
   * Given a sweep ordering index, the outgoing face counter, the outgoing face dof, this function
   * computes the location of this position's upwind psi in the local upwind psi vector.
   */
  void AddFaceViewToDepLocI(int deplocI, int cell_g_index, int face_slot, const CellFace& face);

  void
  LocalIncidentMapping(const Cell& cell, const SPDS& spds, std::vector<int>& local_so_cell_mapping);

  void InitializeBetaElements(const SPDS& spds, int tag_index = 0);

  /**
   * This cell takes a hierarchy of a cell compact view and serializes it for MPI transmission.
   * This is easy since all the values are integers.
   */
  static void SerializeCellInfo(std::vector<CompactCellView>& cell_views,
                                std::vector<int>& face_indices,
                                int num_face_dofs);

  /// Deserializes face indices.
  static void DeSerializeCellInfo(std::vector<CompactCellView>& cell_views,
                                  std::vector<int>* face_indices,
                                  int& num_face_dofs);

  void NonLocalIncidentMapping(const Cell& cell, const SPDS& spds);
};

} // namespace opensn
