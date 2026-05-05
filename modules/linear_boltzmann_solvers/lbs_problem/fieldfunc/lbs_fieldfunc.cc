// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/lbs_problem/lbs_problem.h"

#include "framework/field_functions/field_function_grid_based.h"
#include "framework/materials/multi_group_xs/multi_group_xs.h"
#include "framework/runtime.h"
#include "framework/utils/error.h"

#include <algorithm>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace opensn
{
namespace
{

std::string
NormalizeReactionRateXSName(const std::string& xs_name)
{
  if (xs_name == "total" || xs_name == "sigt")
    return "sigma_t";
  if (xs_name == "absorption" || xs_name == "siga")
    return "sigma_a";
  if (xs_name == "fission" || xs_name == "sigf")
    return "sigma_f";
  return xs_name;
}

bool
BlockAllowed(const std::vector<int>& block_ids, const int block_id)
{
  return block_ids.empty() ||
         std::find(block_ids.begin(), block_ids.end(), block_id) != block_ids.end();
}

} // namespace

std::shared_ptr<FieldFunctionGridBased>
LBSProblem::CreateScalarFluxFieldFunction(unsigned int g, unsigned int m)
{
  OpenSnLogicalErrorIf(g >= num_groups_, GetName() + ": Group index out of range.");
  OpenSnLogicalErrorIf(m >= num_moments_, GetName() + ": Moment index out of range.");

  auto ff_ptr = CreateEmptyFieldFunction(MakeScalarFluxFieldFunctionName(g, m));
  UpdateScalarFluxFieldFunction(*ff_ptr, g, m);

  const std::weak_ptr<LBSProblem> weak_owner = weak_from_this();
  ff_ptr->SetUpdateCallback(
    [weak_owner, g, m](FieldFunctionGridBased& ff)
    {
      auto owner = weak_owner.lock();
      OpenSnLogicalErrorIf(not owner,
                           "Cannot update field function after its owning problem has "
                           "been destroyed.");
      owner->UpdateScalarFluxFieldFunction(ff, g, m);
    },
    [weak_owner]() { return not weak_owner.expired(); });
  return ff_ptr;
}

std::shared_ptr<FieldFunctionGridBased>
LBSProblem::CreateFieldFunction(const std::string& name,
                                const std::string& xs_name,
                                const double power_normalization_target)
{
  const std::string ff_name = MakeFieldFunctionName(name);
  auto ff_ptr = CreateEmptyFieldFunction(ff_name);

  UpdateDerivedFieldFunction(*ff_ptr, xs_name, power_normalization_target);

  const std::weak_ptr<LBSProblem> weak_owner = weak_from_this();
  auto xs_name_copy = xs_name;
  ff_ptr->SetUpdateCallback(
    [weak_owner, xs_name = std::move(xs_name_copy), power_normalization_target](
      FieldFunctionGridBased& ff)
    {
      auto owner = weak_owner.lock();
      OpenSnLogicalErrorIf(not owner,
                           "Cannot update field function after its owning problem has "
                           "been destroyed.");
      owner->UpdateDerivedFieldFunction(ff, xs_name, power_normalization_target);
    },
    [weak_owner]() { return not weak_owner.expired(); });
  return ff_ptr;
}

std::shared_ptr<FieldFunctionGridBased>
LBSProblem::CreateReactionRateDensityFieldFunction(const std::string& name,
                                                   const std::string& xs_name,
                                                   const int group,
                                                   const std::vector<int>& block_ids)
{
  const std::string ff_name = MakeFieldFunctionName(name);
  auto ff_ptr = CreateEmptyFieldFunction(ff_name);

  UpdateReactionRateDensityFieldFunction(*ff_ptr, xs_name, group, block_ids);

  const std::weak_ptr<LBSProblem> weak_owner = weak_from_this();
  auto xs_name_copy = xs_name;
  auto block_ids_copy = block_ids;
  ff_ptr->SetUpdateCallback(
    [weak_owner, xs_name = std::move(xs_name_copy), group, block_ids = std::move(block_ids_copy)](
      FieldFunctionGridBased& ff)
    {
      auto owner = weak_owner.lock();
      OpenSnLogicalErrorIf(not owner,
                           "Cannot update field function after its owning problem has "
                           "been destroyed.");
      owner->UpdateReactionRateDensityFieldFunction(ff, xs_name, group, block_ids);
    },
    [weak_owner]() { return not weak_owner.expired(); });
  return ff_ptr;
}

std::string
LBSProblem::MakeFieldFunctionName(const std::string& base_name) const
{
  std::string prefix;
  if (options_.field_function_prefix_option == "prefix")
  {
    prefix = options_.field_function_prefix;
    if (not prefix.empty())
      prefix += "_";
  }
  if (options_.field_function_prefix_option == "solver_name")
    prefix = GetName() + "_";

  return prefix + base_name;
}

std::string
LBSProblem::MakeScalarFluxFieldFunctionName(const unsigned int g, const unsigned int m) const
{
  std::ostringstream oss;
  oss << MakeFieldFunctionName("phi_g") << std::setw(3) << std::setfill('0') << static_cast<int>(g)
      << "_m" << std::setw(2) << std::setfill('0') << static_cast<int>(m);
  return oss.str();
}

std::shared_ptr<FieldFunctionGridBased>
LBSProblem::CreateEmptyFieldFunction(const std::string& name) const
{
  auto discretization = discretization_;
  return std::make_shared<FieldFunctionGridBased>(
    name, discretization, Unknown(UnknownType::SCALAR));
}

void
LBSProblem::UpdateScalarFluxFieldFunction(FieldFunctionGridBased& ff,
                                          const unsigned int g,
                                          const unsigned int m)
{
  ff.UpdateFieldVector(ComputeScalarFluxFieldFunctionData(g, m));
}

void
LBSProblem::UpdateDerivedFieldFunction(FieldFunctionGridBased& ff,
                                       const std::string& xs_name,
                                       const double power_normalization_target)
{
  std::vector<double> data_vector_local;
  if (xs_name == "power")
  {
    double local_total_power = 0.0;
    data_vector_local = ComputePowerFieldFunctionData(local_total_power);
  }
  else
  {
    data_vector_local = ComputeXSFieldFunctionData(xs_name);
  }

  if (power_normalization_target > 0.0)
  {
    const double scale_factor = ComputeFieldFunctionPowerScaleFactor(power_normalization_target);
    Scale(data_vector_local, scale_factor);
  }

  ff.UpdateFieldVector(data_vector_local);
}

void
LBSProblem::UpdateReactionRateDensityFieldFunction(FieldFunctionGridBased& ff,
                                                   const std::string& xs_name,
                                                   const int group,
                                                   const std::vector<int>& block_ids)
{
  ff.UpdateFieldVector(ComputeReactionRateDensityFieldFunctionData(xs_name, group, block_ids));
}

std::vector<double>
LBSProblem::ComputeScalarFluxFieldFunctionData(const unsigned int g, const unsigned int m) const
{
  const auto& sdm = *discretization_;
  const auto& phi_uk_man = flux_moments_uk_man_;
  std::vector<double> data_vector_local(local_node_count_, 0.0);

  for (const auto& cell : grid_->local_cells)
  {
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const size_t num_nodes = cell_mapping.GetNumNodes();

    for (size_t i = 0; i < num_nodes; ++i)
    {
      const auto imapA = sdm.MapDOFLocal(cell, i, phi_uk_man, m, g);
      const auto imapB = sdm.MapDOFLocal(cell, i);
      data_vector_local[imapB] = phi_new_local_[imapA];
    }
  }

  return data_vector_local;
}

double
LBSProblem::ComputeFieldFunctionPowerScaleFactor(const double power_normalization_target) const
{
  OpenSnInvalidArgumentIf(power_normalization_target <= 0.0,
                          GetName() + ": power_normalization_target must be positive.");

  double local_total_power = 0.0;
  auto power_vector = ComputePowerFieldFunctionData(local_total_power);
  (void)power_vector;

  double global_total_power = 0.0;
  mpi_comm.all_reduce(local_total_power, global_total_power, mpi::op::sum<double>());
  OpenSnLogicalErrorIf(
    global_total_power <= 0.0,
    GetName() + ": Power normalization requested, but global total power is non-positive.");

  return power_normalization_target / global_total_power;
}

std::vector<double>
LBSProblem::ComputeXSFieldFunctionData(const std::string& xs_name) const
{
  const auto& sdm = *discretization_;
  const auto& phi_uk_man = flux_moments_uk_man_;
  std::vector<double> data_vector_local(local_node_count_, 0.0);

  for (const auto& cell : grid_->local_cells)
  {
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const size_t num_nodes = cell_mapping.GetNumNodes();
    const auto& xs = block_id_to_xs_map_.at(cell.block_id);
    const auto* coeffs = xs->GetByName(xs_name);

    if (coeffs == nullptr)
      continue;

    OpenSnLogicalErrorIf(coeffs->size() != num_groups_,
                         GetName() + ": 1D cross section \"" + xs_name +
                           "\" has incompatible group size for field function generation.");

    for (size_t i = 0; i < num_nodes; ++i)
    {
      const auto imapA = sdm.MapDOFLocal(cell, i);
      const auto imapB = sdm.MapDOFLocal(cell, i, phi_uk_man, 0, 0);

      double nodal_value = 0.0;
      for (unsigned int g = 0; g < num_groups_; ++g)
        nodal_value += coeffs->at(g) * phi_new_local_[imapB + g];

      data_vector_local[imapA] = nodal_value;
    }
  }

  return data_vector_local;
}

std::vector<double>
LBSProblem::ComputeReactionRateDensityFieldFunctionData(const std::string& xs_name,
                                                        const int group,
                                                        const std::vector<int>& block_ids) const
{
  OpenSnInvalidArgumentIf(group < -1, GetName() + ": group must be -1 or a valid group index.");
  OpenSnInvalidArgumentIf(group >= static_cast<int>(num_groups_),
                          GetName() + ": Group index out of range.");

  const std::string normalized_xs_name = NormalizeReactionRateXSName(xs_name);
  const auto& sdm = *discretization_;
  const auto& phi_uk_man = flux_moments_uk_man_;
  std::vector<double> data_vector_local(local_node_count_, 0.0);

  const unsigned int first_group = group < 0 ? 0 : static_cast<unsigned int>(group);
  const unsigned int last_group = group < 0 ? num_groups_ - 1 : static_cast<unsigned int>(group);

  for (const auto& cell : grid_->local_cells)
  {
    if (not BlockAllowed(block_ids, cell.block_id))
      continue;

    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const size_t num_nodes = cell_mapping.GetNumNodes();
    const auto& xs = block_id_to_xs_map_.at(cell.block_id);
    const auto* coeffs = xs->GetByName(normalized_xs_name);

    if (coeffs == nullptr)
      continue;

    OpenSnLogicalErrorIf(coeffs->size() != num_groups_,
                         GetName() + ": 1D cross section \"" + xs_name +
                           "\" has incompatible group size for field function generation.");

    for (size_t i = 0; i < num_nodes; ++i)
    {
      const auto imapA = sdm.MapDOFLocal(cell, i);
      double nodal_value = 0.0;

      for (unsigned int g = first_group; g <= last_group; ++g)
      {
        const auto imapB = sdm.MapDOFLocal(cell, i, phi_uk_man, 0, g);
        nodal_value += coeffs->at(g) * phi_new_local_[imapB];
      }

      data_vector_local[imapA] = nodal_value;
    }
  }

  return data_vector_local;
}

std::vector<double>
LBSProblem::ComputePowerFieldFunctionData(double& local_total_power) const
{
  const auto& sdm = *discretization_;
  const auto& phi_uk_man = flux_moments_uk_man_;
  std::vector<double> data_vector_power_local(local_node_count_, 0.0);
  local_total_power = 0.0;

  for (const auto& cell : grid_->local_cells)
  {
    const auto& cell_mapping = sdm.GetCellMapping(cell);
    const size_t num_nodes = cell_mapping.GetNumNodes();

    const auto& Vi = unit_cell_matrices_[cell.local_id].intV_shapeI;
    const auto& xs = block_id_to_xs_map_.at(cell.block_id);

    if (not xs->IsFissionable())
      continue;

    for (size_t i = 0; i < num_nodes; ++i)
    {
      const auto imapA = sdm.MapDOFLocal(cell, i);
      const auto imapB = sdm.MapDOFLocal(cell, i, phi_uk_man, 0, 0);

      double nodal_power = 0.0;
      for (unsigned int g = 0; g < num_groups_; ++g)
      {
        const double sigma_fg = xs->GetSigmaFission()[g];
        const double kappa_g = options_.power_default_kappa;
        nodal_power += kappa_g * sigma_fg * phi_new_local_[imapB + g];
      }

      data_vector_power_local[imapA] = nodal_power;
      local_total_power += nodal_power * Vi(i);
    }
  }

  return data_vector_power_local;
}

} // namespace opensn
