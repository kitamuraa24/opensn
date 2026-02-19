// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "modules/linear_boltzmann_solvers/lbs_problem/postprocessors/reaction_rate_postprocessor.h"

#include "framework/object_factory.h"
#include "framework/materials/multi_group_xs/multi_group_xs.h"

#include <stdexcept>

namespace opensn
{

OpenSnRegisterObjectInNamespace(lbs, ReactionRatePostprocessor);

namespace
{
const std::vector<double>&
SelectSigma(const MultiGroupXS& xs, const std::string& reaction)
{
  if (reaction == "total")
    return xs.GetSigmaTotal();
  if (reaction == "absorption")
    return xs.GetSigmaAbsorption();
  if (reaction == "fission")
    return xs.GetSigmaFission();
  if (reaction == "nu-fission")
    return xs.GetNuSigmaF();

  if (xs.HasCustomXS(reaction))
    return xs.GetCustomXS(reaction);

  throw std::invalid_argument("Unknown reaction '" + reaction + "'");
}
} // namespace

InputParameters
ReactionRatePostprocessor::GetInputParameters()
{
  // Start with the base volume-postprocessor params, then add reaction selection.
  auto params = VolumePostprocessor::GetInputParameters();
  params.AddRequiredParameter<std::string>(
    "reaction",
    "Reaction XS name. Built-ins: 'total', 'absorption', 'fission', 'nu-fission'. "
    "Otherwise must match a custom XS name present in MultiGroupXS.");
  return params;
}

std::shared_ptr<ReactionRatePostprocessor>
ReactionRatePostprocessor::Create(const ParameterBlock& params)
{
  auto& factory = opensn::ObjectFactory::GetInstance();
  return factory.Create<ReactionRatePostprocessor>("lbs::ReactionRatePostprocessor", params);
}

ReactionRatePostprocessor::ReactionRatePostprocessor(const InputParameters& params)
  : VolumePostprocessor(params), reaction_(params.GetParamValue<std::string>("reaction"))
{
}

double
ReactionRatePostprocessor::CellGroupMultiplier(const Cell& cell, unsigned int g) const
{
  const auto& lbs = *GetLBSProblem();
  const auto& xs_map = lbs.GetBlockID2XSMap();

  const auto it = xs_map.find(cell.block_id);
  if (it == xs_map.end() || not it->second)
    throw std::runtime_error("ReactionRatePostprocessor: no XS found for block_id=" +
                             std::to_string(cell.block_id));

  const MultiGroupXS& xs = *(it->second);
  const auto& sigma = SelectSigma(xs, reaction_);

  if (g >= sigma.size())
    throw std::out_of_range("ReactionRatePostprocessor: group index out of range for reaction '" +
                            reaction_ + "'");

  return sigma[g];
}

} // namespace opensn