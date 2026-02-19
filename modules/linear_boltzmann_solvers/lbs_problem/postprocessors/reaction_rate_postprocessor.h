// SPDX-FileCopyrightText: 2026 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#pragma once

#include "modules/linear_boltzmann_solvers/lbs_problem/postprocessors/volume_postprocessor.h"

namespace opensn
{

/**
 * Computes reaction-rate quantities by integrating \f$\Sigma_g\,\phi_g\f$ over
 * volume, optionally restricted by block IDs and/or logical volumes.
 *
 * The reaction cross section is obtained from the LBSProblem block-id-to-XS map
 * (BlockID2XSMap). The reaction can be one of the built-in names:
 *   - "total"
 *   - "absorption"
 *   - "fission"
 *   - "nu-fission"
 * or any custom XS name present in MultiGroupXS.
 */
class ReactionRatePostprocessor : public VolumePostprocessor
{
public:
  explicit ReactionRatePostprocessor(const InputParameters& params);

  /// Returns the input parameters for this object.
  static InputParameters GetInputParameters();
  static std::shared_ptr<ReactionRatePostprocessor> Create(const ParameterBlock& params);

protected:
  double CellGroupMultiplier(const Cell& cell, unsigned int g) const override;

private:
  std::string reaction_;
};

} // namespace opensn