// SPDX-FileCopyrightText: 2025 The OpenSn Authors <https://open-sn.github.io/opensn/>
// SPDX-License-Identifier: MIT

#include "python/lib/py_wrappers.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/postprocessors/volume_postprocessor.h"
#include "modules/linear_boltzmann_solvers/lbs_problem/postprocessors/reaction_rate_postprocessor.h"
#include <pybind11/stl.h>

namespace opensn
{

// Wrap post processors
void
WrapPostprocessors(py::module& post)
{
  // clang-format off
  // Volume post-processor value type enum
  py::enum_<VolumePostprocessor::ValueType>(post, "VolumePostprocessorValueType")
    .value("INTEGRAL", VolumePostprocessor::ValueType::INTEGRAL)
    .value("MAX", VolumePostprocessor::ValueType::MAX)
    .value("MIN", VolumePostprocessor::ValueType::MIN)
    .value("AVERAGE", VolumePostprocessor::ValueType::AVERAGE);

  // Volume post-processor
  auto vp = py::class_<VolumePostprocessor, std::shared_ptr<VolumePostprocessor>>(
    post,
    "VolumePostprocessor",
    R"(
    Volume post-processor.

    Wrapper of :cpp:class:`opensn::VolumePostprocessor`.
    )"
  );
  vp.def(
    py::init(
      [](py::kwargs& params)
      {
        return VolumePostprocessor::Create(kwargs_to_param_block(params));
      }
    ),
    R"(
    Construct a volume post processor object.

    Parameters
    ----------
    problem : LBSProblem
        A handle to an existing LBS problem.
    value_type : str, optional
        Type of value to compute: 'integral' (default), 'max', 'min', or 'avg'.
    )"
  );
  vp.def(
    "Execute",
    [](VolumePostprocessor& self){
      self.Execute();
    },
    R"(
      TODO: finish this
    )"
  );
  vp.def(
    "GetValue",
    [](VolumePostprocessor& self)
    {
      return self.GetValue();
    },
    R"(
    TODO: finish this
    )"
  );

  // Reaction rate post processor
  auto rrp = py::class_<ReactionRatePostprocessor,
                        VolumePostprocessor,
                        std::shared_ptr<ReactionRatePostprocessor>>(
  post,
  "ReactionRatePostprocessor",
  R"(
  Reaction-rate post-processor.
  Wrapper of :cpp:class:`opensn::ReactionRatePostprocessor`.
  Computes \f$\int_V \Sigma_g \phi_g\, dV\f$ (or max/min/avg variants)
  over the selected spatial region and energy groups.
  )"
  );
  rrp.def(
    py::init(
      [](py::kwargs& params)
      {
        return ReactionRatePostprocessor::Create(kwargs_to_param_block(params));
      }
    ),
    R"(
    Construct a reaction-rate post processor object.

    Parameters
    ----------
    problem : LBSProblem
        A handle to an existing LBS problem.
    reaction : str
        Reaction XS name. Built-ins: 'total', 'absorption', 'fission', 'nu-fission'.
        Otherwise must match a custom XS name present in MultiGroupXS.
    value_type : str, optional
        Type of value to compute: 'integral' (default), 'max', 'min', or 'avg'.
    group : int, optional
        Single group to compute (mutually exclusive with groupset).
    groupset : int, optional
        Single groupset to compute (mutually exclusive with group).
    block_ids : list[int], optional
        Block restriction for the postprocessor.
    logical_volumes : list[LogicalVolume], optional
        Logical volumes to restrict the computation to.
    )"
  );
  rrp.def(
    "Execute",
    [](ReactionRatePostprocessor& self){
      self.Execute();
    },
    R"(
      TODO: finish this
    )"
  );
  rrp.def(
    "GetValue",
    [](ReactionRatePostprocessor& self)
    {
      return self.GetValue();
    },
    R"(
    TODO: finish this
    )"
  );
  // clang-format on
}

// Wrap the post-processing components of OpenSn.
void
py_post(py::module& pyopensn)
{
  py::module post = pyopensn.def_submodule("post", "Post-processing module.");
  WrapPostprocessors(post);
}

} // namespace opensn
