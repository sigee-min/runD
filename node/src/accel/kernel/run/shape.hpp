#pragma once

#include <accel/context/value.hpp>
#include <accel/kernel/run.hpp>

#include "local.hpp"

#include <cstddef>

namespace rund::node::accel::detail {

[[nodiscard]] inline bool
RunRequestShapeOk(const rund::AccelContext &context,
                  const KernelExecution &execution,
                  const rund::AccelRun &run) noexcept {
  return run.tile_count != 0u && !execution.steps.empty() &&
         execution.graph_roles.size() ==
             static_cast<std::size_t>(run.binding_count) &&
         execution.graph_shapes.size() ==
             static_cast<std::size_t>(run.binding_count) &&
         execution.graph_visibilities.size() ==
             static_cast<std::size_t>(run.binding_count) &&
         execution.graph_alias_representatives.size() ==
             static_cast<std::size_t>(run.binding_count) &&
         execution.required_barriers.size() == execution.steps.size() &&
         (run.binding_count == 0u || run.bindings != nullptr) &&
         static_cast<bool>(context.pick.backend) &&
         execution.context_admission.check.ok;
}

} // namespace rund::node::accel::detail
