#pragma once

#include <accel/graph/factory/map.hpp>

#include <accel/context/buffer/descriptor.hpp>
#include <accel/graph/value.hpp>

#include <node/accel/context.hpp>

#include "../../local.hpp"

#include <array>
#include <string_view>

namespace node_accel_contract::fusion {

[[nodiscard]] bool RunCapacityCase(
    const rund::AccelContext &context,
    const rund::compute_dsl::ComputeOp &op) {
  constexpr std::size_t count = 63u;
  constexpr rund::AccelBufferDesc shape{
      .scalar_width_bytes = sizeof(rund::kernel::i32),
      .count = 8u,
      .usage = rund::BufferUsage::ReadWrite,
  };
  std::array<GraphBufferRef, count * 2u> refs{};
  std::array<GraphNode, count> nodes{};
  for (std::size_t index = 0u; index < count; ++index) {
    refs[index * 2u] = GraphBufferRef{
        .shape = shape,
        .logical_id = index + 1u,
        .role = Role::Read,
        .visibility = index == 0u ? Visibility::External
                                  : Visibility::Internal,
    };
    refs[index * 2u + 1u] = GraphBufferRef{
        .shape = shape,
        .logical_id = index + 2u,
        .role = Role::Write,
        .visibility = index + 1u == count ? Visibility::External
                                          : Visibility::Internal,
    };
    nodes[index] = rund::AccelMap(op.ir(), refs.data() + index * 2u, 2u,
                                  shape.count);
  }

  const rund::AccelKernel kernel = rund::node::accel::CompileAccelKernel(
      context, rund::AccelGraph{
                   .nodes = nodes.data(),
                   .node_count = nodes.size(),
                   .scalar = op.ir().scalar,
                   .domain = op.ir().domain,
                   .fixed_format = op.ir().fixed_format,
               });
  if (!kernel.check.ok || kernel.node_count != count) {
    return false;
  }
  const auto execution =
      rund::node::accel::detail::AdmitKernelForExecution(context, kernel);
  if (!execution.admission.check.ok || execution.steps.size() != 2u ||
      execution.removed_dispatch_count != 61u ||
      execution.original_operation_count != count ||
      execution.fused_operation_count != 2u ||
      execution.fusion_rejection_count != 1u ||
      std::string_view{execution.fusion_reason} !=
          "compute_fusion_capacity_boundary") {
    return false;
  }
  for (const auto &step : execution.steps) {
    if (!StepArtifactIsChecked(step, kernel)) {
      return false;
    }
  }
  return true;
}

} // namespace node_accel_contract::fusion
