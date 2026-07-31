#pragma once

#include <accel/context/value.hpp>
#include <accel/graph/value.hpp>
#include <accel/graph/node.hpp>
#include <accel/kernel/check.hpp>
#include <accel/kernel/evidence.hpp>

#include <kernel/program/compute/dsl.hpp>
#include <node/accel/context.hpp>

#include <string_view>

namespace node_accel_contract::collective {

[[nodiscard]] inline bool CheckReason(const rund::AccelKernelCheck &check,
                                      const std::string_view reason) noexcept {
  return !check.ok && std::string_view{check.reason} == reason;
}

[[nodiscard]] inline bool
EvidenceReason(const rund::AccelEvidence &evidence,
               const std::string_view reason) noexcept {
  return !evidence.ok && std::string_view{evidence.reason} == reason;
}

[[nodiscard]] inline bool SingleNodeCompileReason(
    const rund::AccelContext &context, const rund::AccelGraph &graph,
    const rund::AccelGraphNode &node, const std::string_view reason) noexcept {
  const rund::AccelGraph single_node_graph{
      .nodes = &node,
      .node_count = 1u,
      .scalar = graph.scalar,
      .domain = graph.domain,
      .fixed_format = graph.fixed_format,
  };
  return CheckReason(
      rund::node::accel::CompileAccelKernel(context, single_node_graph).check,
      reason);
}

} // namespace node_accel_contract::collective
