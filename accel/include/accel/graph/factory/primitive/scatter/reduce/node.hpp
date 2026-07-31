#pragma once

#include <accel/graph/factory/node/base.hpp>
#include <kernel/program/compute/scatter/reduce/identity.hpp>
#include <kernel/program/compute/scatter/reduce/plan.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode AccelScatterReduce(
    const AccelGraphBufferRef *const refs, const std::uint64_t ref_count,
    const kernel::ScatterReduceDesc &desc) noexcept {
  const kernel::ScatterReducePlan plan = kernel::PlanScatterReduce(desc);
  AccelGraphNode node = accel_graph_factory_detail::PrimitiveNode(
      refs, ref_count, kernel::NodeKind::ScatterReduce,
      kernel::HashScatterReduce(desc), desc.element_count);
  node.scatter_reduce = desc;
  node.signature = kernel::GraphSignatureFor(plan);
  return node;
}

} // namespace rund
