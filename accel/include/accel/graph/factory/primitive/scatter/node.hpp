#pragma once

#include <accel/graph/factory/node/base.hpp>
#include <kernel/program/compute/scatter/identity.hpp>
#include <kernel/program/compute/scatter/plan.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelScatter(const AccelGraphBufferRef *const refs,
             const std::uint64_t ref_count,
             const kernel::ScatterDesc &desc) noexcept {
  const kernel::ScatterPlan plan = kernel::PlanScatter(desc);
  AccelGraphNode node = accel_graph_factory_detail::PrimitiveNode(
      refs, ref_count, kernel::NodeKind::Scatter, kernel::HashScatter(desc),
      desc.element_count);
  node.scatter = desc;
  node.signature = kernel::GraphSignatureFor(plan);
  return node;
}

} // namespace rund
