#pragma once

#include <accel/graph/factory/node/base.hpp>
#include <kernel/program/compute/reduce/identity.hpp>
#include <kernel/program/compute/reduce/plan.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelReduce(const AccelGraphBufferRef *const refs,
            const std::uint64_t ref_count,
            const kernel::ReduceDesc &desc) noexcept {
  const kernel::ReducePlan plan = kernel::PlanReduce(desc);
  AccelGraphNode node = accel_graph_factory_detail::PrimitiveNode(
      refs, ref_count, kernel::NodeKind::Reduce, kernel::HashReduce(desc),
      desc.element_count);
  node.reduce = desc;
  node.signature = kernel::GraphSignatureFor(plan);
  return node;
}

} // namespace rund
