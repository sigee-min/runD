#pragma once

#include <accel/graph/factory/node/base.hpp>
#include <kernel/program/compute/segmented/reduce/identity.hpp>
#include <kernel/program/compute/segmented/reduce/plan.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelSegmentedReduce(const AccelGraphBufferRef *const refs,
                     const std::uint64_t ref_count,
                     const kernel::SegmentedReduceDesc &desc) noexcept {
  const kernel::SegmentedReducePlan plan = kernel::PlanSegmentedReduce(desc);
  AccelGraphNode node = accel_graph_factory_detail::PrimitiveNode(
      refs, ref_count, kernel::NodeKind::SegmentedReduce,
      kernel::HashSegmentedReduce(desc), desc.element_count);
  node.segmented_reduce = desc;
  node.signature = kernel::GraphSignatureFor(plan);
  return node;
}

} // namespace rund
