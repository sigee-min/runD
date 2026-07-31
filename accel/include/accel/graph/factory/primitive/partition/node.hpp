#pragma once

#include <accel/graph/factory/node/base.hpp>
#include <kernel/program/compute/partition/identity.hpp>
#include <kernel/program/compute/partition/plan.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelPartition(const AccelGraphBufferRef *const refs,
               const std::uint64_t ref_count,
               const kernel::PartitionDesc &desc) noexcept {
  const kernel::PartitionPlan plan = kernel::PlanPartition(desc);
  AccelGraphNode node = accel_graph_factory_detail::PrimitiveNode(
      refs, ref_count, kernel::NodeKind::Partition, kernel::HashPartition(desc),
      desc.element_count);
  node.partition = desc;
  node.signature = kernel::GraphSignatureFor(plan);
  return node;
}

} // namespace rund
