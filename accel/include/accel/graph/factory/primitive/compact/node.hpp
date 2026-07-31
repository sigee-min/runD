#pragma once

#include <accel/graph/factory/node/base.hpp>
#include <kernel/program/compute/compact/identity.hpp>
#include <kernel/program/compute/compact/plan.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelCompact(const AccelGraphBufferRef *const refs,
             const std::uint64_t ref_count,
             const kernel::CompactDesc &desc) noexcept {
  const kernel::CompactPlan plan = kernel::PlanCompact(desc);
  AccelGraphNode node = accel_graph_factory_detail::PrimitiveNode(
      refs, ref_count, kernel::NodeKind::Compact, kernel::HashCompact(desc),
      desc.element_count);
  node.compact = desc;
  node.signature = kernel::GraphSignatureFor(plan);
  return node;
}

} // namespace rund
