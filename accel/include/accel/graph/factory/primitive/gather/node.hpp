#pragma once

#include <accel/graph/factory/node/base.hpp>
#include <kernel/program/compute/gather/identity.hpp>
#include <kernel/program/compute/gather/plan.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelGather(const AccelGraphBufferRef *const refs,
            const std::uint64_t ref_count,
            const kernel::GatherDesc &desc) noexcept {
  const kernel::GatherPlan plan = kernel::PlanGather(desc);
  AccelGraphNode node = accel_graph_factory_detail::PrimitiveNode(
      refs, ref_count, kernel::NodeKind::Gather, kernel::HashGather(desc),
      desc.element_count);
  node.gather = desc;
  node.signature = kernel::GraphSignatureFor(plan);
  return node;
}

} // namespace rund
