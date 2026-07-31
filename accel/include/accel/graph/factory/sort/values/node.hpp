#pragma once

#include <accel/graph/factory/node/base.hpp>
#include <kernel/program/compute/sort/identity.hpp>
#include <kernel/program/compute/sort/plan.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelSort(const AccelGraphBufferRef *const refs, const std::uint64_t ref_count,
          const kernel::SortDesc &desc) noexcept {
  const kernel::SortPlan plan = kernel::PlanSort(desc);
  AccelGraphNode node = accel_graph_factory_detail::PrimitiveNode(
      refs, ref_count, kernel::NodeKind::Sort, kernel::HashSort(desc),
      desc.element_count);
  node.sort = desc;
  node.signature = kernel::GraphSignatureFor(plan);
  return node;
}

} // namespace rund
