#pragma once

#include <accel/graph/factory/node/base.hpp>
#include <kernel/program/compute/factor/identity.hpp>
#include <kernel/program/compute/factor/plan.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelFactor(const AccelGraphBufferRef *const refs,
            const std::uint64_t ref_count,
            const kernel::FactorDesc &desc) noexcept {
  const kernel::FactorPlan plan = kernel::PlanFactor(desc);
  AccelGraphNode node = accel_graph_factory_detail::PrimitiveNode(
      refs, ref_count, kernel::NodeKind::Factor, kernel::HashFactor(desc),
      plan.factor_count);
  node.factor = desc;
  node.signature = kernel::GraphSignatureFor(plan);
  return node;
}

} // namespace rund
