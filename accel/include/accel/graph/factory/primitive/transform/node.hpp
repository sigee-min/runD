#pragma once

#include <accel/graph/factory/node/base.hpp>
#include <kernel/program/compute/transform/identity.hpp>
#include <kernel/program/compute/transform/plan.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelTransform(const AccelGraphBufferRef *const refs,
               const std::uint64_t ref_count,
               const kernel::TransformDesc &desc) noexcept {
  const kernel::TransformPlan plan = kernel::PlanTransform(desc);
  AccelGraphNode node = accel_graph_factory_detail::PrimitiveNode(
      refs, ref_count, kernel::NodeKind::Transform, kernel::HashTransform(desc),
      desc.element_count);
  node.transform = desc;
  node.signature = kernel::GraphSignatureFor(plan);
  return node;
}

} // namespace rund
