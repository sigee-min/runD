#pragma once

#include <accel/graph/factory/node/base.hpp>
#include <kernel/program/compute/stencil/identity.hpp>
#include <kernel/program/compute/stencil/plan.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelStencil(const AccelGraphBufferRef *const refs,
             const std::uint64_t ref_count,
             const kernel::StencilDesc &desc) noexcept {
  const kernel::StencilPlan plan = kernel::PlanStencil(desc);
  AccelGraphNode node = accel_graph_factory_detail::PrimitiveNode(
      refs, ref_count, kernel::NodeKind::Stencil, kernel::HashStencil(desc),
      desc.element_count);
  node.stencil = desc;
  node.signature = kernel::GraphSignatureFor(plan);
  return node;
}

} // namespace rund
