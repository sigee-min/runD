#pragma once

#include <accel/graph/factory/node/base.hpp>
#include <kernel/program/compute/window/identity.hpp>
#include <kernel/program/compute/window/plan.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelWindow(const AccelGraphBufferRef *const refs,
            const std::uint64_t ref_count,
            const kernel::WindowDesc &desc) noexcept {
  const kernel::WindowPlan plan = kernel::PlanWindow(desc);
  AccelGraphNode node = accel_graph_factory_detail::PrimitiveNode(
      refs, ref_count, kernel::NodeKind::Window, kernel::HashWindow(desc),
      desc.output_count);
  node.window = desc;
  node.signature = kernel::GraphSignatureFor(plan);
  return node;
}

} // namespace rund
