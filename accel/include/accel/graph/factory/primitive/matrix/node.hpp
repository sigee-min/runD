#pragma once

#include <accel/graph/factory/node/base.hpp>
#include <kernel/program/compute/matrix/identity.hpp>
#include <kernel/program/compute/matrix/plan.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelMatrix(const AccelGraphBufferRef *const refs,
            const std::uint64_t ref_count,
            const kernel::MatrixDesc &desc) noexcept {
  const kernel::MatrixPlan plan = kernel::PlanMatrix(desc);
  AccelGraphNode node = accel_graph_factory_detail::PrimitiveNode(
      refs, ref_count, kernel::NodeKind::Matrix, kernel::HashMatrix(desc),
      plan.output_count);
  node.matrix = desc;
  node.signature = kernel::GraphSignatureFor(plan);
  return node;
}

} // namespace rund
