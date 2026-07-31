#pragma once

#include <accel/graph/factory/node/base.hpp>
#include <kernel/program/compute/solve/identity.hpp>
#include <kernel/program/compute/solve/plan.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelSolve(const AccelGraphBufferRef *const refs, const std::uint64_t ref_count,
           const kernel::SolveDesc &desc) noexcept {
  const kernel::SolvePlan plan = kernel::PlanSolve(desc);
  AccelGraphNode node = accel_graph_factory_detail::PrimitiveNode(
      refs, ref_count, kernel::NodeKind::Solve, kernel::HashSolve(desc),
      plan.output_count);
  node.solve = desc;
  node.signature = kernel::GraphSignatureFor(plan);
  return node;
}

} // namespace rund
