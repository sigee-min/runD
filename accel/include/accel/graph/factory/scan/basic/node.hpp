#pragma once

#include <accel/graph/factory/node/base.hpp>
#include <kernel/program/compute/scan/identity.hpp>
#include <kernel/program/compute/scan/plan.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelScan(const AccelGraphBufferRef *const refs, const std::uint64_t ref_count,
          const kernel::ScanDesc &desc) noexcept {
  const kernel::ScanPlan plan = kernel::PlanScan(desc);
  AccelGraphNode node = accel_graph_factory_detail::PrimitiveNode(
      refs, ref_count, kernel::NodeKind::Scan, kernel::HashScan(desc),
      desc.element_count);
  node.scan = desc;
  node.signature = kernel::GraphSignatureFor(plan);
  return node;
}

} // namespace rund
