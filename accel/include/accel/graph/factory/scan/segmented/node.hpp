#pragma once

#include <accel/graph/factory/node/base.hpp>
#include <kernel/program/compute/segmented/scan/identity.hpp>
#include <kernel/program/compute/segmented/scan/plan.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelSegmentedScan(const AccelGraphBufferRef *const refs,
                   const std::uint64_t ref_count,
                   const kernel::SegmentedScanDesc &desc) noexcept {
  const kernel::SegmentedScanPlan plan = kernel::PlanSegmentedScan(desc);
  AccelGraphNode node = accel_graph_factory_detail::PrimitiveNode(
      refs, ref_count, kernel::NodeKind::SegmentedScan,
      kernel::HashSegmentedScan(desc), desc.element_count);
  node.segmented_scan = desc;
  node.signature = kernel::GraphSignatureFor(plan);
  return node;
}

} // namespace rund
