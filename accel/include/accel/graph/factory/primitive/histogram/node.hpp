#pragma once

#include <accel/graph/factory/node/base.hpp>
#include <kernel/program/compute/histogram/identity.hpp>
#include <kernel/program/compute/histogram/plan.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelHistogram(const AccelGraphBufferRef *const refs,
               const std::uint64_t ref_count,
               const kernel::HistogramDesc &desc) noexcept {
  const kernel::HistogramPlan plan = kernel::PlanHistogram(desc);
  AccelGraphNode node = accel_graph_factory_detail::PrimitiveNode(
      refs, ref_count, kernel::NodeKind::Histogram, kernel::HashHistogram(desc),
      desc.element_count);
  node.histogram = desc;
  node.signature = kernel::GraphSignatureFor(plan);
  return node;
}

} // namespace rund
