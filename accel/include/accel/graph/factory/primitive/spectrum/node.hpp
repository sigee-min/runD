#pragma once

#include <accel/graph/factory/node/base.hpp>
#include <kernel/program/compute/spectrum/identity.hpp>
#include <kernel/program/compute/spectrum/plan.hpp>

namespace rund {

[[nodiscard]] inline AccelGraphNode
AccelSpectrum(const AccelGraphBufferRef *const refs,
              const std::uint64_t ref_count,
              const kernel::SpectrumDesc &desc) noexcept {
  const kernel::SpectrumPlan plan = kernel::PlanSpectrum(desc);
  AccelGraphNode node = accel_graph_factory_detail::PrimitiveNode(
      refs, ref_count, kernel::NodeKind::Spectrum, kernel::HashSpectrum(desc),
      plan.value_count);
  node.spectrum = desc;
  node.signature = kernel::GraphSignatureFor(plan);
  return node;
}

} // namespace rund
