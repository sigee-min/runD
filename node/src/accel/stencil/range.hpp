#pragma once

#include "../range_aggregate/model/plan.hpp"

#include <kernel/program/compute/stencil/plan.hpp>

#include <optional>

namespace rund::node::accel::detail {

// Kernel Stencil semantics remain the graph descriptor and hash authority.
// This projection derives the generic Range semantic input without selecting
// or executing a physical candidate.
[[nodiscard]] constexpr std::optional<RangeShape>
ProjectStencilRange(const rund::kernel::StencilPlan &semantic,
                    const rund::kernel::ComputeDomain domain) noexcept {
  if (!semantic.ok ||
      semantic.boundary != rund::kernel::StencilBoundary::Clamp) {
    return std::nullopt;
  }

  std::optional<RangeTraits> traits;
  switch (semantic.op) {
  case rund::kernel::StencilOp::Sum:
    traits = RangeTraits::sum_modulo(domain);
    break;
  case rund::kernel::StencilOp::Min:
    traits = RangeTraits::minimum(domain);
    break;
  case rund::kernel::StencilOp::Max:
    traits = RangeTraits::maximum(domain);
    break;
  }
  rund::kernel::u64 twice_radius = 0u;
  rund::kernel::u64 window_size = 0u;
  return traits.has_value() &&
                 rund::kernel::checked::mul(semantic.radius, 2u,
                                            twice_radius) &&
                 rund::kernel::checked::add(twice_radius, 1u, window_size)
             ? RangeShape::affine(
                   *traits, RangeBoundary::Clamp, semantic.element_count,
                   semantic.element_count, window_size, 1u, semantic.radius,
                   static_cast<rund::kernel::u32>(semantic.element_bytes))
             : std::nullopt;
}

[[nodiscard]] constexpr bool
StencilRangeMatches(const rund::kernel::StencilPlan &semantic,
                    const rund::kernel::ComputeDomain domain,
                    const RangePlan &range) noexcept {
  const std::optional<RangeShape> projected =
      ProjectStencilRange(semantic, domain);
  return projected.has_value() && range.ok() &&
         range.shape().input_count() == projected->input_count() &&
         range.shape().output_count() == projected->output_count() &&
         range.shape().window_size() == projected->window_size() &&
         range.shape().stride() == projected->stride() &&
         range.shape().padding() == projected->padding() &&
         range.shape().boundary() == projected->boundary() &&
         range.shape().count() == projected->count() &&
         range.shape().element_bytes() == projected->element_bytes() &&
         range.shape().traits().operation() ==
             projected->traits().operation() &&
         range.shape().traits().domain() == projected->traits().domain() &&
         range.shape().traits().arithmetic_law() ==
             projected->traits().arithmetic_law();
}

} // namespace rund::node::accel::detail
