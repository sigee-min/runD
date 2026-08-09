#pragma once

#include "../kernel/bindings/range.hpp"
#include "../primitive/shape.hpp"
#include "../range_aggregate/execution.hpp"

#include <kernel/program/compute/stencil/plan.hpp>

#include <cstdint>
#include <optional>

namespace rund::node::accel::detail {

// Kernel Stencil semantics remain the graph descriptor and hash authority.
// This adapter proves that the frozen physical plan came from unchanged
// primitive semantics before the Range execution contract is used.
[[nodiscard]] constexpr std::optional<RangeShape>
StencilRangeShape(const rund::kernel::StencilPlan &semantic,
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
StencilRangePlanMatches(const rund::kernel::StencilPlan &semantic,
                        const rund::kernel::ComputeDomain domain,
                        const RangePlan &range) noexcept {
  const std::optional<RangeShape> shape = StencilRangeShape(semantic, domain);
  return shape.has_value() && range.ok() &&
         range.shape().input_count() == shape->input_count() &&
         range.shape().output_count() == shape->output_count() &&
         range.shape().window_size() == shape->window_size() &&
         range.shape().stride() == shape->stride() &&
         range.shape().padding() == shape->padding() &&
         range.shape().boundary() == shape->boundary() &&
         range.shape().count() == shape->count() &&
         range.shape().element_bytes() == shape->element_bytes() &&
         range.shape().traits().operation() == shape->traits().operation() &&
         range.shape().traits().domain() == shape->traits().domain() &&
         range.shape().traits().arithmetic_law() ==
             shape->traits().arithmetic_law();
}

[[nodiscard]] bool StencilShapeOk(const rund::kernel::StencilDesc &desc,
                                  const rund::kernel::StencilPlan &plan,
                                  const RangeBinds &bindings) noexcept;

} // namespace rund::node::accel::detail
