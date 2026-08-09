#pragma once

#include "../kernel/bindings/range.hpp"
#include "../primitive/shape.hpp"
#include "../range_aggregate/execution.hpp"

#include <kernel/program/compute/stencil/plan.hpp>

#include <cstdint>
#include <optional>

namespace rund::node::accel::detail {

// Stencil owns this semantic-width projection.  The generic executor consumes
// the already-frozen RangeAggregate shape and never reinterprets a Stencil
// descriptor or public binding.
[[nodiscard]] constexpr rund::kernel::u32
StencilElementBytes(const rund::kernel::StencilElement element) noexcept {
  if (element == rund::kernel::StencilElement::U32) {
    return 4u;
  }
  return element == rund::kernel::StencilElement::U64 ? 8u : 0u;
}

// Kernel Stencil semantics remain the graph descriptor and hash authority.
// This adapter proves that the frozen physical plan came from unchanged
// primitive semantics before the RangeAggregate execution contract is used.
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
  return traits.has_value()
             ? RangeShape::window(
                   *traits, RangeBoundary::Clamp, semantic.element_count,
                   semantic.radius,
                   static_cast<rund::kernel::u32>(semantic.element_bytes))
             : std::nullopt;
}

[[nodiscard]] constexpr bool
StencilRangePlanMatches(const rund::kernel::StencilPlan &semantic,
                        const rund::kernel::ComputeDomain domain,
                        const RangePlan &range) noexcept {
  const std::optional<RangeShape> shape = StencilRangeShape(semantic, domain);
  return shape.has_value() && range.ok() &&
         range.shape().element_count() == shape->element_count() &&
         range.shape().radius() == shape->radius() &&
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
