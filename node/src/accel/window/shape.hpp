#pragma once

#include "../kernel/bindings/range.hpp"
#include "../primitive/shape.hpp"
#include "../range_aggregate/model/plan.hpp"

#include <kernel/program/compute/window/plan.hpp>

#include <optional>

namespace rund::node::accel::detail {

[[nodiscard]] constexpr std::optional<RangeShape>
WindowRangeShape(const rund::kernel::WindowPlan &semantic) noexcept {
  if (!semantic.ok) {
    return std::nullopt;
  }

  std::optional<RangeTraits> traits;
  switch (semantic.op) {
  case rund::kernel::WindowOp::Sum:
    if (semantic.domain == rund::kernel::ComputeDomain::Fixed &&
        semantic.fixed_format.overflow != rund::kernel::ComputeOverflow::Wrap) {
      traits = RangeTraits::sum_saturating(semantic.domain);
    } else {
      traits = RangeTraits::sum_modulo(semantic.domain);
    }
    break;
  case rund::kernel::WindowOp::Min:
    traits = RangeTraits::minimum(semantic.domain);
    break;
  case rund::kernel::WindowOp::Max:
    traits = RangeTraits::maximum(semantic.domain);
    break;
  }
  if (!traits.has_value()) {
    return std::nullopt;
  }

  const RangeBoundary boundary =
      semantic.boundary == rund::kernel::WindowBoundary::Clamp
          ? RangeBoundary::Clamp
          : RangeBoundary::Clip;
  const RangeCount count =
      semantic.count_source == rund::kernel::ComputeCountSource::BufferU32
          ? RangeCount::U32
          : (semantic.count_source ==
                     rund::kernel::ComputeCountSource::BufferU64
                 ? RangeCount::U64
                 : RangeCount::Descriptor);
  return RangeShape::affine(
      *traits, boundary, semantic.input_count, semantic.output_count,
      semantic.window_size, semantic.stride, semantic.pad_left,
      static_cast<rund::kernel::u32>(semantic.element_bytes), count);
}

[[nodiscard]] constexpr bool
WindowRangePlanMatches(const rund::kernel::WindowPlan &semantic,
                       const RangePlan &range) noexcept {
  const std::optional<RangeShape> shape = WindowRangeShape(semantic);
  if (!shape.has_value() || !range.ok()) {
    return false;
  }
  const RangeShape &actual = range.shape();
  return actual.boundary() == shape->boundary() &&
         actual.count() == shape->count() &&
         actual.input_count() == shape->input_count() &&
         actual.output_count() == shape->output_count() &&
         actual.window_size() == shape->window_size() &&
         actual.stride() == shape->stride() &&
         actual.padding() == shape->padding() &&
         actual.element_bytes() == shape->element_bytes() &&
         actual.traits().operation() == shape->traits().operation() &&
         actual.traits().domain() == shape->traits().domain() &&
         actual.traits().arithmetic_law() == shape->traits().arithmetic_law();
}

[[nodiscard]] bool WindowShapeOk(const rund::kernel::WindowDesc &desc,
                                 const rund::kernel::WindowPlan &plan,
                                 const RangeBinds &bindings) noexcept;

} // namespace rund::node::accel::detail
