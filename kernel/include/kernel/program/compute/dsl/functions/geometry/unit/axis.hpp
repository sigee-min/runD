#pragma once

#include <kernel/program/compute/dsl/axis.hpp>
#include <kernel/program/compute/dsl/functions/geometry/unit/component.hpp>
#include <kernel/program/compute/dsl/functions/metric/length/euclidean.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue unit(Axis::XTag, const ComputeValue x,
                                       const ComputeValue y) noexcept {
  return detail::UnitComponent(x, len(x, y));
}

[[nodiscard]] inline ComputeValue unit(Axis::YTag, const ComputeValue x,
                                       const ComputeValue y) noexcept {
  return detail::UnitComponent(y, len(x, y));
}

[[nodiscard]] inline ComputeValue unit(Axis::XTag, const ComputeValue x,
                                       const ComputeValue y,
                                       const ComputeValue z) noexcept {
  return detail::UnitComponent(x, len(x, y, z));
}

[[nodiscard]] inline ComputeValue unit(Axis::YTag, const ComputeValue x,
                                       const ComputeValue y,
                                       const ComputeValue z) noexcept {
  return detail::UnitComponent(y, len(x, y, z));
}

[[nodiscard]] inline ComputeValue unit(Axis::ZTag, const ComputeValue x,
                                       const ComputeValue y,
                                       const ComputeValue z) noexcept {
  return detail::UnitComponent(z, len(x, y, z));
}

} // namespace rund::compute_dsl
