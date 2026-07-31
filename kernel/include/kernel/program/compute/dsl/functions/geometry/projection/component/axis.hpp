#pragma once

#include <kernel/program/compute/dsl/axis.hpp>
#include <kernel/program/compute/dsl/functions/geometry/projection/component/value.hpp>
#include <kernel/program/compute/dsl/functions/linear/dot/value.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
proj(Axis::XTag, const ComputeValue ax, const ComputeValue ay,
     const ComputeValue bx, const ComputeValue by) noexcept {
  const ComputeValue dot_value = dot(ax, ay, bx, by);
  const ComputeValue denom = dot(bx, by, bx, by);
  return detail::ProjectComponent(bx, dot_value, denom);
}

[[nodiscard]] inline ComputeValue
proj(Axis::YTag, const ComputeValue ax, const ComputeValue ay,
     const ComputeValue bx, const ComputeValue by) noexcept {
  const ComputeValue dot_value = dot(ax, ay, bx, by);
  const ComputeValue denom = dot(bx, by, bx, by);
  return detail::ProjectComponent(by, dot_value, denom);
}

[[nodiscard]] inline ComputeValue
proj(Axis::XTag, const ComputeValue ax, const ComputeValue ay,
     const ComputeValue az, const ComputeValue bx, const ComputeValue by,
     const ComputeValue bz) noexcept {
  const ComputeValue dot_value = dot(ax, ay, az, bx, by, bz);
  const ComputeValue denom = dot(bx, by, bz, bx, by, bz);
  return detail::ProjectComponent(bx, dot_value, denom);
}

[[nodiscard]] inline ComputeValue
proj(Axis::YTag, const ComputeValue ax, const ComputeValue ay,
     const ComputeValue az, const ComputeValue bx, const ComputeValue by,
     const ComputeValue bz) noexcept {
  const ComputeValue dot_value = dot(ax, ay, az, bx, by, bz);
  const ComputeValue denom = dot(bx, by, bz, bx, by, bz);
  return detail::ProjectComponent(by, dot_value, denom);
}

[[nodiscard]] inline ComputeValue
proj(Axis::ZTag, const ComputeValue ax, const ComputeValue ay,
     const ComputeValue az, const ComputeValue bx, const ComputeValue by,
     const ComputeValue bz) noexcept {
  const ComputeValue dot_value = dot(ax, ay, az, bx, by, bz);
  const ComputeValue denom = dot(bx, by, bz, bx, by, bz);
  return detail::ProjectComponent(bz, dot_value, denom);
}

} // namespace rund::compute_dsl
