#pragma once

#include <kernel/program/compute/dsl/axis.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
proportion(Axis::XTag, const ComputeValue x,
           const ComputeValue y) noexcept {
  return proportion(x, sum(x, y));
}

[[nodiscard]] inline ComputeValue
proportion(Axis::YTag, const ComputeValue x,
           const ComputeValue y) noexcept {
  return proportion(y, sum(x, y));
}

[[nodiscard]] inline ComputeValue
proportion(Axis::XTag, const ComputeValue x, const ComputeValue y,
           const ComputeValue z) noexcept {
  return proportion(x, sum(x, y, z));
}

[[nodiscard]] inline ComputeValue
proportion(Axis::YTag, const ComputeValue x, const ComputeValue y,
           const ComputeValue z) noexcept {
  return proportion(y, sum(x, y, z));
}

[[nodiscard]] inline ComputeValue
proportion(Axis::ZTag, const ComputeValue x, const ComputeValue y,
           const ComputeValue z) noexcept {
  return proportion(z, sum(x, y, z));
}

} // namespace rund::compute_dsl
