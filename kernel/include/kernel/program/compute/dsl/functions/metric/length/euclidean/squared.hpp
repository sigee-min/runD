#pragma once

#include <kernel/program/compute/dsl/metric.hpp>

namespace rund::compute_dsl {
namespace detail {

[[nodiscard]] inline ComputeValue LengthSquared(const ComputeValue x,
                                                const ComputeValue y) noexcept {
  return dot(x, y, x, y);
}

[[nodiscard]] inline ComputeValue LengthSquared(const ComputeValue x,
                                                const ComputeValue y,
                                                const ComputeValue z) noexcept {
  return dot(x, y, z, x, y, z);
}

} // namespace detail

[[nodiscard]] inline ComputeValue len(const MetricOpSquared,
                                      const ComputeValue x,
                                      const ComputeValue y) noexcept {
  return detail::LengthSquared(x, y);
}

[[nodiscard]] inline ComputeValue len(const MetricOpSquared,
                                      const ComputeValue x,
                                      const ComputeValue y,
                                      const ComputeValue z) noexcept {
  return detail::LengthSquared(x, y, z);
}

} // namespace rund::compute_dsl
