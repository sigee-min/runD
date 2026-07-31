#pragma once

#include <kernel/program/compute/dsl/functions/metric/length/euclidean/squared.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue len(const ComputeValue x,
                                      const ComputeValue y) noexcept {
  return sqrt(len(MetricOp::Squared, x, y));
}

[[nodiscard]] inline ComputeValue len(const ComputeValue x,
                                      const ComputeValue y,
                                      const ComputeValue z) noexcept {
  return sqrt(len(MetricOp::Squared, x, y, z));
}

} // namespace rund::compute_dsl
