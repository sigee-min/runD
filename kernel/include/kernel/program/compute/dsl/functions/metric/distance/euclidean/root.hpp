#pragma once

#include <kernel/program/compute/dsl/functions/metric/distance/euclidean/squared.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
dist(const ComputeValue ax, const ComputeValue ay, const ComputeValue bx,
     const ComputeValue by) noexcept {
  return sqrt(dist(MetricOp::Squared, ax, ay, bx, by));
}

[[nodiscard]] inline ComputeValue
dist(const ComputeValue ax, const ComputeValue ay, const ComputeValue az,
     const ComputeValue bx, const ComputeValue by,
     const ComputeValue bz) noexcept {
  return sqrt(dist(MetricOp::Squared, ax, ay, az, bx, by, bz));
}

} // namespace rund::compute_dsl
