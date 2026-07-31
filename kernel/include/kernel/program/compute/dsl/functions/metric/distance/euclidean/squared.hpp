#pragma once

#include <kernel/program/compute/dsl/functions/metric/length/euclidean.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
dist(const MetricOpSquared op, const ComputeValue ax, const ComputeValue ay,
     const ComputeValue bx, const ComputeValue by) noexcept {
  return len(op, sub_sat(ax, bx), sub_sat(ay, by));
}

[[nodiscard]] inline ComputeValue
dist(const MetricOpSquared op, const ComputeValue ax, const ComputeValue ay,
     const ComputeValue az, const ComputeValue bx, const ComputeValue by,
     const ComputeValue bz) noexcept {
  return len(op, sub_sat(ax, bx), sub_sat(ay, by), sub_sat(az, bz));
}

} // namespace rund::compute_dsl
