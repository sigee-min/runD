#pragma once

#include <kernel/program/compute/dsl/functions/ratio/value.hpp>

namespace rund::compute_dsl::detail {

[[nodiscard]] inline ComputeValue
WeightedMeanRatio(const ComputeValue total,
                  const ComputeValue weight_sum) noexcept {
  return ratio(total, weight_sum);
}

} // namespace rund::compute_dsl::detail
