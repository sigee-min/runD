#pragma once

#include <kernel/program/compute/dsl/functions/range/order/minmax.hpp>
#include <kernel/program/compute/dsl/functions/stats/constants.hpp>
#include <kernel/program/compute/dsl/functions/stats/mean/value.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue midrange(const ComputeValue a,
                                           const ComputeValue b,
                                           const ComputeValue c) noexcept {
  return mean(min(a, b, c), max(a, b, c));
}

}  // namespace rund::compute_dsl
