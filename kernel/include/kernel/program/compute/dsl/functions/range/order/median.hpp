#pragma once

#include <kernel/program/compute/dsl/functions/range/order/minmax.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue median(const ComputeValue a,
                                         const ComputeValue b,
                                         const ComputeValue c) noexcept {
  return max(min(a, b), min(max(a, b), c));
}

}  // namespace rund::compute_dsl
