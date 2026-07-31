#pragma once

#include <kernel/program/compute/dsl/functions/range/order/minmax.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue spread(const ComputeValue a,
                                         const ComputeValue b,
                                         const ComputeValue c) noexcept {
  return sub_sat(max(a, b, c), min(a, b, c));
}

}  // namespace rund::compute_dsl
