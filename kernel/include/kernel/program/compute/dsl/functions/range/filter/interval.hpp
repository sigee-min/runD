#pragma once

#include <kernel/program/compute/dsl/functions/range/predicate.hpp>

namespace rund::compute_dsl::detail {

[[nodiscard]] inline ComputeValue
SortedIntervalPredicate(const ComputeValue value, const ComputeValue lo,
                        const ComputeValue hi) noexcept {
  return in_range(value, min(lo, hi), max(lo, hi));
}

} // namespace rund::compute_dsl::detail
