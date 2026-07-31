#pragma once

#include <kernel/program/compute/dsl/functions/mask/select.hpp>
#include <kernel/program/compute/dsl/functions/range/filter/interval.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue bandstop(const ComputeValue value,
                                           const ComputeValue lo,
                                           const ComputeValue hi) noexcept {
  return zero_if(value, detail::SortedIntervalPredicate(value, lo, hi));
}

} // namespace rund::compute_dsl
