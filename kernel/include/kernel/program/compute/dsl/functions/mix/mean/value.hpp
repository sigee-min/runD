#pragma once

#include <kernel/program/compute/dsl/functions/aggregate/sum.hpp>
#include <kernel/program/compute/dsl/functions/mix/sum.hpp>
#include <kernel/program/compute/dsl/functions/mix/mean/ratio.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
weighted_mean(const ComputeValue a, const ComputeValue b,
              const ComputeValue wa, const ComputeValue wb) noexcept {
  return detail::WeightedMeanRatio(mix(a, b, wa, wb), sum(wa, wb));
}

[[nodiscard]] inline ComputeValue
weighted_mean(const ComputeValue a, const ComputeValue b, const ComputeValue c,
              const ComputeValue wa, const ComputeValue wb,
              const ComputeValue wc) noexcept {
  return detail::WeightedMeanRatio(mix(a, b, c, wa, wb, wc), sum(wa, wb, wc));
}

[[nodiscard]] inline ComputeValue
weighted_mean(const ComputeValue a, const ComputeValue b, const ComputeValue c,
              const ComputeValue d, const ComputeValue wa,
              const ComputeValue wb, const ComputeValue wc,
              const ComputeValue wd) noexcept {
  return detail::WeightedMeanRatio(mix(a, b, c, d, wa, wb, wc, wd),
                                   sum(wa, wb, wc, wd));
}

} // namespace rund::compute_dsl
