#pragma once

#include <kernel/program/compute/dsl/functions/stats/constants.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue huber(const ComputeValue value,
                                        const ComputeValue delta) noexcept {
  const ComputeValue bound = detail::StorageQuantize(abs(delta));
  const ComputeValue magnitude = detail::StorageQuantize(abs(value));
  const ComputeValue half = fixed(FixedOp::Half, value);
  const ComputeValue quadratic = mul_fixed(half, mul_fixed(value, value));
  const ComputeValue linear_offset = mul_fixed(half, bound);
  const ComputeValue linear = mul_fixed(bound, sub_sat(magnitude, linear_offset));
  return select(le(magnitude, bound), quadratic, linear);
}

} // namespace rund::compute_dsl
