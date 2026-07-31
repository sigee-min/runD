#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
softsign(const ComputeValue value, const ComputeValue scale) noexcept {
  const ComputeValue positive_scale =
      max(detail::StorageQuantize(abs(scale)),
          fixed(FixedOp::Half, value));
  const ComputeValue denom =
      add_sat(positive_scale,
              mul_fixed(detail::StorageQuantize(abs(value)),
                        fixed(FixedOp::Half, value)));
  return ratio(value, denom);
}

[[nodiscard]] inline ComputeValue softsign(const ComputeValue value) noexcept {
  return softsign(value, fixed(FixedOp::Half, value));
}

} // namespace rund::compute_dsl
