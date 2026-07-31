#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue is_neg(const ComputeValue value) noexcept {
  return lt(value, 0);
}

[[nodiscard]] inline ComputeValue is_pos(const ComputeValue value) noexcept {
  return gt(value, 0);
}

[[nodiscard]] inline ComputeValue is_nonneg(const ComputeValue value) noexcept {
  return ge(value, 0);
}

[[nodiscard]] inline ComputeValue is_nonpos(const ComputeValue value) noexcept {
  return le(value, 0);
}

} // namespace rund::compute_dsl
