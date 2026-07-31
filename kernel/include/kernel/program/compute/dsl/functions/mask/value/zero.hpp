#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue is_zero(const ComputeValue value) noexcept {
  return eq(value, 0);
}

[[nodiscard]] inline ComputeValue nonzero(const ComputeValue value) noexcept {
  return ne(value, 0);
}

} // namespace rund::compute_dsl
