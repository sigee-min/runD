#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue operator+(const ComputeValue lhs,
                                            const ComputeValue rhs) noexcept {
  return detail::operator+(lhs, rhs);
}

[[nodiscard]] inline ComputeValue operator-(const ComputeValue lhs,
                                            const ComputeValue rhs) noexcept {
  return detail::operator-(lhs, rhs);
}

[[nodiscard]] inline ComputeValue operator*(const ComputeValue lhs,
                                            const ComputeValue rhs) noexcept {
  return detail::operator*(lhs, rhs);
}

[[nodiscard]] inline ComputeValue operator-(const ComputeValue value) noexcept {
  return detail::operator-(value);
}

} // namespace rund::compute_dsl
