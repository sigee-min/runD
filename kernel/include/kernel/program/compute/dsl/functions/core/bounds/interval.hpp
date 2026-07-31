#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue clamp(const ComputeValue value,
                                        const ComputeValue lo,
                                        const ComputeValue hi) noexcept {
  return detail::Ternary(rund::kernel::IrOp::Clamp, value, lo, hi);
}

[[nodiscard]] inline ComputeValue saturate(const ComputeValue value) noexcept {
  return clamp(value, fixed_zero(value), fixed_one(value));
}

} // namespace rund::compute_dsl
