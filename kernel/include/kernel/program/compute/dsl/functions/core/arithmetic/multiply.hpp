#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue mul_fixed(const ComputeValue lhs,
                                            const ComputeValue rhs) noexcept {
  return detail::Binary(rund::kernel::IrOp::MulFixed, lhs, rhs);
}

[[nodiscard]] inline ComputeValue
mul_fixed_scaled(const ComputeValue value,
                 const ComputeValue scaled_coefficient) noexcept {
  return detail::Binary(rund::kernel::IrOp::MulFixedScaled, value,
                        scaled_coefficient);
}

[[nodiscard]] inline ComputeValue
mul_unsigned_fixed(const ComputeValue lhs, const ComputeValue rhs) noexcept {
  return detail::Binary(rund::kernel::IrOp::MulUnsignedFixed, lhs, rhs);
}

[[nodiscard]] inline ComputeValue
mul_add_fixed(const ComputeValue lhs, const ComputeValue rhs,
              const ComputeValue addend) noexcept {
  return detail::Ternary(rund::kernel::IrOp::MulAddFixed, lhs, rhs, addend);
}

} // namespace rund::compute_dsl
