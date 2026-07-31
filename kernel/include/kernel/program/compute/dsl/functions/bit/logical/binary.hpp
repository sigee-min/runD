#pragma once

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue bit_and(const ComputeValue lhs,
                                          const ComputeValue rhs) noexcept {
  return detail::Binary(rund::kernel::IrOp::BitAnd, lhs, rhs);
}

[[nodiscard]] inline ComputeValue bit_or(const ComputeValue lhs,
                                         const ComputeValue rhs) noexcept {
  return detail::Binary(rund::kernel::IrOp::BitOr, lhs, rhs);
}

[[nodiscard]] inline ComputeValue bit_xor(const ComputeValue lhs,
                                          const ComputeValue rhs) noexcept {
  return detail::Binary(rund::kernel::IrOp::BitXor, lhs, rhs);
}

} // namespace rund::compute_dsl
