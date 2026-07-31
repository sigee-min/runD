#pragma once

namespace rund::compute_dsl::detail {

[[nodiscard]] inline ComputeValue CompareValue(const rund::kernel::IrOp op,
                                               const ComputeValue lhs,
                                               const ComputeValue rhs) noexcept {
  return Binary(op, lhs, rhs);
}

template <ConstantLiteral T>
[[nodiscard]] inline ComputeValue CompareRight(const rund::kernel::IrOp op,
                                               const ComputeValue lhs,
                                               const T rhs) noexcept {
  return CompareValue(op, lhs, ConstantValue(lhs, rhs));
}

template <ConstantLiteral T>
[[nodiscard]] inline ComputeValue CompareLeft(const rund::kernel::IrOp op,
                                              const T lhs,
                                              const ComputeValue rhs) noexcept {
  return CompareValue(op, ConstantValue(rhs, lhs), rhs);
}

} // namespace rund::compute_dsl::detail
