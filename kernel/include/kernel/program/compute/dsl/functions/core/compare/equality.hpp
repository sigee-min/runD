#pragma once

#include <kernel/program/compute/dsl/functions/core/compare/value.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue eq(const ComputeValue lhs,
                                     const ComputeValue rhs) noexcept {
  return detail::CompareValue(rund::kernel::IrOp::Eq, lhs, rhs);
}

template <detail::ConstantLiteral T>
[[nodiscard]] inline ComputeValue eq(const ComputeValue lhs,
                                     const T rhs) noexcept {
  return detail::CompareRight(rund::kernel::IrOp::Eq, lhs, rhs);
}

template <detail::ConstantLiteral T>
[[nodiscard]] inline ComputeValue eq(const T lhs,
                                     const ComputeValue rhs) noexcept {
  return detail::CompareLeft(rund::kernel::IrOp::Eq, lhs, rhs);
}

[[nodiscard]] inline ComputeValue ne(const ComputeValue lhs,
                                     const ComputeValue rhs) noexcept {
  return detail::CompareValue(rund::kernel::IrOp::Ne, lhs, rhs);
}

template <detail::ConstantLiteral T>
[[nodiscard]] inline ComputeValue ne(const ComputeValue lhs,
                                     const T rhs) noexcept {
  return detail::CompareRight(rund::kernel::IrOp::Ne, lhs, rhs);
}

template <detail::ConstantLiteral T>
[[nodiscard]] inline ComputeValue ne(const T lhs,
                                     const ComputeValue rhs) noexcept {
  return detail::CompareLeft(rund::kernel::IrOp::Ne, lhs, rhs);
}

} // namespace rund::compute_dsl
