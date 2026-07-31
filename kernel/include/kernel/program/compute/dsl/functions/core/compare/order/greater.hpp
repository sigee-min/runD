#pragma once

#include <kernel/program/compute/dsl/functions/core/compare/value.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue gt(const ComputeValue lhs,
                                     const ComputeValue rhs) noexcept {
  return detail::CompareValue(rund::kernel::IrOp::Gt, lhs, rhs);
}

template <detail::ConstantLiteral T>
[[nodiscard]] inline ComputeValue gt(const ComputeValue lhs,
                                     const T rhs) noexcept {
  return detail::CompareRight(rund::kernel::IrOp::Gt, lhs, rhs);
}

template <detail::ConstantLiteral T>
[[nodiscard]] inline ComputeValue gt(const T lhs,
                                     const ComputeValue rhs) noexcept {
  return detail::CompareLeft(rund::kernel::IrOp::Gt, lhs, rhs);
}

[[nodiscard]] inline ComputeValue ge(const ComputeValue lhs,
                                     const ComputeValue rhs) noexcept {
  return detail::CompareValue(rund::kernel::IrOp::Ge, lhs, rhs);
}

template <detail::ConstantLiteral T>
[[nodiscard]] inline ComputeValue ge(const ComputeValue lhs,
                                     const T rhs) noexcept {
  return detail::CompareRight(rund::kernel::IrOp::Ge, lhs, rhs);
}

template <detail::ConstantLiteral T>
[[nodiscard]] inline ComputeValue ge(const T lhs,
                                     const ComputeValue rhs) noexcept {
  return detail::CompareLeft(rund::kernel::IrOp::Ge, lhs, rhs);
}

} // namespace rund::compute_dsl
