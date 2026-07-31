#pragma once

#include <kernel/program/compute/dsl/functions/core/compare/value.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue lt(const ComputeValue lhs,
                                     const ComputeValue rhs) noexcept {
  return detail::CompareValue(rund::kernel::IrOp::Lt, lhs, rhs);
}

template <detail::ConstantLiteral T>
[[nodiscard]] inline ComputeValue lt(const ComputeValue lhs,
                                     const T rhs) noexcept {
  return detail::CompareRight(rund::kernel::IrOp::Lt, lhs, rhs);
}

template <detail::ConstantLiteral T>
[[nodiscard]] inline ComputeValue lt(const T lhs,
                                     const ComputeValue rhs) noexcept {
  return detail::CompareLeft(rund::kernel::IrOp::Lt, lhs, rhs);
}

[[nodiscard]] inline ComputeValue le(const ComputeValue lhs,
                                     const ComputeValue rhs) noexcept {
  return detail::CompareValue(rund::kernel::IrOp::Le, lhs, rhs);
}

template <detail::ConstantLiteral T>
[[nodiscard]] inline ComputeValue le(const ComputeValue lhs,
                                     const T rhs) noexcept {
  return detail::CompareRight(rund::kernel::IrOp::Le, lhs, rhs);
}

template <detail::ConstantLiteral T>
[[nodiscard]] inline ComputeValue le(const T lhs,
                                     const ComputeValue rhs) noexcept {
  return detail::CompareLeft(rund::kernel::IrOp::Le, lhs, rhs);
}

} // namespace rund::compute_dsl
