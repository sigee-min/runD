#pragma once

#include <kernel/program/compute/dsl/expression/literal.hpp>

namespace rund::compute_dsl::detail {

[[nodiscard]] Expr operator+(Expr lhs, Expr rhs) noexcept;
[[nodiscard]] Expr operator-(Expr lhs, Expr rhs) noexcept;
[[nodiscard]] Expr operator-(Expr value) noexcept;
[[nodiscard]] Expr operator*(Expr lhs, Expr rhs) noexcept;
[[nodiscard]] Expr operator/(Expr lhs, Expr rhs) noexcept;

template <ConstantLiteral T>
[[nodiscard]] inline Expr operator+(const Expr lhs, const T rhs) noexcept {
  return Binary(rund::kernel::IrOp::Add, lhs, ConstantValue(lhs, rhs));
}

template <ConstantLiteral T>
[[nodiscard]] inline Expr operator+(const T lhs, const Expr rhs) noexcept {
  return Binary(rund::kernel::IrOp::Add, ConstantValue(rhs, lhs), rhs);
}

template <ConstantLiteral T>
[[nodiscard]] inline Expr operator-(const Expr lhs, const T rhs) noexcept {
  return Binary(rund::kernel::IrOp::Sub, lhs, ConstantValue(lhs, rhs));
}

template <ConstantLiteral T>
[[nodiscard]] inline Expr operator-(const T lhs, const Expr rhs) noexcept {
  return Binary(rund::kernel::IrOp::Sub, ConstantValue(rhs, lhs), rhs);
}

template <ConstantLiteral T>
[[nodiscard]] inline Expr operator*(const Expr lhs, const T rhs) noexcept {
  return Binary(rund::kernel::IrOp::Mul, lhs, ConstantValue(lhs, rhs));
}

template <ConstantLiteral T>
[[nodiscard]] inline Expr operator*(const T lhs, const Expr rhs) noexcept {
  return Binary(rund::kernel::IrOp::Mul, ConstantValue(rhs, lhs), rhs);
}

template <ConstantLiteral T>
[[nodiscard]] inline Expr operator/(const Expr lhs, const T rhs) noexcept {
  return lhs / ConstantValue(lhs, rhs);
}

template <ConstantLiteral T>
[[nodiscard]] inline Expr operator/(const T lhs, const Expr rhs) noexcept {
  return ConstantValue(rhs, lhs) / rhs;
}

} // namespace rund::compute_dsl::detail
