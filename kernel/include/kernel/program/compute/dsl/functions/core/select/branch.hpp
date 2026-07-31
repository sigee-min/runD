#pragma once

#include <kernel/program/compute/dsl/functions/core/select/value.hpp>

namespace rund::compute_dsl {

[[nodiscard]] inline ComputeValue
select(const ComputeValue condition, const ComputeValue when_true,
       const ComputeValue when_false) noexcept {
  return detail::SelectValue(condition, when_true, when_false);
}

template <detail::ConstantLiteral T>
[[nodiscard]] inline ComputeValue select(const ComputeValue condition,
                                         const ComputeValue when_true,
                                         const T when_false) noexcept {
  return detail::SelectFalse(condition, when_true, when_false);
}

template <detail::ConstantLiteral T>
[[nodiscard]] inline ComputeValue
select(const ComputeValue condition, const T when_true,
       const ComputeValue when_false) noexcept {
  return detail::SelectTrue(condition, when_true, when_false);
}

template <detail::ConstantLiteral T, detail::ConstantLiteral U>
[[nodiscard]] inline ComputeValue select(const ComputeValue condition,
                                         const T when_true,
                                         const U when_false) noexcept {
  return detail::SelectBranches(condition, when_true, when_false);
}

} // namespace rund::compute_dsl
