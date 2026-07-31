#pragma once

#include <math64/simd/model.hpp>

namespace rund::math64::simd {

[[nodiscard]] inline Mask64x Eq(const I64x lhs, const I64x rhs) noexcept { return static_cast<Mask64x>(lhs == rhs); }
[[nodiscard]] inline Mask64x Ne(const I64x lhs, const I64x rhs) noexcept { return static_cast<Mask64x>(lhs != rhs); }
[[nodiscard]] inline Mask64x Lt(const I64x lhs, const I64x rhs) noexcept { return static_cast<Mask64x>(lhs < rhs); }
[[nodiscard]] inline Mask64x Le(const I64x lhs, const I64x rhs) noexcept { return static_cast<Mask64x>(lhs <= rhs); }
[[nodiscard]] inline Mask64x Gt(const I64x lhs, const I64x rhs) noexcept { return static_cast<Mask64x>(lhs > rhs); }
[[nodiscard]] inline Mask64x Ge(const I64x lhs, const I64x rhs) noexcept { return static_cast<Mask64x>(lhs >= rhs); }

[[nodiscard]] inline Mask64x Eq(const U64x lhs, const U64x rhs) noexcept { return static_cast<Mask64x>(lhs == rhs); }
[[nodiscard]] inline Mask64x Ne(const U64x lhs, const U64x rhs) noexcept { return static_cast<Mask64x>(lhs != rhs); }
[[nodiscard]] inline Mask64x Lt(const U64x lhs, const U64x rhs) noexcept { return static_cast<Mask64x>(lhs < rhs); }
[[nodiscard]] inline Mask64x Le(const U64x lhs, const U64x rhs) noexcept { return static_cast<Mask64x>(lhs <= rhs); }
[[nodiscard]] inline Mask64x Gt(const U64x lhs, const U64x rhs) noexcept { return static_cast<Mask64x>(lhs > rhs); }
[[nodiscard]] inline Mask64x Ge(const U64x lhs, const U64x rhs) noexcept { return static_cast<Mask64x>(lhs >= rhs); }

}  // namespace rund::math64::simd
