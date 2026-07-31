#pragma once

#include <math32/simd/model.hpp>

namespace rund::math32::simd {

[[nodiscard]] inline Mask32x Eq(const I32x lhs, const I32x rhs) noexcept { return static_cast<Mask32x>(lhs == rhs); }
[[nodiscard]] inline Mask32x Ne(const I32x lhs, const I32x rhs) noexcept { return static_cast<Mask32x>(lhs != rhs); }
[[nodiscard]] inline Mask32x Lt(const I32x lhs, const I32x rhs) noexcept { return static_cast<Mask32x>(lhs < rhs); }
[[nodiscard]] inline Mask32x Le(const I32x lhs, const I32x rhs) noexcept { return static_cast<Mask32x>(lhs <= rhs); }
[[nodiscard]] inline Mask32x Gt(const I32x lhs, const I32x rhs) noexcept { return static_cast<Mask32x>(lhs > rhs); }
[[nodiscard]] inline Mask32x Ge(const I32x lhs, const I32x rhs) noexcept { return static_cast<Mask32x>(lhs >= rhs); }

[[nodiscard]] inline Mask32x Eq(const U32x lhs, const U32x rhs) noexcept { return static_cast<Mask32x>(lhs == rhs); }
[[nodiscard]] inline Mask32x Ne(const U32x lhs, const U32x rhs) noexcept { return static_cast<Mask32x>(lhs != rhs); }
[[nodiscard]] inline Mask32x Lt(const U32x lhs, const U32x rhs) noexcept { return static_cast<Mask32x>(lhs < rhs); }
[[nodiscard]] inline Mask32x Le(const U32x lhs, const U32x rhs) noexcept { return static_cast<Mask32x>(lhs <= rhs); }
[[nodiscard]] inline Mask32x Gt(const U32x lhs, const U32x rhs) noexcept { return static_cast<Mask32x>(lhs > rhs); }
[[nodiscard]] inline Mask32x Ge(const U32x lhs, const U32x rhs) noexcept { return static_cast<Mask32x>(lhs >= rhs); }

}  // namespace rund::math32::simd
