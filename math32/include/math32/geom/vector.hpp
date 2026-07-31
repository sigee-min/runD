#pragma once

#include <math32/geom/vec3x.hpp>
#include <math32/fixed/constants.hpp>
#include <math32/simd/arithmetic.hpp>
#include <math32/simd/compare.hpp>
#include <math32/simd/select.hpp>

namespace rund::math32::geom {
namespace detail {
inline simd::I32x AbsLane(const simd::I32x value) noexcept {
  const simd::I32x zero = simd::SplatI32(0);
  const simd::I32x negated = simd::Sub(zero, value);
  const simd::I32x wrapped = simd::Select(simd::Lt(value, zero), negated, value);
  return simd::Select(simd::Eq(value, simd::SplatI32(FixedMin)), simd::SplatI32(FixedMax), wrapped);
}
}  // namespace detail

[[nodiscard]] inline Vec2x Add(const Vec2x lhs, const Vec2x rhs) noexcept {
  return Vec2x{.x = simd::Add(lhs.x, rhs.x), .y = simd::Add(lhs.y, rhs.y)};
}
[[nodiscard]] inline Vec3x Add(const Vec3x lhs, const Vec3x rhs) noexcept {
  return Vec3x{.x = simd::Add(lhs.x, rhs.x), .y = simd::Add(lhs.y, rhs.y), .z = simd::Add(lhs.z, rhs.z)};
}
[[nodiscard]] inline Vec2x Sub(const Vec2x lhs, const Vec2x rhs) noexcept {
  return Vec2x{.x = simd::Sub(lhs.x, rhs.x), .y = simd::Sub(lhs.y, rhs.y)};
}
[[nodiscard]] inline Vec3x Sub(const Vec3x lhs, const Vec3x rhs) noexcept {
  return Vec3x{.x = simd::Sub(lhs.x, rhs.x), .y = simd::Sub(lhs.y, rhs.y), .z = simd::Sub(lhs.z, rhs.z)};
}
[[nodiscard]] inline Vec2x Neg(const Vec2x value) noexcept {
  const simd::I32x zero = simd::SplatI32(0);
  return Vec2x{.x = simd::Sub(zero, value.x), .y = simd::Sub(zero, value.y)};
}
[[nodiscard]] inline Vec3x Neg(const Vec3x value) noexcept {
  const simd::I32x zero = simd::SplatI32(0);
  return Vec3x{.x = simd::Sub(zero, value.x), .y = simd::Sub(zero, value.y), .z = simd::Sub(zero, value.z)};
}
[[nodiscard]] inline Vec2x Abs(const Vec2x value) noexcept {
  return Vec2x{.x = detail::AbsLane(value.x), .y = detail::AbsLane(value.y)};
}
[[nodiscard]] inline Vec3x Abs(const Vec3x value) noexcept {
  return Vec3x{.x = detail::AbsLane(value.x), .y = detail::AbsLane(value.y), .z = detail::AbsLane(value.z)};
}
[[nodiscard]] inline Vec2x Min(const Vec2x lhs, const Vec2x rhs) noexcept {
  return Vec2x{.x = simd::Min(lhs.x, rhs.x), .y = simd::Min(lhs.y, rhs.y)};
}
[[nodiscard]] inline Vec3x Min(const Vec3x lhs, const Vec3x rhs) noexcept {
  return Vec3x{.x = simd::Min(lhs.x, rhs.x), .y = simd::Min(lhs.y, rhs.y), .z = simd::Min(lhs.z, rhs.z)};
}
[[nodiscard]] inline Vec2x Max(const Vec2x lhs, const Vec2x rhs) noexcept {
  return Vec2x{.x = simd::Max(lhs.x, rhs.x), .y = simd::Max(lhs.y, rhs.y)};
}
[[nodiscard]] inline Vec3x Max(const Vec3x lhs, const Vec3x rhs) noexcept {
  return Vec3x{.x = simd::Max(lhs.x, rhs.x), .y = simd::Max(lhs.y, rhs.y), .z = simd::Max(lhs.z, rhs.z)};
}
[[nodiscard]] inline Vec2x Clamp(const Vec2x value, const Vec2x lower, const Vec2x upper) noexcept {
  return Min(Max(value, lower), upper);
}
[[nodiscard]] inline Vec3x Clamp(const Vec3x value, const Vec3x lower, const Vec3x upper) noexcept {
  return Min(Max(value, lower), upper);
}
[[nodiscard]] inline Mask2x Lt(const Vec2x lhs, const Vec2x rhs) noexcept {
  return Mask2x{.x = simd::Lt(lhs.x, rhs.x), .y = simd::Lt(lhs.y, rhs.y)};
}
[[nodiscard]] inline Mask3x Lt(const Vec3x lhs, const Vec3x rhs) noexcept {
  return Mask3x{.x = simd::Lt(lhs.x, rhs.x), .y = simd::Lt(lhs.y, rhs.y), .z = simd::Lt(lhs.z, rhs.z)};
}
[[nodiscard]] inline Mask2x Le(const Vec2x lhs, const Vec2x rhs) noexcept {
  return Mask2x{.x = simd::Le(lhs.x, rhs.x), .y = simd::Le(lhs.y, rhs.y)};
}
[[nodiscard]] inline Mask3x Le(const Vec3x lhs, const Vec3x rhs) noexcept {
  return Mask3x{.x = simd::Le(lhs.x, rhs.x), .y = simd::Le(lhs.y, rhs.y), .z = simd::Le(lhs.z, rhs.z)};
}
[[nodiscard]] inline Mask2x Eq(const Vec2x lhs, const Vec2x rhs) noexcept {
  return Mask2x{.x = simd::Eq(lhs.x, rhs.x), .y = simd::Eq(lhs.y, rhs.y)};
}
[[nodiscard]] inline Mask3x Eq(const Vec3x lhs, const Vec3x rhs) noexcept {
  return Mask3x{.x = simd::Eq(lhs.x, rhs.x), .y = simd::Eq(lhs.y, rhs.y), .z = simd::Eq(lhs.z, rhs.z)};
}
[[nodiscard]] inline Vec2x Select(const Mask2x mask, const Vec2x a, const Vec2x b) noexcept {
  return Vec2x{.x = simd::Select(mask.x, a.x, b.x), .y = simd::Select(mask.y, a.y, b.y)};
}
[[nodiscard]] inline Vec3x Select(const Mask3x mask, const Vec3x a, const Vec3x b) noexcept {
  return Vec3x{.x = simd::Select(mask.x, a.x, b.x), .y = simd::Select(mask.y, a.y, b.y), .z = simd::Select(mask.z, a.z, b.z)};
}
}  // namespace rund::math32::geom
