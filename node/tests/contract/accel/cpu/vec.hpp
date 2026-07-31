#pragma once

#include "fixed.hpp"
#include "local.hpp"

namespace node_accel_contract::cpu::vec {

[[nodiscard]] inline rund::kernel::i32
Dot2_32(const rund::kernel::i32 ax, const rund::kernel::i32 ay,
        const rund::kernel::i32 bx, const rund::kernel::i32 by) noexcept {
  return rund::math32::detail::ScalarAddSat(
      fixed::QuantizeProduct(ax, bx), fixed::QuantizeProduct(ay, by));
}

[[nodiscard]] inline rund::kernel::i32
Dot3_32(const rund::kernel::i32 ax, const rund::kernel::i32 ay,
        const rund::kernel::i32 az, const rund::kernel::i32 bx,
        const rund::kernel::i32 by, const rund::kernel::i32 bz) noexcept {
  return rund::math32::detail::ScalarAddSat(
      Dot2_32(ax, ay, bx, by), fixed::QuantizeProduct(az, bz));
}

[[nodiscard]] inline rund::kernel::i32
Len2_32(const rund::kernel::i32 x, const rund::kernel::i32 y) noexcept {
  return fixed::SqrtFloor(Dot2_32(x, y, x, y));
}

[[nodiscard]] inline rund::kernel::i32
Len3_32(const rund::kernel::i32 x, const rund::kernel::i32 y,
        const rund::kernel::i32 z) noexcept {
  return fixed::SqrtFloor(Dot3_32(x, y, z, x, y, z));
}

[[nodiscard]] inline rund::kernel::i32
Dist2_32(const rund::kernel::i32 ax, const rund::kernel::i32 ay,
         const rund::kernel::i32 bx, const rund::kernel::i32 by) noexcept {
  return Len2_32(rund::math32::detail::ScalarSubSat(ax, bx),
                 rund::math32::detail::ScalarSubSat(ay, by));
}

[[nodiscard]] inline rund::kernel::i32
Dist3_32(const rund::kernel::i32 ax, const rund::kernel::i32 ay,
         const rund::kernel::i32 az, const rund::kernel::i32 bx,
         const rund::kernel::i32 by, const rund::kernel::i32 bz) noexcept {
  return Len3_32(rund::math32::detail::ScalarSubSat(ax, bx),
                 rund::math32::detail::ScalarSubSat(ay, by),
                 rund::math32::detail::ScalarSubSat(az, bz));
}

[[nodiscard]] inline rund::kernel::i64
Dot2_64(const rund::kernel::i64 ax, const rund::kernel::i64 ay,
        const rund::kernel::i64 bx, const rund::kernel::i64 by) noexcept {
  return rund::math64::detail::ScalarAddSat(
      fixed::QuantizeProduct(ax, bx), fixed::QuantizeProduct(ay, by));
}

[[nodiscard]] inline rund::kernel::i64
Dot3_64(const rund::kernel::i64 ax, const rund::kernel::i64 ay,
        const rund::kernel::i64 az, const rund::kernel::i64 bx,
        const rund::kernel::i64 by, const rund::kernel::i64 bz) noexcept {
  return rund::math64::detail::ScalarAddSat(
      Dot2_64(ax, ay, bx, by), fixed::QuantizeProduct(az, bz));
}

[[nodiscard]] inline rund::kernel::i64
Len2_64(const rund::kernel::i64 x, const rund::kernel::i64 y) noexcept {
  return fixed::SqrtFloor(Dot2_64(x, y, x, y));
}

[[nodiscard]] inline rund::kernel::i64
Len3_64(const rund::kernel::i64 x, const rund::kernel::i64 y,
        const rund::kernel::i64 z) noexcept {
  return fixed::SqrtFloor(Dot3_64(x, y, z, x, y, z));
}

[[nodiscard]] inline rund::kernel::i64
Dist2_64(const rund::kernel::i64 ax, const rund::kernel::i64 ay,
         const rund::kernel::i64 bx, const rund::kernel::i64 by) noexcept {
  return Len2_64(rund::math64::detail::ScalarSubSat(ax, bx),
                 rund::math64::detail::ScalarSubSat(ay, by));
}

[[nodiscard]] inline rund::kernel::i64
Dist3_64(const rund::kernel::i64 ax, const rund::kernel::i64 ay,
         const rund::kernel::i64 az, const rund::kernel::i64 bx,
         const rund::kernel::i64 by, const rund::kernel::i64 bz) noexcept {
  return Len3_64(rund::math64::detail::ScalarSubSat(ax, bx),
                 rund::math64::detail::ScalarSubSat(ay, by),
                 rund::math64::detail::ScalarSubSat(az, bz));
}

} // namespace node_accel_contract::cpu::vec
