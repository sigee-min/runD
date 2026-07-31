#pragma once

#include "fixed.hpp"
#include "proj.hpp"

namespace node_accel_contract::cpu::cross {

[[nodiscard]] inline rund::kernel::i32
Cross2_32(const rund::kernel::i32 ax, const rund::kernel::i32 ay,
          const rund::kernel::i32 bx, const rund::kernel::i32 by) noexcept {
  return rund::math32::detail::ScalarSubSat(
      fixed::QuantizeProduct(ax, by), fixed::QuantizeProduct(ay, bx));
}

[[nodiscard]] inline rund::kernel::i32
Cross3X_32(const rund::kernel::i32, const rund::kernel::i32 ay,
           const rund::kernel::i32 az, const rund::kernel::i32,
           const rund::kernel::i32 by, const rund::kernel::i32 bz) noexcept {
  return Cross2_32(ay, az, by, bz);
}

[[nodiscard]] inline rund::kernel::i32
Cross3Y_32(const rund::kernel::i32 ax, const rund::kernel::i32,
           const rund::kernel::i32 az, const rund::kernel::i32 bx,
           const rund::kernel::i32, const rund::kernel::i32 bz) noexcept {
  return Cross2_32(az, ax, bz, bx);
}

[[nodiscard]] inline rund::kernel::i32
Cross3Z_32(const rund::kernel::i32 ax, const rund::kernel::i32 ay,
           const rund::kernel::i32, const rund::kernel::i32 bx,
           const rund::kernel::i32 by, const rund::kernel::i32) noexcept {
  return Cross2_32(ax, ay, bx, by);
}

[[nodiscard]] inline rund::kernel::i32
Reject2X_32(const rund::kernel::i32 ax, const rund::kernel::i32 ay,
            const rund::kernel::i32 bx, const rund::kernel::i32 by) noexcept {
  return rund::math32::detail::ScalarSubSat(ax,
                                            proj::Proj2X_32(ax, ay, bx, by));
}

[[nodiscard]] inline rund::kernel::i32
Reject2Y_32(const rund::kernel::i32 ax, const rund::kernel::i32 ay,
            const rund::kernel::i32 bx, const rund::kernel::i32 by) noexcept {
  return rund::math32::detail::ScalarSubSat(ay,
                                            proj::Proj2Y_32(ax, ay, bx, by));
}

[[nodiscard]] inline rund::kernel::i32
Reject3X_32(const rund::kernel::i32 ax, const rund::kernel::i32 ay,
            const rund::kernel::i32 az, const rund::kernel::i32 bx,
            const rund::kernel::i32 by, const rund::kernel::i32 bz) noexcept {
  return rund::math32::detail::ScalarSubSat(
      ax, proj::Proj3X_32(ax, ay, az, bx, by, bz));
}

[[nodiscard]] inline rund::kernel::i32
Reject3Y_32(const rund::kernel::i32 ax, const rund::kernel::i32 ay,
            const rund::kernel::i32 az, const rund::kernel::i32 bx,
            const rund::kernel::i32 by, const rund::kernel::i32 bz) noexcept {
  return rund::math32::detail::ScalarSubSat(
      ay, proj::Proj3Y_32(ax, ay, az, bx, by, bz));
}

[[nodiscard]] inline rund::kernel::i32
Reject3Z_32(const rund::kernel::i32 ax, const rund::kernel::i32 ay,
            const rund::kernel::i32 az, const rund::kernel::i32 bx,
            const rund::kernel::i32 by, const rund::kernel::i32 bz) noexcept {
  return rund::math32::detail::ScalarSubSat(
      az, proj::Proj3Z_32(ax, ay, az, bx, by, bz));
}

[[nodiscard]] inline rund::kernel::i64
Cross2_64(const rund::kernel::i64 ax, const rund::kernel::i64 ay,
          const rund::kernel::i64 bx, const rund::kernel::i64 by) noexcept {
  return rund::math64::detail::ScalarSubSat(
      fixed::QuantizeProduct(ax, by), fixed::QuantizeProduct(ay, bx));
}

[[nodiscard]] inline rund::kernel::i64
Cross3X_64(const rund::kernel::i64, const rund::kernel::i64 ay,
           const rund::kernel::i64 az, const rund::kernel::i64,
           const rund::kernel::i64 by, const rund::kernel::i64 bz) noexcept {
  return Cross2_64(ay, az, by, bz);
}

[[nodiscard]] inline rund::kernel::i64
Cross3Y_64(const rund::kernel::i64 ax, const rund::kernel::i64,
           const rund::kernel::i64 az, const rund::kernel::i64 bx,
           const rund::kernel::i64, const rund::kernel::i64 bz) noexcept {
  return Cross2_64(az, ax, bz, bx);
}

[[nodiscard]] inline rund::kernel::i64
Cross3Z_64(const rund::kernel::i64 ax, const rund::kernel::i64 ay,
           const rund::kernel::i64, const rund::kernel::i64 bx,
           const rund::kernel::i64 by, const rund::kernel::i64) noexcept {
  return Cross2_64(ax, ay, bx, by);
}

[[nodiscard]] inline rund::kernel::i64
Reject2X_64(const rund::kernel::i64 ax, const rund::kernel::i64 ay,
            const rund::kernel::i64 bx, const rund::kernel::i64 by) noexcept {
  return rund::math64::detail::ScalarSubSat(ax,
                                            proj::Proj2X_64(ax, ay, bx, by));
}

[[nodiscard]] inline rund::kernel::i64
Reject2Y_64(const rund::kernel::i64 ax, const rund::kernel::i64 ay,
            const rund::kernel::i64 bx, const rund::kernel::i64 by) noexcept {
  return rund::math64::detail::ScalarSubSat(ay,
                                            proj::Proj2Y_64(ax, ay, bx, by));
}

[[nodiscard]] inline rund::kernel::i64
Reject3X_64(const rund::kernel::i64 ax, const rund::kernel::i64 ay,
            const rund::kernel::i64 az, const rund::kernel::i64 bx,
            const rund::kernel::i64 by, const rund::kernel::i64 bz) noexcept {
  return rund::math64::detail::ScalarSubSat(
      ax, proj::Proj3X_64(ax, ay, az, bx, by, bz));
}

[[nodiscard]] inline rund::kernel::i64
Reject3Y_64(const rund::kernel::i64 ax, const rund::kernel::i64 ay,
            const rund::kernel::i64 az, const rund::kernel::i64 bx,
            const rund::kernel::i64 by, const rund::kernel::i64 bz) noexcept {
  return rund::math64::detail::ScalarSubSat(
      ay, proj::Proj3Y_64(ax, ay, az, bx, by, bz));
}

[[nodiscard]] inline rund::kernel::i64
Reject3Z_64(const rund::kernel::i64 ax, const rund::kernel::i64 ay,
            const rund::kernel::i64 az, const rund::kernel::i64 bx,
            const rund::kernel::i64 by, const rund::kernel::i64 bz) noexcept {
  return rund::math64::detail::ScalarSubSat(
      az, proj::Proj3Z_64(ax, ay, az, bx, by, bz));
}

} // namespace node_accel_contract::cpu::cross
