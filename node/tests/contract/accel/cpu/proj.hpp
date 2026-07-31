#pragma once

#include "fixed.hpp"
#include "vec.hpp"

namespace node_accel_contract::cpu::proj {

[[nodiscard]] inline rund::kernel::i32
Unit32(const rund::kernel::i32 value,
       const rund::kernel::i32 length) noexcept {
  return fixed::DivNearestEven(value, length);
}

[[nodiscard]] inline rund::kernel::i64
Unit64(const rund::kernel::i64 value,
       const rund::kernel::i64 length) noexcept {
  return fixed::DivNearestEven(value, length);
}

[[nodiscard]] inline rund::kernel::i32
Unit2X_32(const rund::kernel::i32 x, const rund::kernel::i32 y) noexcept {
  return Unit32(x, vec::Len2_32(x, y));
}

[[nodiscard]] inline rund::kernel::i32
Unit2Y_32(const rund::kernel::i32 x, const rund::kernel::i32 y) noexcept {
  return Unit32(y, vec::Len2_32(x, y));
}

[[nodiscard]] inline rund::kernel::i32
Unit3X_32(const rund::kernel::i32 x, const rund::kernel::i32 y,
          const rund::kernel::i32 z) noexcept {
  return Unit32(x, vec::Len3_32(x, y, z));
}

[[nodiscard]] inline rund::kernel::i32
Unit3Y_32(const rund::kernel::i32 x, const rund::kernel::i32 y,
          const rund::kernel::i32 z) noexcept {
  return Unit32(y, vec::Len3_32(x, y, z));
}

[[nodiscard]] inline rund::kernel::i32
Unit3Z_32(const rund::kernel::i32 x, const rund::kernel::i32 y,
          const rund::kernel::i32 z) noexcept {
  return Unit32(z, vec::Len3_32(x, y, z));
}

[[nodiscard]] inline rund::kernel::i32
Scale2_32(const rund::kernel::i32 ax, const rund::kernel::i32 ay,
          const rund::kernel::i32 bx, const rund::kernel::i32 by) noexcept {
  const rund::kernel::i32 denom = vec::Dot2_32(bx, by, bx, by);
  return fixed::DivNearestEven(vec::Dot2_32(ax, ay, bx, by), denom);
}

[[nodiscard]] inline rund::kernel::i32
Proj2X_32(const rund::kernel::i32 ax, const rund::kernel::i32 ay,
          const rund::kernel::i32 bx, const rund::kernel::i32 by) noexcept {
  return fixed::QuantizeProduct(bx, Scale2_32(ax, ay, bx, by));
}

[[nodiscard]] inline rund::kernel::i32
Proj2Y_32(const rund::kernel::i32 ax, const rund::kernel::i32 ay,
          const rund::kernel::i32 bx, const rund::kernel::i32 by) noexcept {
  return fixed::QuantizeProduct(by, Scale2_32(ax, ay, bx, by));
}

[[nodiscard]] inline rund::kernel::i32
Scale3_32(const rund::kernel::i32 ax, const rund::kernel::i32 ay,
          const rund::kernel::i32 az, const rund::kernel::i32 bx,
          const rund::kernel::i32 by, const rund::kernel::i32 bz) noexcept {
  const rund::kernel::i32 denom = vec::Dot3_32(bx, by, bz, bx, by, bz);
  return fixed::DivNearestEven(vec::Dot3_32(ax, ay, az, bx, by, bz), denom);
}

[[nodiscard]] inline rund::kernel::i32
Proj3X_32(const rund::kernel::i32 ax, const rund::kernel::i32 ay,
          const rund::kernel::i32 az, const rund::kernel::i32 bx,
          const rund::kernel::i32 by, const rund::kernel::i32 bz) noexcept {
  return fixed::QuantizeProduct(bx, Scale3_32(ax, ay, az, bx, by, bz));
}

[[nodiscard]] inline rund::kernel::i32
Proj3Y_32(const rund::kernel::i32 ax, const rund::kernel::i32 ay,
          const rund::kernel::i32 az, const rund::kernel::i32 bx,
          const rund::kernel::i32 by, const rund::kernel::i32 bz) noexcept {
  return fixed::QuantizeProduct(by, Scale3_32(ax, ay, az, bx, by, bz));
}

[[nodiscard]] inline rund::kernel::i32
Proj3Z_32(const rund::kernel::i32 ax, const rund::kernel::i32 ay,
          const rund::kernel::i32 az, const rund::kernel::i32 bx,
          const rund::kernel::i32 by, const rund::kernel::i32 bz) noexcept {
  return fixed::QuantizeProduct(bz, Scale3_32(ax, ay, az, bx, by, bz));
}

[[nodiscard]] inline rund::kernel::i64
Unit2X_64(const rund::kernel::i64 x, const rund::kernel::i64 y) noexcept {
  return Unit64(x, vec::Len2_64(x, y));
}

[[nodiscard]] inline rund::kernel::i64
Unit2Y_64(const rund::kernel::i64 x, const rund::kernel::i64 y) noexcept {
  return Unit64(y, vec::Len2_64(x, y));
}

[[nodiscard]] inline rund::kernel::i64
Unit3X_64(const rund::kernel::i64 x, const rund::kernel::i64 y,
          const rund::kernel::i64 z) noexcept {
  return Unit64(x, vec::Len3_64(x, y, z));
}

[[nodiscard]] inline rund::kernel::i64
Unit3Y_64(const rund::kernel::i64 x, const rund::kernel::i64 y,
          const rund::kernel::i64 z) noexcept {
  return Unit64(y, vec::Len3_64(x, y, z));
}

[[nodiscard]] inline rund::kernel::i64
Unit3Z_64(const rund::kernel::i64 x, const rund::kernel::i64 y,
          const rund::kernel::i64 z) noexcept {
  return Unit64(z, vec::Len3_64(x, y, z));
}

[[nodiscard]] inline rund::kernel::i64
Scale2_64(const rund::kernel::i64 ax, const rund::kernel::i64 ay,
          const rund::kernel::i64 bx, const rund::kernel::i64 by) noexcept {
  const rund::kernel::i64 denom = vec::Dot2_64(bx, by, bx, by);
  return fixed::DivNearestEven(vec::Dot2_64(ax, ay, bx, by), denom);
}

[[nodiscard]] inline rund::kernel::i64
Proj2X_64(const rund::kernel::i64 ax, const rund::kernel::i64 ay,
          const rund::kernel::i64 bx, const rund::kernel::i64 by) noexcept {
  return fixed::QuantizeProduct(bx, Scale2_64(ax, ay, bx, by));
}

[[nodiscard]] inline rund::kernel::i64
Proj2Y_64(const rund::kernel::i64 ax, const rund::kernel::i64 ay,
          const rund::kernel::i64 bx, const rund::kernel::i64 by) noexcept {
  return fixed::QuantizeProduct(by, Scale2_64(ax, ay, bx, by));
}

[[nodiscard]] inline rund::kernel::i64
Scale3_64(const rund::kernel::i64 ax, const rund::kernel::i64 ay,
          const rund::kernel::i64 az, const rund::kernel::i64 bx,
          const rund::kernel::i64 by, const rund::kernel::i64 bz) noexcept {
  const rund::kernel::i64 denom = vec::Dot3_64(bx, by, bz, bx, by, bz);
  return fixed::DivNearestEven(vec::Dot3_64(ax, ay, az, bx, by, bz), denom);
}

[[nodiscard]] inline rund::kernel::i64
Proj3X_64(const rund::kernel::i64 ax, const rund::kernel::i64 ay,
          const rund::kernel::i64 az, const rund::kernel::i64 bx,
          const rund::kernel::i64 by, const rund::kernel::i64 bz) noexcept {
  return fixed::QuantizeProduct(bx, Scale3_64(ax, ay, az, bx, by, bz));
}

[[nodiscard]] inline rund::kernel::i64
Proj3Y_64(const rund::kernel::i64 ax, const rund::kernel::i64 ay,
          const rund::kernel::i64 az, const rund::kernel::i64 bx,
          const rund::kernel::i64 by, const rund::kernel::i64 bz) noexcept {
  return fixed::QuantizeProduct(by, Scale3_64(ax, ay, az, bx, by, bz));
}

[[nodiscard]] inline rund::kernel::i64
Proj3Z_64(const rund::kernel::i64 ax, const rund::kernel::i64 ay,
          const rund::kernel::i64 az, const rund::kernel::i64 bx,
          const rund::kernel::i64 by, const rund::kernel::i64 bz) noexcept {
  return fixed::QuantizeProduct(bz, Scale3_64(ax, ay, az, bx, by, bz));
}

} // namespace node_accel_contract::cpu::proj
