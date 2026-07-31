#pragma once

#include "fixed.hpp"

#include <bit>

namespace node_accel_contract::cpu::noise {

[[nodiscard]] inline rund::kernel::u32
Hash32Bits(rund::kernel::u32 value) noexcept {
  value ^= value >> 16u;
  value *= 0x7feb352du;
  value ^= value >> 15u;
  value *= 0x846ca68bu;
  value ^= value >> 16u;
  return value;
}

[[nodiscard]] inline rund::kernel::u64
Hash64Bits(rund::kernel::u64 value) noexcept {
  value ^= value >> 33u;
  value *= 0xff51afd7ed558ccdull;
  value ^= value >> 33u;
  value *= 0xc4ceb9fe1a85ec53ull;
  value ^= value >> 33u;
  return value;
}

[[nodiscard]] inline rund::kernel::i32
UnitHash32(const rund::kernel::i32 value,
           const rund::kernel::i32 seed = 0) noexcept {
  const rund::kernel::u32 bits = std::bit_cast<rund::kernel::u32>(value);
  const rund::kernel::u32 seed_bits = std::bit_cast<rund::kernel::u32>(seed);
  return std::bit_cast<rund::kernel::i32>(Hash32Bits(bits ^ seed_bits) &
                                          0x7fffffffu);
}

[[nodiscard]] inline rund::kernel::i64
UnitHash64(const rund::kernel::i64 value,
           const rund::kernel::i64 seed = 0) noexcept {
  const rund::kernel::u64 bits = std::bit_cast<rund::kernel::u64>(value);
  const rund::kernel::u64 seed_bits = std::bit_cast<rund::kernel::u64>(seed);
  return std::bit_cast<rund::kernel::i64>(Hash64Bits(bits ^ seed_bits) &
                                          0x7fffffffffffffffull);
}

[[nodiscard]] inline rund::kernel::i32
Noise(const rund::kernel::i32 cell, const rund::kernel::i32 amount,
          const rund::kernel::i32 seed = 0) noexcept {
  const rund::kernel::i32 next = rund::math32::detail::ScalarAddWrap(cell, 1);
  return fixed::Lerp32(UnitHash32(cell, seed), UnitHash32(next, seed),
                       fixed::Fade32(amount));
}

[[nodiscard]] inline rund::kernel::i64
Noise(const rund::kernel::i64 cell, const rund::kernel::i64 amount,
          const rund::kernel::i64 seed = 0) noexcept {
  const rund::kernel::i64 next = rund::math64::detail::ScalarAddWrap(cell, 1);
  return fixed::Lerp64(UnitHash64(cell, seed), UnitHash64(next, seed),
                       fixed::Fade64(amount));
}

[[nodiscard]] inline rund::kernel::i32
Noise(const rund::kernel::i32 x, const rund::kernel::i32 y,
          const rund::kernel::i32 tx, const rund::kernel::i32 ty,
          const rund::kernel::i32 seed = 0) noexcept {
  const rund::kernel::i32 x1 = rund::math32::detail::ScalarAddWrap(x, 1);
  const rund::kernel::i32 y1 = rund::math32::detail::ScalarAddWrap(y, 1);
  const rund::kernel::i32 y0_seed = std::bit_cast<rund::kernel::i32>(
      std::bit_cast<rund::kernel::u32>(y) ^
      std::bit_cast<rund::kernel::u32>(seed));
  const rund::kernel::i32 y1_seed = std::bit_cast<rund::kernel::i32>(
      std::bit_cast<rund::kernel::u32>(y1) ^
      std::bit_cast<rund::kernel::u32>(seed));
  const rund::kernel::i32 row0 =
      fixed::Lerp32(UnitHash32(x, y0_seed), UnitHash32(x1, y0_seed),
                    fixed::Fade32(tx));
  const rund::kernel::i32 row1 =
      fixed::Lerp32(UnitHash32(x, y1_seed), UnitHash32(x1, y1_seed),
                    fixed::Fade32(tx));
  return fixed::Lerp32(row0, row1, fixed::Fade32(ty));
}

[[nodiscard]] inline rund::kernel::i64
Noise(const rund::kernel::i64 x, const rund::kernel::i64 y,
          const rund::kernel::i64 tx, const rund::kernel::i64 ty,
          const rund::kernel::i64 seed = 0) noexcept {
  const rund::kernel::i64 x1 = rund::math64::detail::ScalarAddWrap(x, 1);
  const rund::kernel::i64 y1 = rund::math64::detail::ScalarAddWrap(y, 1);
  const rund::kernel::i64 y0_seed = std::bit_cast<rund::kernel::i64>(
      std::bit_cast<rund::kernel::u64>(y) ^
      std::bit_cast<rund::kernel::u64>(seed));
  const rund::kernel::i64 y1_seed = std::bit_cast<rund::kernel::i64>(
      std::bit_cast<rund::kernel::u64>(y1) ^
      std::bit_cast<rund::kernel::u64>(seed));
  const rund::kernel::i64 row0 =
      fixed::Lerp64(UnitHash64(x, y0_seed), UnitHash64(x1, y0_seed),
                    fixed::Fade64(tx));
  const rund::kernel::i64 row1 =
      fixed::Lerp64(UnitHash64(x, y1_seed), UnitHash64(x1, y1_seed),
                    fixed::Fade64(tx));
  return fixed::Lerp64(row0, row1, fixed::Fade64(ty));
}

template <typename T>
[[nodiscard]] inline T XorBits(const T lhs, const T rhs) noexcept {
  return fixed::XorBits(lhs, rhs);
}

} // namespace node_accel_contract::cpu::noise
