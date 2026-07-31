#pragma once

#include "../local.hpp"

#include <array>

namespace node_accel_contract::cpu::pick {

inline constexpr std::size_t kTileCount = 8u;

struct Work {
  std::array<rund::kernel::i32, kTileCount> lhs{
      0x40000000, -0x20000000, 0x10000000, -0x08000000,
      0x02000000, -0x01000000, 1234567, -7654321};
  std::array<rund::kernel::i32, kTileCount> rhs{
      0x20000000, 0x10000000, -0x08000000, 0x04000000,
      -0x02000000, 0x01000000, -333333, 777777};
  std::array<rund::kernel::i32, kTileCount> out{};
};

}  // namespace node_accel_contract::cpu::pick
