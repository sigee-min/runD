#pragma once

#include "../../local.hpp"

#include <array>

namespace node_accel_contract::cpu::ops64 {

inline constexpr std::size_t kTileCount = 4u;

struct Work {
  std::array<rund::kernel::i64, kTileCount> lhs{};
  std::array<rund::kernel::i64, kTileCount> rhs{};
  std::array<rund::kernel::i64, kTileCount> lo{};
  std::array<rund::kernel::i64, kTileCount> hi{};
  std::array<rund::kernel::i64, kTileCount> addend{};
  std::array<rund::kernel::i64, kTileCount> positive{};
  std::array<rund::kernel::i64, kTileCount> out{};
};

[[nodiscard]] Work MakeWork() {
  return Work{
      .lhs = {0x1000000000000000ll, -0x2000000000000000ll,
              0x3000000000000000ll, -0x0100000000000000ll},
      .rhs = {0x0800000000000000ll, 0x0400000000000000ll,
              0x3000000000000000ll, 0x0100000000000000ll},
      .lo = {-0x1000000000000000ll, -0x1000000000000000ll,
             -0x1000000000000000ll, -0x1000000000000000ll},
      .hi = {0x2000000000000000ll, 0x2000000000000000ll,
             0x2000000000000000ll, 0x2000000000000000ll},
      .addend = {11, -13, 17, -19},
      .positive = {0x4000000000000000ll, 0x2000000000000000ll,
                   0x1000000000000000ll, 0x0800000000000000ll},
  };
}

}  // namespace node_accel_contract::cpu::ops64
