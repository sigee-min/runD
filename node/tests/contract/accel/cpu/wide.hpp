#pragma once

#include "local.hpp"

#include <array>
#include <cstddef>

namespace node_accel_contract::cpu {

template <std::size_t N>
[[nodiscard]] std::array<rund::kernel::i64, N>
Widen(const std::array<rund::kernel::i32, N>& values) {
  std::array<rund::kernel::i64, N> out{};
  for (std::size_t i = 0u; i < out.size(); ++i) {
    out[i] = values[i];
  }
  return out;
}

} // namespace node_accel_contract::cpu
