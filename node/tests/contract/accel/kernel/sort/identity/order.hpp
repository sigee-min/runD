#pragma once

#include "../identity.hpp"

namespace node_accel_contract::sort_identity {

[[nodiscard]] inline bool StableEqualKeyOrderValid(
    const std::array<rund::kernel::u32, 8u>& keys,
    const std::array<rund::kernel::u32, 8u>& values) noexcept {
  for (std::size_t index = 1u; index < keys.size(); ++index) {
    if (keys[index - 1u] == keys[index] &&
        values[index - 1u] > values[index]) {
      return false;
    }
  }
  return true;
}

}  // namespace node_accel_contract::sort_identity
