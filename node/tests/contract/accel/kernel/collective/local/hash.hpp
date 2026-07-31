#pragma once

#include "scan.hpp"

#include <cstdint>

namespace node_accel_contract::collective {

template <typename T>
[[nodiscard]] std::uint64_t HashValues(const T* const values,
                                       const std::size_t count) noexcept {
  std::uint64_t hash = 1469598103934665603ull;
  const auto* const bytes = reinterpret_cast<const std::uint8_t*>(values);
  const std::size_t byte_count = count * sizeof(T);
  for (std::size_t index = 0u; index < byte_count; ++index) {
    hash ^= bytes[index];
    hash *= 1099511628211ull;
  }
  return hash;
}

}  // namespace node_accel_contract::collective
