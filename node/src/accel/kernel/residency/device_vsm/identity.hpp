#pragma once

#include <cstdint>

namespace rund::node::accel::detail {

struct DeviceVsmIdentity final {
  std::uint64_t hi{};
  std::uint64_t lo{};

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return hi != 0u || lo != 0u;
  }

  [[nodiscard]] constexpr bool
  operator==(const DeviceVsmIdentity &) const noexcept = default;
};

} // namespace rund::node::accel::detail
