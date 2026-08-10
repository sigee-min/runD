#pragma once

#include <rund/compute/virtual.hpp>

#include <algorithm>
#include <cstdint>
#include <mutex>

namespace rund::compute::detail {

// The backing, not a VirtualBuffer wrapper, owns mutation serialization and
// partial-write validity. Multiple typed views over one logical dataset
// therefore cannot publish competing poison or callback order.
struct VirtualBackingState final {
  std::mutex gate;
  // Zero is clean. A nonzero value is the output prefix that a successful
  // retry must overwrite completely before this backing can become readable.
  std::uint64_t recovery_bytes{};
};

struct VirtualBackingAccess final {
  [[nodiscard]] static std::mutex &gate(VirtualBacking &backing) noexcept {
    return backing.state_->gate;
  }

  [[nodiscard]] static std::uint64_t
  recovery_bytes(const VirtualBacking &backing) noexcept {
    return backing.state_->recovery_bytes;
  }

  static void require_recovery(VirtualBacking &backing,
                               const std::uint64_t bytes) noexcept {
    backing.state_->recovery_bytes =
        std::max(backing.state_->recovery_bytes, bytes);
  }

  static void clear_recovery(VirtualBacking &backing) noexcept {
    backing.state_->recovery_bytes = 0u;
  }
};

} // namespace rund::compute::detail
