#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

// Frozen primitive executable tuple. Range PrefixDifference has at
// most 24 hierarchy/fix-up/window stages for the legal 64-lane width. Route-
// owned buffers and mutable dispatch state never enter this Program-level
// owner.
struct MetalKernelImmutablePipelines final {
  std::array<std::shared_ptr<void>, 24u> stages{};
  std::shared_ptr<void> control{};
  std::uint32_t count{};

  [[nodiscard]] bool ready(const std::uint32_t expected,
                           const bool requires_control = false) const noexcept {
    if (count != expected || count == 0u || count > stages.size()) {
      return false;
    }
    for (std::size_t index = 0u; index < count; ++index) {
      if (stages[index] == nullptr) {
        return false;
      }
    }
    return !requires_control || control != nullptr;
  }
};

} // namespace rund::node::accel::detail
