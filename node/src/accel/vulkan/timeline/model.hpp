#pragma once

#include <cstdint>

namespace rund::node::accel::detail {

enum class VulkanTimelineRoute : std::uint8_t {
  Unsupported,
  Core,
  Extension,
};

enum class VulkanTimelineInitState : std::uint8_t {
  Uninitialized,
  Initialized,
  Failed,
};

struct VulkanTimelineSupport final {
  VulkanTimelineRoute route{VulkanTimelineRoute::Unsupported};
  std::uint64_t max_difference{};

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return route != VulkanTimelineRoute::Unsupported;
  }
};

struct VulkanTimelinePoint final {
  std::uint32_t generation{};
  // Device-done is one globally ordered timeline. Host-ready has one
  // independent monotonic sequence per fixed cell so alternating Q sizes
  // never requires an unbounded semaphore-value jump on an idle cell.
  std::uint64_t value{};
  std::uint64_t ready_value{};
  std::uint8_t ready_cell{};

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return generation != 0u && value != 0u && ready_value != 0u &&
           ready_cell < 4u;
  }
};

inline constexpr const char *kVulkanTimelineExtension =
    "VK_KHR_timeline_semaphore";

} // namespace rund::node::accel::detail
