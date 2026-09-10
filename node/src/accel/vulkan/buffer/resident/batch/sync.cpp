#include "local.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] std::uint64_t
latest_sequence(const VulkanAdapter &adapter) noexcept {
  std::uint64_t sequence = 0u;
  for (std::size_t slot = 0u; slot < adapter.command_ring.phases.size();
       ++slot) {
    if (adapter.command_ring.phases[slot] == VulkanCommandPhase::Submitted) {
      sequence = std::max(sequence, adapter.command_ring.sequences[slot]);
    }
  }
  return sequence;
}

[[nodiscard]] rund::AccelCheck wait_sequence(VulkanAdapter &adapter,
                                             std::unique_lock<std::mutex> &lock,
                                             const std::uint64_t sequence) {
  if (sequence == 0u) {
    return {true, "ok"};
  }
  adapter.command_cv.wait(lock, [&adapter, sequence] {
    for (std::size_t slot = 0u; slot < adapter.command_ring.phases.size();
         ++slot) {
      if (adapter.command_ring.phases[slot] == VulkanCommandPhase::Submitted &&
          adapter.command_ring.sequences[slot] <= sequence) {
        return false;
      }
    }
    return true;
  });
  for (std::size_t slot = 0u; slot < adapter.completed_sequences.size();
       ++slot) {
    if (adapter.completed_sequences[slot] == sequence) {
      return adapter.completed_status[slot];
    }
  }
  return {true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
