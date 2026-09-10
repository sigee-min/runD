#include "../registry.hpp"
#include "internal.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>

namespace rund::compute::detail::residency {

bool Authority::probe(const std::span<const CacheKey> keys,
                      const std::span<std::uint8_t> resident,
                      const std::uint32_t first_frame,
                      const std::uint32_t frame_count) const noexcept {
  if (keys.size() != resident.size()) {
    return false;
  }
  std::lock_guard lock{gate_};
  if (view_commit_state_locked() != registry_model::ViewCommitState::Idle ||
      execution_state_.slot.token != 0u) {
    return false;
  }
  const std::size_t first = first_frame;
  const std::size_t count = frame_count;
  if (count == 0u || first > frames_.size() || count > frames_.size() - first) {
    return false;
  }
  for (std::size_t index = 0u; index < keys.size(); ++index) {
    resident[index] = static_cast<std::uint8_t>(
        find_frame(frames_, keys[index], first, count) != frames_.size());
  }
  return true;
}

std::uint32_t Authority::frame_capacity() const noexcept {
  std::lock_guard lock{gate_};
  return static_cast<std::uint32_t>(frames_.size());
}

} // namespace rund::compute::detail::residency
