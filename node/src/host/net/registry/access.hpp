#pragma once

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>

namespace rund::net {

struct SocketRegistryOwner {
  static constexpr std::uint64_t external_id =
      std::numeric_limits<std::uint64_t>::max();

  std::uint64_t scheduler_id = 0u;
  std::shared_ptr<std::atomic<std::uint32_t>> live_entries{};

  [[nodiscard]] bool external() const noexcept {
    return scheduler_id == external_id && live_entries == nullptr;
  }

  [[nodiscard]] bool active() const noexcept {
    return external() || (scheduler_id != 0u && live_entries != nullptr);
  }
};

struct SocketRegistryAccess {
  [[nodiscard]] static SocketRegistryOwner ActiveOwner() noexcept;
  [[nodiscard]] static bool TryAdmit(SocketRegistryOwner owner) noexcept;
  static void ReleaseOwner(SocketRegistryOwner owner) noexcept;
};

} // namespace rund::net
