#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

inline constexpr std::size_t kVulkanCommandCapacity = 8u;
inline constexpr std::size_t kInvalidVulkanCommand = kVulkanCommandCapacity;

enum class VulkanCommandPhase : std::uint8_t {
  Free,
  Recording,
  Submitted,
};

struct VulkanCommandLease final {
  std::size_t slot = kInvalidVulkanCommand;
  std::uint64_t sequence = 0u;

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return slot < kVulkanCommandCapacity;
  }
};

struct VulkanCommandRing final {
  std::array<VulkanCommandPhase, kVulkanCommandCapacity> phases{};
  std::array<std::uint64_t, kVulkanCommandCapacity> sequences{};
  std::uint64_t next_sequence = 1u;
  std::uint64_t retired_sequence = 0u;
  std::size_t cursor = 0u;
  std::size_t active = 0u;

  [[nodiscard]] constexpr VulkanCommandLease claim() noexcept {
    if (active == phases.size()) {
      return {};
    }
    for (std::size_t offset = 0u; offset < phases.size(); ++offset) {
      const std::size_t slot = (cursor + offset) % phases.size();
      if (phases[slot] != VulkanCommandPhase::Free) {
        continue;
      }
      phases[slot] = VulkanCommandPhase::Recording;
      sequences[slot] = 0u;
      cursor = (slot + 1u) % phases.size();
      ++active;
      return VulkanCommandLease{.slot = slot};
    }
    return {};
  }

  [[nodiscard]] constexpr bool publish(VulkanCommandLease &lease) noexcept {
    if (!lease || phases[lease.slot] != VulkanCommandPhase::Recording ||
        next_sequence == std::numeric_limits<std::uint64_t>::max()) {
      return false;
    }
    lease.sequence = next_sequence++;
    phases[lease.slot] = VulkanCommandPhase::Submitted;
    sequences[lease.slot] = lease.sequence;
    return true;
  }

  [[nodiscard]] constexpr bool cancel(const VulkanCommandLease lease) noexcept {
    if (!lease || phases[lease.slot] == VulkanCommandPhase::Free ||
        (lease.sequence != 0u && sequences[lease.slot] != lease.sequence)) {
      return false;
    }
    phases[lease.slot] = VulkanCommandPhase::Free;
    sequences[lease.slot] = 0u;
    --active;
    return true;
  }

  [[nodiscard]] constexpr bool
  retirable(const VulkanCommandLease lease) const noexcept {
    if (!lease || lease.sequence == 0u ||
        phases[lease.slot] != VulkanCommandPhase::Submitted ||
        sequences[lease.slot] != lease.sequence ||
        lease.sequence <= retired_sequence) {
      return false;
    }
    for (std::size_t slot = 0u; slot < phases.size(); ++slot) {
      if (phases[slot] == VulkanCommandPhase::Submitted &&
          sequences[slot] < lease.sequence) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] constexpr bool retire(const VulkanCommandLease lease) noexcept {
    if (!retirable(lease)) {
      return false;
    }
    retired_sequence = lease.sequence;
    phases[lease.slot] = VulkanCommandPhase::Free;
    sequences[lease.slot] = 0u;
    --active;
    return true;
  }

  [[nodiscard]] constexpr bool empty() const noexcept { return active == 0u; }
};

static_assert(kVulkanCommandCapacity > 1u);

} // namespace rund::node::accel::detail
