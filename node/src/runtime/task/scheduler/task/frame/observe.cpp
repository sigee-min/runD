#include "internal.hpp"

#include <mutex>

namespace rund::node {

std::uint32_t FrameArena::frame_bytes(void *const frame) noexcept {
  if (frame == nullptr) {
    return 0u;
  }
  const auto *const header = reinterpret_cast<const Header *>(
      static_cast<std::byte *>(frame) - sizeof(Header));
  Store *const store = header->store;
  if (store == nullptr) {
    return 0u;
  }
  const std::uint8_t tier =
      (header->slot_and_tier & frame_detail::TierBit) == 0u ? 0u : 1u;
  const std::uint32_t slot = header->slot_and_tier & frame_detail::SlotMask;
  std::lock_guard lock{store->mutex};
  return tier < store->tier_count && slot < store->tiers[tier].bytes.size()
             ? store->tiers[tier].bytes[slot]
             : 0u;
}

bool FrameArena::frame_reused(void *const frame) noexcept {
  if (frame == nullptr) {
    return false;
  }
  const auto *const header = reinterpret_cast<const Header *>(
      static_cast<std::byte *>(frame) - sizeof(Header));
  return header->store != nullptr && header->generation != 1u;
}

FrameStats FrameArena::stats() const noexcept {
  Store *const store = store_;
  if (store == nullptr) {
    return {};
  }
  std::lock_guard lock{store->mutex};
  return store->stats;
}

bool FrameArena::ready() const noexcept {
  Store *const store = store_;
  if (store == nullptr) {
    return false;
  }
  std::lock_guard lock{store->mutex};
  return !store->retired;
}

ReasonCode FrameArena::code() const noexcept {
  Store *const store = store_;
  if (store == nullptr) {
    return code_;
  }
  std::lock_guard lock{store->mutex};
  return store->code;
}

} // namespace rund::node
