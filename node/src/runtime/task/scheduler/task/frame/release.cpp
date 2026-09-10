#include "internal.hpp"

#include <mutex>

namespace rund::node {

void FrameArena::release(const FrameLease lease) noexcept {
  release_lease(lease);
}

void FrameArena::release_lease(const FrameLease lease) noexcept {
  if (!lease) {
    return;
  }
  auto *const store = static_cast<Store *>(lease.authority);
  if (store == nullptr) {
    return;
  }
  bool released = false;
  {
    std::lock_guard lock{store->mutex};
    if (lease.tier >= store->tier_count) {
      ++store->stats.failures;
      return;
    }
    Store::Tier &tier = store->tiers[lease.tier];
    if (lease.slot >= tier.live.size()) {
      ++store->stats.failures;
      return;
    }
    const void *const expected =
        (lease.tier == 0u
             ? static_cast<std::byte *>(
                   tier.pages[lease.slot / frame_detail::FramePageSlots]) +
                   (lease.slot % frame_detail::FramePageSlots) *
                       tier.stride_bytes
             : static_cast<std::byte *>(tier.blocks[lease.slot])) +
        store->prefix_bytes;
    if (lease.data != expected || tier.live[lease.slot] == 0u ||
        tier.generation[lease.slot] != lease.generation) {
      ++store->stats.failures;
      return;
    }
    tier.live[lease.slot] = 0u;
    tier.bytes[lease.slot] = 0u;
    ++tier.generation[lease.slot];
    if (tier.generation[lease.slot] == 0u) {
      tier.generation[lease.slot] = 1u;
    }
    tier.free.push_back(lease.slot);
    if (store->stats.live != 0u) {
      --store->stats.live;
    }
    released = true;
  }
  if (released) {
    drop(store);
  }
}

void FrameArena::release_frame(void *const frame) noexcept {
  if (frame == nullptr) {
    return;
  }
  const auto *const header = reinterpret_cast<const Header *>(
      static_cast<std::byte *>(frame) - sizeof(Header));
  if (header->store == nullptr) {
    return;
  }
  const std::uint8_t tier =
      (header->slot_and_tier & frame_detail::TierBit) == 0u ? 0u : 1u;
  release_lease(
      FrameLease{.data = frame,
                 .authority = header->store,
                 .slot = header->slot_and_tier & frame_detail::SlotMask,
                 .generation = header->generation,
                 .tier = tier});
}

} // namespace rund::node
