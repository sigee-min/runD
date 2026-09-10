#include "internal.hpp"

#include "../alignment.hpp"

#include <algorithm>
#include <limits>
#include <mutex>
#include <new>

namespace rund::node {

FrameLease FrameArena::acquire(const std::size_t bytes,
                               const std::size_t alignment) noexcept {
  Store *const store = store_;
  if (store == nullptr) {
    code_ = ReasonCode::TaskFrameNotConfigured;
    return {};
  }
  std::lock_guard lock{store->mutex};
  if (store->retired) {
    store->code = ReasonCode::TaskFrameNotConfigured;
    ++store->stats.failures;
    return {};
  }
  if (bytes > store->limits.bytes) {
    store->code = ReasonCode::TaskFrameTooLarge;
    ++store->stats.failures;
    return {};
  }
  if (alignment == 0u || alignment > store->limits.alignment ||
      !alignment::power(alignment)) {
    store->code = ReasonCode::TaskFrameAlignment;
    ++store->stats.failures;
    return {};
  }
  if (store->stats.live >= store->limits.capacity) {
    store->code = ReasonCode::TaskFrameCapacity;
    ++store->stats.failures;
    return {};
  }
  const std::uint8_t tier_index =
      bytes <= store->tiers[0].frame_bytes ? 0u : 1u;
  Store::Tier &tier = store->tiers[tier_index];
  if (tier_index >= store->tier_count) {
    store->code = ReasonCode::TaskFrameCapacity;
    ++store->stats.failures;
    return {};
  }
  std::uint32_t slot{};
  if (!tier.free.empty()) {
    slot = tier.free.back();
    tier.free.pop_back();
  } else if (tier_index == 0u &&
             tier.generation.size() < store->limits.capacity) {
    const std::uint32_t first =
        static_cast<std::uint32_t>(tier.generation.size());
    const std::uint32_t count =
        std::min(frame_detail::FramePageSlots, store->limits.capacity - first);
    void *const page =
        ::operator new(tier.stride_bytes * count,
                       std::align_val_t{store->limits.alignment}, std::nothrow);
    if (page == nullptr) {
      store->code = ReasonCode::TaskFrameCapacity;
      ++store->stats.failures;
      return {};
    }
    try {
      tier.pages.push_back(page);
      for (std::uint32_t index = 0u; index < count; ++index) {
        tier.generation.push_back(1u);
        tier.bytes.push_back(0u);
        tier.live.push_back(0u);
      }
      for (std::uint32_t index = count; index != 1u; --index) {
        tier.free.push_back(first + index - 1u);
      }
    } catch (...) {
      if (!tier.pages.empty() && tier.pages.back() == page) {
        tier.pages.pop_back();
      }
      ::operator delete(page, std::align_val_t{store->limits.alignment});
      store->code = ReasonCode::TaskFrameCapacity;
      ++store->stats.failures;
      return {};
    }
    store->stats.resident_slots += count;
    store->stats.resident_bytes += tier.stride_bytes * count;
    slot = first;
  } else if (tier_index != 0u && tier.blocks.size() < store->limits.capacity) {
    void *const block =
        ::operator new(tier.stride_bytes,
                       std::align_val_t{store->limits.alignment}, std::nothrow);
    if (block == nullptr) {
      store->code = ReasonCode::TaskFrameCapacity;
      ++store->stats.failures;
      return {};
    }
    slot = static_cast<std::uint32_t>(tier.blocks.size());
    try {
      tier.blocks.push_back(block);
      tier.generation.push_back(1u);
      tier.bytes.push_back(0u);
      tier.live.push_back(0u);
    } catch (...) {
      tier.blocks.resize(slot);
      tier.generation.resize(slot);
      tier.bytes.resize(slot);
      tier.live.resize(slot);
      ::operator delete(block, std::align_val_t{store->limits.alignment});
      store->code = ReasonCode::TaskFrameCapacity;
      ++store->stats.failures;
      return {};
    }
    ++store->stats.resident_slots;
    store->stats.resident_bytes += tier.stride_bytes;
  } else {
    store->code = ReasonCode::TaskFrameCapacity;
    ++store->stats.failures;
    return {};
  }
  const bool reused = tier.generation[slot] != 1u;
  tier.live[slot] = 1u;
  tier.bytes[slot] = static_cast<std::uint32_t>(bytes);
  auto *const base =
      tier_index == 0u
          ? static_cast<std::byte *>(
                tier.pages[slot / frame_detail::FramePageSlots]) +
                (slot % frame_detail::FramePageSlots) * tier.stride_bytes
          : static_cast<std::byte *>(tier.blocks[slot]);
  void *const frame = base + store->prefix_bytes;
  auto *const header = reinterpret_cast<Header *>(
      static_cast<std::byte *>(frame) - sizeof(Header));
  *header = Header{.store = store,
                   .slot_and_tier =
                       slot | (tier_index == 0u ? 0u : frame_detail::TierBit),
                   .generation = tier.generation[slot]};
  store->refs.fetch_add(1u, std::memory_order_relaxed);
  ++store->stats.allocations;
  if (reused) {
    ++store->stats.reuses;
  }
  if (tier_index == 0u) {
    ++store->stats.compact_allocations;
  } else {
    ++store->stats.wide_allocations;
  }
  ++store->stats.live;
  store->stats.high_water =
      std::max(store->stats.high_water, store->stats.live);
  store->code = ReasonCode::Ok;
  return FrameLease{.data = frame,
                    .authority = store,
                    .slot = slot,
                    .generation = tier.generation[slot],
                    .bytes = static_cast<std::uint32_t>(bytes),
                    .alignment = static_cast<std::uint32_t>(alignment),
                    .tier = tier_index};
}

} // namespace rund::node
