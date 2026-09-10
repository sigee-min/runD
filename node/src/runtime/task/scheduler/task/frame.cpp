#include "frame/internal.hpp"

#include "alignment.hpp"

#include <algorithm>
#include <limits>
#include <mutex>
#include <new>

namespace rund::node {

FrameArena::Store::~Store() {
  for (Tier &tier : tiers) {
    for (void *const page : tier.pages) {
      ::operator delete(page, std::align_val_t{limits.alignment});
    }
    for (void *const block : tier.blocks) {
      ::operator delete(block, std::align_val_t{limits.alignment});
    }
  }
}

FrameArena::~FrameArena() { reset(); }

void FrameArena::drop(Store *const store) noexcept {
  if (store != nullptr &&
      store->refs.fetch_sub(1u, std::memory_order_acq_rel) == 1u) {
    delete store;
  }
}

task::Status FrameArena::configure(const FrameLimits limits) noexcept {
  static_assert(sizeof(Header) == 16u);
  reset();
  if (limits.capacity == 0u || limits.capacity > frame_detail::SlotMask ||
      limits.bytes == 0u || limits.alignment < frame_detail::FrameAlignment ||
      !alignment::power(limits.alignment)) {
    code_ = ReasonCode::TaskFrameLimitsInvalid;
    return task::Status::fail(code_);
  }
  std::size_t prefix = 0u;
  if (!alignment::up(sizeof(Header), limits.alignment, prefix)) {
    code_ = ReasonCode::TaskFrameLimitsInvalid;
    return task::Status::fail(code_);
  }
  Store *const store = new (std::nothrow) Store{};
  if (store == nullptr) {
    code_ = ReasonCode::TaskFrameCapacity;
    return task::Status::fail(code_);
  }
  store->prefix_bytes = prefix;
  store->limits = limits;
  store->tier_count = limits.bytes <= frame_detail::CompactFrameBytes ? 1u : 2u;
  store->tiers[0].frame_bytes =
      std::min(limits.bytes, frame_detail::CompactFrameBytes);
  store->tiers[1].frame_bytes = limits.bytes;
  try {
    for (std::uint8_t tier_index = 0u; tier_index < store->tier_count;
         ++tier_index) {
      Store::Tier &tier = store->tiers[tier_index];
      std::size_t stride = 0u;
      if (tier.frame_bytes > std::numeric_limits<std::size_t>::max() - prefix ||
          !alignment::up(prefix + tier.frame_bytes, limits.alignment, stride) ||
          stride > std::numeric_limits<std::size_t>::max() / limits.capacity) {
        delete store;
        code_ = ReasonCode::TaskFrameLimitsInvalid;
        return task::Status::fail(code_);
      }
      tier.stride_bytes = stride;
      tier.free.reserve(limits.capacity);
      if (tier_index == 0u) {
        tier.generation.reserve(limits.capacity);
        tier.bytes.reserve(limits.capacity);
        tier.live.reserve(limits.capacity);
        tier.pages.reserve(
            (limits.capacity + frame_detail::FramePageSlots - 1u) /
            frame_detail::FramePageSlots);
      }
    }
  } catch (...) {
    delete store;
    code_ = ReasonCode::TaskFrameCapacity;
    return task::Status::fail(code_);
  }
  store_ = store;
  code_ = ReasonCode::Ok;
  return task::Status::success();
}

void FrameArena::begin_epoch() noexcept {
  Store *const store = store_;
  if (store == nullptr) {
    return;
  }
  std::lock_guard lock{store->mutex};
  const std::uint64_t live = store->stats.live;
  const std::uint64_t resident_slots = store->stats.resident_slots;
  const std::uint64_t resident_bytes = store->stats.resident_bytes;
  store->stats = FrameStats{
      .live = live,
      .high_water = live,
      .resident_slots = resident_slots,
      .resident_bytes = resident_bytes,
  };
}

void FrameArena::reset() noexcept {
  Store *const store = store_;
  store_ = nullptr;
  code_ = ReasonCode::TaskFrameNotConfigured;
  if (store == nullptr) {
    return;
  }
  {
    std::lock_guard lock{store->mutex};
    store->retired = true;
  }
  drop(store);
}

} // namespace rund::node
