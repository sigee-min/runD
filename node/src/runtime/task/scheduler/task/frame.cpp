#include "frame.hpp"
#include "alignment.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <mutex>
#include <new>
#include <vector>

namespace rund::node {
namespace {

constexpr std::size_t kFrameAlignment = 16u;
constexpr std::uint32_t kCompactFrameBytes = 160u;
constexpr std::uint32_t kFramePageSlots = 256u;
constexpr std::uint32_t kTierBit = 1u << 31u;
constexpr std::uint32_t kSlotMask = kTierBit - 1u;

} // namespace

struct FrameArena::Store final {
  struct Tier final {
    std::vector<void *> pages{};
    std::vector<void *> blocks{};
    std::size_t stride_bytes{};
    std::uint32_t frame_bytes{};
    std::vector<std::uint32_t> free{};
    std::vector<std::uint32_t> generation{};
    std::vector<std::uint32_t> bytes{};
    std::vector<std::uint8_t> live{};
  };

  ~Store() {
    for (Tier &tier : tiers) {
      for (void *const page : tier.pages) {
        ::operator delete(page, std::align_val_t{limits.alignment});
      }
      for (void *const block : tier.blocks) {
        ::operator delete(block, std::align_val_t{limits.alignment});
      }
    }
  }

  std::atomic<std::uint64_t> refs{1u};
  std::mutex mutex{};
  std::size_t prefix_bytes{};
  std::array<Tier, 2u> tiers{};
  std::uint8_t tier_count{};
  FrameLimits limits{};
  FrameStats stats{};
  ReasonCode code{ReasonCode::Ok};
  bool retired{};
};

struct FrameArena::Header final {
  Store *store{};
  std::uint32_t slot_and_tier{};
  std::uint32_t generation{};
};

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
  if (limits.capacity == 0u || limits.capacity > kSlotMask ||
      limits.bytes == 0u || limits.alignment < kFrameAlignment ||
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
  store->tier_count = limits.bytes <= kCompactFrameBytes ? 1u : 2u;
  store->tiers[0].frame_bytes = std::min(limits.bytes, kCompactFrameBytes);
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
        tier.pages.reserve((limits.capacity + kFramePageSlots - 1u) /
                           kFramePageSlots);
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
        std::min(kFramePageSlots, store->limits.capacity - first);
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
          ? static_cast<std::byte *>(tier.pages[slot / kFramePageSlots]) +
                (slot % kFramePageSlots) * tier.stride_bytes
          : static_cast<std::byte *>(tier.blocks[slot]);
  void *const frame = base + store->prefix_bytes;
  auto *const header = reinterpret_cast<Header *>(
      static_cast<std::byte *>(frame) - sizeof(Header));
  *header = Header{.store = store,
                   .slot_and_tier = slot | (tier_index == 0u ? 0u : kTierBit),
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
                   tier.pages[lease.slot / kFramePageSlots]) +
                   (lease.slot % kFramePageSlots) * tier.stride_bytes
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
  const std::uint8_t tier = (header->slot_and_tier & kTierBit) == 0u ? 0u : 1u;
  release_lease(FrameLease{.data = frame,
                           .authority = header->store,
                           .slot = header->slot_and_tier & kSlotMask,
                           .generation = header->generation,
                           .tier = tier});
}

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
  const std::uint8_t tier = (header->slot_and_tier & kTierBit) == 0u ? 0u : 1u;
  const std::uint32_t slot = header->slot_and_tier & kSlotMask;
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
