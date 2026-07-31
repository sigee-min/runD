#pragma once

#include <rund/task/stats/storage.hpp>

#include <cstddef>
#include <cstdint>

namespace rund::detail::task {

inline constexpr std::uint64_t kTraceHashSeed = 1469598103934665603ull;

// Slot identity and numeric layout come only from schema/slots.def.
enum class StatSlot : std::uint16_t {
#define RUND_SCHEDULER_STAT_SLOT(slot, index) slot = index,
#include <rund/task/stats/schema/slots.def>
#undef RUND_SCHEDULER_STAT_SLOT
  Count = kStatCount,
};

[[nodiscard]] constexpr std::size_t SlotIndex(const StatSlot slot) noexcept {
  return static_cast<std::size_t>(slot);
}

[[nodiscard]] constexpr std::uint64_t &Stat(StatStorage &storage,
                                            const StatSlot slot) noexcept {
  return storage.counter(SlotIndex(slot));
}

[[nodiscard]] constexpr std::uint64_t Stat(const StatStorage &storage,
                                           const StatSlot slot) noexcept {
  return storage.counter(SlotIndex(slot));
}

[[nodiscard]] consteval bool StatSlotsAreContiguous() noexcept {
  std::size_t expected = 0u;
#define RUND_SCHEDULER_STAT_SLOT(slot, index)                                  \
  if (index != expected++) {                                                   \
    return false;                                                              \
  }
#include <rund/task/stats/schema/slots.def>
#undef RUND_SCHEDULER_STAT_SLOT
  return expected == kStatCount;
}

static_assert(SlotIndex(StatSlot::Count) == kStatCount);
static_assert(StatSlotsAreContiguous());

} // namespace rund::detail::task
