#pragma once

#include <cstddef>
#include <cstdint>

namespace rund::detail::task {

inline constexpr std::size_t kStatCount = 0u
#define RUND_SCHEDULER_STAT_SLOT(slot, index) +1u
#include <rund/task/stats/schema/slots.def>
#undef RUND_SCHEDULER_STAT_SLOT
    ;

struct StatStorage {
  std::uint64_t counters[kStatCount]{};

  [[nodiscard]] constexpr std::uint64_t &
  counter(const std::size_t index) noexcept {
    return counters[index];
  }
  [[nodiscard]] constexpr std::uint64_t
  counter(const std::size_t index) const noexcept {
    return counters[index];
  }
};

static_assert(sizeof(StatStorage) == 1872u);
static_assert(alignof(StatStorage) == alignof(std::uint64_t));

} // namespace rund::detail::task
