#pragma once

#include <rund/task/stats.hpp>

namespace rund::detail::task {

// The installed SDK only sees the incomplete friend declaration. Live
// materialization and replay decoding use this source-private authority.
struct StatsAccess {
  [[nodiscard]] static constexpr ::rund::task::Stats
  Snapshot(const StatStorage &storage) noexcept {
    return ::rund::task::Stats{storage};
  }

  [[nodiscard]] static constexpr std::uint64_t &
  Counter(::rund::task::Stats &stats, const StatSlot slot) noexcept {
    return Stat(stats.values_, slot);
  }

  [[nodiscard]] static constexpr std::uint64_t
  Value(const ::rund::task::Stats &stats, const StatSlot slot) noexcept {
    return Stat(stats.values_, slot);
  }
};

} // namespace rund::detail::task
