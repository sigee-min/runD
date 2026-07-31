#pragma once

#include <rund/task/stats/network.hpp>
#include <rund/task/stats/reactor.hpp>
#include <rund/task/stats/resource.hpp>
#include <rund/task/stats/slots.hpp>

#include <cstdint>

namespace rund::detail::task {
struct StatsAccess;

inline constexpr std::size_t kRootPublicStatCount = 0u
#define RUND_SCHEDULER_PUBLIC_STAT(public_name, storage_slot) +1u
#include <rund/task/stats/schema/public/root.def>
#undef RUND_SCHEDULER_PUBLIC_STAT
    ;
} // namespace rund::detail::task

namespace rund::task {

// A Stats value is a read-only snapshot. The scheduler mutates the
// fixed-size source storage and materializes this value once at the report
// boundary; SDK callers cannot create a second live mutation authority.
class Stats {
public:
  constexpr Stats() noexcept {
    detail::task::Stat(values_, detail::task::StatSlot::TaskWorkers) = 1u;
    detail::task::Stat(values_, detail::task::StatSlot::TraceHash) =
        detail::task::kTraceHashSeed;
  }

  [[nodiscard]] constexpr ReactorStats reactor() const noexcept {
    return ReactorStats{values_};
  }
  [[nodiscard]] constexpr NetworkStats network() const noexcept {
    return NetworkStats{values_};
  }
  [[nodiscard]] constexpr ResourceStats resources() const noexcept {
    return ResourceStats{values_};
  }

#define RUND_SCHEDULER_PUBLIC_STAT(public_name, storage_slot)                  \
  [[nodiscard]] constexpr std::uint64_t public_name() const noexcept {         \
    return detail::task::Stat(values_, detail::task::StatSlot::storage_slot);  \
  }
#include <rund/task/stats/schema/public/root.def>
#undef RUND_SCHEDULER_PUBLIC_STAT

private:
  constexpr explicit Stats(const detail::task::StatStorage &values) noexcept
      : values_(values) {}

  detail::task::StatStorage values_{};

  friend struct detail::task::StatsAccess;
};

static_assert(sizeof(Stats) == sizeof(detail::task::StatStorage));
static_assert(alignof(Stats) == alignof(detail::task::StatStorage));
static_assert(detail::task::kRootPublicStatCount +
                  detail::task::kReactorPublicStatCount +
                  detail::task::kNetworkPublicStatCount +
                  detail::task::kResourcePublicStatCount ==
              detail::task::kStatCount);

} // namespace rund::task
