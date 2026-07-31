#pragma once

#include <rund/task/stats/slots.hpp>

#include <cstddef>
#include <cstdint>

namespace rund::detail::task {

inline constexpr std::size_t kNetworkPublicStatCount = 0u
#define RUND_SCHEDULER_PUBLIC_STAT(public_name, storage_slot) +1u
#include <rund/task/stats/schema/public/network.def>
#undef RUND_SCHEDULER_PUBLIC_STAT
    ;

} // namespace rund::detail::task

namespace rund::task {

class Stats;

class NetworkStats {
public:
  constexpr NetworkStats() noexcept : values_{} {}

#define RUND_SCHEDULER_PUBLIC_STAT(public_name, storage_slot)                  \
  [[nodiscard]] constexpr std::uint64_t public_name() const noexcept {         \
    return values_[static_cast<std::size_t>(Field::public_name)];              \
  }
#include <rund/task/stats/schema/public/network.def>
#undef RUND_SCHEDULER_PUBLIC_STAT

private:
  enum class Field : std::size_t {
#define RUND_SCHEDULER_PUBLIC_STAT(public_name, storage_slot) public_name,
#include <rund/task/stats/schema/public/network.def>
#undef RUND_SCHEDULER_PUBLIC_STAT
    Count,
  };

  static_assert(static_cast<std::size_t>(Field::Count) ==
                detail::task::kNetworkPublicStatCount);

  constexpr explicit NetworkStats(
      const detail::task::StatStorage &source) noexcept
      : values_{} {
    std::size_t index = 0u;
#define RUND_SCHEDULER_PUBLIC_STAT(public_name, storage_slot)                  \
  values_[index++] =                                                           \
      detail::task::Stat(source, detail::task::StatSlot::storage_slot);
#include <rund/task/stats/schema/public/network.def>
#undef RUND_SCHEDULER_PUBLIC_STAT
  }

  std::uint64_t values_[detail::task::kNetworkPublicStatCount];

  friend class Stats;
};

static_assert(sizeof(NetworkStats) ==
              detail::task::kNetworkPublicStatCount * sizeof(std::uint64_t));

} // namespace rund::task
