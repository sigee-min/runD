#pragma once

#include <rund/task/stats/slots.hpp>

#include <rund/task/stats/storage.hpp>

#include <cstdint>

namespace rund::node {

inline void RecordLaneResidualCompletionPublish(
    ::rund::detail::task::StatStorage &stats) noexcept {
  ++::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::LaneResidualCompletionPublishes);
}

inline void
RecordLaneResidualContextInstalls(::rund::detail::task::StatStorage &stats,
                                  const std::uint64_t count) noexcept {
  ::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::LaneResidualContextInstalls) +=
      count;
}

inline void
RecordLaneResidualJoinOwnerChecks(::rund::detail::task::StatStorage &stats,
                                  const std::uint64_t count) noexcept {
  ::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::LaneResidualJoinOwnerChecks) +=
      count;
}

inline void
RecordLaneResidualWakeNotifications(::rund::detail::task::StatStorage &stats,
                                    const std::uint64_t count) noexcept {
  ::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::LaneResidualWakeNotifications) +=
      count;
}

inline void
RecordLaneResidualPolicyDecision(::rund::detail::task::StatStorage &stats,
                                 const bool accepted) noexcept {
  if (accepted) {
    ++::rund::detail::task::Stat(
        stats, ::rund::detail::task::StatSlot::LaneResidualPolicyAccepts);
  } else {
    ++::rund::detail::task::Stat(
        stats, ::rund::detail::task::StatSlot::LaneResidualPolicyRejections);
  }
}

[[nodiscard]] inline bool ShouldRecordLaneResidualSegmentMetrics(
    const std::uint64_t participating_lanes) noexcept {
  return participating_lanes > 1u;
}

} // namespace rund::node
