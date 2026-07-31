#pragma once

#include <rund/task/stats/slots.hpp>

#include <array>
#include <cstddef>

namespace rund::node::replay_detail {

using SchedulerStatSlot = ::rund::detail::task::StatSlot;

inline constexpr std::array<SchedulerStatSlot, 13u> kSchedulerReplaySlots{
    SchedulerStatSlot::Spawned,
    SchedulerStatSlot::Completed,
    SchedulerStatSlot::Failed,
    SchedulerStatSlot::Yields,
    SchedulerStatSlot::Joins,
    SchedulerStatSlot::Timers,
    SchedulerStatSlot::ReactorWaits,
    SchedulerStatSlot::ChannelSends,
    SchedulerStatSlot::ChannelRecvs,
    SchedulerStatSlot::ChannelCloses,
    SchedulerStatSlot::Observations,
    SchedulerStatSlot::ObservationDropped,
    SchedulerStatSlot::TraceHash,
};

[[nodiscard]] consteval bool SchedulerReplaySlotsAreValid() noexcept {
  for (std::size_t index = 0u; index < kSchedulerReplaySlots.size(); ++index) {
    if (::rund::detail::task::SlotIndex(kSchedulerReplaySlots[index]) >=
        ::rund::detail::task::kStatCount) {
      return false;
    }
    for (std::size_t other = index + 1u;
         other < kSchedulerReplaySlots.size(); ++other) {
      if (kSchedulerReplaySlots[index] == kSchedulerReplaySlots[other]) {
        return false;
      }
    }
  }
  return true;
}

static_assert(SchedulerReplaySlotsAreValid());

} // namespace rund::node::replay_detail
