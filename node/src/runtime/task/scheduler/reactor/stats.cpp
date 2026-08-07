#include <rund/counter.hpp>
#include <rund/task/stats/slots.hpp>

#include "stats.hpp"

#include <algorithm>

namespace rund::node {

void RecordReactorWaitRegistered(
    ::rund::detail::task::StatStorage &stats) noexcept {
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(
          stats, ::rund::detail::task::StatSlot::ReactorWaitsRegistered),
      1u);
}

void RecordReactorWaitsCanceled(::rund::detail::task::StatStorage &stats,
                                const std::uint64_t count) noexcept {
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(
          stats, ::rund::detail::task::StatSlot::ReactorWaitsCanceled),
      count);
}

void RecordReactorTimedWaitRegistered(
    ::rund::detail::task::StatStorage &stats) noexcept {
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(
          stats, ::rund::detail::task::StatSlot::ReactorTimedWaitsRegistered),
      1u);
}

void RecordReactorTimeoutTimerCancel(
    ::rund::detail::task::StatStorage &stats) noexcept {
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(
          stats, ::rund::detail::task::StatSlot::ReactorTimeoutTimerCancels),
      1u);
}

void RecordReactorTimeoutCleanupFailure(
    ::rund::detail::task::StatStorage &stats) noexcept {
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(
          stats, ::rund::detail::task::StatSlot::ReactorTimeoutCleanupFailures),
      1u);
}

void RecordReactorReadyEvents(::rund::detail::task::StatStorage &stats,
                              const std::size_t count) noexcept {
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(
          stats, ::rund::detail::task::StatSlot::ReactorReadyEvents),
      count);
}

void RecordReactorReadyManyRequest(
    ::rund::detail::task::StatStorage &stats) noexcept {
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(
          stats, ::rund::detail::task::StatSlot::ReactorReadyManyRequests),
      1u);
}

void RecordReactorReadyManyEvents(::rund::detail::task::StatStorage &stats,
                                  const std::size_t count) noexcept {
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(
          stats, ::rund::detail::task::StatSlot::ReactorReadyManyEvents),
      count);
}

void RecordReactorReadySetCreate(
    ::rund::detail::task::StatStorage &stats) noexcept {
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(
          stats, ::rund::detail::task::StatSlot::ReactorReadySetCreates),
      1u);
}

void RecordReactorReadySetDestroy(
    ::rund::detail::task::StatStorage &stats) noexcept {
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(
          stats, ::rund::detail::task::StatSlot::ReactorReadySetDestroys),
      1u);
}

void RecordReactorReadySetClear(
    ::rund::detail::task::StatStorage &stats) noexcept {
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(
          stats, ::rund::detail::task::StatSlot::ReactorReadySetClears),
      1u);
}

void RecordReactorReadySetMemberAdded(
    ::rund::detail::task::StatStorage &stats) noexcept {
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(
          stats, ::rund::detail::task::StatSlot::ReactorReadySetMembers),
      1u);
}

void RecordReactorReadySetMemberRemoved(
    ::rund::detail::task::StatStorage &stats) noexcept {
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(
          stats, ::rund::detail::task::StatSlot::ReactorReadySetMembersRemoved),
      1u);
}

void RecordReactorReadySetWait(
    ::rund::detail::task::StatStorage &stats) noexcept {
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(
          stats, ::rund::detail::task::StatSlot::ReactorReadySetWaits),
      1u);
}

void RecordReactorReadySetEvents(::rund::detail::task::StatStorage &stats,
                                 const std::size_t count) noexcept {
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(
          stats, ::rund::detail::task::StatSlot::ReactorReadySetReadyEvents),
      count);
}

void RecordReactorReadySetInvalidations(
    ::rund::detail::task::StatStorage &stats,
    const std::size_t count) noexcept {
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(
          stats, ::rund::detail::task::StatSlot::ReactorReadySetInvalidations),
      count);
}

void RecordReactorCloseInvalidatedWaits(
    ::rund::detail::task::StatStorage &stats,
    const std::size_t count) noexcept {
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(
          stats, ::rund::detail::task::StatSlot::ReactorCloseInvalidatedWaits),
      count);
}

void RecordReactorBacklogPush(::rund::detail::task::StatStorage &stats,
                              const std::size_t count) noexcept {
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(
          stats, ::rund::detail::task::StatSlot::ReactorBacklogPushes),
      count);
}

void RecordReactorBacklogDrain(::rund::detail::task::StatStorage &stats,
                               const std::size_t count) noexcept {
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(
          stats, ::rund::detail::task::StatSlot::ReactorBacklogDrains),
      count);
}

void RecordReactorBacklogDepth(::rund::detail::task::StatStorage &stats,
                               const std::size_t depth) noexcept {
  ::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::ReactorMaxBacklogDepth) =
      std::max<std::uint64_t>(
          ::rund::detail::task::Stat(
              stats, ::rund::detail::task::StatSlot::ReactorMaxBacklogDepth),
          depth);
}

void RecordReactorRegistrationApplyCall(
    ::rund::detail::task::StatStorage &stats) noexcept {
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(
          stats, ::rund::detail::task::StatSlot::ReactorRegistrationApplyCalls),
      1u);
}

void RecordReactorRegistrationChangeApplied(
    ::rund::detail::task::StatStorage &stats) noexcept {
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(
          stats,
          ::rund::detail::task::StatSlot::ReactorRegistrationChangesApplied),
      1u);
}

} // namespace rund::node
