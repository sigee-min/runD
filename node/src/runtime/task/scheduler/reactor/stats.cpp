#include <rund/task/stats/slots.hpp>

#include "stats.hpp"

#include <algorithm>

namespace rund::node {

void RecordReactorWaitRegistered(
    ::rund::detail::task::StatStorage &stats) noexcept {
  ++::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::ReactorWaitsRegistered);
}

void RecordReactorWaitsCanceled(::rund::detail::task::StatStorage &stats,
                                const std::uint64_t count) noexcept {
  ::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::ReactorWaitsCanceled) += count;
}

void RecordReactorTimedWaitRegistered(
    ::rund::detail::task::StatStorage &stats) noexcept {
  ++::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::ReactorTimedWaitsRegistered);
}

void RecordReactorTimeoutTimerCancel(
    ::rund::detail::task::StatStorage &stats) noexcept {
  ++::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::ReactorTimeoutTimerCancels);
}

void RecordReactorTimeoutCleanupFailure(
    ::rund::detail::task::StatStorage &stats) noexcept {
  ++::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::ReactorTimeoutCleanupFailures);
}

void RecordReactorReadyEvents(::rund::detail::task::StatStorage &stats,
                              const std::size_t count) noexcept {
  ::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::ReactorReadyEvents) += count;
}

void RecordReactorReadyManyRequest(
    ::rund::detail::task::StatStorage &stats) noexcept {
  ++::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::ReactorReadyManyRequests);
}

void RecordReactorReadyManyEvents(::rund::detail::task::StatStorage &stats,
                                  const std::size_t count) noexcept {
  ::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::ReactorReadyManyEvents) += count;
}

void RecordReactorReadySetCreate(
    ::rund::detail::task::StatStorage &stats) noexcept {
  ++::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::ReactorReadySetCreates);
}

void RecordReactorReadySetDestroy(
    ::rund::detail::task::StatStorage &stats) noexcept {
  ++::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::ReactorReadySetDestroys);
}

void RecordReactorReadySetClear(
    ::rund::detail::task::StatStorage &stats) noexcept {
  ++::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::ReactorReadySetClears);
}

void RecordReactorReadySetMemberAdded(
    ::rund::detail::task::StatStorage &stats) noexcept {
  ++::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::ReactorReadySetMembers);
  ++::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::ReactorReadySetMembersAdded);
}

void RecordReactorReadySetMemberRemoved(
    ::rund::detail::task::StatStorage &stats) noexcept {
  ++::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::ReactorReadySetMembersRemoved);
}

void RecordReactorReadySetWait(
    ::rund::detail::task::StatStorage &stats) noexcept {
  ++::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::ReactorReadySetWaits);
}

void RecordReactorReadySetEvents(::rund::detail::task::StatStorage &stats,
                                 const std::size_t count) noexcept {
  ::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::ReactorReadySetReadyEvents) +=
      count;
  ::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::ReactorReadySetEvents) += count;
}

void RecordReactorReadySetInvalidations(
    ::rund::detail::task::StatStorage &stats,
    const std::size_t count) noexcept {
  ::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::ReactorReadySetInvalidations) +=
      count;
}

void RecordReactorCloseInvalidatedWaits(
    ::rund::detail::task::StatStorage &stats,
    const std::size_t count) noexcept {
  ::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::ReactorCloseInvalidatedWaits) +=
      count;
}

void RecordReactorBacklogPush(::rund::detail::task::StatStorage &stats,
                              const std::size_t count) noexcept {
  ::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::ReactorBacklogPushes) += count;
}

void RecordReactorBacklogDrain(::rund::detail::task::StatStorage &stats,
                               const std::size_t count) noexcept {
  ::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::ReactorBacklogDrains) += count;
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
  ++::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::ReactorRegistrationApplyCalls);
}

void RecordReactorRegistrationChangeApplied(
    ::rund::detail::task::StatStorage &stats) noexcept {
  ++::rund::detail::task::Stat(
      stats, ::rund::detail::task::StatSlot::ReactorRegistrationChangesApplied);
}

} // namespace rund::node
