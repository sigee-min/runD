#include "local.hpp"

#include "src/runtime/task/scheduler/reactor/stats.hpp"
#include "src/runtime/task/stats/access.hpp"

#include <rund/task/stats.hpp>

#include <cstdint>
#include <limits>

namespace {

using ::rund::detail::task::StatSlot;

static_assert(::rund::detail::task::kStatCount == 232u);
static_assert(sizeof(::rund::detail::task::StatStorage) == 1856u);
static_assert(sizeof(::rund::task::Stats) == 1856u);
static_assert(sizeof(::rund::task::ReactorStats) == 176u);
static_assert(
    ::rund::detail::task::SlotIndex(StatSlot::ReactorReadySetMembers) == 185u);
static_assert(::rund::detail::task::SlotIndex(
                  StatSlot::ReactorReadySetMembersRemoved) == 186u);
static_assert(::rund::detail::task::SlotIndex(
                  StatSlot::ReactorReadySetReadyEvents) == 188u);

} // namespace

namespace rund::node::test_contract {

bool NetReadySetTelemetryHasOneAuthority() {
  using ::rund::detail::task::Stat;
  using ::rund::detail::task::StatsAccess;

  constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  constexpr std::uint64_t live_members = 17u;

  ::rund::detail::task::StatStorage storage{};
  Stat(storage, StatSlot::ReactorReadySetMembers) = maximum - 1u;
  Stat(storage, StatSlot::ReactorReadySetReadyEvents) = maximum - 2u;
  Stat(storage, StatSlot::ReactorReadySetMembersRemoved) = 5u;
  Stat(storage, StatSlot::ResourceLiveReadySetMembers) = live_members;
  Stat(storage, StatSlot::ReactorWaitsCanceled) = maximum - 1u;

  RecordReactorReadySetMemberAdded(storage);
  RecordReactorReadySetMemberAdded(storage);
  RecordReactorReadySetEvents(storage, 2u);
  RecordReactorReadySetEvents(storage, 1u);
  RecordReactorReadySetMemberRemoved(storage);
  RecordReactorWaitsCanceled(storage, 2u);
  RecordReactorWaitsCanceled(storage, 1u);

  READY_SET_ASSERT(Stat(storage, StatSlot::ReactorReadySetMembers) == maximum);
  READY_SET_ASSERT(Stat(storage, StatSlot::ReactorReadySetReadyEvents) ==
                   maximum);
  READY_SET_ASSERT(Stat(storage, StatSlot::ReactorReadySetMembersRemoved) ==
                   6u);
  READY_SET_ASSERT(Stat(storage, StatSlot::ResourceLiveReadySetMembers) ==
                   live_members);
  READY_SET_ASSERT(Stat(storage, StatSlot::ReactorWaitsCanceled) == maximum);

  const ::rund::task::Stats snapshot = StatsAccess::Snapshot(storage);
  const ::rund::task::ReactorStats reactor = snapshot.reactor();
  READY_SET_ASSERT(reactor.ready_set_members() == maximum);
  READY_SET_ASSERT(reactor.ready_set_ready_events() == maximum);
  READY_SET_ASSERT(reactor.ready_set_members_removed() == 6u);
  READY_SET_ASSERT(snapshot.resources().live_ready_set_members() ==
                   live_members);
  return true;
}

} // namespace rund::node::test_contract
