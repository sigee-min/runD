#include "local.hpp"

#include "src/runtime/task/scheduler/reactor/stats.hpp"
#include "src/runtime/task/stats/access.hpp"

#include <rund/task/stats.hpp>

#include <array>
#include <cstdint>
#include <limits>

namespace {

using ::rund::detail::task::StatSlot;

constexpr std::array<StatSlot, ::rund::detail::task::kReactorPublicStatCount>
    ReactorPublicProjection{
#define RUND_SCHEDULER_PUBLIC_STAT(public_name, storage_slot)                  \
  StatSlot::storage_slot,
#include <rund/task/stats/schema/public/reactor.def>
#undef RUND_SCHEDULER_PUBLIC_STAT
    };

[[nodiscard]] consteval std::size_t
ProjectionMultiplicity(const StatSlot slot) noexcept {
  std::size_t count = 0u;
  for (const StatSlot projected : ReactorPublicProjection) {
    count += projected == slot ? 1u : 0u;
  }
  return count;
}

static_assert(::rund::detail::task::kStatCount == 234u);
static_assert(sizeof(::rund::detail::task::StatStorage) == 1872u);
static_assert(sizeof(::rund::task::Stats) == 1872u);
static_assert(sizeof(::rund::task::ReactorStats) == 192u);
static_assert(
    ::rund::detail::task::SlotIndex(StatSlot::ReactorReadySetMembers) == 185u);
static_assert(::rund::detail::task::SlotIndex(
                  StatSlot::ReservedReactorReadySetMembersAdded) == 186u);
static_assert(::rund::detail::task::SlotIndex(
                  StatSlot::ReactorReadySetReadyEvents) == 189u);
static_assert(::rund::detail::task::SlotIndex(
                  StatSlot::ReservedReactorReadySetEvents) == 190u);
static_assert(ProjectionMultiplicity(StatSlot::ReactorReadySetMembers) == 2u);
static_assert(ProjectionMultiplicity(StatSlot::ReactorReadySetReadyEvents) ==
              2u);
static_assert(ProjectionMultiplicity(
                  StatSlot::ReservedReactorReadySetMembersAdded) == 0u);
static_assert(ProjectionMultiplicity(StatSlot::ReservedReactorReadySetEvents) ==
              0u);
static_assert(ProjectionMultiplicity(StatSlot::ReactorReadySetMembersRemoved) ==
              1u);

} // namespace

namespace rund::node::test_contract {

bool NetReadySetTelemetryHasOneAuthority() {
  using ::rund::detail::task::Stat;
  using ::rund::detail::task::StatsAccess;

  constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  constexpr std::uint64_t members_reservation_canary = 41u;
  constexpr std::uint64_t events_reservation_canary = 43u;
  constexpr std::uint64_t live_members = 17u;

  ::rund::detail::task::StatStorage storage{};
  Stat(storage, StatSlot::ReactorReadySetMembers) = maximum - 1u;
  Stat(storage, StatSlot::ReservedReactorReadySetMembersAdded) =
      members_reservation_canary;
  Stat(storage, StatSlot::ReactorReadySetReadyEvents) = maximum - 2u;
  Stat(storage, StatSlot::ReservedReactorReadySetEvents) =
      events_reservation_canary;
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
  READY_SET_ASSERT(
      Stat(storage, StatSlot::ReservedReactorReadySetMembersAdded) ==
      members_reservation_canary);
  READY_SET_ASSERT(Stat(storage, StatSlot::ReservedReactorReadySetEvents) ==
                   events_reservation_canary);
  READY_SET_ASSERT(Stat(storage, StatSlot::ReactorReadySetMembersRemoved) ==
                   6u);
  READY_SET_ASSERT(Stat(storage, StatSlot::ResourceLiveReadySetMembers) ==
                   live_members);
  READY_SET_ASSERT(Stat(storage, StatSlot::ReactorWaitsCanceled) == maximum);

  const ::rund::task::Stats snapshot = StatsAccess::Snapshot(storage);
  const ::rund::task::ReactorStats reactor = snapshot.reactor();
  READY_SET_ASSERT(reactor.ready_set_members() == maximum);
  READY_SET_ASSERT(reactor.ready_set_members_added() == maximum);
  READY_SET_ASSERT(reactor.ready_set_ready_events() == maximum);
  READY_SET_ASSERT(reactor.ready_set_events() == maximum);
  READY_SET_ASSERT(reactor.ready_set_members_removed() == 6u);
  READY_SET_ASSERT(snapshot.resources().live_ready_set_members() ==
                   live_members);
  READY_SET_ASSERT(
      StatsAccess::Value(snapshot,
                         StatSlot::ReservedReactorReadySetMembersAdded) ==
      members_reservation_canary);
  READY_SET_ASSERT(
      StatsAccess::Value(snapshot, StatSlot::ReservedReactorReadySetEvents) ==
      events_reservation_canary);
  return true;
}

} // namespace rund::node::test_contract
