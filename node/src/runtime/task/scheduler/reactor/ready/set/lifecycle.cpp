#include "operations.hpp"

#include <algorithm>
#include <vector>

namespace rund::node {

bool Scheduler::CancelReadySetWaitGroups(const ::rund::net::ready::Set handle,
                                         const ReasonCode code) noexcept {
  std::vector<std::uint64_t> &group_ids =
      state_->reactor.reactor_many_group_id_scratch;
  group_ids.clear();
  try {
    for (const ReactorManyGroup &group : state_->reactor.reactor_many_groups) {
      if (!group.completed &&
          ReactorReadySetIdentityOwner::same(group.ready_set, handle)) {
        group_ids.push_back(group.group_id);
      }
    }
  } catch (...) {
    return false;
  }
  bool cleanup_ok = true;
  for (const std::uint64_t group_id : group_ids) {
    ReactorManyGroup *const group =
        ReactorManyFindGroup(state_->reactor.reactor_many_groups, group_id);
    if (group == nullptr || group->completed) {
      continue;
    }
    if (!ReactorCleanupWait(
            *this,
            ReactorCleanupRequest{.wait_id = 0u,
                                  .group_id = group_id,
                                  .reason = code,
                                  .timeout_cleanup =
                                      ReactorTimeoutCleanupPolicy::IfPresent,
                                  .remove_ready_backlog = true,
                                  .cleanup_siblings = true})) {
      cleanup_ok = false;
    }
  }
  return cleanup_ok;
}

::rund::net::ready::Status
Scheduler::CreateReadySet(const ::rund::net::ready::Config options) noexcept {
  (void)TrapLaneOwnedSegmentPrimitive();
  EnsureCurrentCommit();
  const auto finish = [this](const ReasonCode code,
                             const ::rund::net::ready::Set set = {}) noexcept {
    ::rund::net::ready::Status result = ReadySetStatus(code, set);
    CompletePrimitiveCommit();
    return result;
  };

  if (options.max_members == 0u) {
    return finish(ReasonCode::TaskInvalid);
  }
  if (ReactorReadySetLiveCount(state_->reactor.reactor_ready_sets) >=
      state_->resources.limits.net_ready_set_capacity) {
    return finish(ReasonCode::ReactorWaitCapacityExceeded);
  }
  const std::uint32_t max_members = std::min<std::uint32_t>(
      options.max_members,
      state_->resources.limits.net_ready_set_member_capacity);
  if (max_members == 0u) {
    return finish(ReasonCode::ReactorWaitCapacityExceeded);
  }
  ReactorReadySet *slot =
      ReactorReadySetSelectActivationSlot(state_->reactor.reactor_ready_sets);
  const std::size_t set_capacity =
      state_->reactor.reactor_ready_sets.capacity();
  bool added_slot = false;
  try {
    if (slot == nullptr) {
      state_->reactor.reactor_ready_sets.push_back(ReactorReadySet{});
      slot = &state_->reactor.reactor_ready_sets.back();
      added_slot = true;
    }
    const std::size_t member_capacity = slot->members.capacity();
    slot->members.clear();
    slot->members.reserve(max_members);
    state_->reactor.reactor_ready_set_storage_growths +=
        slot->members.capacity() != member_capacity ? 1u : 0u;
  } catch (...) {
    if (added_slot) {
      state_->reactor.reactor_ready_sets.pop_back();
    }
    return finish(ReasonCode::ReactorWaitCapacityExceeded);
  }
  slot->max_members = max_members;
  slot->next_member_index = 0u;
  ::rund::net::ready::Set handle{};
  if (!ProcessReadySetIdentityOwner().activate(slot->identity, &handle)) {
    if (added_slot) {
      state_->reactor.reactor_ready_sets.pop_back();
    }
    return finish(ReasonCode::ReactorWaitCapacityExceeded);
  }
  state_->reactor.reactor_ready_set_storage_growths +=
      state_->reactor.reactor_ready_sets.capacity() != set_capacity ? 1u : 0u;
  RecordReactorReadySetCreate(state_->evidence.metrics);
  return finish(ReasonCode::Ok, handle);
}

::rund::net::ready::Status
Scheduler::DestroyReadySet(const ::rund::net::ready::Set handle) noexcept {
  (void)TrapLaneOwnedSegmentPrimitive();
  EnsureCurrentCommit();
  const auto finish = [this](const ReasonCode code,
                             const ::rund::net::ready::Set set = {}) noexcept {
    ::rund::net::ready::Status result = ReadySetStatus(code, set);
    CompletePrimitiveCommit();
    return result;
  };

  ReactorReadySet *const set =
      ReactorReadySetFind(state_->reactor.reactor_ready_sets, handle);
  if (set == nullptr) {
    return finish(ReasonCode::TaskInvalid);
  }
  const ::rund::net::ready::Set live_handle = set->identity.handle;
  const std::uint32_t invalidated_members = ReactorReadySetClearMembers(*set);
  const bool canceled =
      CancelReadySetWaitGroups(live_handle, ReasonCode::TaskCancelled);
  ::rund::net::ready::Set tombstone{};
  if (!ProcessReadySetIdentityOwner().retire(set->identity, live_handle,
                                             &tombstone)) {
    return finish(ReasonCode::TaskInvalid);
  }
  RecordReactorReadySetDestroy(state_->evidence.metrics);
  RecordReactorReadySetInvalidations(state_->evidence.metrics,
                                     invalidated_members);
  return finish(canceled ? ReasonCode::Ok : ReasonCode::IoPollFailed,
                tombstone);
}

::rund::net::ready::Status
Scheduler::ClearReadySet(const ::rund::net::ready::Set handle) noexcept {
  (void)TrapLaneOwnedSegmentPrimitive();
  EnsureCurrentCommit();
  const auto finish = [this](const ReasonCode code,
                             const ::rund::net::ready::Set set = {}) noexcept {
    ::rund::net::ready::Status result = ReadySetStatus(code, set);
    CompletePrimitiveCommit();
    return result;
  };

  ReactorReadySet *const set =
      ReactorReadySetFind(state_->reactor.reactor_ready_sets, handle);
  if (set == nullptr) {
    return finish(ReasonCode::TaskInvalid);
  }
  const std::uint32_t invalidated_members = ReactorReadySetClearMembers(*set);
  set->next_member_index = 0u;
  const bool canceled =
      CancelReadySetWaitGroups(set->identity.handle, ReasonCode::TaskCancelled);
  RecordReactorReadySetClear(state_->evidence.metrics);
  RecordReactorReadySetInvalidations(state_->evidence.metrics,
                                     invalidated_members);
  return finish(canceled ? ReasonCode::Ok : ReasonCode::IoPollFailed,
                set->identity.handle);
}

} // namespace rund::node
