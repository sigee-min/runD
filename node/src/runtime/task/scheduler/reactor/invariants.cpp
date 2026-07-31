#include "invariants.hpp"

#include "../state/model/timer.hpp"
#include "../state/storage.hpp"
#include "registry.hpp"

namespace rund::node {
namespace {

[[nodiscard]] bool ReactorWaitIdIsLive(const ReactorRuntime &reactor,
                                       const std::uint64_t wait_id) noexcept {
  return ReactorRegistryFindWait(reactor, wait_id) != nullptr;
}

[[nodiscard]] bool
ReactorManyTimerOwnerIsLive(const std::vector<ReactorManyGroup> &groups,
                            const std::uint64_t timer_wait_id) noexcept {
  if (timer_wait_id == 0u) {
    return false;
  }
  for (const ReactorManyGroup &group : groups) {
    if (group.timer_wait_id == timer_wait_id) {
      return true;
    }
  }
  return false;
}

} // namespace

ReactorInvariantSnapshot
Scheduler::ValidateReactorCleanupInvariants() noexcept {
  ReactorInvariantSnapshot snapshot{};
  if (state_ == nullptr) {
    return snapshot;
  }

  snapshot.ok = true;
  snapshot.reason = "ok";
  snapshot.waits = static_cast<std::uint64_t>(
      ReactorRegistrySize(state_->reactor.reactor));
  snapshot.ready_backlog_entries =
      static_cast<std::uint64_t>(state_->reactor.reactor.ready_backlog.size());
  snapshot.many_groups =
      static_cast<std::uint64_t>(state_->reactor.reactor_many_groups.size());
  for (const ReactorReadySet &set : state_->reactor.reactor_ready_sets) {
    snapshot.ready_set_member_storage += set.members.size();
    snapshot.ready_set_member_capacity += set.members.capacity();
  }
  snapshot.many_validation_comparisons =
      state_->reactor.reactor_many_validation_comparisons;
  snapshot.many_request_copies = state_->reactor.reactor_many_request_copies;
  snapshot.many_storage_growths = state_->reactor.reactor_many_storage_growths;
  snapshot.ready_set_storage_growths =
      state_->reactor.reactor_ready_set_storage_growths;

  for (const TimerWait &timer : state_->ready.timers) {
    if (timer.kind == TimerWaitKind::ReactorTimeout ||
        timer.kind == TimerWaitKind::ReactorManyTimeout) {
      ++snapshot.timeout_timers;
    }
  }

  for (const ReactorManyGroup &group : state_->reactor.reactor_many_groups) {
    if (group.ready_set_id != 0u) {
      ++snapshot.ready_set_waits;
    }
  }

  for (const ReactorReady &ready : state_->reactor.reactor.ready_backlog) {
    if (!ReactorWaitIdIsLive(state_->reactor.reactor, ready.wait_id)) {
      snapshot.ok = false;
      snapshot.reason = "reactor_backlog_without_wait";
      return snapshot;
    }
  }

  for (const TimerWait &timer : state_->ready.timers) {
    if (timer.kind == TimerWaitKind::ReactorTimeout &&
        !ReactorWaitIdIsLive(state_->reactor.reactor, timer.wait_id)) {
      snapshot.ok = false;
      snapshot.reason = "reactor_timeout_without_wait";
      return snapshot;
    }
    if (timer.kind == TimerWaitKind::ReactorManyTimeout &&
        !ReactorManyTimerOwnerIsLive(state_->reactor.reactor_many_groups,
                                     timer.wait_id)) {
      snapshot.ok = false;
      snapshot.reason = "reactor_many_timeout_without_group";
      return snapshot;
    }
  }
  return snapshot;
}

ReactorInvariantSnapshot ValidateReactorCleanupInvariantsForTest() noexcept {
  Scheduler *const scheduler = Scheduler::Active();
  if (scheduler == nullptr) {
    return ReactorInvariantSnapshot{};
  }
  return scheduler->ValidateReactorCleanupInvariants();
}

} // namespace rund::node
