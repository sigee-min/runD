#include <algorithm>
#include <vector>

#include "../state/model/task.hpp"
#include "../state/model/timer.hpp"
#include "../state/storage.hpp"
#include "apply/policy.hpp"
#include "backend.hpp"
#include "backlog.hpp"
#include "budget.hpp"
#include "expand.hpp"
#include "generation.hpp"
#include "registration.hpp"
#include "registry.hpp"
#include "scratch.hpp"

#include <chrono>
#include <limits>

namespace rund::node {
namespace {

[[nodiscard]] int RemainingPollTimeout(
    const int configured_timeout_ms,
    const Clock::time_point deadline) noexcept {
  if (configured_timeout_ms <= 0) {
    return configured_timeout_ms;
  }
  const auto remaining = deadline - Clock::now();
  if (remaining <= Clock::duration::zero()) {
    return 0;
  }
  const auto nanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(remaining).count();
  constexpr std::int64_t kNanosecondsPerMillisecond = 1'000'000ll;
  const std::int64_t milliseconds =
      (nanoseconds + kNanosecondsPerMillisecond - 1ll) /
      kNanosecondsPerMillisecond;
  return milliseconds > std::numeric_limits<int>::max()
             ? std::numeric_limits<int>::max()
             : static_cast<int>(milliseconds);
}

} // namespace

bool Scheduler::DrainReadyReactor(const int timeout_ms,
                                  const bool force_apply) noexcept {
  ReactorRuntime &reactor = state_->reactor.reactor;
  bool invalidated = false;
  if (!ReactorGenerationCleanupInvalidWaits(*this, &invalidated)) {
    return false;
  }
  if (invalidated) {
    return true;
  }
  if (state_->ready.ready_depth == 0u &&
      ReactorRegistrationHasDeferredRemoves(reactor)) {
    if (!ReactorRegistrationFlushDeferredRemoves(reactor)) {
      return false;
    }
  }
  if (ReactorRegistryEmpty(reactor)) {
    if (!reactor.changes.empty()) {
      ReactorApplyPolicyRecordFlush(reactor, force_apply);
      (void)ReactorBackendApplyChanges(reactor, state_->evidence.metrics);
    }
    return false;
  }
  if (timeout_ms == 0 && state_->ready.ready_depth > 1u) {
    return false;
  }

  const std::size_t configured_budget =
      state_->resources.limits.reactor_ready_budget == 0u
          ? state_->resources.limits.ready_queue_capacity
          : state_->resources.limits.reactor_ready_budget;
  const std::size_t wake_capacity = state_->resources.limits.task_capacity;
  const std::size_t queue_space =
      state_->ready.ready_depth >= wake_capacity
          ? 0u
          : wake_capacity - state_->ready.ready_depth;
  const std::size_t budget = std::min(configured_budget, queue_space);
  if (budget == 0u) {
    return false;
  }
  if (ReactorBacklogHasReady(reactor)) {
    if (!ReactorBacklogTakePrefix(reactor, state_->evidence.metrics, budget,
                                  reactor.budget_ready_scratch)) {
      return false;
    }
    return DrainReactorReadyBatch(reactor.budget_ready_scratch);
  }

  const bool defer_apply = ReactorApplyPolicyShouldDefer(
      reactor, state_->ready.ready_depth, force_apply);
  if (defer_apply) {
    return false;
  }
  ReactorApplyPolicyRecordFlush(reactor, force_apply);
  const ReactorApplyResult applied =
      ReactorBackendApplyChanges(reactor, state_->evidence.metrics);
  if (applied.disposition() != ReactorApplyDisposition::Success) {
    if (applied.disposition() == ReactorApplyDisposition::Invalid) {
      if (!ReactorExpandInvalidHandle(reactor, applied.invalid_handle())) {
        return false;
      }
    } else if (!ReactorExpandPollFailure(reactor)) {
      return false;
    }
  } else {
    const Clock::time_point poll_deadline =
        timeout_ms > 0 ? Clock::now() + std::chrono::milliseconds{timeout_ms}
                       : Clock::time_point{};
    int poll_timeout_ms = timeout_ms;
    for (;;) {
      const ReactorPlatformPollResult poll = PollReactorPlatform(
          reactor.platform, poll_timeout_ms, ReactorRegistrySize(reactor),
          reactor.platform_ready);
      if (poll.disposition() != ReactorPlatformPollDisposition::Success) {
        if (poll.disposition() == ReactorPlatformPollDisposition::Invalid) {
          if (!ReactorExpandInvalidAll(reactor)) {
            return false;
          }
        } else if (!ReactorExpandPollFailure(reactor)) {
          return false;
        }
        break;
      }
      if (reactor.platform_ready.empty()) {
        reactor.ready.clear();
        return false;
      }
      if (!ReactorExpandPlatformReady(reactor, reactor.platform_ready)) {
        return false;
      }
      if (!reactor.ready.empty()) {
        break;
      }

      // A backend may publish an event that was already queued when its
      // registration was removed. That native event has no logical wait and
      // therefore cannot prove scheduler quiescence. Drain it and poll again
      // inside the original timeout window; restarting the full timeout would
      // move timer ordering, while returning false would report a false
      // deadlock with live reactor waits.
      if (timeout_ms == 0) {
        return true;
      }
      poll_timeout_ms = RemainingPollTimeout(timeout_ms, poll_deadline);
      if (timeout_ms > 0 && poll_timeout_ms == 0) {
        return false;
      }
    }
  }

  if (!ReactorGenerationCleanupInvalidWaits(*this, &invalidated)) {
    return false;
  }
  if (invalidated) {
    reactor.ready.erase(
        std::remove_if(
            reactor.ready.begin(), reactor.ready.end(),
            [&reactor](const ReactorReady &ready) {
              return ReactorRegistryFindWait(reactor, ready.wait_id) ==
                     nullptr;
            }),
        reactor.ready.end());
    if (reactor.ready.empty()) {
      return true;
    }
  }

  if (reactor.ready.empty()) {
    return false;
  }

  if (!ReactorScratchOrderReady(reactor, reactor.ready)) {
    return false;
  }
  const ReactorBudgetSelection selection =
      ReactorBudgetSelect(reactor, reactor.ordered_ready_scratch, budget);
  if (!selection.ok || selection.ready == nullptr) {
    return false;
  }
  std::size_t consumed = selection.consumed;
  const std::vector<ReactorReady> *selected_ready = selection.ready;
  const auto many_group_for_task =
      [this](const std::uint64_t task_id) noexcept -> std::uint64_t {
    const TaskRecord *const record = state_->Find(task_id);
    return record == nullptr ? 0u : record->wait_source_id;
  };
  const auto prefix_contains_many_group =
      [&many_group_for_task,
       &reactor](const std::size_t prefix_count,
                 const std::uint64_t group_id) noexcept -> bool {
    if (group_id == 0u) {
      return false;
    }
    for (std::size_t index = 0u; index < prefix_count; ++index) {
      if (many_group_for_task(reactor.ordered_ready_scratch[index].task_id) ==
          group_id) {
        return true;
      }
    }
    return false;
  };
  if (consumed < reactor.ordered_ready_scratch.size()) {
    const std::size_t original_consumed = consumed;
    std::size_t extended_consumed = original_consumed;
    for (std::size_t index = original_consumed;
         index < reactor.ordered_ready_scratch.size(); ++index) {
      const std::uint64_t group_id =
          many_group_for_task(reactor.ordered_ready_scratch[index].task_id);
      if (prefix_contains_many_group(original_consumed, group_id)) {
        extended_consumed = index + 1u;
      }
    }
    if (extended_consumed != original_consumed) {
      try {
        reactor.budget_ready_scratch.clear();
        reactor.budget_ready_scratch.reserve(extended_consumed);
        reactor.budget_ready_scratch.insert(
            reactor.budget_ready_scratch.end(),
            reactor.ordered_ready_scratch.begin(),
            reactor.ordered_ready_scratch.begin() + extended_consumed);
      } catch (...) {
        reactor.budget_ready_scratch.clear();
        return false;
      }
      consumed = extended_consumed;
      selected_ready = &reactor.budget_ready_scratch;
    }
  }
  if (!ReactorBacklogStoreSuffix(reactor, state_->evidence.metrics,
                                 reactor.ordered_ready_scratch, consumed)) {
    return false;
  }
  const std::vector<ReactorReady> &ordered = *selected_ready;
  return DrainReactorReadyBatch(ordered);
}

} // namespace rund::node
