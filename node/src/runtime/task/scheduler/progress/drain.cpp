#include "../state/model/task.hpp"
#include "../state/storage.hpp"
#include "../reactor/registry.hpp"

namespace rund::node {

void Scheduler::ValidateReplayDrain() noexcept {
  if (state_->plan.mode() == ::rund::replay::detail::scope::Mode::Replay) {
    const auto &expected = *state_->plan.value.expected;
    if (!state_->identity.host_replay_failed &&
        state_->identity.next_expected_host_event !=
            expected.events().size()) {
      state_->identity.host_replay_failed = true;
      state_->identity.host_replay_reason = "host_replay_event_mismatch";
    }
    if (!state_->identity.host_replay_payload_failed &&
        state_->identity.next_expected_host_payload !=
            expected.payloads().host_record_count()) {
      state_->identity.host_replay_failed = true;
      state_->identity.host_replay_reason = "host_replay_payload_mismatch";
      state_->identity.host_replay_payload_failed = true;
      state_->identity.host_replay_payload_reason =
          "host_replay_payload_mismatch";
    }
  }
  if ((state_->plan.mode() == ::rund::replay::detail::scope::Mode::Replay ||
       state_->plan.mode() == ::rund::replay::detail::scope::Mode::Scenario) &&
      !state_->identity.host_replay_payload_failed &&
      state_->identity.next_expected_replay_input !=
          state_->plan.value.expected->payloads().input_record_count()) {
    state_->identity.host_replay_failed = true;
    state_->identity.host_replay_reason = "replay_input_missing";
    state_->identity.host_replay_payload_failed = true;
    state_->identity.host_replay_payload_reason = "replay_input_missing";
  }
}

task::Status Scheduler::Drain() noexcept {
  for (;;) {
    if (!ReadyQueuesEmpty() || !state_->ready.timers.empty() ||
        !ReactorRegistryEmpty(state_->reactor.reactor)) {
      if (Step()) {
        continue;
      }
    }
    if (WaitForDirectJobs()) {
      continue;
    }
    if (WakeDeadlockedTasks()) {
      continue;
    }
    break;
  }
  for (const TaskRecord &record : state_->ready.records) {
    if (record.state != TaskState::Completed &&
        record.state != TaskState::Failed) {
      return FailJoin(ReasonCode::TaskDeadlock);
    }
  }
  ValidateReplayDrain();
  FlushPendingRootJoinRetireBatch();
  const ReasonCode code = state_->FirstFailureCode();
  return code == ReasonCode::Ok ? task::Status::success()
                                : task::Status::fail(code);
}

} // namespace rund::node
