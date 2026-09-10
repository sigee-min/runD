#include "local.hpp"

namespace rund_node_test_pipeline_residency::window {
namespace {

bool schedule_model(const std::uint64_t epochs) {
  const execution::SealResult sealed = plan(epochs);
  residency::Authority authority{};
  execution::Stream schedule{};
  if (!sealed || !frames(authority) ||
      !schedule.begin_schedule(authority, sealed.plan)) {
    return false;
  }
  const residency::ExecutionLease lease = schedule.lease();
  for (std::uint64_t epoch = 0u; epoch < epochs; ++epoch) {
    residency::ExecutionTicket input{};
    residency::ExecutionTicket output{};
    if (!schedule.issue(epoch, execution::Phase::Input, input) ||
        !schedule.terminal(input, residency::ExecutionTerminal::Success,
                           Status::success()) ||
        !schedule.release(release(lease, epoch)) ||
        !schedule.issue(epoch, execution::Phase::Output, output) ||
        !schedule.terminal(output, residency::ExecutionTerminal::Success,
                           Status::success())) {
      return false;
    }
  }
  const execution::ScheduleEvidence terminal{
      .status = Status::success(),
      .terminal = execution::TerminalKind::Known,
      .plan_identity = lease.plan,
      .token = lease.token,
      .generation = lease.generation,
      .epoch_count = epochs,
      .public_handoffs = 1u,
      .native_batches = epochs,
      .queue_calls = 1u,
      .native_inflight_peak = epochs == 1u ? 1u : 2u,
      .released_prefix = epochs,
      .completed_ns = epochs + 700u,
  };
  execution::StreamEvidence evidence{};
  if (!schedule.schedule(terminal) || !schedule.final(evidence) ||
      evidence.chunk_count != 1u || evidence.native_batches != epochs ||
      evidence.queue_calls != 1u || evidence.released_prefix != epochs) {
    return false;
  }
  const residency::ExecutionClose closed = schedule.close();
  return closed && closed.success && !closed.quarantined &&
         closed.progress.input_services == epochs &&
         closed.progress.native_dispatches == epochs &&
         closed.progress.native_completions == epochs &&
         closed.progress.output_services == epochs;
}

bool schedule_known_failure() {
  constexpr std::uint64_t epochs = 101u;
  constexpr std::uint64_t failed_epoch = 2u;
  const execution::SealResult sealed = plan(epochs);
  residency::Authority authority{};
  execution::Stream schedule{};
  if (!sealed || !frames(authority) ||
      !schedule.begin_schedule(authority, sealed.plan)) {
    return false;
  }
  const residency::ExecutionLease lease = schedule.lease();
  const Status failure = Status::fail(Reason::BackendFailed);
  for (std::uint64_t epoch = 0u; epoch < epochs; ++epoch) {
    if (epoch < failed_epoch) {
      residency::ExecutionTicket input{};
      residency::ExecutionTicket output{};
      if (!schedule.issue(epoch, execution::Phase::Input, input) ||
          !schedule.terminal(input, residency::ExecutionTerminal::Success,
                             Status::success()) ||
          !schedule.release(release(lease, epoch)) ||
          !schedule.issue(epoch, execution::Phase::Output, output) ||
          !schedule.terminal(output, residency::ExecutionTerminal::Success,
                             Status::success())) {
        return false;
      }
      continue;
    }
    if (epoch == failed_epoch) {
      residency::ExecutionTicket input{};
      if (!schedule.issue(epoch, execution::Phase::Input, input) ||
          !schedule.terminal(input, residency::ExecutionTerminal::Failure,
                             failure)) {
        return false;
      }
    }
    if (!schedule.release(release(lease, epoch, failure,
                                  execution::TerminalKind::Known, true, true,
                                  false))) {
      return false;
    }
  }
  const execution::ScheduleEvidence terminal{
      .status = failure,
      .terminal = execution::TerminalKind::Known,
      .plan_identity = lease.plan,
      .token = lease.token,
      .generation = lease.generation,
      .epoch_count = epochs,
      .public_handoffs = 1u,
      .native_batches = epochs,
      .queue_calls = 1u,
      .native_inflight_peak = 2u,
      .released_prefix = epochs,
      .completed_ns = 900u,
  };
  if (!schedule.schedule(terminal)) {
    return false;
  }
  const residency::ExecutionClose closed = schedule.close();
  if (!closed || closed.success || !closed.quarantined ||
      closed.progress.native_dispatches != epochs ||
      closed.progress.native_completions != epochs ||
      closed.failure_count != 1u) {
    return false;
  }
  execution::Stream retry{};
  return retry.begin_schedule(authority, sealed.plan) && retry.abandon();
}

bool schedule_unknown_prefix() {
  constexpr std::uint64_t epochs = 9u;
  const execution::SealResult sealed = plan(epochs);
  residency::Authority authority{};
  execution::Stream schedule{};
  if (!sealed || !frames(authority) ||
      !schedule.begin_schedule(authority, sealed.plan)) {
    return false;
  }
  const residency::ExecutionLease lease = schedule.lease();
  residency::ExecutionTicket input{};
  const Status lost = Status::fail(Reason::DeviceLost);
  if (!schedule.issue(0u, execution::Phase::Input, input) ||
      !schedule.terminal(input, residency::ExecutionTerminal::Success,
                         Status::success()) ||
      !schedule.release(release(lease, 0u, lost,
                                execution::TerminalKind::UnknownMayWrite, true,
                                false, true))) {
    return false;
  }
  // All Q batches were accepted by the one native handoff, but only the first
  // terminal is observable after loss. The aggregate authenticates acceptance
  // without fabricating eight completion receipts.
  const execution::ScheduleEvidence terminal{
      .status = lost,
      .terminal = execution::TerminalKind::UnknownMayWrite,
      .plan_identity = lease.plan,
      .token = lease.token,
      .generation = lease.generation,
      .epoch_count = epochs,
      .public_handoffs = 1u,
      .native_batches = epochs,
      .queue_calls = 1u,
      .native_inflight_peak = 1u,
      .released_prefix = 1u,
      .completed_ns = 901u,
  };
  if (!schedule.schedule(terminal)) {
    return false;
  }
  const residency::ExecutionClose closed = schedule.close();
  return closed && !closed.success && closed.quarantined &&
         closed.progress.native_dispatches == epochs &&
         closed.progress.native_completions == 0u;
}

} // namespace

int CheckScheduleWindow() {
  if (!schedule_model(9u)) {
    return 41;
  }
  if (!schedule_model(100'000u)) {
    return 42;
  }
  if (!schedule_known_failure()) {
    return 43;
  }
  if (!schedule_unknown_prefix()) {
    return 44;
  }
  return 0;
}

} // namespace rund_node_test_pipeline_residency::window
