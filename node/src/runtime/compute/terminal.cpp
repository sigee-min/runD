#include "terminal.hpp"

#include "state.hpp"

namespace rund::node::compute_detail {

bool ClaimVirtualStart(std::atomic<TerminalPhase> &phase) noexcept {
  TerminalPhase expected = TerminalPhase::Open;
  return phase.compare_exchange_strong(expected, TerminalPhase::Running,
                                       std::memory_order_acq_rel,
                                       std::memory_order_acquire);
}

FinishClaim ClaimFinish(std::atomic<TerminalPhase> &phase) noexcept {
  TerminalPhase expected = TerminalPhase::Open;
  if (phase.compare_exchange_strong(expected, TerminalPhase::Finishing,
                                    std::memory_order_acq_rel,
                                    std::memory_order_acquire)) {
    return FinishClaim::Finish;
  }
  expected = TerminalPhase::Running;
  if (phase.compare_exchange_strong(expected, TerminalPhase::Finishing,
                                    std::memory_order_acq_rel,
                                    std::memory_order_acquire)) {
    return FinishClaim::Finish;
  }
  return expected == TerminalPhase::Cancelled ? FinishClaim::Cancel
                                              : FinishClaim::Closed;
}

CancelClaim RequestCancel(std::atomic<TerminalPhase> &phase) noexcept {
  TerminalPhase expected = TerminalPhase::Open;
  if (phase.compare_exchange_strong(expected, TerminalPhase::Cancelled,
                                    std::memory_order_acq_rel,
                                    std::memory_order_acquire)) {
    return CancelClaim::Accept;
  }
  if (expected == TerminalPhase::Cancelled) {
    return CancelClaim::Cancelled;
  }
  if (expected == TerminalPhase::Running ||
      expected == TerminalPhase::Finishing ||
      expected == TerminalPhase::Complete) {
    return CancelClaim::Closed;
  }
  return CancelClaim::Invalid;
}

void MarkComplete(std::atomic<TerminalPhase> &phase) noexcept {
  phase.store(TerminalPhase::Complete, std::memory_order_release);
}

namespace {

using ResultOperation = compute::detail::TerminalObservation (*)(
    const Operation &, TaskState &) noexcept;

compute::detail::TerminalObservation
Finish(TaskState &task, const ResultOperation result) noexcept {
  switch (ClaimFinish(task.terminal_phase)) {
  case FinishClaim::Finish:
    return task.operation && result != nullptr
               ? result(task.operation, task)
               : compute::detail::TerminalObservationAccess::from_stats(
                     compute::Status::fail(compute::Reason::TaskInvalid));
  case FinishClaim::Cancel:
    return task.operation && task.operation.table->cancel != nullptr
               ? task.operation.table->cancel(task.operation, task)
               : compute::detail::TerminalObservationAccess::from_stats(
                     compute::Status::fail(compute::Reason::TaskInvalid));
  case FinishClaim::Closed:
    return compute::detail::TerminalObservationAccess::from_stats(
        compute::Status::fail(compute::Reason::AlreadyCompleted));
  }
  return compute::detail::TerminalObservationAccess::from_stats(
      compute::Status::fail(compute::Reason::CompletionInvalid));
}

} // namespace

compute::detail::TerminalObservation FinishCpu(TaskState &task) noexcept {
  return Finish(task,
                task.operation ? task.operation.table->result_cpu : nullptr);
}

compute::detail::TerminalObservation FinishAccel(TaskState &task) noexcept {
  return Finish(task,
                task.operation ? task.operation.table->result_accel : nullptr);
}

compute::detail::TerminalObservation
FinishFailure(TaskState &task, const compute::Status status) noexcept {
  switch (ClaimFinish(task.terminal_phase)) {
  case FinishClaim::Finish:
    return task.operation && task.operation.table->fail != nullptr
               ? task.operation.table->fail(task.operation, task, status)
               : compute::detail::TerminalObservationAccess::from_stats(
                     compute::Status::fail(compute::Reason::TaskInvalid));
  case FinishClaim::Cancel:
    return task.operation && task.operation.table->cancel != nullptr
               ? task.operation.table->cancel(task.operation, task)
               : compute::detail::TerminalObservationAccess::from_stats(
                     compute::Status::fail(compute::Reason::TaskInvalid));
  case FinishClaim::Closed:
    return compute::detail::TerminalObservationAccess::from_stats(
        compute::Status::fail(compute::Reason::AlreadyCompleted));
  }
  return compute::detail::TerminalObservationAccess::from_stats(
      compute::Status::fail(compute::Reason::CompletionInvalid));
}

} // namespace rund::node::compute_detail
