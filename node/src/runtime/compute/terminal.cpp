#include "terminal.hpp"

#include "state.hpp"

namespace rund::node::compute_detail {

FinishClaim ClaimFinish(std::atomic<TerminalPhase> &phase) noexcept {
  TerminalPhase expected = TerminalPhase::Open;
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
  if (expected == TerminalPhase::Finishing ||
      expected == TerminalPhase::Complete) {
    return CancelClaim::Closed;
  }
  return CancelClaim::Invalid;
}

void MarkComplete(std::atomic<TerminalPhase> &phase) noexcept {
  phase.store(TerminalPhase::Complete, std::memory_order_release);
}

namespace {

using ResultOperation = compute::Status (*)(const Operation &,
                                             TaskState &) noexcept;

compute::Status Finish(TaskState &task,
                       const ResultOperation result) noexcept {
  switch (ClaimFinish(task.terminal_phase)) {
  case FinishClaim::Finish:
    return task.operation && result != nullptr
               ? result(task.operation, task)
               : compute::Status::fail(compute::Reason::TaskInvalid);
  case FinishClaim::Cancel:
    return task.operation && task.operation.table->cancel != nullptr
               ? task.operation.table->cancel(task.operation.owner)
               : compute::Status::fail(compute::Reason::TaskInvalid);
  case FinishClaim::Closed:
    return compute::Status::fail(compute::Reason::AlreadyCompleted);
  }
  return compute::Status::fail(compute::Reason::CompletionInvalid);
}

} // namespace

compute::Status FinishCpu(TaskState &task) noexcept {
  return Finish(task, task.operation ? task.operation.table->result_cpu
                                     : nullptr);
}

compute::Status FinishAccel(TaskState &task) noexcept {
  return Finish(task, task.operation ? task.operation.table->result_accel
                                     : nullptr);
}

compute::Status FinishFailure(TaskState &task,
                              const compute::Status status) noexcept {
  switch (ClaimFinish(task.terminal_phase)) {
  case FinishClaim::Finish:
    return task.operation && task.operation.table->fail != nullptr
               ? task.operation.table->fail(task.operation.owner, status)
               : compute::Status::fail(compute::Reason::TaskInvalid);
  case FinishClaim::Cancel:
    return task.operation && task.operation.table->cancel != nullptr
               ? task.operation.table->cancel(task.operation.owner)
               : compute::Status::fail(compute::Reason::TaskInvalid);
  case FinishClaim::Closed:
    return compute::Status::fail(compute::Reason::AlreadyCompleted);
  }
  return compute::Status::fail(compute::Reason::CompletionInvalid);
}

} // namespace rund::node::compute_detail
