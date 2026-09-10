#include "callback.hpp"

#include "../../../backend.hpp"
#include "../../../status.hpp"
#include "../../state.hpp"

#include <mutex>
#include <utility>

namespace rund::compute::detail {
namespace {

[[nodiscard]] Status backend_status(const rund::AccelCheck check,
                                    const Reason fallback) noexcept {
  return check.ok ? Status::success()
                  : Status::fail(project_reason(check.reason, fallback));
}

} // namespace

void complete_schedule_release(
    void *const raw,
    node::accel::detail::PreparedResidencyScheduleRelease &&native) noexcept {
  auto *const control = static_cast<PipelineResidencyScheduleControl *>(raw);
  if (control == nullptr) {
    return;
  }
  PipelineResidencyWindowReleaseCompletion completion = nullptr;
  void *user = nullptr;
  PipelineWindowRelease projected{};
  {
    std::lock_guard lock{control->gate};
    const std::uint64_t epoch = native.receipt.epoch;
    if (!control->active || control->plan == nullptr ||
        epoch >= control->lease.epochs) {
      return;
    }
    const std::size_t role = static_cast<std::size_t>(
        epoch % node::accel::detail::ResidencyScheduleRoleCapacity);
    PipelineExecutionAttempt &attempt = control->attempts[role];
    std::shared_ptr<PipelineState> &pipeline = control->pipelines[role];
    const bool known =
        native.receipt.terminal == node::accel::detail::NativeTerminal::Known;
    Status status = backend_status(native.receipt.check, Reason::BackendFailed);
    const std::uint64_t attempt_generation = attempt.attempt_generation;
    PipelineExecutionTerminal terminal{};
    if (attempt.started && attempt.signalled && pipeline != nullptr) {
      terminal = complete_pipeline_execution_attempt(
          *pipeline, attempt, native.receipt, native.evidence);
      status = terminal.status;
    } else {
      const bool collapsed_unknown =
          !native.receipt.check.ok && !known && native.receipt.dispatched &&
          !native.receipt.completed && native.receipt.may_write;
      if (!collapsed_unknown &&
          (native.receipt.may_write || native.receipt.check.ok || !known ||
           !native.receipt.dispatched || !native.receipt.completed)) {
        status = Status::fail(Reason::CompletionInvalid);
      }
      attempt = {};
    }
    projected = PipelineWindowRelease{
        .release =
            residency::execution::Release{
                .status = status,
                .terminal =
                    known ? residency::execution::TerminalKind::Known
                          : residency::execution::TerminalKind::UnknownMayWrite,
                .plan_identity = control->lease.plan,
                .token = control->lease.token,
                .generation = control->lease.generation,
                .epoch = epoch,
                .backend_sequence = native.receipt.backend_sequence,
                .bank = native.receipt.bank,
                .dispatched = native.receipt.dispatched,
                .completed = native.receipt.completed,
                .may_write = native.receipt.may_write,
            },
        .attempt_generation = attempt_generation,
        .published_generation = terminal.published_generation,
        .terminal_published = terminal.terminal_published,
        .reseeded = terminal.reseeded,
    };
    completion = control->release;
    user = control->user;
  }
  completion(user, std::move(projected));
}

void complete_schedule_final(
    void *const raw,
    node::accel::detail::PreparedResidencyScheduleFinal &&native) noexcept {
  auto *const control = static_cast<PipelineResidencyScheduleControl *>(raw);
  if (control == nullptr) {
    return;
  }
  PipelineResidencyScheduleFinalCompletion completion = nullptr;
  void *user = nullptr;
  residency::execution::ScheduleEvidence evidence{};
  {
    std::lock_guard lock{control->gate};
    if (!control->active || control->plan == nullptr) {
      return;
    }
    evidence = residency::execution::ScheduleEvidence{
        .status = backend_status(native.evidence.check, Reason::BackendFailed),
        .terminal = native.evidence.terminal ==
                            node::accel::detail::NativeTerminal::UnknownMayWrite
                        ? residency::execution::TerminalKind::UnknownMayWrite
                        : residency::execution::TerminalKind::Known,
        .plan_identity = native.evidence.plan_identity,
        .token = native.evidence.token,
        .generation = native.evidence.generation,
        .epoch_count = native.evidence.epoch_count,
        .public_handoffs = native.evidence.public_handoffs,
        .native_batches = native.evidence.native_batches,
        .queue_calls = native.evidence.queue_calls,
        .native_inflight_peak = native.evidence.native_inflight_peak,
        .released_prefix = native.evidence.released_prefix,
        .completed_ns = native.evidence.completed_ns,
    };
    completion = control->final;
    user = control->user;
    control->plan = nullptr;
    control->lease = {};
    control->pipelines.fill(nullptr);
    control->attempts = {};
    control->release = nullptr;
    control->final = nullptr;
    control->user = nullptr;
    control->active = false;
  }
  completion(user, std::move(evidence));
}

} // namespace rund::compute::detail
