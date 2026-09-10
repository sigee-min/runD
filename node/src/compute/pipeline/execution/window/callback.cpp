#include "internal.hpp"

#include "../prepare.hpp"
#include "../submit.hpp"

#include "../../../backend.hpp"
#include "../../../status.hpp"
#include "../../claim.hpp"
#include "../../local.hpp"
#include "../../state.hpp"

#include <mutex>
#include <utility>

namespace rund::compute::detail::pipeline_window_detail {
namespace {

[[nodiscard]] Status backend_status(const rund::AccelCheck check,
                                    const Reason fallback) noexcept {
  return check.ok ? Status::success()
                  : Status::fail(project_reason(check.reason, fallback));
}

} // namespace

void complete_release(
    void *const raw,
    node::accel::detail::PreparedResidencyWindowRelease &&native) noexcept {
  auto *const control = static_cast<PipelineResidencyWindowControl *>(raw);
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
        epoch < control->native.bound.first_epoch ||
        epoch - control->native.bound.first_epoch >=
            control->native.bound.batch_count ||
        epoch >= control->lease.epochs) {
      return;
    }
    const std::size_t index =
        static_cast<std::size_t>(epoch - control->native.bound.first_epoch);
    PipelineExecutionAttempt &attempt = control->attempts[index];
    std::shared_ptr<PipelineState> &pipeline = control->pipelines[index];
    const bool known =
        native.receipt.terminal == node::accel::detail::NativeTerminal::Known;
    Status status = backend_status(native.receipt.check, Reason::BackendFailed);
    const std::uint64_t attempt_generation = attempt.attempt_generation;
    std::uint64_t published_generation = 0u;
    bool terminal_published = false;
    bool reseeded = false;
    if (attempt.started && attempt.signalled && pipeline != nullptr) {
      const PipelineExecutionTerminal terminal =
          complete_pipeline_execution_attempt(*pipeline, attempt,
                                              native.receipt, native.evidence);
      status = terminal.status;
      published_generation = terminal.published_generation;
      terminal_published = terminal.terminal_published;
      reseeded = terminal.reseeded;
    } else {
      const bool collapsed_unknown =
          !native.receipt.check.ok && !known && native.receipt.dispatched &&
          !native.receipt.completed && native.receipt.may_write;
      if (!collapsed_unknown &&
          (native.receipt.may_write || native.receipt.check.ok || !known ||
           !native.receipt.dispatched || !native.receipt.completed)) {
        status = Status::fail(Reason::CompletionInvalid);
      }
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
        .published_generation = published_generation,
        .terminal_published = terminal_published,
        .reseeded = reseeded,
    };
    if (!control->evidence.terminal(projected)) {
      projected.release.status = Status::fail(Reason::CompletionInvalid);
      projected.release.terminal =
          residency::execution::TerminalKind::UnknownMayWrite;
      projected.release.may_write = native.receipt.dispatched;
    }
    control->unknown_seen =
        control->unknown_seen ||
        projected.release.terminal ==
            residency::execution::TerminalKind::UnknownMayWrite;
    completion = control->release;
    user = control->user;
  }
  if (completion != nullptr) {
    completion(user, std::move(projected));
  }
}

void complete_final(
    void *const raw,
    node::accel::detail::PreparedResidencyWindowFinal &&native) noexcept {
  auto *const control = static_cast<PipelineResidencyWindowControl *>(raw);
  if (control == nullptr) {
    return;
  }
  PipelineResidencyWindowFinalCompletion completion = nullptr;
  void *user = nullptr;
  residency::execution::WindowEvidence evidence{};
  {
    std::lock_guard lock{control->gate};
    if (!control->active || control->plan == nullptr) {
      return;
    }
    const bool projected = control->evidence.final(
        native.evidence.queue_calls, native.evidence.native_inflight_peak,
        native.evidence.completed_ns, evidence);
    if (!projected) {
      const bool certain = control->evidence.integrity(
          native.evidence.queue_calls, native.evidence.native_inflight_peak,
          native.evidence.completed_ns, evidence);
      if (!certain) {
        evidence = residency::execution::WindowEvidence{
            .status = Status::fail(Reason::CompletionInvalid),
            .terminal = residency::execution::TerminalKind::UnknownMayWrite,
            .plan_identity = control->lease.plan,
            .token = control->lease.token,
            .generation = control->lease.generation,
            .first_epoch = control->native.bound.first_epoch,
            .epoch_count = control->lease.epochs,
            .public_handoffs = 1u,
            .native_batches = native.evidence.native_batches,
            .queue_calls = native.evidence.queue_calls,
            .native_inflight_peak = native.evidence.native_inflight_peak,
            .release_count = control->native.bound.batch_count,
            .completed_ns = native.evidence.completed_ns,
        };
        control->unknown_seen = true;
      }
    }
    if (!native.evidence.check.ok ||
        native.evidence.terminal ==
            node::accel::detail::NativeTerminal::UnknownMayWrite) {
      evidence.status =
          backend_status(native.evidence.check, Reason::BackendFailed);
      evidence.terminal =
          native.evidence.terminal ==
                  node::accel::detail::NativeTerminal::UnknownMayWrite
              ? residency::execution::TerminalKind::UnknownMayWrite
              : residency::execution::TerminalKind::Known;
    }
    evidence.first_epoch = native.evidence.first_epoch;
    completion = control->final;
    user = control->user;
    control->active = false;
  }
  if (completion != nullptr) {
    completion(user, std::move(evidence));
  }
}

} // namespace rund::compute::detail::pipeline_window_detail
