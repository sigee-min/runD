#include "../../../../../compute/device/residency/registry/direct_recurrence_owner.hpp"
#include "internal.hpp"

#include "../../../../pipeline/claim.hpp"
#include "../../../../pipeline/local.hpp"

#include <algorithm>
#include <utility>

namespace rund::compute::detail::accel_backend::service_free_direct {
namespace {

void record(Run &run) noexcept {
  const auto registration = run.registration->snapshot();
  run.pipeline->service_free_direct = ServiceFreeDirectProductEvidence{
      .iterations = run.final.evidence.iterations,
      .completed_iterations = run.final.evidence.completed_iterations,
      .public_handoff_count = 1u,
      .native_submit_count = run.final.evidence.native_submit_count,
      .epoch_native_submit_count = run.final.evidence.epoch_native_submit_count,
      .payload_dispatch_count = run.final.evidence.payload_dispatch_count,
      .host_service_turn_count = run.final.evidence.host_service_turn_count,
      .host_epoch_callback_count = run.final.evidence.host_epoch_callback_count,
      .final_callback_count = run.final.evidence.final_callback_count,
      .authority_publication_count = 0u,
      .registered_state_count = static_cast<std::uint32_t>(registration.count),
      .selected = true,
      .fixed_native_storage = run.preparation.capability.fixed_native_storage,
      .fixed_common_storage = run.preparation.capability.fixed_common_storage,
  };
  run.pipeline->stats.command_submits = run.final.evidence.native_submit_count;
  run.pipeline->stats.dispatches = run.final.evidence.payload_dispatch_count;
  run.pipeline->stats.final_dispatches =
      run.final.evidence.payload_dispatch_count;
  run.pipeline->stats.control.iteration_count =
      run.final.evidence.completed_iterations;
  run.pipeline->stats.kernel_ns = run.final.evidence.completed_ns;
  run.pipeline->attempt.backend_submitted =
      run.final.evidence.native_submit_count != 0u;
  run.pipeline->attempt.writes_possible = run.final.evidence.may_write;
}

void publish(void *const raw, const bool success) noexcept {
  auto *const run = static_cast<Run *>(raw);
  if (run == nullptr || run->pipeline == nullptr) {
    return;
  }
  ++run->publication_count;
  run->publication_success = success;
  run->pipeline->service_free_direct.authority_publication_count =
      run->publication_count;
  const bool expected = run->final.check.ok;
  const Reason reason = success == expected
                            ? expected ? Reason::Ok
                                       : project_reason(run->final.check.reason,
                                                        Reason::BackendFailed)
                            : Reason::CompletionInvalid;
  publish_shared_pipeline_terminal_transaction(
      *run->pipeline,
      PipelineTerminal{
          .reason = reason,
          .verified = static_cast<std::size_t>(
              std::min<std::uint64_t>(run->final.evidence.completed_iterations,
                                      run->pipeline->steps.size())),
          .failed_step = static_cast<std::size_t>(std::min<std::uint64_t>(
              run->final.evidence.completed_iterations,
              run->pipeline->steps.empty() ? 0u
                                           : run->pipeline->steps.size() - 1u)),
          .failure_step_known =
              !success && run->final.evidence.completed_iterations <
                              run->pipeline->steps.size(),
          .writes_possible = run->final.evidence.may_write,
          .publication_suppressed = !success,
      });
}

[[nodiscard]] bool project_profile(Run &run) noexcept {
  if (run.pipeline == nullptr || run.pipeline->profile == nullptr ||
      run.final.terminal !=
          node::accel::detail::ServiceFreeDirectTerminal::Known) {
    return true;
  }
  if (run.final.profile_owner == nullptr) {
    return false;
  }
  const bool identity =
      run.pipeline->prepared.owner != nullptr &&
      run.final.profile_owner.get() == run.pipeline->prepared.owner.get() &&
      run.final.active_step_count == run.pipeline->active_step_count;
  const node::accel::detail::PreparedPipelineEvidence evidence{
      .check = run.final.check,
      .profile = run.final.profile,
      .active_step_count = run.final.active_step_count,
      .submitted = run.final.evidence.native_submit_count != 0u,
  };
  return capture_accel_pipeline_profile(*run.pipeline, evidence, identity);
}

} // namespace

Status close(Run &run) noexcept {
  namespace accel = node::accel::detail;
  if (run.pipeline == nullptr || run.registration == nullptr ||
      !run.lease.has_value() || !*run.lease ||
      !shared_pipeline_terminal_ready(*run.pipeline)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (run.pipeline->profile != nullptr &&
      run.final.terminal == accel::ServiceFreeDirectTerminal::Known) {
    const bool profile_ok = project_profile(run);
    if (run.final.check.ok && !profile_ok) {
      run.final.check = {false, "compute_completion_invalid"};
    }
  }
  record(run);
  const bool success = run.final.check.ok;
  const residency::execution::TerminalKind terminal =
      run.final.terminal == accel::ServiceFreeDirectTerminal::Known
          ? residency::execution::TerminalKind::Known
          : residency::execution::TerminalKind::UnknownMayWrite;
  residency::DirectRecurrenceFinal prepared{};
  const Status status =
      success ? Status::success()
              : Status::fail(project_reason(run.final.check.reason,
                                            Reason::BackendFailed));
  auto &authority = run.pipeline->device->residency->authority();
  if (!authority.direct_recurrences().prepare_direct_recurrence_final(
          *run.lease, status, terminal, run.final.evidence.may_write,
          run.final.evidence.completed_iterations, prepared)) {
    const residency::DirectAbort aborted =
        authority.direct_recurrences().abort_direct_recurrence(
            *run.lease, Status::fail(Reason::CompletionInvalid), terminal,
            run.final.evidence.may_write);
    if (aborted == residency::DirectAbort::Quarantined) {
      run.pipeline->service_free_direct.quarantined = true;
      return Status::fail(Reason::DeviceLost);
    }
    return Status::fail(Reason::CompletionInvalid);
  }
  const residency::ExecutionClose staged =
      authority.direct_recurrences().stage_direct_recurrence_final(
          std::move(prepared));
  if (!staged) {
    if (staged.quarantined) {
      run.pipeline->service_free_direct.quarantined = true;
      return Status::fail(Reason::DeviceLost);
    }
    const residency::DirectAbort aborted =
        authority.direct_recurrences().abort_direct_recurrence_frozen(
            std::move(prepared));
    if (aborted == residency::DirectAbort::Quarantined) {
      run.pipeline->service_free_direct.quarantined = true;
      return Status::fail(Reason::DeviceLost);
    }
    return Status::fail(Reason::CompletionInvalid);
  }
  if (staged.quarantined) {
    run.pipeline->service_free_direct.quarantined = true;
    return Status::fail(Reason::DeviceLost);
  }
  if (run.registration->release_pending(*run.lease) !=
      residency::RegistrationResult::Done) {
    run.pipeline->service_free_direct.quarantined = true;
    return Status::fail(Reason::DeviceLost);
  }
  const residency::ExecutionClose closed =
      run.registration->finish_pending(*run.lease, &run, publish);
  if (!closed || closed.quarantined || run.publication_count != 1u ||
      run.publication_success != run.final.check.ok) {
    if (closed.quarantined) {
      run.pipeline->service_free_direct.quarantined = true;
    }
    return Status::fail(Reason::CompletionInvalid);
  }
  return status;
}

void cancel(Run &run, const Status failure) noexcept {
  const auto registration = run.registration->snapshot();
  run.final = node::accel::detail::ServiceFreeDirectFinal{
      .check = {false,
                failure ? "accel_kernel_run_invalid" : failure.error().data()},
      .terminal = node::accel::detail::ServiceFreeDirectTerminal::Known,
      .evidence =
          node::accel::detail::ServiceFreeDirectEvidence{
              .proof = registration.proof == nullptr
                           ? node::accel::detail::ServiceFreeDirectIdentity{}
                           : registration.proof->identity,
              .token = run.lease->token(),
              .generation = run.lease->generation(),
              .nonce = run.lease->owner(),
              .iterations = run.lease->iterations(),
              .native_submit_count = 0u,
              .epoch_native_submit_count = 0u,
              .payload_dispatch_count = 0u,
              .final_callback_count = 0u,
              .may_write = false,
          },
  };
  static_cast<void>(close(run));
}

} // namespace rund::compute::detail::accel_backend::service_free_direct
