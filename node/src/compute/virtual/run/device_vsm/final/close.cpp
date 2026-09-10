#include "../../../../../accel/clock.hpp"
#include "../../../../../compute/device/residency/registry/direct_recurrence_owner.hpp"
#include "../internal.hpp"
#include "../staging/hash.hpp"

namespace rund::compute::detail::device_vsm_product_detail {
namespace {

void unlock_publication(DeviceVsmPublication &publication) noexcept {
  for (std::size_t index = 0u; index < publication.pipeline_lock_count;
       ++index) {
    if (publication.pipeline_locks[index].owns_lock()) {
      publication.pipeline_locks[index].unlock();
    }
  }
  publication.pipeline_lock_count = 0u;
}

void reject_after_publication(DeviceVsmProductRun &run,
                              DeviceVsmPublication &publication,
                              const Status failure) noexcept {
  unlock_publication(publication);
  reject_pipelines(run, failure);
}

} // namespace

VirtualExecutionResult finish(DeviceVsmProductRun &run,
                              Status status) noexcept {
  namespace accel = node::accel::detail;
  VirtualExecutionResult result{};
  result.status = status;
  if (run.owner == nullptr || run.registration == nullptr ||
      !run.lease.has_value() || !*run.lease || run.projection == nullptr ||
      run.output == nullptr) {
    result.status = Status::fail(Reason::CompletionInvalid);
    result.poison_pipeline = true;
    result.certainty = VirtualRunWriteCertainty::KnownNoWrite;
    reject_pipelines(run, result.status);
    return result;
  }

  const std::uint32_t submission_count = run.submission_control.count();
  residency::execution::TerminalKind terminal =
      residency::execution::TerminalKind::Known;
  bool may_write = false;
  std::uint64_t completed = 0u;
  if (submission_count == 1u) {
    terminal = run.final.terminal == accel::DeviceVsmTerminal::Known
                   ? residency::execution::TerminalKind::Known
                   : residency::execution::TerminalKind::UnknownMayWrite;
    may_write = run.final.evidence.may_write;
    completed = run.final.evidence.completed_epochs;
    status = status_from(run.final.check);
  }
  const bool unknown =
      terminal == residency::execution::TerminalKind::UnknownMayWrite;
  result.certainty = unknown ? VirtualRunWriteCertainty::UnknownMayWrite
                             : VirtualRunWriteCertainty::KnownNoWrite;
  if (unknown) {
    status = Status::fail(Reason::DeviceLost);
    result.poison_pipeline = true;
  }
  bool device_lost = status.reason() == Reason::DeviceLost;
  if (device_lost) {
    quarantine_owner(run.owner);
    result.poison_pipeline = true;
  }
  if (status) {
    status = stage_output(run);
    may_write = true;
    device_lost = device_lost || status.reason() == Reason::DeviceLost;
    if (device_lost) {
      quarantine_owner(run.owner);
      result.poison_pipeline = true;
    }
  }
  if (status && consume_virtual_execution_close_failure_once()) {
    status = Status::fail(Reason::PipelineInvalid);
  }

  DeviceVsmPublication publication{};
  if (status) {
    if (!prepare_success_publication(run, publication)) {
      status = Status::fail(Reason::CompletionInvalid);
      run.backing_may_write = true;
      if (publication.run == nullptr &&
          !prepare_failure_publication(run, publication, status)) {
        reject_after_publication(run, publication, status);
        result.status = status;
        result.poison_pipeline = true;
        return result;
      }
      publication.status = status;
    }
  } else if (!prepare_failure_publication(run, publication, status)) {
    reject_after_publication(run, publication, status);
    result.status = status;
    result.poison_pipeline = true;
    return result;
  }

  residency::DirectRecurrenceFinal prepared{};
  auto &authority = run.state->pipeline->device->residency->authority();
  if (!authority.direct_recurrences().prepare_direct_recurrence_final(
          *run.lease, status, terminal, may_write, completed, prepared)) {
    const residency::DirectAbort aborted =
        authority.direct_recurrences().abort_direct_recurrence(
            *run.lease, Status::fail(Reason::CompletionInvalid), terminal,
            may_write);
    if (aborted == residency::DirectAbort::Quarantined) {
      quarantine_owner(run.owner);
      reject_after_publication(run, publication,
                               Status::fail(Reason::DeviceLost));
      result.status = Status::fail(Reason::DeviceLost);
      result.poison_pipeline = true;
      return result;
    }
    reject_after_publication(run, publication,
                             Status::fail(Reason::CompletionInvalid));
    result.status = Status::fail(Reason::CompletionInvalid);
    result.poison_pipeline = true;
    return result;
  }
  const residency::ExecutionClose staged =
      authority.direct_recurrences().stage_direct_recurrence_final(
          std::move(prepared));
  if (!staged) {
    if (staged.quarantined) {
      run.backing_may_write = true;
      quarantine_owner(run.owner);
      reject_after_publication(run, publication,
                               Status::fail(Reason::DeviceLost));
      result.status = Status::fail(Reason::DeviceLost);
      result.poison_pipeline = true;
      return result;
    }
    const residency::DirectAbort aborted =
        authority.direct_recurrences().abort_direct_recurrence_frozen(
            std::move(prepared));
    if (aborted == residency::DirectAbort::Quarantined) {
      run.backing_may_write = true;
      quarantine_owner(run.owner);
      reject_after_publication(run, publication,
                               Status::fail(Reason::DeviceLost));
      result.status = Status::fail(Reason::DeviceLost);
      result.poison_pipeline = true;
      return result;
    }
    reject_after_publication(run, publication,
                             Status::fail(Reason::CompletionInvalid));
    result.status = Status::fail(Reason::CompletionInvalid);
    result.poison_pipeline = true;
    return result;
  }
  if (staged.quarantined) {
    run.backing_may_write = true;
    quarantine_owner(run.owner);
    reject_after_publication(run, publication,
                             Status::fail(Reason::DeviceLost));
    result.status = Status::fail(Reason::DeviceLost);
    result.poison_pipeline = true;
  } else {
    const residency::ExecutionClose closed = run.registration->finish_pending(
        *run.lease, &publication, commit_publication);
    if (!closed) {
      if (closed.quarantined) {
        unlock_publication(publication);
        run.backing_may_write = true;
        quarantine_owner(run.owner);
        result.status = Status::fail(Reason::DeviceLost);
        result.poison_pipeline = true;
        return result;
      }
      reject_after_publication(run, publication,
                               Status::fail(Reason::CompletionInvalid));
      result.status = Status::fail(Reason::CompletionInvalid);
      result.poison_pipeline = true;
      return result;
    }
    if (closed.quarantined) {
      unlock_publication(publication);
      run.backing_may_write = true;
      quarantine_owner(run.owner);
      result.status = Status::fail(Reason::DeviceLost);
      result.poison_pipeline = true;
      return result;
    }
    result.status = status;
    result.poison_pipeline =
        device_lost ||
        result.certainty == VirtualRunWriteCertainty::UnknownMayWrite;
    if (result.status) {
      commit_output_hash(run);
    }
  }
  const bool strict_scan_overflow =
      submission_count == 1u && !run.projection->graph_execution() &&
      run.projection->scan() && status.reason() == Reason::ScanSumOverflow;
  const bool strict_reduce_overflow =
      submission_count == 1u && !run.projection->graph_execution() &&
      run.projection->reduction() &&
      status.reason() == Reason::ReduceSumOverflow;
  result.failed_page =
      result.status ? ResidencyStats::no_failed_page
      : strict_scan_overflow || strict_reduce_overflow
          ? run.final.evidence.failed_page
      : run.failed_page != ResidencyStats::no_failed_page
          ? run.failed_page
          : std::min(completed,
                     (run.projection->graph_execution()
                          ? run.projection->active.graph.page_count()
                          : run.projection->active.stream.page_count()) -
                         1u);
  result.output_hash = result.status ? run.output_hash : 0u;
  fold_stats(run);
  const std::uint64_t final_ns = node::accel::detail::MonotonicNanoseconds();
  run.final.evidence.final_ns = final_ns;
  if (run.owner != nullptr && run.owner->evidence != nullptr) {
    run.owner->evidence->native.final_ns = final_ns;
  }
  return result;
}

} // namespace rund::compute::detail::device_vsm_product_detail
