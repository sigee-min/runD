#include "attempt.hpp"
#include "../../../accel/kernel/residency/window.hpp"

#include "prepare.hpp"
#include "submit.hpp"

#include "../../status.hpp"
#include "../claim.hpp"
#include "../local.hpp"
#include "../state.hpp"

#include <array>
#include <mutex>

namespace rund::compute::detail {

const node::accel::detail::PreparedKernelPipeline *
prepared_pipeline_for(const PipelineState &pipeline,
                      const std::uint8_t parity) noexcept {
  return pipeline.transactional && parity != 0u ? &pipeline.alternate_prepared
                                                : &pipeline.prepared;
}

Status snapshot_pipeline_execution(
    const std::array<std::shared_ptr<PipelineState>,
                     residency::execution::BankCapacity> &pipelines,
    PipelineExecutionSnapshot &snapshot) noexcept {
  snapshot = {};
  for (std::size_t bank = 0u; bank < pipelines.size(); ++bank) {
    const std::shared_ptr<PipelineState> &pipeline = pipelines[bank];
    if (!valid_pipeline(pipeline) || pipeline->device == nullptr ||
        pipeline->device->backend == Backend::Cpu ||
        pipeline->device->ops == nullptr || pipeline->residency_bank != bank ||
        pipeline->residency_stage != PipelineResidencyStage::Direct ||
        pipeline->residency_graph_stage != residency::NoGraphStage ||
        pipeline->residency_port_count != 0u ||
        !pipeline->publications.empty() || !pipeline->windows.empty()) {
      return Status::fail(Reason::BackendUnsupported);
    }
    std::lock_guard pipeline_lock{pipeline->gate};
    std::lock_guard publication_lock{pipeline->publication->gate};
    if (pipeline->phase != PipelinePhase::Ready ||
        pipeline->publication->attempt_active ||
        pipeline->publication->device_lost) {
      return Status::fail(pipeline->publication->device_lost
                              ? Reason::DeviceLost
                              : Reason::PipelineBusy);
    }
    snapshot.generation[bank] = pipeline->publication->generation;
    snapshot.parity[bank] = pipeline->publication->parity;
  }
  if (pipelines[0u]->device != pipelines[1u]->device) {
    return Status::fail(Reason::BindingDeviceMismatch);
  }
  return Status::success();
}

Status
begin_pipeline_execution_attempt(PipelineState &pipeline,
                                 const residency::execution::Node &dispatch,
                                 const std::span<const std::uint32_t> locals,
                                 const std::uint32_t control_generation,
                                 PipelineExecutionAttempt &attempt) noexcept {
  if (dispatch.id.phase != residency::execution::Phase::Dispatch ||
      dispatch.domain != residency::execution::Domain::Native ||
      dispatch.input_count == 0u ||
      dispatch.input_count != dispatch.output_count ||
      dispatch.input_count != locals.size() || attempt.started ||
      attempt.signalled) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::lock_guard lock{pipeline.gate};
  Status status =
      start_pipeline(pipeline, PipelineClaimAuthority::PrivateResidency);
  if (status) {
    attempt.epoch = dispatch.id.epoch;
    attempt.attempt_generation = pipeline.attempt.generation;
    attempt.issued_steps = dispatch.input_count;
    const PipelineOutcome outcome =
        prepare_residency_pipeline_execution(pipeline, locals);
    status = outcome.status;
    if (!status && pipeline.phase == PipelinePhase::Running) {
      static_cast<void>(publish_residency_pipeline_execution(
          pipeline, dispatch.input_count, outcome));
    }
  }
  if (status && control_generation != pipeline.attempt.generation + 1u) {
    PipelineOutcome mismatch{
        .status = Status::fail(Reason::CompletionInvalid),
        .publication_suppressed = true,
    };
    static_cast<void>(publish_residency_pipeline_execution(
        pipeline, dispatch.input_count, mismatch));
    status = mismatch.status;
  }
  if (!status) {
    attempt = {};
    return status;
  }
  attempt.started = true;
  return Status::success();
}

PipelineExecutionTerminal complete_pipeline_execution_attempt(
    PipelineState &pipeline, PipelineExecutionAttempt &attempt,
    const node::accel::detail::BackendResidencyWindowReceipt &receipt,
    const node::accel::detail::PreparedPipelineEvidence &evidence) noexcept {
  PipelineExecutionTerminal terminal{};
  terminal.status = receipt.check.ok
                        ? Status::success()
                        : Status::fail(project_reason(receipt.check.reason,
                                                      Reason::BackendFailed));
  std::lock_guard lock{pipeline.gate};
  if (!attempt.started || !attempt.signalled ||
      attempt.epoch != receipt.epoch ||
      pipeline.phase != PipelinePhase::Running || attempt.issued_steps == 0u ||
      attempt.issued_steps > pipeline.steps.size()) {
    terminal.status = Status::fail(Reason::CompletionInvalid);
    return terminal;
  }
  PipelineOutcome outcome{
      .status = terminal.status,
      .writes_possible = receipt.may_write,
      .native_completion = receipt.dispatched
                               ? PipelineNativeCompletion::Known
                               : PipelineNativeCompletion::NotSubmitted,
      .publication_suppressed = !terminal.status,
  };
  std::array<std::uint32_t, residency::execution::UseCapacity> locals{};
  for (std::size_t local = 0u; local < attempt.issued_steps; ++local) {
    locals[local] = static_cast<std::uint32_t>(local);
  }
  outcome = finish_residency_pipeline_execution(
      pipeline,
      std::span<const std::uint32_t>{locals.data(), attempt.issued_steps},
      evidence, outcome);
  terminal.status = publish_residency_pipeline_execution(
      pipeline, attempt.issued_steps, outcome);
  terminal.terminal_published = true;
  {
    std::lock_guard publication_lock{pipeline.publication->gate};
    terminal.published_generation = pipeline.publication->generation;
    terminal.reseeded =
        pipeline.native_generation == terminal.published_generation &&
        pipeline.native_parity == pipeline.publication->parity;
  }
  attempt = {};
  return terminal;
}

void reject_pipeline_execution_attempt(PipelineState &pipeline,
                                       PipelineExecutionAttempt &attempt,
                                       const Status failure) noexcept {
  std::lock_guard lock{pipeline.gate};
  if (attempt.started && pipeline.phase == PipelinePhase::Running) {
    PipelineOutcome rejected{
        .status = failure,
        .publication_suppressed = true,
    };
    static_cast<void>(publish_residency_pipeline_execution(
        pipeline, attempt.issued_steps, rejected));
  }
  attempt = {};
}

} // namespace rund::compute::detail
