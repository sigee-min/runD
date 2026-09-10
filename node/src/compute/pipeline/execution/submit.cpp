#include "submit/local.hpp"

#include "../claim.hpp"
#include "../local.hpp"
#include "../state.hpp"

#include "../../backend.hpp"
#include "../../device/residency/pool.hpp"
#include "../../status.hpp"

#include <algorithm>
#include <limits>
#include <mutex>
#include <utility>

namespace rund::compute::detail {
namespace {

[[nodiscard]] Status select_pipeline_execution(
    const std::shared_ptr<PipelineState> &state,
    const residency::execution::Owner &owner,
    const residency::execution::Plan &plan,
    PipelineExecutionSubmission &submission,
    const node::accel::detail::PreparedKernelPipeline *&prepared,
    const DeviceOps *&ops) noexcept {
  prepared = nullptr;
  ops = nullptr;
  residency::execution::Node dispatch{};
  if (!valid_pipeline(state) || state->device->backend == Backend::Cpu ||
      !owner || plan.identity() == 0u || plan.epoch_count() != 1u ||
      submission.phase.load(std::memory_order_acquire) !=
          PipelineExecutionPhase::Submitting ||
      !plan.project(
          residency::execution::NodeId{
              .epoch = 0u, .phase = residency::execution::Phase::Dispatch},
          dispatch) ||
      dispatch.domain != residency::execution::Domain::Native ||
      dispatch.input_count == 0u ||
      dispatch.input_count != dispatch.output_count ||
      dispatch.input_count > submission.locals.size() ||
      state->residency_pool == nullptr ||
      state->residency_stage != PipelineResidencyStage::Direct ||
      state->residency_graph_stage != residency::NoGraphStage ||
      state->residency_port_count != 0u || !state->publications.empty() ||
      !state->windows.empty() ||
      state->residency_bank >= residency::Pool::BankCount) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const residency::Pool &pool = *state->residency_pool;
  if (pool.device != state->device ||
      dispatch.input_count > pool.layout.frame_capacity ||
      dispatch.input_count > state->steps.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const residency::FrameRegion input =
      pool.input_regions[state->residency_bank];
  const residency::FrameRegion output{
      .tier = input.tier,
      .role = residency::FrameRole::Output,
      .first = pool.first_output_frame +
               state->residency_bank * pool.layout.frame_capacity,
      .count = pool.layout.frame_capacity,
  };
  if (dispatch.route.source != input || dispatch.route.target != output ||
      dispatch.active_mask !=
          (dispatch.input_count == residency::execution::UseCapacity
               ? std::numeric_limits<std::uint32_t>::max()
               : (std::uint32_t{1u} << dispatch.input_count) - 1u)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  for (std::size_t local = 0u; local < dispatch.input_count; ++local) {
    submission.locals[local] = static_cast<std::uint32_t>(local);
  }
  submission.issued_steps = dispatch.input_count;
  ops = state->device->ops;
  prepared = state->transactional && state->attempt.parity != 0u
                 ? &state->alternate_prepared
                 : &state->prepared;
  if (ops == nullptr || ops->residency.submit_residency_pipeline == nullptr ||
      prepared == nullptr || !prepared->ok || owner.native != prepared->owner) {
    prepared = nullptr;
    ops = nullptr;
    return Status::fail(Reason::PipelineInvalid);
  }
  return Status::success();
}

void publish_rejected(PipelineExecutionSubmission &submission,
                      const Status failure) noexcept {
  const std::shared_ptr<PipelineState> state = submission.pipeline;
  if (state == nullptr) {
    return;
  }
  std::lock_guard lock{state->gate};
  if (state->phase == PipelinePhase::Running) {
    PipelineOutcome rejected{
        .status = failure,
        .writes_possible = false,
        .publication_suppressed = true,
    };
    static_cast<void>(publish_residency_pipeline_execution(
        *state, submission.issued_steps, rejected));
  }
}

} // namespace

Status
submit_pipeline_execution(const std::shared_ptr<PipelineState> &state,
                          const residency::execution::Owner &owner,
                          const residency::execution::Plan &plan,
                          residency::execution::Control &control,
                          PipelineExecutionSubmission &submission) noexcept {
  if (plan.identity() != 0u && plan.epoch_count() != 1u) {
    return Status::fail(Reason::BackendUnsupported);
  }
  if (!control || submission.active()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (state == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  bool inactive = false;
  if (!control.active.compare_exchange_strong(inactive, true,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
    return Status::fail(Reason::PipelineBusy);
  }
  control.bound_plan = plan.identity();
  control.bound_token = control.token;
  control.bound_generation = control.generation;
  control.bound_evidence = control.evidence;
  control.bound_completion = control.completion;
  control.bound_user = control.user;
  submission.pipeline = state;
  submission.control = &control;
  submission.phase.store(PipelineExecutionPhase::Submitting,
                         std::memory_order_release);

  const node::accel::detail::PreparedKernelPipeline *prepared = nullptr;
  const DeviceOps *ops = nullptr;
  Status admitted = Status::fail(Reason::PipelineInvalid);
  {
    std::unique_lock lock{state->gate, std::try_to_lock};
    if (!lock.owns_lock()) {
      admitted = Status::fail(Reason::PipelineBusy);
    } else {
      admitted = select_pipeline_execution(state, owner, plan, submission,
                                           prepared, ops);
      if (admitted) {
        admitted =
            start_pipeline(*state, PipelineClaimAuthority::PrivateResidency);
      }
      if (admitted) {
        const PipelineOutcome prepared_outcome =
            prepare_residency_pipeline_execution(
                *state, std::span<const std::uint32_t>{
                            submission.locals.data(), submission.issued_steps});
        submission.writes_possible = prepared_outcome.writes_possible;
        admitted = prepared_outcome.status;
        if (!admitted && state->phase == PipelinePhase::Running) {
          static_cast<void>(publish_residency_pipeline_execution(
              *state, submission.issued_steps, prepared_outcome));
        }
      }
    }
  }
  if (!admitted) {
    clear_pipeline_execution_control(control);
    submission.reset();
    return admitted;
  }

  const rund::AccelCheck submitted = ops->residency.submit_residency_pipeline(
      *state->device, *prepared,
      std::span<const std::uint32_t>{submission.locals.data(),
                                     submission.issued_steps},
      submission.pipeline, complete_pipeline_execution_callback, &submission);
  if (submitted.ok) {
    PipelineExecutionPhase expected = PipelineExecutionPhase::Submitting;
    static_cast<void>(submission.phase.compare_exchange_strong(
        expected, PipelineExecutionPhase::Submitted, std::memory_order_acq_rel,
        std::memory_order_acquire));
    return Status::success();
  }

  PipelineExecutionPhase expected = PipelineExecutionPhase::Submitting;
  if (submission.phase.compare_exchange_strong(
          expected, PipelineExecutionPhase::Completed,
          std::memory_order_acq_rel, std::memory_order_acquire)) {
    const Status failure =
        Status::fail(project_reason(submitted.reason, Reason::BackendFailed));
    publish_rejected(submission, failure);
    clear_pipeline_execution_control(control);
    submission.reset();
    return failure;
  }
  return Status::fail(Reason::CompletionInvalid);
}

} // namespace rund::compute::detail
