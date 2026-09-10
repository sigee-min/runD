#include "../window.hpp"

#include "../../../backend.hpp"
#include "../../../status.hpp"
#include "../../claim.hpp"
#include "../../state.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>

namespace rund::compute::detail {

Status signal_pipeline_execution_window(PipelineResidencyWindowControl &control,
                                        const std::uint64_t epoch,
                                        Status admission) noexcept {
  std::shared_ptr<PipelineState> pipeline{};
  node::accel::detail::PreparedKernelPipeline prepared{};
  node::accel::detail::BackendResidencyWindowSignal signal{};
  {
    std::lock_guard lock{control.gate};
    if (!control.active || control.plan == nullptr ||
        epoch < control.native.bound.first_epoch ||
        epoch - control.native.bound.first_epoch >=
            control.native.bound.batch_count ||
        epoch >= control.lease.epochs) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::size_t index =
        static_cast<std::size_t>(epoch - control.native.bound.first_epoch);
    PipelineExecutionAttempt &attempt = control.attempts[index];
    if (attempt.started || attempt.signalled) {
      return Status::fail(Reason::PipelineBusy);
    }
    pipeline = control.pipelines[index];
    if (pipeline == nullptr || pipeline->device == nullptr ||
        pipeline->device->ops == nullptr ||
        pipeline->device->ops->residency.signal_residency_window == nullptr) {
      return Status::fail(Reason::PipelineInvalid);
    }
    residency::execution::Node dispatch{};
    if (!control.plan->project(
            residency::execution::NodeId{
                .epoch = epoch, .phase = residency::execution::Phase::Dispatch},
            dispatch)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    signal = node::accel::detail::BackendResidencyWindowSignal{
        .admission =
            admission
                ? rund::AccelCheck{true, "ok"}
                : rund::AccelCheck{false,
                                   reason_message(admission.reason()).data()},
        .plan_identity = control.lease.plan,
        .token = control.lease.token,
        .generation = control.lease.generation,
        .epoch = epoch,
        .control_generation =
            control.native.bound.batches[index].control_generation,
        .bank = static_cast<std::uint8_t>(dispatch.bank),
    };
    prepared = control.native.bound.batches[index].pipeline;
    if (admission) {
      const Status started = begin_pipeline_execution_attempt(
          *pipeline, dispatch,
          std::span<const std::uint32_t>{
              control.native.bound.batches[index].locals.data(),
              dispatch.input_count},
          signal.control_generation, attempt);
      if (!started) {
        admission = started;
        signal.admission = {false, reason_message(started.reason()).data()};
      }
    }
    attempt.signalled = true;
  }

  Status signalled = pipeline->device->ops->residency.signal_residency_window(
      *pipeline->device, prepared, signal);
  if (!signalled && signal.admission.ok) {
    {
      std::lock_guard lock{control.gate};
      const std::size_t index =
          static_cast<std::size_t>(epoch - control.native.bound.first_epoch);
      PipelineExecutionAttempt &attempt = control.attempts[index];
      reject_pipeline_execution_attempt(*pipeline, attempt, signalled);
      attempt.signalled = true;
    }
    signal.admission = {false, reason_message(signalled.reason()).data()};
    signalled = pipeline->device->ops->residency.signal_residency_window(
        *pipeline->device, prepared, signal);
  }
  if (!signalled) {
    std::lock_guard lock{control.gate};
    control
        .attempts[static_cast<std::size_t>(epoch -
                                           control.native.bound.first_epoch)]
        .signalled = false;
  }
  return signalled;
}

Status abort_pipeline_execution_window(PipelineResidencyWindowControl &control,
                                       const Status failure) noexcept {
  std::shared_ptr<PipelineState> leader{};
  node::accel::detail::BackendResidencyWindowAbort abort{};
  {
    std::lock_guard lock{control.gate};
    if (failure || !control.active || control.plan == nullptr ||
        control.lease.token == 0u || control.lease.generation == 0u) {
      return Status::fail(Reason::PipelineInvalid);
    }
    leader = control.pipelines[0u];
    if (leader == nullptr || leader->device == nullptr ||
        leader->device->ops == nullptr ||
        leader->device->ops->residency.abort_residency_window == nullptr) {
      return Status::fail(Reason::BackendUnsupported);
    }
    abort = node::accel::detail::BackendResidencyWindowAbort{
        .failure = {false, reason_message(failure.reason()).data()},
        .plan_identity = control.lease.plan,
        .token = control.lease.token,
        .generation = control.lease.generation,
    };
  }
  return leader->device->ops->residency.abort_residency_window(
      *leader->device, control.native, abort);
}

Status rearm_pipeline_execution_window(
    PipelineResidencyWindowControl &control) noexcept {
  std::scoped_lock lock{control.gate, control.native.gate};
  if (control.active || control.native.active || control.native.quarantined) {
    return Status::fail(Reason::PipelineBusy);
  }
  control.evidence.reset();
  control.plan = nullptr;
  control.lease = {};
  control.pipelines.fill(nullptr);
  control.attempts = {};
  control.release = nullptr;
  control.final = nullptr;
  control.user = nullptr;
  control.unknown_seen = false;
  return Status::success();
}

} // namespace rund::compute::detail
