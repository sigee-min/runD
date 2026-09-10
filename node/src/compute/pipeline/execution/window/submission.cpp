#include "internal.hpp"

#include "../../../backend.hpp"
#include "../../local.hpp"
#include "../../state.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>

namespace rund::compute::detail {
namespace {

[[nodiscard]] Status submit_chunk_impl(
    const residency::execution::Plan &plan,
    const residency::ExecutionLease &lease, const std::uint64_t first_epoch,
    const std::size_t count,
    const std::array<std::shared_ptr<PipelineState>,
                     residency::execution::BankCapacity> &pipelines,
    const PipelineResidencyWindowReleaseCompletion release,
    const PipelineResidencyWindowFinalCompletion final, void *const user,
    PipelineResidencyWindowControl &control,
    node::accel::detail::PreparedResidencyStreamControl *const
        stream) noexcept {
  if (plan.identity() == 0u || count == 0u ||
      count > residency::execution::WindowCapacity || !lease ||
      lease.plan != plan.identity() || lease.epochs != plan.epoch_count() ||
      first_epoch >= plan.epoch_count() ||
      count > plan.epoch_count() - first_epoch || release == nullptr ||
      final == nullptr || user == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::lock_guard control_lock{control.gate};
  if (control.active) {
    return Status::fail(Reason::PipelineBusy);
  }
  node::accel::detail::PreparedResidencyWindowRequest request{
      .plan_identity = plan.identity(),
      .token = lease.token,
      .generation = lease.generation,
      .first_epoch = first_epoch,
      .batch_count = count,
      .release = pipeline_window_detail::complete_release,
      .final = pipeline_window_detail::complete_final,
      .user = &control,
  };
  PipelineExecutionSnapshot snapshot{};
  const Status snapshotted = snapshot_pipeline_execution(pipelines, snapshot);
  if (!snapshotted) {
    return snapshotted;
  }
  for (std::size_t bank = 0u; bank < pipelines.size(); ++bank) {
    const std::shared_ptr<PipelineState> &pipeline = pipelines[bank];
    if (((stream == nullptr &&
          pipeline->device->ops->residency.submit_residency_window ==
              nullptr) ||
         (stream != nullptr &&
          pipeline->device->ops->residency.submit_residency_stream_window ==
              nullptr)) ||
        pipeline->device->ops->residency.signal_residency_window == nullptr) {
      return Status::fail(Reason::BackendUnsupported);
    }
  }
  for (std::size_t slot = 0u; slot < count; ++slot) {
    const std::uint64_t epoch = first_epoch + slot;
    residency::execution::Node dispatch{};
    if (!plan.project(
            residency::execution::NodeId{
                .epoch = epoch, .phase = residency::execution::Phase::Dispatch},
            dispatch) ||
        dispatch.domain != residency::execution::Domain::Native ||
        dispatch.input_count == 0u ||
        dispatch.input_count != dispatch.output_count ||
        dispatch.input_count > residency::execution::UseCapacity) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::size_t index = slot;
    const std::size_t bank = static_cast<std::size_t>(dispatch.bank);
    if (bank >= pipelines.size() ||
        dispatch.input_count > pipelines[bank]->steps.size() ||
        snapshot.generation[bank] >=
            std::numeric_limits<std::uint32_t>::max()) {
      return Status::fail(Reason::PipelineCapacity);
    }
    const node::accel::detail::PreparedKernelPipeline *const prepared =
        prepared_pipeline_for(*pipelines[bank], snapshot.parity[bank]);
    if (prepared == nullptr || !prepared->ok) {
      return Status::fail(Reason::PipelineInvalid);
    }
    node::accel::detail::PreparedResidencyWindowBatch &batch =
        request.batches[index];
    batch.pipeline = *prepared;
    batch.local_count = dispatch.input_count;
    batch.epoch = epoch;
    batch.control_generation =
        static_cast<std::uint32_t>(snapshot.generation[bank] + 1u);
    batch.bank = static_cast<std::uint8_t>(bank);
    for (std::size_t local = 0u; local < dispatch.input_count; ++local) {
      batch.locals[local] = static_cast<std::uint32_t>(local);
    }
    control.pipelines[index] = pipelines[bank];
    snapshot.generation[bank] += 1u;
    if (pipelines[bank]->transactional) {
      snapshot.parity[bank] ^= 1u;
    }
  }
  if (!control.evidence.bind(plan, lease, first_epoch, count)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  control.plan = &plan;
  control.lease = lease;
  control.release = release;
  control.final = final;
  control.user = user;
  control.active = true;
  control.unknown_seen = false;
  const Status submitted =
      stream == nullptr
          ? pipelines[0u]->device->ops->residency.submit_residency_window(
                *pipelines[0u]->device, request, control.native)
          : pipelines[0u]
                ->device->ops->residency.submit_residency_stream_window(
                    *pipelines[0u]->device, request, control.native, *stream);
  if (!submitted) {
    control.evidence.reset();
    control.plan = nullptr;
    control.lease = {};
    control.pipelines.fill(nullptr);
    control.attempts = {};
    control.release = nullptr;
    control.final = nullptr;
    control.user = nullptr;
    control.active = false;
    control.unknown_seen = false;
  }
  return submitted;
}

} // namespace

Status submit_pipeline_execution_window(
    const residency::execution::Plan &plan,
    const residency::ExecutionLease &lease,
    const std::array<std::shared_ptr<PipelineState>,
                     residency::execution::BankCapacity> &pipelines,
    const PipelineResidencyWindowReleaseCompletion release,
    const PipelineResidencyWindowFinalCompletion final, void *const user,
    PipelineResidencyWindowControl &control) noexcept {
  return submit_pipeline_execution_window_chunk(
      plan, lease, 0u, static_cast<std::size_t>(plan.epoch_count()), pipelines,
      release, final, user, control);
}

Status submit_pipeline_execution_window_chunk(
    const residency::execution::Plan &plan,
    const residency::ExecutionLease &lease, const std::uint64_t first_epoch,
    const std::size_t count,
    const std::array<std::shared_ptr<PipelineState>,
                     residency::execution::BankCapacity> &pipelines,
    const PipelineResidencyWindowReleaseCompletion release,
    const PipelineResidencyWindowFinalCompletion final, void *const user,
    PipelineResidencyWindowControl &control) noexcept {
  return submit_chunk_impl(plan, lease, first_epoch, count, pipelines, release,
                           final, user, control, nullptr);
}

Status submit_pipeline_execution_stream_chunk(
    const residency::execution::Plan &plan,
    const residency::ExecutionLease &lease, const std::uint64_t first_epoch,
    const std::size_t count,
    const std::array<std::shared_ptr<PipelineState>,
                     residency::execution::BankCapacity> &pipelines,
    const PipelineResidencyWindowReleaseCompletion release,
    const PipelineResidencyWindowFinalCompletion final, void *const user,
    PipelineResidencyWindowControl &control,
    node::accel::detail::PreparedResidencyStreamControl &stream) noexcept {
  return submit_chunk_impl(plan, lease, first_epoch, count, pipelines, release,
                           final, user, control, &stream);
}

} // namespace rund::compute::detail
