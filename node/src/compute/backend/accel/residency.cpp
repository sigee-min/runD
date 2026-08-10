#include "local.hpp"

#include "../../pipeline/state.hpp"
#include "../../status.hpp"

#include <kernel/core/checked.hpp>

#include <memory>
#include <utility>

namespace rund::compute::detail::accel_backend {
namespace {

[[nodiscard]] bool
add_submission_memory(PipelinePlan &plan,
                      const std::uint64_t retained_bytes) noexcept {
  return kernel::checked::add(plan.prepared_native_bytes, retained_bytes,
                              plan.prepared_native_bytes) &&
         kernel::checked::add(plan.prepared_bytes, retained_bytes,
                              plan.prepared_bytes) &&
         kernel::checked::add(plan.peak_bytes, retained_bytes,
                              plan.peak_bytes) &&
         kernel::checked::add(plan.committed_peak_bytes, retained_bytes,
                              plan.committed_peak_bytes) &&
         kernel::checked::add(plan.total_bytes, retained_bytes,
                              plan.total_bytes) &&
         kernel::checked::add(plan.logical_bytes, retained_bytes,
                              plan.logical_bytes) &&
         kernel::checked::add(plan.live_bytes, retained_bytes,
                              plan.live_bytes) &&
         kernel::checked::add(plan.physical_bytes, retained_bytes,
                              plan.physical_bytes);
}

} // namespace

Status prepare_pipeline_residency(PipelineState &pipeline) noexcept {
  const AccelDeviceState *const accel =
      pipeline.device == nullptr ? nullptr : accel_device(*pipeline.device);
  if (accel == nullptr || pipeline.residency_submission_memory) {
    return Status::fail(Reason::PipelineInvalid);
  }

  bool primary_supported = false;
  const rund::AccelCheck primary_query =
      node::accel::detail::QueryPreparedKernelPipelineResidency(
          pipeline.prepared, primary_supported);
  if (!primary_query.ok) {
    return Status::fail(
        project_reason(primary_query.reason, Reason::PipelineMemoryBudget));
  }

  bool alternate_supported = primary_supported;
  if (pipeline.transactional) {
    const rund::AccelCheck alternate_query =
        node::accel::detail::QueryPreparedKernelPipelineResidency(
            pipeline.alternate_prepared, alternate_supported);
    if (!alternate_query.ok) {
      return Status::fail(
          project_reason(alternate_query.reason, Reason::PipelineMemoryBudget));
    }
    if (alternate_supported != primary_supported) {
      return Status::fail(Reason::PipelineInvalid);
    }
  }
  if (!primary_supported) {
    return Status::success();
  }

  const storage::Report report =
      pipeline.device->pipeline_memory_budget.report();
  if (!report || report.available_bytes == 0u) {
    return Status::fail(Reason::DevicePipelineMemoryCapacity);
  }
  storage::Reservation capacity_gate =
      pipeline.device->pipeline_memory_budget.reserve(report.available_bytes);
  if (!capacity_gate) {
    return Status::fail(Reason::DevicePipelineMemoryCapacity);
  }

  std::shared_ptr<void> primary_candidate;
  std::uint64_t primary_bytes = 0u;
  const rund::AccelCheck prepared =
      node::accel::detail::StagePreparedKernelPipelineResidency(
          pipeline.prepared, primary_candidate, primary_bytes);
  if (!prepared.ok) {
    return Status::fail(
        project_reason(prepared.reason, Reason::PipelineMemoryBudget));
  }
  if (primary_candidate == nullptr) {
    return Status::fail(Reason::PipelineMemoryBudget);
  }

  std::uint64_t retained_bytes = primary_bytes;
  std::shared_ptr<void> alternate_candidate;
  if (pipeline.transactional) {
    std::uint64_t alternate_bytes = 0u;
    const rund::AccelCheck alternate =
        node::accel::detail::StagePreparedKernelPipelineResidency(
            pipeline.alternate_prepared, alternate_candidate, alternate_bytes);
    if (!alternate.ok) {
      return Status::fail(
          project_reason(alternate.reason, Reason::PipelineMemoryBudget));
    }
    if (alternate_candidate == nullptr) {
      return Status::fail(Reason::PipelineMemoryBudget);
    }
    if (!kernel::checked::add(retained_bytes, alternate_bytes,
                              retained_bytes)) {
      return Status::fail(Reason::PipelineMemoryBudget);
    }
  }
  if (retained_bytes > capacity_gate.max_allocated_bytes()) {
    return Status::fail(Reason::DevicePipelineMemoryCapacity);
  }

  PipelinePlan published_plan = pipeline.plan;
  if (!add_submission_memory(published_plan, retained_bytes)) {
    return Status::fail(Reason::PipelineMemoryBudget);
  }
  storage::Reservation admission = capacity_gate.partition(retained_bytes);
  if (!admission) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (!capacity_gate.refund()) {
    alternate_candidate.reset();
    primary_candidate.reset();
    static_cast<void>(admission.refund());
    return Status::fail(Reason::PipelineInvalid);
  }
  const storage::Status committed = admission.commit(storage::Usage{
      .physical_bytes = 0u,
      .allocated_bytes = retained_bytes,
  });
  if (!committed) {
    alternate_candidate.reset();
    primary_candidate.reset();
    static_cast<void>(admission.refund());
    return Status::fail(Reason::PipelineInvalid);
  }

  pipeline.residency_submission_memory = std::move(admission);
  pipeline.plan = published_plan;
  node::accel::detail::CommitPreparedKernelPipelineResidency(
      pipeline.prepared, std::move(primary_candidate));
  if (pipeline.transactional) {
    node::accel::detail::CommitPreparedKernelPipelineResidency(
        pipeline.alternate_prepared, std::move(alternate_candidate));
  }
  return Status::success();
}

} // namespace rund::compute::detail::accel_backend
