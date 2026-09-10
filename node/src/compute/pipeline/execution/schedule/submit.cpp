#include "callback.hpp"
#include "local.hpp"

#include "../../../backend.hpp"
#include "../../../status.hpp"
#include "../../state.hpp"

#include <mutex>

namespace rund::compute::detail {
namespace {

[[nodiscard]] bool
same_schedule(const PipelineExecutionSchedulePrepared &left,
              const PipelineExecutionSchedulePrepared &right) noexcept {
  if (left.request.plan_identity != right.request.plan_identity ||
      left.request.epoch_count != right.request.epoch_count ||
      left.request.role_count != right.request.role_count ||
      left.request.tail_local_count != right.request.tail_local_count) {
    return false;
  }
  for (std::size_t role = 0u; role < left.request.role_count; ++role) {
    const auto &a = left.request.roles[role];
    const auto &b = right.request.roles[role];
    if (a.pipeline.owner != b.pipeline.owner || a.locals != b.locals ||
        a.local_count != b.local_count ||
        a.first_control_generation != b.first_control_generation ||
        a.control_generation_stride != b.control_generation_stride ||
        a.role != b.role || a.bank != b.bank ||
        left.pipelines[role] != right.pipelines[role]) {
      return false;
    }
  }
  return true;
}

} // namespace

Status submit_pipeline_execution_schedule_impl(
    const residency::execution::Plan &plan,
    const PipelineExecutionSchedulePrepared &prepared,
    const residency::ExecutionLease &lease,
    const PipelineResidencyWindowReleaseCompletion release,
    const PipelineResidencyScheduleFinalCompletion final, void *const user,
    PipelineResidencyScheduleControl &control,
    node::accel::detail::PreparedResidencyStreamControl &stream) noexcept {
  if (!prepared || !lease || lease.plan != plan.identity() ||
      lease.epochs != plan.epoch_count() || release == nullptr ||
      final == nullptr || user == nullptr) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::lock_guard control_lock{control.gate};
  if (control.active) {
    return Status::fail(Reason::PipelineBusy);
  }
  PipelineExecutionSchedulePrepared current{};
  const std::array<std::shared_ptr<PipelineState>,
                   residency::execution::BankCapacity>
      banks{prepared.pipelines[0u], prepared.pipelines[1u]};
  const Status rebuilt =
      build_pipeline_execution_schedule(plan, banks, current, false);
  if (!rebuilt || !same_schedule(prepared, current)) {
    return rebuilt ? Status::fail(Reason::PipelineBusy) : rebuilt;
  }
  node::accel::detail::PreparedResidencyScheduleRequest request =
      prepared.request;
  request.token = lease.token;
  request.generation = lease.generation;
  request.release = complete_schedule_release;
  request.final = complete_schedule_final;
  request.user = &control;
  control.pipelines = prepared.pipelines;
  control.lowering = prepared.lowering;
  control.plan = &plan;
  control.lease = lease;
  control.release = release;
  control.final = final;
  control.user = user;
  control.active = true;
  const std::shared_ptr<DeviceState> &device = prepared.pipelines[0u]->device;
  const Status submitted = device->ops->residency.submit_residency_schedule(
      *device, request, control.native, stream);
  if (!submitted) {
    control.plan = nullptr;
    control.lease = {};
    control.pipelines.fill(nullptr);
    control.attempts = {};
    control.release = nullptr;
    control.final = nullptr;
    control.user = nullptr;
    control.active = false;
  }
  return submitted;
}

} // namespace rund::compute::detail
