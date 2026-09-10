#include "local.hpp"

#include "../../../backend.hpp"
#include "../../../status.hpp"
#include "../../state.hpp"

#include <mutex>

namespace rund::compute::detail {

Status abort_pipeline_execution_schedule_impl(
    PipelineResidencyScheduleControl &control, const Status failure) noexcept {
  std::shared_ptr<PipelineState> leader{};
  node::accel::detail::BackendResidencyWindowAbort abort{};
  {
    std::lock_guard lock{control.gate};
    if (failure || !control.active || control.plan == nullptr) {
      return Status::fail(Reason::PipelineInvalid);
    }
    leader = control.pipelines[0u];
    if (leader == nullptr || leader->device == nullptr ||
        leader->device->ops == nullptr ||
        leader->device->ops->residency.abort_residency_schedule == nullptr) {
      return Status::fail(Reason::BackendUnsupported);
    }
    abort = node::accel::detail::BackendResidencyWindowAbort{
        .failure = {false, reason_message(failure.reason()).data()},
        .plan_identity = control.lease.plan,
        .token = control.lease.token,
        .generation = control.lease.generation,
    };
  }
  return leader->device->ops->residency.abort_residency_schedule(
      *leader->device, control.native, abort);
}

} // namespace rund::compute::detail
