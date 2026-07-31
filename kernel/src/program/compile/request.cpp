#include "local.hpp"

#include <span>

namespace rund::kernel::program_detail {

ScheduleCompileRequest BuildProgramScheduleRequest(
    const KernelProgramCompileRequest& request,
    const WorkerBackendCapabilities& capabilities) {
  ScheduleCompileRequest schedule_request = request.schedule;
  if (request.require_no_allocation) {
    schedule_request.allocation = AllocationPolicy::NoGrowth;
  }
  if (capabilities.worker_capacity_milli != nullptr &&
      capabilities.worker_capacity_count >= schedule_request.execution_width) {
    schedule_request.worker_capacity_milli =
        std::span<const u32>(capabilities.worker_capacity_milli,
                             capabilities.worker_capacity_count);
    schedule_request.trust_worker_capacity =
        capabilities.worker_capacity_truth_level == WorkerTruthLevel::Verified;
  }
  return schedule_request;
}

WorkspaceReservation BuildProgramWorkspaceReservation(
    const KernelProgramCompileRequest& request,
    const ScheduleCompileRequest& schedule_request) {
  return KernelProgramWorkspaceReservation(KernelProgramCompileRequest{
      .schedule = schedule_request,
      .worker_backend = request.worker_backend,
      .require_no_allocation = request.require_no_allocation,
      .collect_worker_stats = request.collect_worker_stats,
      .require_dispatch_backend = request.require_dispatch_backend,
      .physical_tile_policy = request.physical_tile_policy,
      .fold_operation = request.fold_operation,
      .strict_float_reduction = request.strict_float_reduction,
  });
}

} // namespace rund::kernel::program_detail
