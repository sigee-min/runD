#include "dispatch/local.hpp"

#include <kernel/internal/program/contract.hpp>

namespace rund::kernel::program_detail {

const char* ValidateDispatchRequirements(const KernelProgramCompileRequest& request,
                                         const ScheduleCompileRequest& schedule_request,
                                         const Workspace& workspace,
                                         const WorkerBackendCapabilities& capabilities) {
  if (!request.require_dispatch_backend) {
    return nullptr;
  }
  if (schedule_request.execution_width > 1u &&
      (!request.worker_backend || !capabilities.width_matches_request)) {
    return "pool_width_mismatch";
  }
  if (capabilities.is_nested) {
    return "pool_nested_dispatch";
  }
  if (schedule_request.execution_width > 1u &&
      schedule_request.intent == PartitionIntent::StaticWidth &&
      !capabilities.supports_claim_free_static_tiles) {
    return "static_tile_backend_required";
  }
  const bool no_growth_required = internal::ProgramRequiresNoGrowth(request, schedule_request);
  if (no_growth_required && request.collect_worker_stats &&
      GetWorkspaceCapacity(workspace).worker_stats_capacity < schedule_request.execution_width) {
    return "worker_stats_capacity_exceeded";
  }
  if (no_growth_required && request.collect_worker_stats &&
      schedule_request.execution_width > 1u && !capabilities.supports_no_alloc_worker_stats) {
    return "worker_stats_would_allocate";
  }
  return nullptr;
}

} // namespace rund::kernel::program_detail
