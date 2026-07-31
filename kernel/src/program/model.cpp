#include "model/local.hpp"
#include "model/tile.hpp"

#include <kernel/internal/program/contract.hpp>

namespace rund::kernel::program_detail {

KernelProgramTelemetrySchema
BuildProgramTelemetrySchema(const Telemetry &telemetry,
                            const KernelProgramCapacityProof &capacity,
                            const KernelProgramPlacementMetadata &placement,
                            const ScheduleView schedule) {
  return BuildKernelProgramTelemetrySchema(telemetry, capacity, placement,
                                           schedule);
}

KernelProgram
BuildProgramView(const Workspace &workspace,
                 const CurrentKernelProgramViews &current_views,
                 const KernelProgramDispatchContract dispatch,
                 const KernelProgramCapacityProof &capacity,
                 const KernelProgramPlacementMetadata &placement,
                 const WorkerBackendCapabilities &backend_capabilities,
                 const KernelProgramPhysicalTilePolicy &physical_tile_policy,
                 const bool ok, const char *const reason) {
  const KernelProgramTelemetrySchema telemetry_schema =
      BuildProgramTelemetrySchema(workspace.telemetry, capacity, placement,
                                  current_views.schedule);
  return KernelProgram{
      .generation = workspace.program_generation,
      .schedule = current_views.schedule,
      .exec_kernel = BuildKernelProgramTilePlan(
          current_views.schedule, backend_capabilities, physical_tile_policy),
      .fold_graph = current_views.fold_graph,
      .ordered_packet_indices = current_views.ordered_packet_indices,
      .ordered_packet_count = current_views.ordered_packet_count,
      .dispatch = dispatch,
      .capacity = capacity,
      .placement = placement,
      .telemetry_schema = telemetry_schema,
      .backend_capabilities = backend_capabilities,
      .ok = ok,
      .reason = reason,
  };
}

KernelProgramDispatchContract
BuildDispatchContract(const KernelProgramCompileRequest &request,
                      const ScheduleCompileRequest &schedule_request,
                      const WorkerBackendCapabilities &capabilities,
                      const KernelProgramCapacityProof &proof) {
  const bool static_tile_dispatch =
      schedule_request.intent == PartitionIntent::StaticWidth &&
      capabilities.supports_static_tile_map;
  const bool no_growth_required =
      internal::ProgramRequiresNoGrowth(request, schedule_request);
  return KernelProgramDispatchContract{
      .backend_width = capabilities.backend_width,
      .static_tile_map = static_tile_dispatch,
      .global_claim_sync_elided =
          static_tile_dispatch && capabilities.supports_claim_free_static_tiles,
      .require_no_allocation = no_growth_required,
      .collect_worker_stats = request.collect_worker_stats,
      .require_dispatch_backend = request.require_dispatch_backend,
      .no_allocation_verified = !no_growth_required || proof.satisfied,
  };
}

} // namespace rund::kernel::program_detail
