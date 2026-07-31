#include "local.hpp"
#include "../timing/local.hpp"

#include "../../schedule/workspace/placement/local.hpp"

namespace rund::kernel::program_detail {

KernelProgramPlacementMetadata BuildTimedProgramPlacement(
    Workspace& workspace,
    const ScheduleCompileRequest& schedule_request,
    const CurrentKernelProgramViews& current_views,
    const WorkerBackendCapabilities& capabilities) {
  const TimePoint placement_start = Now();
  KernelProgramPlacementMetadata placement = current_views.placement;
  if (current_views.schedule.packet_count != schedule_request.packet_count ||
      current_views.schedule.execution_width != schedule_request.execution_width) {
    placement = BuildKernelProgramPlacementMetadata(schedule_request,
                                                    current_views.schedule,
                                                    current_views.resolved_work_units,
                                                    capabilities);
  } else {
    placement.affinity_truth_level = capabilities.affinity_truth_level;
    placement.affinity_hint_only =
        capabilities.affinity_truth_level == WorkerTruthLevel::HintOnly;
    placement.affinity_placement_reason =
        workspace_placement::AffinityPlacementReason(capabilities);
  }
  RecordPlacementCost(workspace, placement_start);
  return placement;
}

} // namespace rund::kernel::program_detail
