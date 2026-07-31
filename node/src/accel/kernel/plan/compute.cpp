#include <accel/kernel/run.hpp>

#include "local.hpp"

#include <kernel/program/compute/plan.hpp>

namespace rund::node::accel::detail {

const rund::kernel::ExecutionMetadata &
StepMetadata(const KernelExecutionStep &step) noexcept {
  return step.artifact.metadata;
}

rund::kernel::TilePhaseDescription PhaseFor(const KernelExecution &execution,
                                            const KernelExecutionStep &step,
                                            const rund::AccelRun &run,
                                            const std::uint64_t step_index) {
  const std::uint64_t tile_count = step.kind() == rund::kernel::NodeKind::Map
                                       ? step.element_count
                                       : run.tile_count;
  return rund::kernel::TilePhaseDescription{
      .phase_id = execution.admission.kernel_id + step_index,
      .tile_count = tile_count,
      .capacity =
          rund::kernel::TilePhaseCapacityRequirement{
              .scratch_bytes_per_tile = 0u,
              .scratch_alignment = 1u,
              .output_shards = tile_count,
              .queue_slots = tile_count,
              .task_slots = tile_count,
          },
  };
}

rund::kernel::ComputePlan PlanStep(const KernelExecution &execution,
                                   const KernelExecutionStep &step,
                                   const rund::AccelRun &run,
                                   const std::uint64_t step_index) {
  rund::kernel::ComputeMap map = StepMetadata(step).map;
  map.api = execution.admission.frozen_caps.api;
  return rund::kernel::PlanCompute(
      PhaseFor(execution, step, run, step_index), map,
      execution.admission.frozen_caps,
      rund::kernel::ComputeLimit{
          .staging_bytes = execution.admission.frozen_caps.staging_bytes,
          .max_window_tiles = execution.admission.frozen_caps.max_window_tiles,
      });
}

void AdoptPlanStatus(PlannedStep &planned, const bool ok,
                     const char *const reason) noexcept {
  if (!ok) {
    planned.plan.ok = false;
    planned.plan.reason = reason;
    return;
  }
  planned.plan.ok = true;
  planned.plan.reason = "ok";
}

} // namespace rund::node::accel::detail
