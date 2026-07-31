#pragma once

#include <accel/api.hpp>
#include <accel/device.hpp>

#include "../../local.hpp"
#include "../hash/run.hpp"
#include "resources.hpp"

#include "src/accel/context/internal.hpp"
#include <kernel/program/compute/lowering/artifact/admission.hpp>
#include <kernel/program/compute/plan.hpp>

namespace node_accel_contract::cpu_context {

[[nodiscard]] inline bool
CpuContextMapEvidenceCountersMatch(const rund::AccelDevice &pick) {
  const MapRun run = ContextMapHash(pick);
  const MapRunResources resources = MakeMapRunResources(pick);
  if (!resources.ok) {
    return false;
  }
  const auto execution = rund::node::accel::detail::AdmitKernelForExecution(
      resources.context, resources.kernel);
  if (!execution.admission.check.ok || execution.steps.size() != 1u) {
    return false;
  }
  const auto &step = execution.steps.front();
  const rund::kernel::ComputePlan plan = rund::kernel::PlanCompute(
      rund::kernel::TilePhaseDescription{
          .phase_id = resources.kernel.kernel_id,
          .tile_count = step.element_count,
          .capacity =
              rund::kernel::TilePhaseCapacityRequirement{
                  .scratch_alignment = 1u,
                  .output_shards = step.element_count,
                  .queue_slots = step.element_count,
                  .task_slots = step.element_count,
              },
      },
      step.artifact.metadata.map, resources.kernel.frozen_caps,
      rund::kernel::ComputeLimit{
          .staging_bytes = resources.kernel.frozen_caps.staging_bytes,
          .max_window_tiles = resources.kernel.frozen_caps.max_window_tiles,
      });
  const auto warm =
      rund::kernel::compute_lowering_detail::AdmitRetained(
          plan, step.artifact, &step.cpu_input);
  return run.ok && run.evidence.backend == rund::AccelApi::Cpu &&
         run.evidence.dispatch_count == 1u &&
         run.evidence.command_submit_count == 0u &&
         run.evidence.original_dispatch_count == 1u &&
         run.evidence.final_dispatch_count == 1u &&
         run.evidence.host_to_device_bytes == 0u &&
         run.evidence.device_to_host_bytes == 0u && plan.ok &&
         step.artifact.canonical_ir_bytes.empty() &&
         step.artifact.source_text.empty() && step.cpu_input.ok &&
         step.cpu_input.retained_dynamic_memory_bytes() != 0u && warm.ok &&
         warm.parse_count == 0u && warm.emission_count == 0u;
}

} // namespace node_accel_contract::cpu_context
