#pragma once

#include <accel/device.hpp>

#include "../device.hpp"
#include "op.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/lowering/artifact/admission.hpp>
#include <kernel/program/compute/lowering/entry.hpp>
#include <kernel/program/compute/lowering/fusion/build.hpp>
#include <kernel/program/compute/plan.hpp>

namespace node_accel_contract::cpu::pick {

struct Resources {
  rund::AccelDevice pick{};
  rund::kernel::TilePhaseDescription phase{};
  rund::kernel::ComputePlan plan{};
  rund::kernel::LoweringArtifact artifact{};
  rund::kernel::BindingSet bindings{};
  rund::kernel::ComputeDispatchWindow window{};
};

[[nodiscard]] inline rund::kernel::TilePhaseDescription Phase() {
  return rund::kernel::TilePhaseDescription{
      .phase_id = 26u,
      .tile_count = kTileCount,
      .capacity =
          rund::kernel::TilePhaseCapacityRequirement{
              .scratch_bytes_per_tile = 0u,
              .scratch_alignment = 1u,
              .output_shards = kTileCount,
              .queue_slots = kTileCount,
              .task_slots = kTileCount,
          },
  };
}

[[nodiscard]] inline Resources
BuildResources(const rund::compute_dsl::ComputeOp &op) {
  Resources out{};
  out.pick = node_accel_contract::PickCpu(cpu::CpuOnlyPolicy());
  out.phase = Phase();
  rund::kernel::ComputeMap map = op.map();
  map.api = rund::kernel::ComputeApi::Cpu;
  out.plan = rund::kernel::PlanCompute(
      out.phase, map, out.pick.caps,
      rund::kernel::ComputeLimit{
          .staging_bytes = out.pick.caps.staging_bytes,
          .max_window_tiles = out.pick.caps.max_window_tiles,
      });
  out.artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  out.bindings = op.bindings<rund::kernel::i32>(
      out.phase.phase_id, out.pick.cpu_caps.fixed_lane32_lanes,
      rund::kernel::ComputeApi::Cpu);
  out.window = rund::kernel::ComputeDispatchWindow{
      .begin_sequence = 0u,
      .tile_count = kTileCount,
  };
  return out;
}

} // namespace node_accel_contract::cpu::pick
