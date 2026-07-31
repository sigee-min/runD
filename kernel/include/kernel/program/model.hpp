#pragma once

#include <kernel/dispatch/worker/capabilities.hpp>
#include <kernel/program/capacity.hpp>
#include <kernel/program/dispatch.hpp>
#include <kernel/program/placement.hpp>
#include <kernel/program/telemetry.hpp>
#include <kernel/program/tile.hpp>
#include <kernel/reduction/fold/graph/state.hpp>
#include <kernel/schedule/planner/view.hpp>

namespace rund::kernel {

struct KernelProgram {
  u64 generation = 0u;
  ScheduleView schedule{};
  KernelProgramTilePlan exec_kernel{};
  FoldGraphView fold_graph{};
  const u32* ordered_packet_indices = nullptr;
  u32 ordered_packet_count = 0u;
  KernelProgramDispatchContract dispatch{};
  KernelProgramCapacityProof capacity{};
  KernelProgramPlacementMetadata placement{};
  KernelProgramTelemetrySchema telemetry_schema{};
  WorkerBackendCapabilities backend_capabilities{};
  bool ok = false;
  const char* reason = "not_run";
};

} // namespace rund::kernel
