#pragma once

#include <kernel/dispatch/worker/backend.hpp>
#include <kernel/program/tile.hpp>
#include <kernel/reduction/fold/strict.hpp>
#include <kernel/schedule/planner/request.hpp>

namespace rund::kernel {

struct KernelProgramCompileRequest {
  ScheduleCompileRequest schedule{};
  WorkerBackend worker_backend{};
  bool require_no_allocation = false;
  bool collect_worker_stats = false;
  bool require_dispatch_backend = true;
  KernelProgramPhysicalTilePolicy physical_tile_policy{};
  // Program compile owns fixed-tree graph construction from this fold policy.
  // External FoldGraphView injection is intentionally not part of this request.
  FoldOperation fold_operation = FoldOperation::FixedBinaryTreeHash;
  StrictFloatReductionPolicy strict_float_reduction{};
};

} // namespace rund::kernel
