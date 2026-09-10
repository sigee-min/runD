#include "local.hpp"

#include "../../../cpu/collective/execute.hpp"
#include "../../../cpu/map.hpp"
#include "../../../cpu/run/state.hpp"
#include "../../../status.hpp"

#include <atomic>
#include <string_view>

namespace rund::compute::detail {

CpuPassResult finish_graph_pass(JobState &job,
                                const kernel::ComputeTileRunResult *const tiles,
                                const std::atomic_bool *const cancel) noexcept {
  CpuRun &run = *job.cpu;
  if (run.pass == CpuPass::Primitive) {
    cpu_tile_detail::finish_trace_dispatch(run, true);
    run.primitive_ready = nullptr;
    run.primitive_ready_context = nullptr;
    if (cancel != nullptr && cancel->load(std::memory_order_acquire)) {
      return CpuPassResult::failed(Status::fail(Reason::Cancelled));
    }
    if (!run.primitive_status) {
      return CpuPassResult::failed(run.primitive_status);
    }
    ::rund::detail::counter::Accumulate(run.stats.dispatches, 1u);
    ++run.step;
    return CpuPassResult::next();
  }
  cpu_tile_detail::finish_trace_dispatch(
      run, tiles != nullptr && tiles->backend_dispatch_count != 0u);
  if (tiles == nullptr || !tiles->ok) {
    return CpuPassResult::failed(Status::fail(project_reason(
        tiles == nullptr ? "compute_cpu_step_invalid" : tiles->reason,
        Reason::TileBackendFailed)));
  }
  if (cancel != nullptr && cancel->load(std::memory_order_acquire)) {
    return CpuPassResult::failed(Status::fail(Reason::Cancelled));
  }
  const kernel::ComputeTileExecutor *const executor = active_tiles(job);
  if (executor == nullptr) {
    return CpuPassResult::failed(Status::fail(Reason::CpuStepInvalid));
  }
  const std::uint64_t dispatches =
      executor->count() == 0u ? 0u : tiles->backend_dispatch_count;
  const std::uint64_t tile_size = active_tile_size(job);
  if (run.pass == CpuPass::Map) {
    cpu_tile_detail::add_tiles(run.stats, *tiles, dispatches, tile_size);
    const CpuSimdCount simd =
        sum_simd(*cpu_map_run(*run.graph->storage, run.step));
    ::rund::detail::counter::Accumulate(run.stats.vector_chunks, simd.vectors);
    ::rund::detail::counter::Accumulate(run.stats.tail_chunks, simd.tails);
    ++run.step;
    return CpuPassResult::next();
  }
  if (run.pass == CpuPass::ScanLocal) {
    const char *reason = "compute_domain_unsupported";
    switch (run.tile.type) {
    case Type::I32:
    case Type::FixedLane32:
      reason = MergeScanTiles<std::int32_t>(*run.tile.run);
      break;
    case Type::U32:
      reason = MergeScanTiles<std::uint32_t>(*run.tile.run);
      break;
    case Type::I64:
    case Type::FixedLane64:
      reason = MergeScanTiles<std::int64_t>(*run.tile.run);
      break;
    case Type::U64:
      reason = MergeScanTiles<std::uint64_t>(*run.tile.run);
      break;
    }
    if (std::string_view{reason} != "pass") {
      return CpuPassResult::failed(
          Status::fail(project_reason(reason, Reason::TileBackendFailed)));
    }
    run.pending_dispatches = dispatches;
    run.pass = CpuPass::ScanCorrect;
    run.tile.pass = CpuPass::ScanCorrect;
    return CpuPassResult::repeat();
  }
  if (run.pass == CpuPass::ScanCorrect) {
    cpu_tile_detail::add_tiles(run.stats, *tiles, dispatches, tile_size);
    ::rund::detail::counter::Accumulate(run.stats.dispatches,
                                        run.pending_dispatches);
    run.pending_dispatches = 0u;
    ++run.step;
    return CpuPassResult::next();
  }
  if (run.pass != CpuPass::ReduceLocal) {
    return CpuPassResult::failed(Status::fail(Reason::CpuStepInvalid));
  }
  cpu_tile_detail::add_tiles(run.stats, *tiles, dispatches, tile_size);
  const char *reason = "compute_domain_unsupported";
  switch (run.tile.type) {
  case Type::I32:
  case Type::FixedLane32:
    reason = MergeReduceTiles(*run.tile.run, run.tile.reduce,
                              static_cast<std::int32_t *>(run.tile.output));
    break;
  case Type::U32:
    reason = MergeReduceTiles(*run.tile.run, run.tile.reduce,
                              static_cast<std::uint32_t *>(run.tile.output));
    break;
  case Type::I64:
  case Type::FixedLane64:
    reason = MergeReduceTiles(*run.tile.run, run.tile.reduce,
                              static_cast<std::int64_t *>(run.tile.output));
    break;
  case Type::U64:
    reason = MergeReduceTiles(*run.tile.run, run.tile.reduce,
                              static_cast<std::uint64_t *>(run.tile.output));
    break;
  }
  if (std::string_view{reason} != "pass") {
    return CpuPassResult::failed(
        Status::fail(project_reason(reason, Reason::TileBackendFailed)));
  }
  ++run.step;
  return CpuPassResult::next();
}

} // namespace rund::compute::detail
