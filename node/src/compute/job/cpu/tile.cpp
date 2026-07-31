#include "model.hpp"

#include "../../cpu/collective/execute.hpp"
#include "../../cpu/map.hpp"
#include "../../cpu/run/state.hpp"
#include "../../status.hpp"

#include <kernel/dispatch/kernel.hpp>

#include <algorithm>
#include <cstdint>
#include <string_view>

namespace rund::compute::detail {

namespace {

void primitive_worker(void *const raw, const kernel::Partition &) noexcept {
  auto *const job = static_cast<JobState *>(raw);
  if (job == nullptr || job->cpu == nullptr) {
    return;
  }
  job->cpu->primitive_status = execute_cpu_primitive(*job, job->cpu->step);
}

void primitive_ready(void *const raw, const bool ok) noexcept {
  auto *const run = static_cast<CpuRun *>(raw);
  if (run == nullptr) {
    return;
  }
  if (!ok) {
    run->primitive_status = Status::fail(Reason::PrimitiveBackendFailed);
  }
  if (run->primitive_ready != nullptr) {
    run->primitive_ready(run->primitive_ready_context);
  }
}

[[nodiscard]] Status
submit_primitive(JobState &job, const kernel::WorkerBackend backend,
                 void *const ready_context,
                 void (*const ready)(void *context) noexcept) noexcept {
  if (!backend || backend.submit_partitions == nullptr || job.cpu == nullptr ||
      ready == nullptr) {
    return Status::fail(Reason::PrimitiveBackendInvalid);
  }
  const kernel::u32 width = backend.worker_count == nullptr
                                ? 0u
                                : backend.worker_count(backend.context);
  const kernel::WorkerBackendCapabilities caps =
      kernel::InspectWorkerBackend(backend, width);
  if (!caps.width_matches_request || !caps.supports_async_partitions ||
      caps.is_nested) {
    return Status::fail(Reason::PrimitiveBackendInvalid);
  }
  CpuRun &run = *job.cpu;
  run.primitive_partition =
      kernel::Partition{.worker_index = 0u, .begin = 0u, .end = 1u};
  run.primitive_status = Status::fail(Reason::PrimitiveNotReady);
  run.primitive_ready_context = ready_context;
  run.primitive_ready = ready;
  run.primitive_submission.remaining.store(0u, std::memory_order_relaxed);
  run.primitive_submission.failed.store(false, std::memory_order_relaxed);
  run.primitive_submission.completion = kernel::WorkerCompletion{
      .context = &run,
      .invoke = primitive_ready,
  };
  const bool submitted = backend.submit_partitions(
      backend.context, &run.primitive_partition, 1u,
      kernel::WorkerTask{.context = &job, .invoke = primitive_worker}, nullptr,
      &run.primitive_submission);
  if (!submitted) {
    run.primitive_ready = nullptr;
    run.primitive_ready_context = nullptr;
    return Status::fail(Reason::PrimitiveSubmitFailed);
  }
  return Status::success();
}

kernel::ComputeTileCallbackResult
collective_tile(const void *const raw,
                const kernel::ComputeTile &tile) noexcept {
  auto &context = *const_cast<CpuCollectiveTileContext *>(
      static_cast<const CpuCollectiveTileContext *>(raw));
  if (context.run == nullptr) {
    return {false, "compute_cpu_buffer_invalid"};
  }
  if (context.kind == CpuCollectiveKind::Scan) {
    switch (context.type) {
    case Type::I32:
    case Type::FixedLane32:
      return RunScanTile(*context.run, context.scan, context.pass,
                         static_cast<const std::int32_t *>(context.input),
                         static_cast<std::int32_t *>(context.output), tile,
                         context.cancel);
    case Type::U32:
      return RunScanTile(*context.run, context.scan, context.pass,
                         static_cast<const std::uint32_t *>(context.input),
                         static_cast<std::uint32_t *>(context.output), tile,
                         context.cancel);
    case Type::I64:
    case Type::FixedLane64:
      return RunScanTile(*context.run, context.scan, context.pass,
                         static_cast<const std::int64_t *>(context.input),
                         static_cast<std::int64_t *>(context.output), tile,
                         context.cancel);
    case Type::U64:
      return RunScanTile(*context.run, context.scan, context.pass,
                         static_cast<const std::uint64_t *>(context.input),
                         static_cast<std::uint64_t *>(context.output), tile,
                         context.cancel);
    }
  }
  switch (context.type) {
  case Type::I32:
  case Type::FixedLane32:
    return RunReduceTile(*context.run, context.reduce,
                         static_cast<const std::int32_t *>(context.input), tile,
                         context.cancel);
  case Type::U32:
    return RunReduceTile(*context.run, context.reduce,
                         static_cast<const std::uint32_t *>(context.input),
                         tile, context.cancel);
  case Type::I64:
  case Type::FixedLane64:
    return RunReduceTile(*context.run, context.reduce,
                         static_cast<const std::int64_t *>(context.input), tile,
                         context.cancel);
  case Type::U64:
    return RunReduceTile(*context.run, context.reduce,
                         static_cast<const std::uint64_t *>(context.input),
                         tile, context.cancel);
  }
  return {false, "compute_domain_unsupported"};
}

[[nodiscard]] Status
submit_tiles(kernel::ComputeTileExecutor &tiles,
             const kernel::WorkerBackend backend, const void *const context,
             const kernel::ComputeTileCallback callback,
             void *const ready_context,
             void (*const ready)(void *context) noexcept) noexcept {
  const kernel::ComputeTileSubmitResult submitted = tiles.submit_with_erased(
      backend, context, callback, ready_context, ready);
  return submitted.ok ? Status::success()
                      : Status::fail(project_reason(submitted.reason,
                                                    Reason::TileBackendFailed));
}

void add_tiles(Stats &stats, const kernel::ComputeTileRunResult &tiles,
               const std::uint64_t dispatches,
               const std::uint64_t tile_size) noexcept {
  ::rund::detail::counter::Accumulate(stats.dispatches, dispatches);
  stats.worker_count = tiles.worker_count;
  stats.participating_workers = tiles.participating_workers;
  ::rund::detail::counter::Accumulate(stats.tile_count,
                                      tiles.worker_tile_count);
  stats.tile_size = std::max(stats.tile_size, tile_size);
}

} // namespace

kernel::ComputeTileExecutor *active_tiles(JobState &job) noexcept {
  if (job.cpu == nullptr || job.cpu->graph == nullptr) {
    return nullptr;
  }
  CpuRun &run = *job.cpu;
  if (run.pass == CpuPass::Map) {
    if (run.step >= run.graph->maps.size() ||
        run.graph->maps[run.step] == nullptr) {
      return nullptr;
    }
    return &run.graph->maps[run.step]->tiles;
  }
  if (run.pass == CpuPass::ScanLocal || run.pass == CpuPass::ScanCorrect ||
      run.pass == CpuPass::ReduceLocal) {
    if (run.step >= run.graph->collectives.size() ||
        run.graph->collectives[run.step] == nullptr) {
      return nullptr;
    }
    return &run.graph->collectives[run.step]->tiles;
  }
  return nullptr;
}

std::uint64_t active_tile_size(const JobState &job) noexcept {
  if (job.cpu == nullptr || job.cpu->graph == nullptr ||
      job.program == nullptr || job.program->cpu_graph == nullptr) {
    return 0u;
  }
  const CpuRun &run = *job.cpu;
  if (run.pass == CpuPass::Map) {
    return run.step < job.program->cpu_graph->maps.size() &&
                   job.program->cpu_graph->maps[run.step] != nullptr
               ? job.program->cpu_graph->maps[run.step]->tile_size
               : 0u;
  }
  return run.step < run.graph->collectives.size() &&
                 run.graph->collectives[run.step] != nullptr
             ? run.graph->collectives[run.step]->tile_size
             : 0u;
}

Status submit_graph_pass(JobState &job, const kernel::WorkerBackend backend,
                         void *const ready_context,
                         void (*const ready)(void *context) noexcept) noexcept {
  CpuRun &run = *job.cpu;
  if (run.pass == CpuPass::Primitive) {
    return submit_primitive(job, backend, ready_context, ready);
  }
  kernel::ComputeTileExecutor *const tiles = active_tiles(job);
  if (tiles == nullptr) {
    return Status::fail(Reason::CpuStepInvalid);
  }
  if (run.pass == CpuPass::Map) {
    CpuMapRun &map = *run.graph->maps[run.step];
    return submit_tiles(map.tiles, backend, &map.tile, run_cpu_map_tile,
                        ready_context, ready);
  }
  return submit_tiles(*tiles, backend, &run.tile, collective_tile,
                      ready_context, ready);
}

kernel::ComputeTileRunResult run_graph_pass(JobState &job) {
  CpuRun &run = *job.cpu;
  kernel::ComputeTileExecutor *const tiles = active_tiles(job);
  if (tiles == nullptr) {
    return {.reason = "compute_cpu_step_invalid"};
  }
  if (run.pass == CpuPass::Map) {
    CpuMapRun &map = *run.graph->maps[run.step];
    return map.tiles.run([&](const kernel::ComputeTile &tile) {
      return run_cpu_map_tile(&map.tile, tile);
    });
  }
  return tiles->run([&](const kernel::ComputeTile &tile) {
    return collective_tile(&run.tile, tile);
  });
}

PassResult finish_graph_pass(JobState &job,
                             const kernel::ComputeTileRunResult *const tiles,
                             const std::atomic_bool *const cancel) noexcept {
  CpuRun &run = *job.cpu;
  if (run.pass == CpuPass::Primitive) {
    run.primitive_ready = nullptr;
    run.primitive_ready_context = nullptr;
    if (cancel != nullptr && cancel->load(std::memory_order_acquire)) {
      return {.status = Status::fail(Reason::Cancelled)};
    }
    if (!run.primitive_status) {
      return {.status = run.primitive_status};
    }
    ::rund::detail::counter::Accumulate(run.stats.dispatches, 1u);
    ++run.step;
    return {};
  }
  if (tiles == nullptr || !tiles->ok) {
    return {.status = Status::fail(project_reason(
                tiles == nullptr ? "compute_cpu_step_invalid" : tiles->reason,
                Reason::TileBackendFailed))};
  }
  if (cancel != nullptr && cancel->load(std::memory_order_acquire)) {
    return {.status = Status::fail(Reason::Cancelled)};
  }
  const kernel::ComputeTileExecutor *const executor = active_tiles(job);
  if (executor == nullptr) {
    return {.status = Status::fail(Reason::CpuStepInvalid)};
  }
  const std::uint64_t dispatches =
      executor->count() == 0u ? 0u : tiles->backend_dispatch_count;
  const std::uint64_t tile_size = active_tile_size(job);
  if (run.pass == CpuPass::Map) {
    add_tiles(run.stats, *tiles, dispatches, tile_size);
    const CpuSimdCount simd = sum_simd(*run.graph->maps[run.step]);
    ::rund::detail::counter::Accumulate(run.stats.vector_chunks, simd.vectors);
    ::rund::detail::counter::Accumulate(run.stats.tail_chunks, simd.tails);
    ++run.step;
    return {};
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
      return {.status = Status::fail(
                  project_reason(reason, Reason::TileBackendFailed))};
    }
    run.pending_dispatches = dispatches;
    run.pass = CpuPass::ScanCorrect;
    run.tile.pass = CpuPass::ScanCorrect;
    return {.flow = PassFlow::Repeat};
  }
  if (run.pass == CpuPass::ScanCorrect) {
    add_tiles(run.stats, *tiles, dispatches, tile_size);
    ::rund::detail::counter::Accumulate(run.stats.dispatches,
                                        run.pending_dispatches);
    run.pending_dispatches = 0u;
    ++run.step;
    return {};
  }
  if (run.pass != CpuPass::ReduceLocal) {
    return {.status = Status::fail(Reason::CpuStepInvalid)};
  }
  add_tiles(run.stats, *tiles, dispatches, tile_size);
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
    return {.status = Status::fail(
                project_reason(reason, Reason::TileBackendFailed))};
  }
  ++run.step;
  return {};
}

} // namespace rund::compute::detail
