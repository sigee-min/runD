#include "local.hpp"

#include "../../../cpu/collective/execute.hpp"
#include "../../../status.hpp"

#include <algorithm>

namespace rund::compute::detail::cpu_tile_detail {

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

Status submit_tiles(kernel::ComputeTileExecutor &tiles,
                    const kernel::WorkerBackend backend,
                    const void *const context,
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

} // namespace rund::compute::detail::cpu_tile_detail
