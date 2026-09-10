#include "tile/local.hpp"

#include "../../cpu/map.hpp"
#include "../../cpu/run/state.hpp"

namespace rund::compute::detail {

Status submit_graph_pass(JobState &job, const kernel::WorkerBackend backend,
                         void *const ready_context,
                         void (*const ready)(void *context) noexcept) noexcept {
  CpuRun &run = *job.cpu;
  cpu_tile_detail::begin_trace_dispatch(run);
  Status submitted = Status::fail(Reason::CpuStepInvalid);
  if (run.pass == CpuPass::Primitive) {
    submitted =
        cpu_tile_detail::submit_primitive(job, backend, ready_context, ready);
  } else if (kernel::ComputeTileExecutor *const tiles = active_tiles(job);
             tiles == nullptr) {
    submitted = Status::fail(Reason::CpuStepInvalid);
  } else if (run.pass == CpuPass::Map) {
    CpuMapRun &map = *cpu_map_run(*run.graph->storage, run.step);
    CpuMapRoute *const route = cpu_map_route(*run.graph, run.step);
    if (route == nullptr) {
      submitted = Status::fail(Reason::CpuStepInvalid);
    } else {
      submitted =
          cpu_tile_detail::submit_tiles(map.tiles, backend, &route->tile,
                                        run_cpu_map_tile, ready_context, ready);
    }
  } else {
    submitted = cpu_tile_detail::submit_tiles(*tiles, backend, &run.tile,
                                              cpu_tile_detail::collective_tile,
                                              ready_context, ready);
  }
  if (!submitted) {
    cpu_tile_detail::cancel_trace_dispatch(run);
  }
  return submitted;
}

} // namespace rund::compute::detail
