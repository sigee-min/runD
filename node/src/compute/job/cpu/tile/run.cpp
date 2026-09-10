#include "local.hpp"

#include "../../../cpu/map.hpp"
#include "../../../cpu/run/state.hpp"

namespace rund::compute::detail {

kernel::ComputeTileRunResult run_graph_pass(JobState &job) {
  CpuRun &run = *job.cpu;
  kernel::ComputeTileExecutor *const tiles = active_tiles(job);
  if (tiles == nullptr) {
    return {.reason = "compute_cpu_step_invalid"};
  }
  if (run.pass == CpuPass::Map) {
    CpuMapRun &map = *cpu_map_run(*run.graph->storage, run.step);
    CpuMapRoute *const route = cpu_map_route(*run.graph, run.step);
    if (route == nullptr) {
      return {.reason = "compute_cpu_step_invalid"};
    }
    return map.tiles.run([&](const kernel::ComputeTile &tile) {
      return run_cpu_map_tile(&route->tile, tile);
    });
  }
  return tiles->run([&](const kernel::ComputeTile &tile) {
    return cpu_tile_detail::collective_tile(&run.tile, tile);
  });
}

} // namespace rund::compute::detail
