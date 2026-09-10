#include "../../../cpu/state/program.hpp"
#include "local.hpp"

#include "../../../cpu/map.hpp"
#include "../../../cpu/run/state.hpp"

namespace rund::compute::detail {

kernel::ComputeTileExecutor *active_tiles(JobState &job) noexcept {
  if (job.cpu == nullptr || job.cpu->graph == nullptr ||
      job.cpu->graph->storage == nullptr) {
    return nullptr;
  }
  CpuRun &run = *job.cpu;
  if (run.pass == CpuPass::Map) {
    CpuMapRun *const map = cpu_map_run(*run.graph->storage, run.step);
    return map == nullptr ? nullptr : &map->tiles;
  }
  if (run.pass == CpuPass::ScanLocal || run.pass == CpuPass::ScanCorrect ||
      run.pass == CpuPass::ReduceLocal) {
    CpuCollectiveRun *const collective =
        cpu_collective_run(*run.graph->storage, run.step);
    return collective == nullptr ? nullptr : &collective->tiles;
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
  const CpuCollectiveRun *const collective =
      run.graph->storage == nullptr
          ? nullptr
          : cpu_collective_run(*run.graph->storage, run.step);
  return collective == nullptr ? 0u : collective->tile_size;
}

} // namespace rund::compute::detail
