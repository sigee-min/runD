#include "../../state/program.hpp"
#include "local.hpp"

#include "../../../device/state.hpp"
#include "../../graph.hpp"

namespace rund::compute::detail {

using cpu_run_state_detail::add_count;

Result<CpuRunRoutePlan>
plan_cpu_run_route(const std::shared_ptr<ProgramState> &program) noexcept {
  if (program == nullptr || program->cpu_graph == nullptr) {
    return Result<CpuRunRoutePlan>::success({});
  }
  const CpuGraphProgram &graph_program = *program->cpu_graph;
  if (graph_program.runtime == nullptr) {
    return Result<CpuRunRoutePlan>::fail(Reason::CpuRuntimeInvalid);
  }
  const CpuRuntimeGraph &graph = *graph_program.runtime;
  CpuRunRoutePlan plan{
      .program = &graph_program,
      .runtime = &graph,
      .graph_hash = graph_program.graph_hash,
      .step_count = graph.steps.size(),
  };
  for (const CpuRuntimeStep &step : graph.steps) {
    const auto *const map = std::get_if<CpuRuntimeMap>(&step);
    if (map == nullptr) {
      continue;
    }
    if (!add_count(plan.map_count, 1u) ||
        !add_count(plan.read_count, map->inputs.size()) ||
        !add_count(plan.write_count, map->outputs.size())) {
      return Result<CpuRunRoutePlan>::fail(Reason::ProgramCapacity);
    }
  }
  return Result<CpuRunRoutePlan>::success(plan);
}

bool append_cpu_run_route_slice(CpuPreparedArenaPlan &arena,
                                const CpuRunRoutePlan &route,
                                CpuRunRouteSlice &slice) noexcept {
  if (arena.layout.sealed || route.map_count > route.step_count) {
    return false;
  }
  const CpuRunRouteSlice next{
      .map_begin = arena.map_count,
      .map_count = route.map_count,
      .read_begin = arena.read_count,
      .read_count = route.read_count,
      .write_begin = arena.write_count,
      .write_count = route.write_count,
  };
  std::size_t maps = arena.map_count;
  std::size_t reads = arena.read_count;
  std::size_t writes = arena.write_count;
  if (!add_count(maps, route.map_count) ||
      !add_count(reads, route.read_count) ||
      !add_count(writes, route.write_count)) {
    return false;
  }
  arena.map_count = maps;
  arena.read_count = reads;
  arena.write_count = writes;
  slice = next;
  return true;
}

} // namespace rund::compute::detail
