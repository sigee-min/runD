#include "../../state/program.hpp"
#include "local.hpp"

#include "../../../device/state.hpp"
#include "../../../exception.hpp"
#include "../../graph.hpp"

#include <utility>

namespace rund::compute::detail {
namespace {

using cpu_run_state_detail::add_count;

[[nodiscard]] bool
valid_route_plan(const std::shared_ptr<ProgramState> &program,
                 const CpuRunRoutePlan &plan) noexcept {
  if (program == nullptr || program->cpu_graph == nullptr) {
    return plan == CpuRunRoutePlan{};
  }
  const CpuGraphProgram &graph = *program->cpu_graph;
  if (graph.runtime == nullptr || plan.program != &graph ||
      plan.runtime != graph.runtime.get() ||
      plan.graph_hash != graph.graph_hash ||
      plan.step_count != graph.runtime->steps.size() ||
      plan.map_count > plan.step_count) {
    return false;
  }
  std::size_t map_count = 0u;
  std::size_t read_count = 0u;
  std::size_t write_count = 0u;
  for (const CpuRuntimeStep &step : graph.runtime->steps) {
    const auto *const map = std::get_if<CpuRuntimeMap>(&step);
    if (map != nullptr && (!add_count(map_count, 1u) ||
                           !add_count(read_count, map->inputs.size()) ||
                           !add_count(write_count, map->outputs.size()))) {
      return false;
    }
  }
  return plan.map_count == map_count && plan.read_count == read_count &&
         plan.write_count == write_count;
}

} // namespace

Status materialize_cpu_run(
    CpuRun &run, const std::shared_ptr<ProgramState> &program,
    const std::span<const std::shared_ptr<BufferState>> workspace) {
  const auto storage_plan = plan_cpu_graph_storage(program);
  const auto route_plan = plan_cpu_run_route(program);
  if (!storage_plan || !route_plan) {
    return Status::fail(storage_plan ? route_plan.reason()
                                     : storage_plan.reason());
  }
  if (route_plan->program == nullptr) {
    return run.graph == nullptr ? Status::success()
                                : Status::fail(Reason::CpuRuntimeInvalid);
  }
  if (program == nullptr || program->device == nullptr ||
      program->device->host_page_bytes == 0u) {
    return Status::fail(Reason::CpuRuntimeInvalid);
  }
  CpuPreparedArenaPlan arena_plan{};
  arena_plan.execution = storage_plan->execution;
  CpuRunRouteSlice route_slice{};
  if (!append_cpu_run_route_slice(arena_plan, *route_plan, route_slice) ||
      !seal_cpu_prepared_arena_plan(arena_plan,
                                    program->device->host_page_bytes)) {
    return Status::fail(Reason::ProgramCapacity);
  }
  auto prepared = make_cpu_prepared_arena(arena_plan);
  if (!prepared) {
    return Status::fail(prepared.reason());
  }
  auto storage = make_cpu_graph_storage(program, *storage_plan, *prepared);
  if (!storage) {
    return Status::fail(storage.reason());
  }
  if (!(*prepared)->claims_complete(storage_plan->execution)) {
    return Status::fail(Reason::CpuRuntimeInvalid);
  }
  return materialize_cpu_run(run, program, workspace,
                             std::move(storage).value(), *route_plan,
                             std::move(prepared).value(), route_slice);
}

Status materialize_cpu_run(
    CpuRun &cpu, const std::shared_ptr<ProgramState> &program,
    const std::span<const std::shared_ptr<BufferState>> workspace,
    std::shared_ptr<CpuGraphStorage> storage, const CpuRunRoutePlan &plan,
    std::shared_ptr<CpuPreparedArena> prepared_arena,
    const CpuRunRouteSlice &route_slice) {
  if (!valid_route_plan(program, plan)) {
    return Status::fail(Reason::CpuRuntimeInvalid);
  }
  if (plan.program == nullptr) {
    return cpu.graph == nullptr ? Status::success()
                                : Status::fail(Reason::CpuRuntimeInvalid);
  }
  if (workspace.size() != program->chunks.size()) {
    return Status::fail(Reason::CpuRuntimeInvalid);
  }
  if (storage == nullptr || storage->program != program->cpu_graph.get() ||
      prepared_arena == nullptr || storage->prepared_arena != prepared_arena ||
      route_slice.map_count != plan.map_count ||
      route_slice.read_count != plan.read_count ||
      route_slice.write_count != plan.write_count) {
    return Status::fail(Reason::CpuRuntimeInvalid);
  }
  try {
    if (cpu.graph != nullptr) {
      return Status::fail(Reason::CpuRuntimeInvalid);
    }
    CpuGraphRun &run = cpu.graph.emplace();
    run.storage = std::move(storage);
    if (!prepared_arena->view(route_slice, run.maps, run.reads, run.writes)) {
      cpu.graph.reset();
      return Status::fail(Reason::CpuRuntimeInvalid);
    }
    run.buffers = workspace;
    return Status::success();
  } catch (...) {
    compute_exception::rethrow_unless_capacity_exception();
    cpu.graph.reset();
    return Status::fail(Reason::BufferCapacity);
  }
}

} // namespace rund::compute::detail
