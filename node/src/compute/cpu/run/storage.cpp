#include "../state/program.hpp"
#include "state.hpp"

#include "storage/local.hpp"

#include "../scratch.hpp"

namespace rund::compute::detail {

Result<CpuGraphStoragePlan>
plan_cpu_graph_storage(const std::shared_ptr<ProgramState> &program) noexcept {
  if (program == nullptr || program->cpu_graph == nullptr) {
    return Result<CpuGraphStoragePlan>::success({});
  }
  const CpuGraphProgram &graph_program = *program->cpu_graph;
  if (graph_program.runtime == nullptr ||
      graph_program.maps.size() != graph_program.runtime->steps.size() ||
      graph_program.collectives.size() != graph_program.runtime->steps.size()) {
    return Result<CpuGraphStoragePlan>::fail(Reason::CpuRuntimeInvalid);
  }
  const CpuRuntimeGraph &graph = *graph_program.runtime;
  CpuGraphStoragePlan plan{
      .program = &graph_program,
      .runtime = &graph,
      .graph_hash = graph_program.graph_hash,
      .step_count = graph.steps.size(),
  };
  for (std::size_t index = 0u; index < graph.steps.size(); ++index) {
    if (graph_program.maps[index] != nullptr) {
      const auto map = plan_cpu_map_run(*graph_program.maps[index]);
      if (!map) {
        return Result<CpuGraphStoragePlan>::fail(map.reason());
      }
      ++plan.map_count;
      const CpuExecutionStoragePlan execution{
          .tiles = map->tiles,
          .map_scratch_count = map->scratch_word_count,
          .simd_count = map->workers,
      };
      if (!add_cpu_storage_bytes(plan.maps, map->bytes) ||
          !merge_cpu_execution_storage_plan(plan.execution, execution)) {
        return Result<CpuGraphStoragePlan>::fail(Reason::ProgramCapacity);
      }
    }
    if (graph_program.collectives[index] != nullptr) {
      const auto collective =
          plan_cpu_collective_run(*graph_program.collectives[index]);
      if (!collective) {
        return Result<CpuGraphStoragePlan>::fail(collective.reason());
      }
      ++plan.collective_count;
      const CpuExecutionStoragePlan execution{
          .tiles = collective->tiles,
          .collective_total_count = collective->tile_count,
          .collective_prefix_count =
              collective->needs_prefixes ? collective->tile_count : 0u,
      };
      if (!add_cpu_storage_bytes(plan.collectives, collective->bytes) ||
          !merge_cpu_execution_storage_plan(plan.execution, execution)) {
        return Result<CpuGraphStoragePlan>::fail(Reason::ProgramCapacity);
      }
    }
    const auto *const primitive =
        std::get_if<CpuRuntimePrimitive>(&graph.steps[index]);
    if (primitive == nullptr) {
      continue;
    }
    const auto scratch = plan_cpu_scratch(*primitive);
    if (!scratch) {
      return Result<CpuGraphStoragePlan>::fail(scratch.reason());
    }
    if (scratch->shape == CpuPrimitiveScratchShape::None) {
      continue;
    }
    ++plan.scratch_count;
    const Status appended =
        append_cpu_primitive_arena_plan(plan.execution, *scratch);
    if (!appended) {
      return Result<CpuGraphStoragePlan>::fail(Reason::ProgramCapacity);
    }
  }
  plan.scratch_slots = plan.scratch_count == 0u ? 0u : plan.step_count;
  const auto containers =
      plan_cpu_graph_containers(plan.step_count, plan.scratch_slots);
  if (!containers) {
    return Result<CpuGraphStoragePlan>::fail(containers.reason());
  }
  plan.containers = *containers;
  plan.private_total = plan.containers;
  if (!add_cpu_storage_bytes(plan.private_total, plan.maps) ||
      !add_cpu_storage_bytes(plan.private_total, plan.collectives)) {
    return Result<CpuGraphStoragePlan>::fail(Reason::ProgramCapacity);
  }
  return Result<CpuGraphStoragePlan>::success(plan);
}

} // namespace rund::compute::detail
