#include "../../state/program.hpp"
#include "local.hpp"

#include "../../../buffer/local.hpp"
#include "../../../exception.hpp"
#include "../../../memory/cpu.hpp"
#include "../../scratch.hpp"

#include <kernel/program/compute/tile/run.hpp>

#include <utility>
#include <variant>

namespace rund::compute::detail {
namespace {

[[nodiscard]] bool
same_executor_memory(const kernel::ComputeTileRetainedMemory &left,
                     const kernel::ComputeTileRetainedMemory &right) noexcept {
  return left.state_bytes == right.state_bytes &&
         left.workspace_bytes == right.workspace_bytes &&
         left.failure_slot_bytes == right.failure_slot_bytes &&
         left.worker_tile_bytes == right.worker_tile_bytes &&
         left.async_context_bytes == right.async_context_bytes &&
         left.total_bytes == right.total_bytes;
}

[[nodiscard]] bool same_tile_storage_plan(
    const kernel::ComputeTileRunStoragePlan &left,
    const kernel::ComputeTileRunStoragePlan &right) noexcept {
  return left.failure_slot_capacity == right.failure_slot_capacity &&
         left.worker_capacity == right.worker_capacity && left.ok == right.ok &&
         left.retained.ok == right.retained.ok &&
         same_executor_memory(left.retained.memory, right.retained.memory);
}

[[nodiscard]] bool
valid_storage_plan(const std::shared_ptr<ProgramState> &program,
                   const CpuGraphStoragePlan &plan) noexcept {
  if (program == nullptr || program->cpu_graph == nullptr) {
    return plan == CpuGraphStoragePlan{};
  }
  const CpuGraphProgram &graph = *program->cpu_graph;
  if (graph.runtime == nullptr || plan.program != &graph ||
      plan.runtime != graph.runtime.get() ||
      plan.graph_hash != graph.graph_hash ||
      plan.step_count != graph.runtime->steps.size() ||
      graph.maps.size() != plan.step_count ||
      graph.collectives.size() != plan.step_count ||
      plan.map_count > plan.step_count ||
      plan.collective_count > plan.step_count ||
      plan.scratch_count > plan.step_count ||
      plan.scratch_slots != (plan.scratch_count == 0u ? 0u : plan.step_count)) {
    return false;
  }
  const auto containers =
      plan_cpu_graph_containers(plan.step_count, plan.scratch_slots);
  if (!containers || *containers != plan.containers) {
    return false;
  }
  CpuStorageBytes private_total = plan.containers;
  return add_cpu_storage_bytes(private_total, plan.maps) &&
         add_cpu_storage_bytes(private_total, plan.collectives) &&
         private_total == plan.private_total &&
         ((plan.map_count == 0u && plan.collective_count == 0u) ||
          plan.execution.tiles.ok);
}

[[nodiscard]] Result<CpuMapRun> make_map_run(const CpuProgram &program,
                                             const CpuMapRunPlan &plan,
                                             CpuPreparedArena *execution) {
  if (plan.program != &program || execution == nullptr) {
    return Result<CpuMapRun>::fail(Reason::CpuRuntimeInvalid);
  }
  try {
    CpuMapRun run{};
    run.tile_plan = program.tile_plan.run_plan();
    if (!run.tile_plan.prepared() ||
        !same_tile_storage_plan(run.tile_plan.storage_plan(), plan.tiles) ||
        plan.workers > execution->simd().size() ||
        plan.scratch_word_count > execution->map_scratch().size()) {
      return Result<CpuMapRun>::fail(Reason::TileRunCapacity);
    }
    run.simd = execution->simd().first(plan.workers);
    run.scratch = execution->map_scratch().first(plan.scratch_word_count);
    run.execution = execution;
    run.scratch_words = plan.scratch_words;
    run.workers = plan.workers;
    return Result<CpuMapRun>::success(std::move(run));
  } catch (...) {
    compute_exception::rethrow_unless_capacity_exception();
    return Result<CpuMapRun>::fail(Reason::TileRunCapacity);
  }
}

[[nodiscard]] Result<CpuCollectiveRun>
make_collective_run(const CpuCollective &program,
                    const CpuCollectiveRunPlan &plan,
                    CpuPreparedArena *execution) {
  if (plan.program != &program || execution == nullptr) {
    return Result<CpuCollectiveRun>::fail(Reason::CpuRuntimeInvalid);
  }
  try {
    CpuCollectiveRun run{};
    run.tile_plan = program.tile_plan.run_plan();
    if (!run.tile_plan.prepared() ||
        !same_tile_storage_plan(run.tile_plan.storage_plan(), plan.tiles) ||
        plan.tile_count > execution->collective_totals().size() ||
        (plan.needs_prefixes &&
         plan.tile_count > execution->collective_prefixes().size())) {
      return Result<CpuCollectiveRun>::fail(Reason::TileRunCapacity);
    }
    run.total_capacity = execution->collective_totals().first(plan.tile_count);
    run.prefix_capacity =
        plan.needs_prefixes
            ? execution->collective_prefixes().first(plan.tile_count)
            : std::span<CpuCollectiveWide>{};
    run.totals = run.total_capacity;
    run.prefixes = run.prefix_capacity;
    run.execution = execution;
    run.tile_size = program.tile_size;
    run.needs_prefixes = plan.needs_prefixes;
    return Result<CpuCollectiveRun>::success(std::move(run));
  } catch (...) {
    compute_exception::rethrow_unless_capacity_exception();
    return Result<CpuCollectiveRun>::fail(Reason::TileRunCapacity);
  }
}

} // namespace

Result<std::shared_ptr<CpuGraphStorage>>
make_cpu_graph_storage(const std::shared_ptr<ProgramState> &program,
                       const CpuGraphStoragePlan &plan,
                       std::shared_ptr<CpuPreparedArena> prepared_arena) {
  if (!valid_storage_plan(program, plan)) {
    return Result<std::shared_ptr<CpuGraphStorage>>::fail(
        Reason::CpuRuntimeInvalid);
  }
  if (plan.program == nullptr) {
    return Result<std::shared_ptr<CpuGraphStorage>>::success(nullptr);
  }
  if (prepared_arena == nullptr || !prepared_arena->supports(plan.execution)) {
    return Result<std::shared_ptr<CpuGraphStorage>>::fail(
        Reason::CpuRuntimeInvalid);
  }
  try {
    const CpuGraphProgram &graph_program = *plan.program;
    const CpuRuntimeGraph &graph = *plan.runtime;
    auto storage = std::make_shared<CpuGraphStorage>();
    storage->program = plan.program;
    storage->prepared_arena = std::move(prepared_arena);
    storage->maps.reserve(plan.map_count);
    storage->collectives.reserve(plan.collective_count);
    storage->map_by_step.resize(plan.step_count, NoCpuGraphStorageIndex);
    storage->collective_by_step.resize(plan.step_count, NoCpuGraphStorageIndex);
    storage->scratch.resize(plan.scratch_slots);
    std::size_t map_count = 0u;
    std::size_t collective_count = 0u;
    std::size_t scratch_count = 0u;
    CpuStorageBytes map_bytes{};
    CpuStorageBytes collective_bytes{};
    CpuExecutionStoragePlan execution_plan{};
    for (std::size_t index = 0u; index < plan.step_count; ++index) {
      if (graph_program.maps[index] != nullptr) {
        const auto map_plan = plan_cpu_map_run(*graph_program.maps[index]);
        if (!map_plan) {
          return Result<std::shared_ptr<CpuGraphStorage>>::fail(
              map_plan.reason());
        }
        auto map = make_map_run(*graph_program.maps[index], *map_plan,
                                storage->prepared_arena.get());
        const CpuExecutionStoragePlan map_execution{
            .tiles = map_plan->tiles,
            .map_scratch_count = map_plan->scratch_word_count,
            .simd_count = map_plan->workers,
        };
        if (!map || !add_cpu_storage_bytes(map_bytes, map_plan->bytes) ||
            !merge_cpu_execution_storage_plan(execution_plan, map_execution)) {
          return Result<std::shared_ptr<CpuGraphStorage>>::fail(
              map ? Reason::ProgramCapacity : map.reason());
        }
        storage->maps.emplace_back(std::move(map).value());
        storage->map_by_step[index] = storage->maps.size() - 1u;
        ++map_count;
      }
      if (graph_program.collectives[index] != nullptr) {
        const auto collective_plan =
            plan_cpu_collective_run(*graph_program.collectives[index]);
        if (!collective_plan) {
          return Result<std::shared_ptr<CpuGraphStorage>>::fail(
              collective_plan.reason());
        }
        auto collective = make_collective_run(*graph_program.collectives[index],
                                              *collective_plan,
                                              storage->prepared_arena.get());
        const CpuExecutionStoragePlan collective_execution{
            .tiles = collective_plan->tiles,
            .collective_total_count = collective_plan->tile_count,
            .collective_prefix_count = collective_plan->needs_prefixes
                                           ? collective_plan->tile_count
                                           : 0u,
        };
        if (!collective ||
            !add_cpu_storage_bytes(collective_bytes, collective_plan->bytes) ||
            !merge_cpu_execution_storage_plan(execution_plan,
                                              collective_execution)) {
          return Result<std::shared_ptr<CpuGraphStorage>>::fail(
              collective ? Reason::ProgramCapacity : collective.reason());
        }
        storage->collectives.emplace_back(std::move(collective).value());
        storage->collective_by_step[index] = storage->collectives.size() - 1u;
        ++collective_count;
      }
      const auto *const primitive =
          std::get_if<CpuRuntimePrimitive>(&graph.steps[index]);
      if (primitive == nullptr) {
        continue;
      }
      const auto scratch_plan = plan_cpu_scratch(*primitive);
      if (!scratch_plan) {
        return Result<std::shared_ptr<CpuGraphStorage>>::fail(
            scratch_plan.reason());
      }
      if (scratch_plan->shape == CpuPrimitiveScratchShape::None) {
        continue;
      }
      if (storage->prepared_arena == nullptr) {
        return Result<std::shared_ptr<CpuGraphStorage>>::fail(
            Reason::CpuRuntimeInvalid);
      }
      auto scratch = prepare_cpu_scratch(*primitive, *scratch_plan,
                                         *storage->prepared_arena);
      const Status appended =
          append_cpu_primitive_arena_plan(execution_plan, *scratch_plan);
      if (!scratch || std::holds_alternative<std::monostate>(scratch.value()) ||
          !appended) {
        return Result<std::shared_ptr<CpuGraphStorage>>::fail(
            scratch ? Reason::ProgramCapacity : scratch.reason());
      }
      storage->scratch[index] = std::move(scratch).value();
      ++scratch_count;
    }
    if (map_count != plan.map_count || storage->maps.size() != plan.map_count ||
        collective_count != plan.collective_count ||
        storage->collectives.size() != plan.collective_count ||
        scratch_count != plan.scratch_count || map_bytes != plan.maps ||
        collective_bytes != plan.collectives ||
        execution_plan != plan.execution) {
      return Result<std::shared_ptr<CpuGraphStorage>>::fail(
          Reason::CpuRuntimeInvalid);
    }
    const CpuStorageBytes observed =
        cpu_graph_storage_private_memory(storage.get());
    if (observed.host != plan.private_total.host ||
        observed.tile != plan.private_total.tile) {
      return Result<std::shared_ptr<CpuGraphStorage>>::fail(
          Reason::BufferCapacity);
    }
    return Result<std::shared_ptr<CpuGraphStorage>>::success(
        std::move(storage));
  } catch (...) {
    compute_exception::rethrow_unless_capacity_exception();
    return Result<std::shared_ptr<CpuGraphStorage>>::fail(
        Reason::BufferCapacity);
  }
}

} // namespace rund::compute::detail
