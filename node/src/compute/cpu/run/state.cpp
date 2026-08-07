#include "state.hpp"

#include "../../buffer/local.hpp"
#include "../../device/state.hpp"
#include "../../exception.hpp"
#include "../../memory/cpu.hpp"
#include "../graph.hpp"
#include "../map.hpp"
#include "../scratch.hpp"

#include <kernel/core/checked.hpp>

#include <array>
#include <limits>
#include <utility>

namespace rund::compute::detail {

CpuGraphProgram::~CpuGraphProgram() = default;

namespace {

[[nodiscard]] bool to_u64(const std::size_t value,
                          std::uint64_t &result) noexcept {
  if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
    if (value > std::numeric_limits<std::uint64_t>::max()) {
      return false;
    }
  }
  result = static_cast<std::uint64_t>(value);
  return true;
}

[[nodiscard]] bool extent_bytes(const std::size_t count,
                                const std::uint64_t width,
                                std::uint64_t &result) noexcept {
  std::uint64_t count64 = 0u;
  return to_u64(count, count64) && kernel::checked::mul(count64, width, result);
}

[[nodiscard]] bool add_count(std::size_t &total,
                             const std::size_t value) noexcept {
  if (value > std::numeric_limits<std::size_t>::max() - total) {
    return false;
  }
  total += value;
  return true;
}

[[nodiscard]] bool add_bytes(CpuStorageBytes &total,
                             const CpuStorageBytes value) noexcept {
  return kernel::checked::add(total.host, value.host, total.host) &&
         kernel::checked::add(total.tile, value.tile, total.tile);
}

[[nodiscard]] bool add_host(CpuStorageBytes &total,
                            const std::uint64_t bytes) noexcept {
  return kernel::checked::add(total.host, bytes, total.host);
}

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

struct CpuMapRunPlan final {
  const CpuProgram *program = nullptr;
  std::size_t workers{};
  std::size_t scratch_words{};
  std::size_t scratch_word_count{};
  kernel::ComputeTileRunStoragePlan tiles{};
  CpuStorageBytes bytes{};
};

[[nodiscard]] Result<CpuMapRunPlan>
plan_map_run(const CpuProgram &program) noexcept {
  const kernel::ComputeTileRunStoragePlan tiles =
      program.tile_plan.planned_run_storage();
  if (!tiles) {
    return Result<CpuMapRunPlan>::fail(Reason::TileRunCapacity);
  }
  CpuMapRunPlan plan{
      .program = &program,
      .workers = static_cast<std::size_t>(program.workers),
      .scratch_words = program.scratch_words,
      .tiles = tiles,
      .bytes = CpuStorageBytes{.host = sizeof(CpuMapRun)},
  };
  if (plan.scratch_words != 0u &&
      plan.workers >
          std::numeric_limits<std::size_t>::max() / plan.scratch_words) {
    return Result<CpuMapRunPlan>::fail(Reason::ProgramCapacity);
  }
  plan.scratch_word_count = plan.workers * plan.scratch_words;
  return Result<CpuMapRunPlan>::success(plan);
}

struct CpuCollectiveRunPlan final {
  const CpuCollective *program = nullptr;
  std::size_t tile_count{};
  bool needs_prefixes{};
  kernel::ComputeTileRunStoragePlan tiles{};
  CpuStorageBytes bytes{};
};

[[nodiscard]] Result<CpuCollectiveRunPlan>
plan_collective_run(const CpuCollective &program) noexcept {
  const kernel::ComputeTileRunStoragePlan tiles =
      program.tile_plan.planned_run_storage();
  if (!tiles) {
    return Result<CpuCollectiveRunPlan>::fail(Reason::TileRunCapacity);
  }
  CpuCollectiveRunPlan plan{
      .program = &program,
      .tile_count = static_cast<std::size_t>(program.tile_count),
      .needs_prefixes = program.needs_prefixes,
      .tiles = tiles,
      .bytes = CpuStorageBytes{.host = sizeof(CpuCollectiveRun)},
  };
  return Result<CpuCollectiveRunPlan>::success(plan);
}

[[nodiscard]] Result<CpuStorageBytes>
plan_graph_containers(const std::size_t steps,
                      const std::size_t scratch_slots) noexcept {
  CpuStorageBytes bytes{.host = sizeof(CpuGraphStorage)};
  std::uint64_t maps = 0u;
  std::uint64_t collectives = 0u;
  std::uint64_t scratch = 0u;
  if (!extent_bytes(steps, sizeof(std::size_t), maps) ||
      !extent_bytes(steps, sizeof(std::size_t), collectives) ||
      !extent_bytes(scratch_slots, sizeof(CpuPrimitiveScratch), scratch) ||
      !add_host(bytes, maps) || !add_host(bytes, collectives) ||
      !add_host(bytes, scratch)) {
    return Result<CpuStorageBytes>::fail(Reason::ProgramCapacity);
  }
  return Result<CpuStorageBytes>::success(bytes);
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
      plan_graph_containers(plan.step_count, plan.scratch_slots);
  if (!containers || *containers != plan.containers) {
    return false;
  }
  CpuStorageBytes private_total = plan.containers;
  return add_bytes(private_total, plan.maps) &&
         add_bytes(private_total, plan.collectives) &&
         private_total == plan.private_total &&
         ((plan.map_count == 0u && plan.collective_count == 0u) ||
          plan.execution.tiles.ok);
}

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
    if (map != nullptr &&
        (!add_count(map_count, 1u) ||
         !add_count(read_count, map->inputs.size()) ||
         !add_count(write_count, map->outputs.size()))) {
      return false;
    }
  }
  if (plan.map_count != map_count || plan.read_count != read_count ||
      plan.write_count != write_count) {
    return false;
  }
  return true;
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

[[nodiscard]] Status freeze_cpu_map_bindings(JobState &job,
                                             CpuGraphRun &run) noexcept {
  if (job.program == nullptr || job.program->device == nullptr ||
      job.program->cpu_graph == nullptr ||
      job.program->cpu_graph->runtime == nullptr || run.storage == nullptr ||
      run.storage->program != job.program->cpu_graph.get()) {
    return Status::fail(Reason::CpuRuntimeInvalid);
  }
  CpuGraphProgram &program = *job.program->cpu_graph;
  const CpuRuntimeGraph &runtime = *program.runtime;
  if (program.maps.size() != runtime.steps.size() ||
      run.storage->map_by_step.size() != runtime.steps.size() ||
      run.maps.size() != run.storage->maps.size()) {
    return Status::fail(Reason::CpuRuntimeInvalid);
  }
  const auto buffer = [&](const std::uint32_t value) noexcept {
    return graph_value_buffer_id(*job.program, value, job.inputs, job.outputs,
                                 run.buffers);
  };
  const auto view = [&](const std::uint32_t value,
                        BufferState *const owner) noexcept {
    return owner == nullptr ? JobBufferView{}
                            : job_value_view(job, value, *owner);
  };
  std::size_t read_begin = 0u;
  std::size_t write_begin = 0u;
  for (std::size_t step = 0u; step < runtime.steps.size(); ++step) {
    const auto *const map = std::get_if<CpuRuntimeMap>(&runtime.steps[step]);
    if (map == nullptr) {
      continue;
    }
    CpuProgram *const map_program = program.maps[step].get();
    CpuMapRun *const map_run = cpu_map_run(*run.storage, step);
    CpuMapRoute *const map_route = cpu_map_route(run, step);
    if (map_program == nullptr || map_run == nullptr || map_route == nullptr ||
        map->inputs.size() > kernel::kMaxComputeBindingCount ||
        map->outputs.empty() || map->outputs.size() > MaxOutputs ||
        read_begin > run.reads.size() ||
        map->inputs.size() > run.reads.size() - read_begin ||
        write_begin > run.writes.size() ||
        map->outputs.size() > run.writes.size() - write_begin) {
      return Status::fail(Reason::GraphBindingInvalid);
    }
    std::array<BufferState *, kernel::kMaxComputeBindingCount> inputs{};
    std::array<BufferState *, MaxOutputs> outputs{};
    std::array<JobBufferView, kernel::kMaxComputeBindingCount> input_views{};
    std::array<JobBufferView, MaxOutputs> output_views{};
    for (std::size_t index = 0u; index < map->inputs.size(); ++index) {
      inputs[index] = buffer(map->inputs[index]);
      input_views[index] = view(map->inputs[index], inputs[index]);
    }
    for (std::size_t index = 0u; index < map->outputs.size(); ++index) {
      outputs[index] = buffer(map->outputs[index]);
      output_views[index] = view(map->outputs[index], outputs[index]);
    }
    const Status prepared = prepare_cpu_map_bindings(
        *map_program, job.program->device, *map_run, *map_route,
        run.reads.subspan(read_begin, map->inputs.size()),
        run.writes.subspan(write_begin, map->outputs.size()),
        std::span<BufferState *const>{inputs.data(), map->inputs.size()},
        std::span<BufferState *const>{outputs.data(), map->outputs.size()},
        std::span<const JobBufferView>{input_views.data(), map->inputs.size()},
        std::span<const JobBufferView>{output_views.data(),
                                       map->outputs.size()});
    if (!prepared) {
      return prepared;
    }
    map_route->bindings_frozen = true;
    read_begin += map->inputs.size();
    write_begin += map->outputs.size();
  }
  if (read_begin != run.reads.size() || write_begin != run.writes.size()) {
    return Status::fail(Reason::CpuRuntimeInvalid);
  }
  run.bound_inputs = job.inputs.data();
  return Status::success();
}

} // namespace

Status refresh_cpu_map_bindings(JobState &job) noexcept {
  return job.cpu == nullptr || job.cpu->graph == nullptr
             ? Status::success()
             : freeze_cpu_map_bindings(job, *job.cpu->graph);
}

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
      const auto map = plan_map_run(*graph_program.maps[index]);
      if (!map) {
        return Result<CpuGraphStoragePlan>::fail(map.reason());
      }
      ++plan.map_count;
      const CpuExecutionStoragePlan execution{
          .tiles = map->tiles,
          .map_scratch_count = map->scratch_word_count,
          .simd_count = map->workers,
      };
      if (!add_bytes(plan.maps, map->bytes) ||
          !merge_cpu_execution_storage_plan(plan.execution, execution)) {
        return Result<CpuGraphStoragePlan>::fail(Reason::ProgramCapacity);
      }
    }
    if (graph_program.collectives[index] != nullptr) {
      const auto collective =
          plan_collective_run(*graph_program.collectives[index]);
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
      if (!add_bytes(plan.collectives, collective->bytes) ||
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
      plan_graph_containers(plan.step_count, plan.scratch_slots);
  if (!containers) {
    return Result<CpuGraphStoragePlan>::fail(containers.reason());
  }
  plan.containers = *containers;
  plan.private_total = plan.containers;
  if (!add_bytes(plan.private_total, plan.maps) ||
      !add_bytes(plan.private_total, plan.collectives)) {
    return Result<CpuGraphStoragePlan>::fail(Reason::ProgramCapacity);
  }
  return Result<CpuGraphStoragePlan>::success(plan);
}

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
  if (prepared_arena == nullptr ||
      !prepared_arena->supports(plan.execution)) {
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
        const auto map_plan = plan_map_run(*graph_program.maps[index]);
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
        if (!map || !add_bytes(map_bytes, map_plan->bytes) ||
            !merge_cpu_execution_storage_plan(execution_plan,
                                            map_execution)) {
          return Result<std::shared_ptr<CpuGraphStorage>>::fail(
              map ? Reason::ProgramCapacity : map.reason());
        }
        storage->maps.emplace_back(std::move(map).value());
        storage->map_by_step[index] = storage->maps.size() - 1u;
        ++map_count;
      }
      if (graph_program.collectives[index] != nullptr) {
        const auto collective_plan =
            plan_collective_run(*graph_program.collectives[index]);
        if (!collective_plan) {
          return Result<std::shared_ptr<CpuGraphStorage>>::fail(
              collective_plan.reason());
        }
        auto collective = make_collective_run(
            *graph_program.collectives[index], *collective_plan,
            storage->prepared_arena.get());
        const CpuExecutionStoragePlan collective_execution{
            .tiles = collective_plan->tiles,
            .collective_total_count = collective_plan->tile_count,
            .collective_prefix_count = collective_plan->needs_prefixes
                                           ? collective_plan->tile_count
                                           : 0u,
        };
        if (!collective ||
            !add_bytes(collective_bytes, collective_plan->bytes) ||
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
        collective_bytes != plan.collectives || execution_plan != plan.execution) {
      return Result<std::shared_ptr<CpuGraphStorage>>::fail(
          Reason::CpuRuntimeInvalid);
    }
    const CpuRetainedMemory observed =
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

bool append_cpu_job_binding_slice(CpuPreparedArenaPlan &arena,
                                  const CpuJobBindingCounts &counts,
                                  CpuJobBindingSlice &slice) noexcept {
  if (arena.layout.sealed) {
    return false;
  }
  std::size_t owners = arena.buffer_owner_count;
  std::size_t views = arena.buffer_view_count;
  std::size_t kernel_views = arena.kernel_view_count;
  std::size_t view_transfers = arena.view_transfer_count;
  CpuJobBindingSlice next{
      .input_begin = owners,
      .input_count = counts.inputs,
  };
  if (!add_count(owners, counts.inputs)) {
    return false;
  }
  next.output_begin = owners;
  next.output_count = counts.outputs;
  if (!add_count(owners, counts.outputs)) {
    return false;
  }
  next.input_view_begin = views;
  next.input_view_count = counts.inputs;
  if (!add_count(views, counts.inputs)) {
    return false;
  }
  next.output_view_begin = views;
  next.output_view_count = counts.outputs;
  if (!add_count(views, counts.outputs)) {
    return false;
  }
  next.kernel_view_begin = kernel_views;
  next.kernel_view_count = counts.kernel_views;
  if (!add_count(kernel_views, counts.kernel_views)) {
    return false;
  }
  next.input_transfer_begin = view_transfers;
  next.input_transfer_count = counts.input_transfers;
  if (!add_count(view_transfers, counts.input_transfers)) {
    return false;
  }
  next.output_transfer_begin = view_transfers;
  next.output_transfer_count = counts.output_transfers;
  if (!add_count(view_transfers, counts.output_transfers)) {
    return false;
  }
  arena.buffer_owner_count = owners;
  arena.buffer_view_count = views;
  arena.kernel_view_count = kernel_views;
  arena.view_transfer_count = view_transfers;
  slice = next;
  return true;
}

bool append_cpu_workspace_slice(CpuPreparedArenaPlan &arena,
                                const std::size_t buffer_count,
                                CpuWorkspaceSlice &slice) noexcept {
  if (arena.layout.sealed) {
    return false;
  }
  const CpuWorkspaceSlice next{
      .workspace_begin = arena.workspace_count,
      .workspace_count = 1u,
      .buffer_begin = arena.buffer_owner_count,
      .buffer_count = buffer_count,
      .offset_begin = arena.workspace_offset_count,
      .offset_count = buffer_count,
  };
  std::size_t workspaces = arena.workspace_count;
  std::size_t buffers = arena.buffer_owner_count;
  std::size_t offsets = arena.workspace_offset_count;
  if (!add_count(workspaces, 1u) || !add_count(buffers, buffer_count) ||
      !add_count(offsets, buffer_count)) {
    return false;
  }
  arena.workspace_count = workspaces;
  arena.buffer_owner_count = buffers;
  arena.workspace_offset_count = offsets;
  slice = next;
  return true;
}

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
      prepared_arena == nullptr ||
      storage->prepared_arena != prepared_arena ||
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

Status prepare_cpu_run(JobState &job) {
  if (job.program != nullptr && job.program->empty()) {
    return Status::success();
  }
  if (job.program == nullptr || job.program->cpu_graph == nullptr) {
    job.cpu.reset();
    return Status::success();
  }
  CpuRun &run = job.cpu.emplace();
  const Status materialized =
      materialize_cpu_run(run, job.program, job_graph_buffers(job));
  if (!materialized) {
    job.cpu.reset();
    return materialized;
  }
  return refresh_cpu_map_bindings(job);
}

Status prepare_cpu_run(JobState &job, std::shared_ptr<CpuGraphStorage> storage,
                       const CpuRunRoutePlan &plan,
                       std::shared_ptr<CpuPreparedArena> prepared_arena,
                       const CpuRunRouteSlice &route_slice) {
  if (job.program != nullptr && job.program->empty()) {
    return Status::success();
  }
  if (plan.program == nullptr) {
    job.cpu.reset();
    return Status::success();
  }
  CpuRun &run = job.cpu.emplace();
  const Status materialized = materialize_cpu_run(
      run, job.program, job_graph_buffers(job), std::move(storage), plan,
      std::move(prepared_arena), route_slice);
  if (!materialized) {
    job.cpu.reset();
    return materialized;
  }
  return refresh_cpu_map_bindings(job);
}

} // namespace rund::compute::detail
