#include "model.hpp"

#include <node/runtime/compute/access.hpp>

#include "../../../../src/compute/cpu/graph.hpp"
#include "../../../../src/compute/cpu/run/state.hpp"
#include "../../../../src/compute/flow/state.hpp"
#include "../../../../src/compute/job/state.hpp"
#include "../../../../src/compute/job/view.hpp"
#include "../../../../src/compute/memory/cpu.hpp"
#include "../../../../src/compute/program/state.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace rund_node_memory_contract {

namespace {

[[nodiscard]] int CheckViewRequirements(
    const std::shared_ptr<rund::compute::detail::ProgramState> &program) {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  const auto first = plan_cpu_view_transfer_requirements(program);
  const auto second = plan_cpu_view_transfer_requirements(program);
  if (!first || !second || *first != *second ||
      first->program != program.get() || !first->inputs.empty() ||
      first->outputs != std::vector<std::uint32_t>{0u}) {
    return 1;
  }
  constexpr std::array<JobBufferView, 1u> strided_input{{
      {.count = 4u, .stride = 2u, .element_bytes = 4u, .alignment = 4u},
  }};
  constexpr std::array<JobBufferView, 1u> strided_output{{
      {.offset = 1u,
       .count = 4u,
       .stride = 3u,
       .element_bytes = 4u,
       .alignment = 4u},
  }};
  const auto first_route =
      plan_cpu_view_transfers(program, strided_input, strided_output, &*first);
  constexpr std::array<JobBufferView, 1u> dense_input{{
      {.count = 4u, .stride = 1u, .element_bytes = 4u, .alignment = 4u},
  }};
  constexpr std::array<JobBufferView, 1u> dense_output{{
      {.count = 4u, .stride = 1u, .element_bytes = 4u, .alignment = 4u},
  }};
  const auto second_route =
      plan_cpu_view_transfers(program, dense_input, dense_output, &*first);
  if (!first_route || !second_route || !first_route->inputs.empty() ||
      first_route->outputs.size() != 1u || first_route->bytes != 16u ||
      !second_route->inputs.empty() || !second_route->outputs.empty() ||
      second_route->bytes != 0u) {
    return 2;
  }
  CpuViewTransferRequirements forged = *first;
  forged.outputs.push_back(0u);
  const auto duplicate =
      plan_cpu_view_transfers(program, strided_input, strided_output, &forged);
  forged = *first;
  forged.outputs.front() = 1u;
  const auto out_of_range =
      plan_cpu_view_transfers(program, strided_input, strided_output, &forged);
  forged = *first;
  ++forged.graph_hash;
  const auto wrong_graph =
      plan_cpu_view_transfers(program, strided_input, strided_output, &forged);
  return !duplicate && !out_of_range && !wrong_graph &&
                 duplicate.reason() == Reason::PipelineInvalid &&
                 out_of_range.reason() == Reason::PipelineInvalid &&
                 wrong_graph.reason() == Reason::PipelineInvalid
             ? 0
             : 3;
}

[[nodiscard]] int CheckStoragePreflight(
    const std::shared_ptr<rund::compute::detail::ProgramState> &program,
    const bool expect_map, const bool expect_collective,
    const bool expect_primitive) {
  using namespace rund::compute::detail;
  node_compute_allocation::Start();
  const auto first_storage = plan_cpu_graph_storage(program);
  const auto second_storage = plan_cpu_graph_storage(program);
  const auto first_route = plan_cpu_run_route(program);
  const auto second_route = plan_cpu_run_route(program);
  node_compute_allocation::Stop();
  if (node_compute_allocation::Count() != 0u || !first_storage ||
      !second_storage || !first_route || !second_route ||
      *first_storage != *second_storage || *first_route != *second_route) {
    return 1;
  }
  const CpuGraphStoragePlan &storage_plan = *first_storage;
  const CpuRunRoutePlan &route_plan = *first_route;
  const CpuStorageBytes execution_payload =
      cpu_execution_storage_payload(storage_plan.execution);
  if (storage_plan.program != program->cpu_graph.get() ||
      storage_plan.runtime != program->cpu_graph->runtime.get() ||
      storage_plan.private_total.host == 0u ||
      (expect_map &&
       (storage_plan.map_count == 0u || storage_plan.maps.host == 0u ||
        storage_plan.maps.tile != 0u)) ||
      (expect_collective && (storage_plan.collective_count == 0u ||
                             storage_plan.collectives.host == 0u ||
                             storage_plan.collectives.tile != 0u)) ||
      (expect_primitive && (storage_plan.scratch_count == 0u ||
                            storage_plan.execution
                                    .primitive_object_payload_bytes == 0u ||
                            execution_payload.host == 0u)) ||
      ((expect_map || expect_collective || expect_primitive) &&
       (!cpu_execution_storage_required(storage_plan.execution) ||
        execution_payload.tile == 0u)) ||
      route_plan.program != storage_plan.program ||
      route_plan.runtime != storage_plan.runtime ||
      route_plan.step_count != storage_plan.step_count ||
      route_plan.map_count != storage_plan.map_count ||
      (expect_map &&
       (route_plan.map_count == 0u || route_plan.read_count == 0u ||
        route_plan.write_count == 0u))) {
    std::fprintf(stderr,
                 "storage preflight mismatch map=%zu/%llu/%llu "
                 "collective=%zu/%llu/%llu primitive=%zu/%zu/%zu "
                 "route=%zu/%zu/%zu\n",
                 storage_plan.map_count,
                 static_cast<unsigned long long>(storage_plan.maps.host),
                 static_cast<unsigned long long>(storage_plan.maps.tile),
                 storage_plan.collective_count,
                 static_cast<unsigned long long>(storage_plan.collectives.host),
                 static_cast<unsigned long long>(storage_plan.collectives.tile),
                 storage_plan.scratch_count,
                 storage_plan.execution.primitive_object_payload_bytes,
                 storage_plan.execution.primitive_object_storage_bytes,
                 route_plan.step_count, route_plan.read_count,
                 route_plan.write_count);
    return 2;
  }
  CpuPreparedArenaPlan arena_plan{};
  arena_plan.execution = storage_plan.execution;
  CpuRunRouteSlice route_slice{};
  if (!append_cpu_run_route_slice(arena_plan, route_plan, route_slice) ||
      !seal_cpu_prepared_arena_plan(arena_plan,
                                    program->device->host_page_bytes)) {
    return 18;
  }
  auto arena = make_cpu_prepared_arena(arena_plan);
  if (!arena) {
    return 19;
  }
  CpuGraphStoragePlan invalid_storage = storage_plan;
  CpuRunRoutePlan invalid_route = route_plan;
  ++invalid_storage.graph_hash;
  ++invalid_route.graph_hash;
  node_compute_allocation::Start();
  const auto rejected_storage =
      make_cpu_graph_storage(program, invalid_storage, *arena);
  CpuRun rejected_run{};
  const auto rejected_route = materialize_cpu_run(
      rejected_run, program, {}, {}, invalid_route, *arena, route_slice);
  node_compute_allocation::Stop();
  if (node_compute_allocation::Count() != 0u || rejected_storage ||
      rejected_route ||
      rejected_storage.reason() != rund::compute::Reason::CpuRuntimeInvalid ||
      rejected_route.reason() != rund::compute::Reason::CpuRuntimeInvalid) {
    return 3;
  }
  auto storage = make_cpu_graph_storage(program, storage_plan, *arena);
  if (!storage || storage.value() == nullptr) {
    return 4;
  }
  if (storage.value()->maps.size() != storage_plan.map_count ||
      storage.value()->collectives.size() != storage_plan.collective_count ||
      storage.value()->map_by_step.size() != storage_plan.step_count ||
      storage.value()->collective_by_step.size() != storage_plan.step_count) {
    return 10;
  }
  std::size_t map_index = 0u;
  std::size_t collective_index = 0u;
  for (std::size_t step = 0u; step < storage_plan.step_count; ++step) {
    const CpuProgram *const map = program->cpu_graph->maps[step].get();
    const CpuMapRun *const run = cpu_map_run(*storage.value(), step);
    if ((map == nullptr) != (run == nullptr)) {
      return 8;
    }
    if (map == nullptr) {
      if (storage.value()->map_by_step[step] != NoCpuGraphStorageIndex) {
        return 11;
      }
    } else {
      if (storage.value()->map_by_step[step] != map_index++) {
        return 12;
      }
      const std::size_t workers = static_cast<std::size_t>(map->workers);
      if ((map->scratch_words != 0u &&
           workers >
               std::numeric_limits<std::size_t>::max() / map->scratch_words) ||
          run->workers != workers || run->scratch_words != map->scratch_words ||
          run->simd.size() != workers ||
          run->scratch.size() != workers * map->scratch_words) {
        return 9;
      }
    }
    const CpuCollective *const collective =
        program->cpu_graph->collectives[step].get();
    const CpuCollectiveRun *const collective_run =
        cpu_collective_run(*storage.value(), step);
    if ((collective == nullptr) != (collective_run == nullptr)) {
      return 13;
    }
    if (collective == nullptr) {
      if (storage.value()->collective_by_step[step] != NoCpuGraphStorageIndex) {
        return 14;
      }
    } else if (storage.value()->collective_by_step[step] !=
               collective_index++) {
      return 15;
    }
  }
  if (map_index != storage_plan.map_count ||
      collective_index != storage_plan.collective_count) {
    return 16;
  }
  const CpuRetainedMemory observed_storage =
      cpu_graph_storage_memory(storage.value().get());
  const CpuStorageBytes arena_payload = cpu_prepared_arena_payload(arena_plan);
  const std::uint64_t expected_storage_host =
      storage_plan.private_total.host + arena_payload.host +
      sizeof(CpuPreparedArena);
  const std::uint64_t expected_storage_tile =
      storage_plan.private_total.tile + arena_payload.tile;
  if (observed_storage.host != expected_storage_host ||
      observed_storage.tile != expected_storage_tile) {
    return 5;
  }
  std::vector<std::shared_ptr<BufferState>> workspace(program->chunks.size());
  CpuRun run{};
  const auto materialized = materialize_cpu_run(
      run, program, workspace, storage.value(), route_plan, *arena,
      route_slice);
  if (!materialized || run.graph == nullptr ||
      run.graph->maps.size() != route_plan.map_count) {
    return 6;
  }
  for (std::size_t step = 0u; step < route_plan.step_count; ++step) {
    const CpuMapRoute *const route = cpu_map_route(*run.graph, step);
    if ((program->cpu_graph->maps[step] == nullptr) != (route == nullptr)) {
      return 17;
    }
  }
  return 0;
}

} // namespace

int CheckCpuGraphStorageFormula() {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 4u> first_input{1u, 2u, 3u, 4u};
  constexpr std::array<std::uint32_t, 4u> second_input{4u, 3u, 2u, 1u};
  constexpr std::uint64_t bytes = first_input.size() * sizeof(std::uint32_t);
  constexpr std::uint64_t internal_values = 1u;
  constexpr std::uint64_t external_values = 2u;
  constexpr std::uint64_t input_values = 1u;
  constexpr std::uint64_t output_values = external_values - input_values;
  constexpr std::uint64_t concurrent_jobs = 2u;
  constexpr std::uint64_t cached_run =
      (input_values + output_values + internal_values) * bytes;
  constexpr std::uint64_t per_job =
      (2u * input_values + output_values + internal_values) * bytes;
  constexpr std::uint64_t total_physical =
      cached_run + concurrent_jobs * per_job;

  auto device = open(Target::cpu(1u));
  if (!device) {
    return 1;
  }
  const std::uint64_t baseline = device->memory().host.current;
  {
    auto program =
        on(*device)
            .input<std::uint32_t>(first_input.size())
            .map("memory-graph-map", [](auto value) { return value + 1u; })
            .scan(Scan::InclusiveSum)
            .compile();
    if (!program) {
      return 2;
    }
    const PhysicalInternal program_internal = PhysicalInternalMemory(*program);
    if (!program_internal.complete || program_internal.count != 0u ||
        program_internal.bytes != 0u ||
        device->memory().host.current != baseline) {
      return 3;
    }
    const std::shared_ptr<detail::ProgramState> &program_state =
        detail::FlowAccess::state(*program);
    if (program_state == nullptr || program_state->cpu_graph == nullptr ||
        program_state->cpu_graph->runtime == nullptr) {
      return 9;
    }
    if (const int view_requirements = CheckViewRequirements(program_state);
        view_requirements != 0) {
      return 40 + view_requirements;
    }
    const detail::CpuRuntimeGraph &runtime_graph =
        *program_state->cpu_graph->runtime;
    const auto *runtime_map =
        runtime_graph.steps.empty()
            ? nullptr
            : std::get_if<detail::CpuRuntimeMap>(&runtime_graph.steps.front());
    if (runtime_graph.values.size() != 3u || runtime_graph.steps.size() != 2u ||
        runtime_map == nullptr || runtime_map->inputs.size() != 1u ||
        runtime_map->outputs.size() != 1u ||
        program_state->cpu_graph->maps.front() == nullptr ||
        program_state->cpu_graph->collectives.back() == nullptr) {
      return 10;
    }
    if (const int preflight =
            CheckStoragePreflight(program_state, true, true, false);
        preflight != 0) {
      return 20 + preflight;
    }
    detail::CpuProgram &map_program = *program_state->cpu_graph->maps.front();
    const std::size_t saved_scratch_words = map_program.scratch_words;
    const std::uint32_t saved_workers = map_program.workers;
    map_program.scratch_words = std::numeric_limits<std::size_t>::max();
    map_program.workers = std::numeric_limits<std::uint32_t>::max();
    node_compute_allocation::Start();
    const auto overflow_first = detail::plan_cpu_graph_storage(program_state);
    const auto overflow_second = detail::plan_cpu_graph_storage(program_state);
    node_compute_allocation::Stop();
    map_program.scratch_words = saved_scratch_words;
    map_program.workers = saved_workers;
    if (node_compute_allocation::Count() != 0u || overflow_first ||
        overflow_second || overflow_first.reason() != Reason::ProgramCapacity ||
        overflow_second.reason() != Reason::ProgramCapacity) {
      return 27;
    }
    auto primitive_program =
        on(*device)
            .map<std::uint32_t>("storage-plan-primitive", first_input.size(),
                                [](auto value) { return value; })
            .sort()
            .compile();
    if (!primitive_program) {
      return 28;
    }
    const auto &primitive_state = detail::FlowAccess::state(*primitive_program);
    if (primitive_state == nullptr) {
      return 29;
    }
    if (const int primitive_preflight =
            CheckStoragePreflight(primitive_state, false, false, true);
        primitive_preflight != 0) {
      return 30 + primitive_preflight;
    }
    const auto vector_bytes = [](const auto &values) {
      return static_cast<std::uint64_t>(values.capacity()) *
             sizeof(typename std::remove_cvref_t<decltype(values)>::value_type);
    };
    std::uint64_t exact_runtime_bytes = sizeof(detail::CpuRuntimeGraph) +
                                        vector_bytes(runtime_graph.values) +
                                        vector_bytes(runtime_graph.steps);
    for (const detail::CpuRuntimeStep &step : runtime_graph.steps) {
      if (const auto *map = std::get_if<detail::CpuRuntimeMap>(&step)) {
        exact_runtime_bytes +=
            vector_bytes(map->inputs) + vector_bytes(map->outputs);
      } else if (const auto *primitive =
                     std::get_if<detail::CpuRuntimePrimitive>(&step)) {
        exact_runtime_bytes += vector_bytes(primitive->inputs);
      }
    }
    if (detail::cpu_runtime_graph_memory(runtime_graph) !=
        exact_runtime_bytes) {
      return 17;
    }
    MemoryStats before_cache{};
    const bool before_cache_allocation_free =
        ReadMemory(*program, before_cache);
    MemoryStats cache_snapshot_stats{};
    const SnapshotAccounting cache_snapshot =
        SnapshotMemory(*program, cache_snapshot_stats);
    if (!before_cache_allocation_free || !cache_snapshot.complete ||
        !cache_snapshot.valid || !cache_snapshot.allocation_free ||
        cache_snapshot.metadata_entries != 1u ||
        cache_snapshot.tile_entries != 1u ||
        cache_snapshot.internal_entries != 0u ||
        !SameStats(before_cache, cache_snapshot_stats) ||
        before_cache.tile.current == 0u) {
      return 11;
    }
    auto first_run = program->run(first_input);
    MemoryStats after_cache{};
    const bool after_cache_allocation_free = ReadMemory(*program, after_cache);
    MemoryStats cache_run_snapshot_stats{};
    const SnapshotAccounting cache_run_snapshot =
        SnapshotMemory(*program, cache_run_snapshot_stats);
    if (!first_run ||
        *first_run != std::vector<std::uint32_t>{2u, 5u, 9u, 14u} ||
        !cache_run_snapshot.complete || !cache_run_snapshot.valid ||
        !cache_run_snapshot.allocation_free || !after_cache_allocation_free ||
        !SameStats(after_cache, cache_run_snapshot_stats) ||
        after_cache.host.current <= before_cache.host.current ||
        after_cache.tile.current <= before_cache.tile.current ||
        cache_run_snapshot.internal_entries != 1u ||
        device->memory().host.current != baseline + cached_run) {
      return 12;
    }
    auto second_run = program->run(second_input);
    MemoryStats second_cache{};
    const bool second_cache_allocation_free =
        ReadMemory(*program, second_cache);
    if (!second_run || !second_cache_allocation_free ||
        *second_run != std::vector<std::uint32_t>{5u, 9u, 12u, 14u} ||
        !SameCounter(after_cache.host, second_cache.host) ||
        after_cache.tile.current != second_cache.tile.current ||
        after_cache.tile.peak != second_cache.tile.peak ||
        after_cache.tile.cumulative != second_cache.tile.cumulative ||
        after_cache.tile.budget != second_cache.tile.budget ||
        after_cache.tile.reused >= second_cache.tile.reused ||
        device->memory().host.current != baseline + cached_run) {
      return 13;
    }
    auto first = program->resident(first_input);
    auto second = program->resident(second_input);
    if (!first || !second) {
      return 4;
    }
    const std::shared_ptr<detail::JobState> first_state =
        detail::JobAccess::state(*first);
    const std::shared_ptr<detail::JobState> second_state =
        detail::JobAccess::state(*second);
    if (first_state == nullptr || second_state == nullptr) {
      return 14;
    }
    MemoryStats first_memory{};
    MemoryStats second_memory{};
    const bool first_memory_allocation_free = ReadMemory(*first, first_memory);
    const bool second_memory_allocation_free =
        ReadMemory(*second, second_memory);
    MemoryStats first_snapshot_stats{};
    MemoryStats second_snapshot_stats{};
    const SnapshotAccounting first_snapshot =
        SnapshotMemory(*first, first_snapshot_stats);
    const SnapshotAccounting second_snapshot =
        SnapshotMemory(*second, second_snapshot_stats);
    if (first_state->cpu == nullptr || second_state->cpu == nullptr ||
        !first_memory_allocation_free || !second_memory_allocation_free ||
        !first_snapshot.complete || !second_snapshot.complete ||
        !first_snapshot.valid || !second_snapshot.valid ||
        !first_snapshot.allocation_free || !second_snapshot.allocation_free ||
        first_snapshot.metadata_entries != 1u ||
        second_snapshot.metadata_entries != 1u ||
        first_snapshot.tile_entries != 1u ||
        second_snapshot.tile_entries != 1u ||
        first_snapshot.internal_entries != 1u ||
        second_snapshot.internal_entries != 1u ||
        !SameStats(first_memory, first_snapshot_stats) ||
        !SameStats(second_memory, second_snapshot_stats) ||
        first_memory.tile.current == 0u || second_memory.tile.current == 0u) {
      return 15;
    }
    const PhysicalInternal first_internal = PhysicalInternalMemory(*first);
    const PhysicalInternal second_internal = PhysicalInternalMemory(*second);
    if (!first_internal.complete || !second_internal.complete ||
        first_internal.count != 1u || second_internal.count != 1u ||
        first_internal.bytes != internal_values * bytes ||
        second_internal.bytes != internal_values * bytes ||
        first_memory.resident.current != per_job ||
        second_memory.resident.current != per_job ||
        device->memory().host.current != baseline + total_physical) {
      return 5;
    }
    if (!first->run() || !second->run()) {
      return 6;
    }
    const MemoryStats first_warm_memory = first->memory();
    const MemoryStats second_warm_memory = second->memory();
    if (first_warm_memory.host.current != first_memory.host.current ||
        second_warm_memory.host.current != second_memory.host.current ||
        first_warm_memory.tile.current != first_memory.tile.current ||
        second_warm_memory.tile.current != second_memory.tile.current) {
      return 16;
    }
    const auto first_output = first->read();
    const auto second_output = second->read();
    const Stats first_stats = first->stats();
    const Stats second_stats = second->stats();
    if (!first_output || !second_output ||
        *first_output != std::vector<std::uint32_t>{2u, 5u, 9u, 14u} ||
        *second_output != std::vector<std::uint32_t>{5u, 9u, 12u, 14u} ||
        first_stats.graph_hash == 0u || second_stats.graph_hash == 0u ||
        first_stats.graph_hash != second_stats.graph_hash ||
        first_stats.output_hash == 0u || second_stats.output_hash == 0u) {
      return 7;
    }
  }
  return device->memory().host.current == baseline ? 0 : 8;
}

} // namespace rund_node_memory_contract
