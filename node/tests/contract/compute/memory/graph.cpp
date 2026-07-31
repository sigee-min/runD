#include "model.hpp"

#include <node/runtime/compute/access.hpp>

#include "../../../../src/compute/cpu/graph.hpp"
#include "../../../../src/compute/flow/state.hpp"
#include "../../../../src/compute/job/state.hpp"
#include "../../../../src/compute/memory/cpu.hpp"
#include "../../../../src/compute/program/state.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace rund_node_memory_contract {

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
