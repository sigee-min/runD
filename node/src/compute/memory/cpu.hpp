#pragma once

#include "../../accel/kernel/memory.hpp"
#include "../cpu/graph.hpp"
#include "../cpu/state.hpp"
#include "local.hpp"

#include <rund/counter.hpp>

namespace rund::compute::detail {

using CpuRetainedMemory = CpuStorageBytes;

[[nodiscard]] inline std::uint64_t
add_cpu_memory_bytes(const std::uint64_t left,
                     const std::uint64_t right) noexcept {
  return ::rund::detail::counter::SaturatingAdd(left, right);
}

inline void add_cpu_memory(CpuRetainedMemory &total,
                           const CpuRetainedMemory value) noexcept {
  total.host = add_cpu_memory_bytes(total.host, value.host);
  total.tile = add_cpu_memory_bytes(total.tile, value.tile);
}

[[nodiscard]] inline CpuRetainedMemory
cpu_executor_memory(const kernel::ComputeTileExecutor &executor) noexcept {
  const kernel::ComputeTileRetainedMemory retained = executor.retained_memory();
  return CpuRetainedMemory{
      .host = add_cpu_memory_bytes(retained.state_bytes,
                                   retained.async_context_bytes),
      .tile = add_cpu_memory_bytes(
          add_cpu_memory_bytes(retained.workspace_bytes,
                               retained.failure_slot_bytes),
          retained.worker_tile_bytes),
  };
}

[[nodiscard]] inline std::uint64_t
cpu_runtime_graph_memory(const CpuRuntimeGraph &graph) noexcept {
  std::uint64_t bytes = sizeof(CpuRuntimeGraph);
  bytes = add_cpu_memory_bytes(bytes, vector_memory(graph.values));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(graph.steps));
  for (const CpuRuntimeStep &step : graph.steps) {
    if (const auto *map = std::get_if<CpuRuntimeMap>(&step)) {
      bytes = add_cpu_memory_bytes(bytes, vector_memory(map->inputs));
      bytes = add_cpu_memory_bytes(bytes, vector_memory(map->outputs));
    } else if (const auto *primitive =
                   std::get_if<CpuRuntimePrimitive>(&step)) {
      bytes = add_cpu_memory_bytes(bytes, vector_memory(primitive->inputs));
    }
  }
  return bytes;
}

[[nodiscard]] inline CpuRetainedMemory
cpu_program_memory(const CpuGraphProgram *const program) noexcept {
  if (program == nullptr) {
    return {};
  }
  CpuRetainedMemory memory{.host = sizeof(CpuGraphProgram)};
  memory.host = add_cpu_memory_bytes(memory.host, vector_memory(program->maps));
  memory.host =
      add_cpu_memory_bytes(memory.host, vector_memory(program->collectives));
  memory.host =
      add_cpu_memory_bytes(memory.host, vector_memory(program->bind_begin));
  memory.host =
      add_cpu_memory_bytes(memory.host, vector_memory(program->bind_count));
  memory.host =
      add_cpu_memory_bytes(memory.host, vector_memory(program->resets));
  if (program->runtime != nullptr) {
    memory.host = add_cpu_memory_bytes(
        memory.host, cpu_runtime_graph_memory(*program->runtime));
  }
  for (const auto &map : program->maps) {
    if (map != nullptr) {
      memory.host = add_cpu_memory_bytes(memory.host, sizeof(CpuProgram));
      memory.host = add_cpu_memory_bytes(
          memory.host, map->dispatch.prepared.retained_dynamic_memory_bytes());
      memory.host =
          add_cpu_memory_bytes(memory.host, vector_memory(map->input_bytes));
      memory.host =
          add_cpu_memory_bytes(memory.host, vector_memory(map->input_counts));
      memory.host =
          add_cpu_memory_bytes(memory.host, vector_memory(map->read_routes));
      add_cpu_memory(memory, cpu_executor_memory(map->tile_plan));
    }
  }
  for (const auto &collective : program->collectives) {
    if (collective != nullptr) {
      memory.host = add_cpu_memory_bytes(memory.host, sizeof(CpuCollective));
      add_cpu_memory(memory, cpu_executor_memory(collective->tile_plan));
    }
  }
  return memory;
}

[[nodiscard]] inline CpuRetainedMemory cpu_graph_storage_private_memory(
    const CpuGraphStorage *const storage) noexcept {
  if (storage == nullptr) {
    return {};
  }
  CpuRetainedMemory memory{.host = sizeof(CpuGraphStorage)};
  memory.host =
      add_cpu_memory_bytes(memory.host, vector_memory(storage->map_by_step));
  memory.host = add_cpu_memory_bytes(
      memory.host, vector_memory(storage->collective_by_step));
  memory.host = add_cpu_memory_bytes(memory.host, vector_memory(storage->maps));
  memory.host =
      add_cpu_memory_bytes(memory.host, vector_memory(storage->collectives));
  memory.host =
      add_cpu_memory_bytes(memory.host, vector_memory(storage->scratch));
  return memory;
}

[[nodiscard]] inline CpuRetainedMemory
cpu_prepared_arena_memory(const CpuPreparedArena *const arena) noexcept {
  return arena == nullptr
             ? CpuRetainedMemory{}
             : CpuRetainedMemory{.host = arena->payload_host_bytes(),
                                 .tile = arena->payload_tile_bytes()};
}

[[nodiscard]] inline CpuRetainedMemory
cpu_graph_storage_memory(const CpuGraphStorage *const storage) noexcept {
  CpuRetainedMemory memory = cpu_graph_storage_private_memory(storage);
  if (storage != nullptr) {
    add_cpu_memory(memory,
                   cpu_prepared_arena_memory(storage->prepared_arena.get()));
    if (storage->prepared_arena != nullptr) {
      memory.host = add_cpu_memory_bytes(memory.host, sizeof(CpuPreparedArena));
    }
  }
  return memory;
}

[[nodiscard]] inline CpuRetainedMemory
cpu_run_memory(const CpuRun *const run) noexcept {
  CpuRetainedMemory memory{};
  if (run != nullptr && run->graph != nullptr) {
    add_cpu_memory(memory, cpu_graph_storage_memory(run->graph->storage.get()));
  }
  return memory;
}

} // namespace rund::compute::detail
