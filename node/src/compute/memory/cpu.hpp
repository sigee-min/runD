#pragma once

#include "../../accel/kernel/memory.hpp"
#include "../cpu/graph.hpp"
#include "../cpu/state.hpp"
#include "local.hpp"

#include <rund/counter.hpp>

#include <limits>
#include <type_traits>

namespace rund::compute::detail {

struct CpuRetainedMemory final {
  std::uint64_t host{};
  std::uint64_t tile{};
};

template <class T>
[[nodiscard]] inline std::uint64_t
vector_memory(const CpuOverwriteBuffer<T> &values) noexcept {
  constexpr std::uint64_t width = sizeof(T);
  return values.capacity() > std::numeric_limits<std::uint64_t>::max() / width
             ? std::numeric_limits<std::uint64_t>::max()
             : static_cast<std::uint64_t>(values.capacity()) * width;
}

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
cpu_map_scratch_memory(const CpuMapRun &run) noexcept {
  std::uint64_t bytes =
      add_cpu_memory_bytes(vector_memory(run.scratch), vector_memory(run.simd));
  for (const auto &scratch : run.scratch) {
    bytes = add_cpu_memory_bytes(bytes, vector_memory(scratch));
  }
  return bytes;
}

[[nodiscard]] inline std::uint64_t
cpu_collective_scratch_memory(const CpuCollectiveRun &run) noexcept {
  return add_cpu_memory_bytes(vector_memory(run.totals),
                              vector_memory(run.prefixes));
}

template <class LaneBuffer, std::size_t Count>
[[nodiscard]] inline std::uint64_t
cpu_lane_array_memory(const std::array<LaneBuffer, Count> &values) noexcept {
  std::uint64_t bytes{};
  for (const auto &value : values) {
    bytes = add_cpu_memory_bytes(bytes, vector_memory(value));
  }
  return bytes;
}

template <class Key>
[[nodiscard]] inline std::uint64_t
cpu_scratch_tile_memory(const CpuSortPrimitiveScratch<Key> &scratch) noexcept {
  return add_cpu_memory_bytes(vector_memory(scratch.keys),
                              vector_memory(scratch.values));
}

[[nodiscard]] inline std::uint64_t
cpu_scratch_tile_memory(const CpuScatterPrimitiveScratch &scratch) noexcept {
  return add_cpu_memory_bytes(vector_memory(scratch.values.keys),
                              vector_memory(scratch.values.marks));
}

[[nodiscard]] inline std::uint64_t cpu_scratch_tile_memory(
    const CpuScatterReducePrimitiveScratch &scratch) noexcept {
  return vector_memory(scratch.sorted_indices);
}

template <class Lane>
[[nodiscard]] inline std::uint64_t
cpu_scratch_tile_memory(const CpuTransformScratch<Lane> &scratch) noexcept {
  return vector_memory(scratch.twiddle);
}

template <class Lane>
[[nodiscard]] inline std::uint64_t
cpu_scratch_tile_memory(const CpuFactorQrScratch<Lane> &scratch) noexcept {
  return cpu_lane_array_memory(scratch.values);
}

template <class Lane>
[[nodiscard]] inline std::uint64_t
cpu_scratch_tile_memory(const CpuSolveLuScratch<Lane> &scratch) noexcept {
  return add_cpu_memory_bytes(vector_memory(scratch.factor),
                              vector_memory(scratch.pivots));
}

template <class Lane>
[[nodiscard]] inline std::uint64_t
cpu_scratch_tile_memory(const CpuSolveCholeskyScratch<Lane> &scratch) noexcept {
  return vector_memory(scratch.factor);
}

template <class Lane>
[[nodiscard]] inline std::uint64_t
cpu_scratch_tile_memory(const CpuSolveQrMatrixScratch<Lane> &scratch) noexcept {
  return add_cpu_memory_bytes(vector_memory(scratch.y),
                              cpu_lane_array_memory(scratch.qr));
}

template <class Lane>
[[nodiscard]] inline std::uint64_t
cpu_scratch_tile_memory(const CpuSolveQrFactorScratch<Lane> &scratch) noexcept {
  return vector_memory(scratch.y);
}

template <class Lane>
[[nodiscard]] inline std::uint64_t
cpu_scratch_tile_memory(const CpuSpectrumEigenScratch<Lane> &scratch) noexcept {
  return cpu_lane_array_memory(scratch.values);
}

template <class Lane>
[[nodiscard]] inline std::uint64_t cpu_scratch_tile_memory(
    const CpuSpectrumSvdValuesScratch<Lane> &scratch) noexcept {
  return add_cpu_memory_bytes(cpu_lane_array_memory(scratch.values),
                              vector_memory(scratch.order));
}

template <class Lane>
[[nodiscard]] inline std::uint64_t cpu_scratch_tile_memory(
    const CpuSpectrumSvdVectorsScratch<Lane> &scratch) noexcept {
  return add_cpu_memory_bytes(cpu_lane_array_memory(scratch.values),
                              vector_memory(scratch.order));
}

[[nodiscard]] inline CpuRetainedMemory
cpu_primitive_scratch_memory(const CpuPrimitiveScratch &scratch) noexcept {
  return std::visit(
      [](const auto &owner) noexcept -> CpuRetainedMemory {
        using Owner = std::remove_cvref_t<decltype(owner)>;
        if constexpr (std::is_same_v<Owner, std::monostate>) {
          return {};
        } else {
          using Scratch = typename Owner::element_type;
          if (owner == nullptr) {
            return {};
          }
          return CpuRetainedMemory{
              .host = sizeof(Scratch),
              .tile = cpu_scratch_tile_memory(*owner),
          };
        }
      },
      scratch);
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

[[nodiscard]] inline CpuRetainedMemory
cpu_run_memory(const CpuRun *const run) noexcept {
  if (run == nullptr) {
    return {};
  }
  CpuRetainedMemory memory{.host = sizeof(CpuRun)};
  if (run->graph == nullptr) {
    return memory;
  }
  memory.host = add_cpu_memory_bytes(memory.host, sizeof(CpuGraphRun));
  memory.host =
      add_cpu_memory_bytes(memory.host, vector_memory(run->graph->maps));
  memory.host =
      add_cpu_memory_bytes(memory.host, vector_memory(run->graph->collectives));
  memory.host =
      add_cpu_memory_bytes(memory.host, vector_memory(run->graph->scratch));
  memory.host = add_cpu_memory_bytes(memory.host,
                                     vector_memory(run->graph->owned_buffers));
  for (const auto &map : run->graph->maps) {
    if (map != nullptr) {
      memory.host = add_cpu_memory_bytes(memory.host, sizeof(CpuMapRun));
      memory.tile =
          add_cpu_memory_bytes(memory.tile, cpu_map_scratch_memory(*map));
      add_cpu_memory(memory, cpu_executor_memory(map->tiles));
    }
  }
  for (const auto &collective : run->graph->collectives) {
    if (collective != nullptr) {
      memory.host = add_cpu_memory_bytes(memory.host, sizeof(CpuCollectiveRun));
      memory.tile = add_cpu_memory_bytes(
          memory.tile, cpu_collective_scratch_memory(*collective));
      add_cpu_memory(memory, cpu_executor_memory(collective->tiles));
    }
  }
  for (const auto &scratch : run->graph->scratch) {
    add_cpu_memory(memory, cpu_primitive_scratch_memory(scratch));
  }
  return memory;
}

} // namespace rund::compute::detail
