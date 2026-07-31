#include "../program/state.hpp"
#include "cpu.hpp"
#include "local.hpp"

#include <kernel/program/compute/retention.hpp>
#include <rund/compute/abi/observe.hpp>

namespace rund::compute::detail {
namespace {

[[nodiscard]] std::uint64_t host_memory(const ProgramState &state) noexcept {
  std::uint64_t bytes = sizeof(ProgramState);
  bytes = add_cpu_memory_bytes(
      bytes,
      kernel::compute_retained_detail::StringExternalStorageBytes(state.name));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(state.input_types));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(state.input_sizes));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(state.input_formats));
  bytes = add_cpu_memory_bytes(bytes,
                               vector_memory(state.bounded_input_capacities));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(state.output_types));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(state.output_sizes));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(state.output_formats));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(state.output_aliases));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(state.chunks));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(state.chunk_order));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(state.graph_value_routes));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(state.graph_bindings));

  bytes =
      add_cpu_memory_bytes(bytes, vector_memory(state.graph_info.resources));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(state.graph_info.nodes));
  for (const graph::Node &node : state.graph_info.nodes) {
    bytes = add_cpu_memory_bytes(bytes, vector_memory(node.accesses));
    bytes = add_cpu_memory_bytes(bytes, vector_memory(node.dependencies));
  }
  bytes = add_cpu_memory_bytes(bytes, vector_memory(state.graph_info.barriers));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(state.graph_info.inputs));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(state.graph_info.outputs));

  if (state.accel != nullptr) {
    bytes = add_cpu_memory_bytes(bytes, sizeof(AccelProgram));
    bytes = add_cpu_memory_bytes(bytes, state.accel->kernel_token_host_bytes);
  }
  return bytes;
}

struct ProgramMemoryView final {
  std::uint64_t host{};
  std::uint64_t tile{};
};

[[nodiscard]] ProgramMemoryView
memory_view(const ProgramState &state) noexcept {
  const CpuRetainedMemory plans = cpu_program_memory(state.cpu_graph.get());
  return ProgramMemoryView{
      .host = add_cpu_memory_bytes(host_memory(state), plans.host),
      .tile = plans.tile,
  };
}

[[nodiscard]] MemoryStats summary(const ProgramState &state,
                                  const ProgramMemoryView view) noexcept {
  MemoryStats stats{.backend = memory_backend(*state.device),
                    .scope = MemoryScope::Program,
                    .host = fixed_memory(view.host),
                    .tile = fixed_memory(view.tile)};
  return stats;
}

[[nodiscard]] MemoryStats
summary(const ProgramState &state, const ProgramMemoryView view,
        const std::shared_ptr<JobState> &cached_job) noexcept {
  MemoryStats stats = summary(state, view);
  if (cached_job != nullptr) {
    merge_memory(stats, job_memory(cached_job));
    stats.scope = MemoryScope::Program;
  }
  return stats;
}

} // namespace

MemoryStats
program_memory(const std::shared_ptr<ProgramState> &state) noexcept {
  if (state == nullptr || state->device == nullptr) {
    return {};
  }
  std::lock_guard lock{state->cache.gate};
  const ProgramMemoryView view = memory_view(*state);
  return summary(*state, view, state->cache.job);
}

MemorySnapshot
program_memory_snapshot(const std::shared_ptr<ProgramState> &state,
                        const std::span<MemoryEntry> entries) noexcept {
  if (state == nullptr || state->device == nullptr) {
    SnapshotWriter writer{MemoryStats{}, entries};
    return writer.finish();
  }
  std::lock_guard lock{state->cache.gate};
  const ProgramMemoryView view = memory_view(*state);
  MemorySnapshot nested{};
  if (state->cache.job != nullptr) {
    const std::span<MemoryEntry> nested_entries =
        entries.size() > 2u ? entries.subspan(2u) : std::span<MemoryEntry>{};
    nested = job_memory_snapshot(state->cache.job, nested_entries);
  }

  std::size_t written = 0u;
  const auto add = [&](const MemoryEntry entry) noexcept {
    if (written < entries.size()) {
      entries[written++] = entry;
    }
  };
  add(MemoryEntry{.category = MemoryCategory::Host,
                  .use = MemoryUse::Metadata,
                  .index = 0u,
                  .bytes = fixed_memory(view.host)});
  add(MemoryEntry{.category = MemoryCategory::Tile,
                  .use = MemoryUse::Scratch,
                  .index = 0u,
                  .bytes = fixed_memory(view.tile)});
  MemoryStats total = summary(*state, view);
  if (state->cache.job != nullptr) {
    merge_memory(total, nested.summary);
    total.scope = MemoryScope::Program;
  }
  return MemorySnapshot{
      .summary = total,
      .written = written + nested.written,
      .total = 2u + nested.total,
  };
}

} // namespace rund::compute::detail
