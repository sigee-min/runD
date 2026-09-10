#include "../graph/state.hpp"
#include "../program/state.hpp"
#include "cpu.hpp"
#include "local.hpp"

#include <kernel/program/compute/retention.hpp>
#include <rund/compute/abi/observe.hpp>

namespace rund::compute::detail {
namespace {

[[nodiscard]] bool
first_expression_owner(const GraphState &graph, const std::size_t step_index,
                       const std::size_t expression_index) noexcept {
  const auto *const current = std::get_if<MapStep>(&graph.steps[step_index]);
  if (current == nullptr || expression_index >= current->expressions.size() ||
      current->expressions[expression_index].state == nullptr) {
    return false;
  }
  const ExprState *const owner =
      current->expressions[expression_index].state.get();
  for (std::size_t prior_step = 0u; prior_step <= step_index; ++prior_step) {
    const auto *const prior = std::get_if<MapStep>(&graph.steps[prior_step]);
    if (prior == nullptr) {
      continue;
    }
    const std::size_t limit =
        prior_step == step_index ? expression_index : prior->expressions.size();
    for (std::size_t prior_expression = 0u; prior_expression < limit;
         ++prior_expression) {
      if (prior->expressions[prior_expression].state.get() == owner) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] std::uint64_t
canonical_graph_memory(const GraphState &graph) noexcept {
  using kernel::compute_retained_detail::CapacityBytes;
  using kernel::compute_retained_detail::StringExternalStorageBytes;
  std::uint64_t bytes = sizeof(GraphState);
  bytes = add_cpu_memory_bytes(bytes, StringExternalStorageBytes(graph.name));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(graph.values));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(graph.inputs));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(graph.bounded_inputs));
  bytes = add_cpu_memory_bytes(
      bytes, CapacityBytes(graph.value_ids.capacity(), sizeof(std::uint32_t)));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(graph.steps));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(graph.outputs));
  bytes = add_cpu_memory_bytes(bytes, vector_memory(graph.identity_outputs));
  for (std::size_t step_index = 0u; step_index < graph.steps.size();
       ++step_index) {
    const auto *const map = std::get_if<MapStep>(&graph.steps[step_index]);
    if (map == nullptr) {
      continue;
    }
    bytes = add_cpu_memory_bytes(bytes, StringExternalStorageBytes(map->name));
    bytes = add_cpu_memory_bytes(bytes, vector_memory(map->expressions));
    bytes = add_cpu_memory_bytes(bytes, vector_memory(map->reads));
    for (std::size_t expression = 0u; expression < map->expressions.size();
         ++expression) {
      if (!first_expression_owner(graph, step_index, expression)) {
        continue;
      }
      const ExprState &state = *map->expressions[expression].state;
      bytes = add_cpu_memory_bytes(bytes, sizeof(ExprState));
      bytes = add_cpu_memory_bytes(bytes, vector_memory(state.nodes));
      bytes = add_cpu_memory_bytes(bytes, vector_memory(state.canonical_slots));
    }
  }
  return bytes;
}

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
  if (state.canonical_graph != nullptr) {
    bytes = add_cpu_memory_bytes(
        bytes, canonical_graph_memory(*state.canonical_graph));
  }

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
  const CpuStorageBytes plans = cpu_program_memory(state.cpu_graph.get());
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
