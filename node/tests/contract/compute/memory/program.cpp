#include "model.hpp"

#include "../../../../src/compute/cpu/graph.hpp"
#include "../../../../src/compute/flow/state.hpp"
#include "../../../../src/compute/graph/state.hpp"
#include "../../../../src/compute/memory/cpu.hpp"
#include "../../../../src/compute/program/state.hpp"

#include <array>
#include <cstdint>

namespace rund_node_memory_contract {

int CheckCpuProgramOwnerDeltas() {
  using namespace rund::compute;
  auto device = open(Target::cpu(1u));
  if (!device) {
    return 1;
  }

  auto simple = on(*device)
                    .input<std::uint32_t>(32u)
                    .map("m", [](auto value) { return value + 1u; })
                    .compile();
  auto long_name =
      on(*device)
          .input<std::uint32_t>(32u)
          .map("this-map-label-is-deliberately-long-and-must-not-be-retained-"
               "by-the-compiled-program-owner-after-lowering",
               [](auto value) { return value + 1u; })
          .compile();
  auto deep_plan =
      on(*device)
          .input<std::uint32_t>(32u)
          .map("m",
               [](auto value) {
                 return (
                     (((((value + 1u) * 3u + value + 5u) * 7u + value + 11u) *
                           13u +
                       value + 17u) *
                          19u +
                      value + 23u) *
                         29u +
                     value);
               })
          .compile();
  auto extra_binding =
      on(*device)
          .input<std::uint32_t>(32u)
          .zip_input<std::uint32_t>(32u)
          .map("m", [](auto left, auto right) { return left + right; })
          .compile();
  auto extra_topology = on(*device)
                            .input<std::uint32_t>(32u)
                            .map("m", [](auto value) { return value + 1u; })
                            .scan(Scan::InclusiveSum)
                            .compile();
  if (!simple || !long_name || !deep_plan || !extra_binding ||
      !extra_topology) {
    return 2;
  }

  const auto &simple_state = detail::FlowAccess::state(*simple);
  const auto &name_state = detail::FlowAccess::state(*long_name);
  const auto &plan_state = detail::FlowAccess::state(*deep_plan);
  const auto &binding_state = detail::FlowAccess::state(*extra_binding);
  const auto &topology_state = detail::FlowAccess::state(*extra_topology);
  const auto first_map = [](const auto &state) -> const detail::CpuProgram * {
    if (state == nullptr || state->cpu_graph == nullptr) {
      return nullptr;
    }
    for (const auto &map : state->cpu_graph->maps) {
      if (map != nullptr) {
        return map.get();
      }
    }
    return nullptr;
  };
  const detail::CpuProgram *const simple_map = first_map(simple_state);
  const detail::CpuProgram *const name_map = first_map(name_state);
  const detail::CpuProgram *const plan_map = first_map(plan_state);
  const detail::CpuProgram *const binding_map = first_map(binding_state);
  if (simple_map == nullptr || name_map == nullptr || plan_map == nullptr ||
      binding_map == nullptr || topology_state == nullptr ||
      topology_state->cpu_graph == nullptr ||
      topology_state->cpu_graph->runtime == nullptr) {
    return 3;
  }

  const auto compact = [](const auto &state) {
    if (state == nullptr || state->cpu_graph == nullptr ||
        state->cpu_graph->runtime == nullptr) {
      return false;
    }
    const detail::CpuRuntimeGraph &graph = *state->cpu_graph->runtime;
    for (const detail::CpuRuntimeStep &step : graph.steps) {
      const auto *primitive = std::get_if<detail::CpuRuntimePrimitive>(&step);
      if (primitive != nullptr &&
          !std::visit([](const auto &plan) { return plan.ok; },
                      primitive->plan)) {
        return false;
      }
    }
    return true;
  };
  if (!compact(simple_state) || !compact(name_state) || !compact(plan_state) ||
      !compact(binding_state) || !compact(topology_state)) {
    return 4;
  }

  const auto prepared_bytes = [](const auto &prepared) {
    return static_cast<std::uint64_t>(prepared.instructions.capacity()) *
               sizeof(rund::node::accel::cpu_simd_detail::PreparedInstruction) +
           static_cast<std::uint64_t>(prepared.value_formats.capacity()) *
               sizeof(rund::kernel::ComputeFixedFormat);
  };
  const auto compact_prepared = [&](const detail::CpuProgram &map) {
    const auto &prepared = map.dispatch.prepared;
    return prepared.ok && !prepared.instructions.empty() &&
           prepared.value_formats.size() == prepared.instructions.size() + 1u &&
           prepared.once_count <= prepared.instructions.size() &&
           prepared.retained_dynamic_memory_bytes() == prepared_bytes(prepared);
  };
  if (!compact_prepared(*simple_map) || !compact_prepared(*name_map) ||
      !compact_prepared(*plan_map) || !compact_prepared(*binding_map)) {
    return 11;
  }

  MemoryStats simple_memory{};
  MemoryStats name_memory{};
  MemoryStats plan_memory{};
  MemoryStats binding_memory{};
  MemoryStats topology_memory{};
  if (!ReadMemory(*simple, simple_memory) ||
      !ReadMemory(*long_name, name_memory) ||
      !ReadMemory(*deep_plan, plan_memory) ||
      !ReadMemory(*extra_binding, binding_memory) ||
      !ReadMemory(*extra_topology, topology_memory)) {
    return 5;
  }
  const auto exact_snapshot = [](const auto &program,
                                 const MemoryStats expected) {
    MemoryStats snapshot_stats{};
    const SnapshotAccounting snapshot = SnapshotMemory(program, snapshot_stats);
    return snapshot.complete && snapshot.valid && snapshot.allocation_free &&
           snapshot.metadata_entries == 1u && snapshot.tile_entries == 1u &&
           snapshot.internal_entries == 0u &&
           SameStats(expected, snapshot_stats);
  };
  if (!exact_snapshot(*simple, simple_memory) ||
      !exact_snapshot(*long_name, name_memory) ||
      !exact_snapshot(*deep_plan, plan_memory) ||
      !exact_snapshot(*extra_binding, binding_memory) ||
      !exact_snapshot(*extra_topology, topology_memory)) {
    return 6;
  }

  // Diagnostic map labels do not affect graph identity or retained bytes.
  const auto &simple_prepared = simple_map->dispatch.prepared;
  const auto &name_prepared = name_map->dispatch.prepared;
  if (!SameCounter(simple_memory.host, name_memory.host) ||
      !SameCounter(simple_memory.tile, name_memory.tile) ||
      simple_state->graph_info.fingerprint !=
          name_state->graph_info.fingerprint ||
      simple_map->map.op_hash_hi != name_map->map.op_hash_hi ||
      simple_map->map.op_hash_lo != name_map->map.op_hash_lo ||
      simple_prepared.instructions.size() !=
          name_prepared.instructions.size() ||
      simple_prepared.value_formats.size() !=
          name_prepared.value_formats.size() ||
      simple_prepared.once_count != name_prepared.once_count ||
      simple_prepared.retained_dynamic_memory_bytes() !=
          name_prepared.retained_dynamic_memory_bytes()) {
    return 7;
  }

  // The topology and binding counts remain equal here; only the expression IR
  // grows. Its retained delta belongs to the prepared instruction/format owner.
  const auto &plan_prepared = plan_map->dispatch.prepared;
  if (simple_state->graph_info.nodes.size() !=
          plan_state->graph_info.nodes.size() ||
      simple_map->map.input_buffer_count != plan_map->map.input_buffer_count ||
      simple_prepared.read_count != plan_prepared.read_count ||
      simple_prepared.write_count != plan_prepared.write_count ||
      plan_prepared.instructions.size() <=
          simple_prepared.instructions.size() ||
      plan_prepared.value_formats.size() <=
          simple_prepared.value_formats.size() ||
      plan_prepared.retained_dynamic_memory_bytes() <=
          simple_prepared.retained_dynamic_memory_bytes() ||
      plan_memory.host.current <= simple_memory.host.current) {
    return 8;
  }

  // Adding one external input changes the compact descriptor, PreparedRun read
  // count, Program route, and graph::Info ownership.
  const auto &binding_prepared = binding_map->dispatch.prepared;
  if (binding_state->input_types.size() !=
          simple_state->input_types.size() + 1u ||
      binding_map->map.input_buffer_count !=
          simple_map->map.input_buffer_count + 1u ||
      binding_prepared.read_count != simple_prepared.read_count + 1u ||
      binding_prepared.write_count != simple_prepared.write_count ||
      binding_state->graph_info.inputs.size() !=
          simple_state->graph_info.inputs.size() + 1u ||
      binding_memory.host.current <= simple_memory.host.current) {
    return 9;
  }

  const detail::CpuRuntimeGraph &topology_graph =
      *topology_state->cpu_graph->runtime;
  if (topology_state->graph_info.nodes.size() !=
          simple_state->graph_info.nodes.size() + 1u ||
      topology_graph.steps.size() !=
          simple_state->cpu_graph->runtime->steps.size() + 1u ||
      topology_state->cpu_graph->collectives.size() < 2u ||
      topology_state->cpu_graph->collectives.back() == nullptr ||
      topology_memory.host.current <= simple_memory.host.current ||
      topology_memory.tile.current <= simple_memory.tile.current) {
    return 10;
  }
  return 0;
}

} // namespace rund_node_memory_contract
