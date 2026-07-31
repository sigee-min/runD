#include "local.hpp"

#include "../../cpu/graph.hpp"
#include "../../fixed/format.hpp"
#include "../../type.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <limits>
#include <new>
#include <tuple>
#include <utility>
#include <vector>

namespace rund::compute::detail::graph_compile {
namespace {

[[nodiscard]] Status fill_interface(ProgramState &program,
                                    const GraphState &graph) {
  std::vector<std::size_t> output_positions;
  try {
    program.graph_value_routes.resize(graph.values.size());
    program.output_types.reserve(graph.outputs.size());
    program.output_sizes.reserve(graph.outputs.size());
    program.output_formats.reserve(graph.outputs.size());
    if (!graph.identity_outputs.empty()) {
      output_positions.assign(graph.values.size(),
                              std::numeric_limits<std::size_t>::max());
    }
    for (std::size_t index = 0u; index < graph.outputs.size(); ++index) {
      const std::uint32_t output = graph.outputs[index];
      if (output == 0u || output > graph.values.size()) {
        return Status::fail(Reason::GraphBindingInvalid);
      }
      const GraphValue &value = graph.values[output - 1u];
      program.output_types.push_back(value.type);
      program.output_sizes.push_back(value.count);
      program.output_formats.push_back(value.fixed_format);
      program.graph_value_routes[output - 1u] = GraphValueRoute{
          GraphBindSource::Output, static_cast<std::uint32_t>(index)};
      if (!output_positions.empty()) {
        output_positions[output - 1u] = index;
      }
    }
    if (!graph.identity_outputs.empty()) {
      program.output_aliases.reserve(graph.identity_outputs.size());
      for (const std::uint32_t output : graph.identity_outputs) {
        if (output == 0u || output > output_positions.size() ||
            output_positions[output - 1u] ==
                std::numeric_limits<std::size_t>::max()) {
          return Status::fail(Reason::GraphBindingInvalid);
        }
        program.output_aliases.push_back(output_positions[output - 1u]);
      }
    }

    program.input_types.reserve(graph.inputs.size());
    program.input_sizes.reserve(graph.inputs.size());
    program.input_formats.reserve(graph.inputs.size());
    program.bounded_input_capacities.assign(graph.inputs.size(), 0u);
    for (std::size_t index = 0u; index < graph.inputs.size(); ++index) {
      const std::uint32_t input = graph.inputs[index];
      if (input == 0u || input > graph.values.size()) {
        return Status::fail(Reason::GraphBindingInvalid);
      }
      const GraphValue &value = graph.values[input - 1u];
      program.input_types.push_back(value.type);
      program.input_sizes.push_back(value.count);
      program.input_formats.push_back(value.fixed_format);
      program.graph_value_routes[input - 1u] = GraphValueRoute{
          GraphBindSource::Input, static_cast<std::uint32_t>(index)};
    }
    for (const BoundedInputSchema bounded : graph.bounded_inputs) {
      if (bounded.count == 0u ||
          bounded.count > program.graph_value_routes.size() ||
          bounded.capacity == 0u) {
        return Status::fail(Reason::BoundedCountInvalid);
      }
      const GraphValueRoute route =
          program.graph_value_routes[bounded.count - 1u];
      if (route.source != GraphBindSource::Input ||
          route.index >= program.bounded_input_capacities.size() ||
          program.bounded_input_capacities[route.index] != 0u) {
        return Status::fail(Reason::BoundedCountInvalid);
      }
      program.bounded_input_capacities[route.index] = bounded.capacity;
    }
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::GraphCapacity);
  }
  return valid_input_shape(program)
             ? Status::success()
             : Status::fail(Reason::ProgramInputShapeInvalid);
}

[[nodiscard]] Status fill_storage(ProgramState &program,
                                  const GraphState &graph,
                                  const graph::Info &layout) {
  if (layout.resources.size() != graph.values.size()) {
    return Status::fail(Reason::GraphInvalid);
  }
  if (layout.memory.allocation_count >
      std::numeric_limits<std::size_t>::max()) {
    return Status::fail(Reason::GraphCapacity);
  }
  try {
    std::vector<std::uint32_t> slots(graph.values.size(),
                                     std::numeric_limits<std::uint32_t>::max());
    std::vector<std::uint64_t> extents(graph.values.size());
    for (std::size_t index = 0u; index < graph.values.size(); ++index) {
      const graph::Resource &resource = layout.resources[index];
      if (program.cpu_graph != nullptr) {
        program.cpu_graph->runtime->values.push_back(
            CpuRuntimeValue{.type = graph.values[index].type,
                            .count = graph.values[index].count});
      }
      if (resource.visibility != graph::Visibility::Internal ||
          resource.first_use == resource::NoNode) {
        continue;
      }
      std::uint64_t resource_end = 0u;
      if (resource.alias_group == 0u || resource.alias_group > extents.size() ||
          !kernel::checked::add(resource.alias_offset_bytes, resource.bytes,
                                resource_end)) {
        return Status::fail(Reason::GraphInvalid);
      }
      std::uint64_t &extent = extents[resource.alias_group - 1u];
      extent = std::max(extent, resource_end);
    }
    for (std::size_t index = 0u; index < graph.values.size(); ++index) {
      GraphValueRoute &route = program.graph_value_routes[index];
      if (route.source != GraphBindSource::Internal) {
        continue;
      }
      const graph::Resource &resource = layout.resources[index];
      if (resource.visibility != graph::Visibility::Internal ||
          resource.alias_group == 0u || resource.alias_group > slots.size()) {
        return Status::fail(Reason::GraphInvalid);
      }
      if (resource.first_use == resource::NoNode) {
        continue;
      }
      std::uint32_t &slot = slots[resource.alias_group - 1u];
      if (slot == std::numeric_limits<std::uint32_t>::max()) {
        if (program.chunks.size() >=
            std::numeric_limits<std::uint32_t>::max()) {
          return Status::fail(Reason::GraphCapacity);
        }
        slot = static_cast<std::uint32_t>(program.chunks.size());
        const std::uint64_t extent = extents[resource.alias_group - 1u];
        if (extent == 0u || extent % sizeof(std::uint32_t) != 0u ||
            extent / sizeof(std::uint32_t) >
                std::numeric_limits<std::size_t>::max()) {
          return Status::fail(Reason::GraphCapacity);
        }
        program.chunks.push_back(Chunk{
            .count = static_cast<std::size_t>(extent / sizeof(std::uint32_t))});
      }
      const GraphValue &value = graph.values[index];
      route.index = slot;
      route.offset_bytes = resource.alias_offset_bytes;
      route.bytes = resource.bytes;
      route.count = value.count;
      route.element_bytes = type_bytes(value.type);
      route.alignment = route.element_bytes;
    }
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::GraphCapacity);
  }
  return program.chunks.size() == layout.memory.allocation_count
             ? Status::success()
             : Status::fail(Reason::GraphInvalid);
}

[[nodiscard]] Status fill_resets(ProgramState &program,
                                 const graph::Info &layout) {
  if (layout.memory.reset_count > std::numeric_limits<std::size_t>::max()) {
    return Status::fail(Reason::GraphCapacity);
  }
  std::vector<ResetRoute> *const routes =
      program.cpu_graph == nullptr ? nullptr : &program.cpu_graph->resets;
  std::size_t count = 0u;
  try {
    if (routes != nullptr) {
      routes->reserve(static_cast<std::size_t>(layout.memory.reset_count));
    }
    for (std::size_t index = 0u; index < layout.resources.size(); ++index) {
      const graph::Resource &resource = layout.resources[index];
      if (!resource.requires_reset()) {
        continue;
      }
      if (resource.reset_node >= layout.nodes.size() ||
          resource.last_use >= layout.nodes.size() ||
          resource.last_use < resource.reset_node) {
        return Status::fail(Reason::GraphInvalid);
      }
      ++count;
      if (routes != nullptr) {
        routes->push_back(ResetRoute{
            .value_index = static_cast<std::uint32_t>(index),
            .step = resource.reset_node,
            .last = resource.last_use,
        });
      }
    }
    if (routes != nullptr) {
      std::sort(routes->begin(), routes->end(),
                [](const ResetRoute left, const ResetRoute right) noexcept {
                  return std::tie(left.step, left.value_index) <
                         std::tie(right.step, right.value_index);
                });
    }
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::GraphCapacity);
  }
  return count == layout.memory.reset_count
             ? Status::success()
             : Status::fail(Reason::GraphInvalid);
}

[[nodiscard]] Status fill_order(ProgramState &program) {
  try {
    program.chunk_order.resize(program.chunks.size());
    for (std::size_t index = 0u; index < program.chunk_order.size(); ++index) {
      if (index > std::numeric_limits<std::uint32_t>::max()) {
        return Status::fail(Reason::GraphCapacity);
      }
      program.chunk_order[index] = static_cast<std::uint32_t>(index);
    }
    std::sort(program.chunk_order.begin(), program.chunk_order.end(),
              [&](const std::uint32_t left, const std::uint32_t right) {
                const Chunk &a = program.chunks[left];
                const Chunk &b = program.chunks[right];
                return a.count != b.count ? a.count > b.count : left < right;
              });
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::GraphCapacity);
  }
  return Status::success();
}

} // namespace

Result<std::shared_ptr<ProgramState>>
prepare(const std::shared_ptr<GraphState> &graph, const graph::Info &layout) {
  if (graph == nullptr || graph->device == nullptr) {
    return Result<std::shared_ptr<ProgramState>>::fail(Reason::GraphInvalid);
  }
  try {
    auto program = std::make_shared<ProgramState>();
    program->device = graph->device;
    program->name = graph->name;
    program->count = graph->count;
    if (graph->device->backend == Backend::Cpu) {
      program->cpu_graph = std::make_unique<CpuGraphProgram>();
      program->cpu_graph->runtime = std::make_unique<CpuRuntimeGraph>();
      program->cpu_graph->runtime->values.reserve(graph->values.size());
      program->cpu_graph->runtime->steps.reserve(graph->steps.size());
      program->cpu_graph->maps.resize(graph->steps.size());
      program->cpu_graph->collectives.resize(graph->steps.size());
      program->cpu_graph->bind_begin.resize(graph->steps.size());
      program->cpu_graph->bind_count.resize(graph->steps.size());
    }
    const Status interface = fill_interface(*program, *graph);
    if (!interface) {
      return Result<std::shared_ptr<ProgramState>>::fail(interface.reason());
    }
    program->chunks.reserve(
        static_cast<std::size_t>(layout.memory.allocation_count));
    program->graph_bindings.reserve(graph->steps.size() * 4u);
    const Status storage = fill_storage(*program, *graph, layout);
    if (!storage) {
      return Result<std::shared_ptr<ProgramState>>::fail(storage.reason());
    }
    const Status order = fill_order(*program);
    if (!order) {
      return Result<std::shared_ptr<ProgramState>>::fail(order.reason());
    }
    const Status resets = fill_resets(*program, layout);
    return resets
               ? Result<std::shared_ptr<ProgramState>>::success(
                     std::move(program))
               : Result<std::shared_ptr<ProgramState>>::fail(resets.reason());
  } catch (const std::bad_alloc &) {
    return Result<std::shared_ptr<ProgramState>>::fail(Reason::GraphCapacity);
  }
}

} // namespace rund::compute::detail::graph_compile
