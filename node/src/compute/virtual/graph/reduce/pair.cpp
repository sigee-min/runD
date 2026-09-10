#include "../reduce.hpp"

#include "wavefront.hpp"

#include "../../backing.hpp"

#include <algorithm>
#include <array>

namespace rund::compute::detail::graph_reduce {

bool graph_wavefront_pair_eligible(const VirtualPipelineState &state,
                                   const VirtualRunProjection &run) noexcept {
  if (!run.graph_execution() || run.graph_reduction() ||
      state.geometry.route != VirtualRoute::GraphPointwise ||
      state.pipeline == nullptr || state.pipeline->device == nullptr ||
      state.pipeline->device->backend == Backend::Cpu ||
      state.pipeline->residency == nullptr ||
      !state.pipeline->residency->graph_tiled() || state.input_count != 3u ||
      run.input_count != 3u || run.device_vsm_required ||
      state.graph_input_resource_count != state.input_count ||
      state.output == nullptr || state.output->backing == nullptr ||
      VirtualBackingAccess::resident(*state.output->backing) != nullptr ||
      !run.active.graph.valid()) {
    return false;
  }
  for (std::size_t input = 0u; input < state.input_count; ++input) {
    const VirtualBufferState *const value = virtual_input(state, input);
    if (value == nullptr || value->backing == nullptr ||
        value->backing->max_parallel_reads() > 1u ||
        VirtualBackingAccess::resident(*value->backing) != nullptr) {
      return false;
    }
  }
  const residency::TiledGraphPlan &graph =
      state.pipeline->residency->tiled_graph();
  const std::span<const residency::TiledGraphStage> stages = graph.stages();
  if (stages.size() != 4u || stages.front().ports.empty() ||
      stages.back().ports.empty() || run.active.graph.batch_count() == 0u) {
    return false;
  }

  std::array<std::uint32_t, 2u> resources{};
  std::array<std::size_t, 2u> inputs{};
  std::uint32_t prefix_resource = 0u;
  std::size_t prefix_input = 0u;
  std::size_t prefix_reads = 0u;
  for (const residency::TiledGraphPort port : stages.front().ports) {
    const residency::TiledGraphResource *const resource =
        graph.resource(port.resource);
    if (!residency::reads(port.access)) {
      continue;
    }
    ++prefix_reads;
    if (resource == nullptr ||
        resource->kind != residency::GraphResourceKind::ExternalInput ||
        resource->persistence != residency::ResourcePersistence::Backing) {
      return false;
    }
    prefix_resource = port.resource;
  }
  const auto prefix = std::find(
      state.graph_input_resources.begin(),
      state.graph_input_resources.begin() +
          static_cast<std::ptrdiff_t>(state.graph_input_resource_count),
      prefix_resource);
  if (prefix_reads != 1u || prefix_resource == 0u ||
      prefix ==
          state.graph_input_resources.begin() +
              static_cast<std::ptrdiff_t>(state.graph_input_resource_count)) {
    return false;
  }
  prefix_input = static_cast<std::size_t>(
      std::distance(state.graph_input_resources.begin(), prefix));
  if (prefix_input >= run.input_count ||
      run.inputs[prefix_input].backing == 0u) {
    return false;
  }
  for (std::size_t middle = 0u; middle < resources.size(); ++middle) {
    const residency::TiledGraphStage &stage = stages[middle + 1u];
    std::size_t read_count = 0u;
    std::size_t external_count = 0u;
    std::uint32_t resource_id = 0u;
    for (const residency::TiledGraphPort port : stage.ports) {
      const residency::TiledGraphResource *const resource =
          graph.resource(port.resource);
      if (!residency::reads(port.access)) {
        continue;
      }
      ++read_count;
      if (resource == nullptr ||
          resource->kind != residency::GraphResourceKind::ExternalInput ||
          resource->persistence != residency::ResourcePersistence::Backing) {
        return false;
      }
      ++external_count;
      resource_id = port.resource;
    }
    if (read_count != 1u || external_count != 1u || resource_id == 0u) {
      return false;
    }
    const auto input = std::find(
        state.graph_input_resources.begin(),
        state.graph_input_resources.begin() +
            static_cast<std::ptrdiff_t>(state.graph_input_resource_count),
        resource_id);
    if (input ==
        state.graph_input_resources.begin() +
            static_cast<std::ptrdiff_t>(state.graph_input_resource_count)) {
      return false;
    }
    resources[middle] = resource_id;
    inputs[middle] = static_cast<std::size_t>(
        std::distance(state.graph_input_resources.begin(), input));
    if (inputs[middle] >= run.input_count ||
        run.inputs[inputs[middle]].backing == 0u) {
      return false;
    }
  }
  if (resources[0u] == resources[1u] || resources[0u] == prefix_resource ||
      resources[1u] == prefix_resource || inputs[0u] == inputs[1u] ||
      inputs[0u] == prefix_input || inputs[1u] == prefix_input) {
    return false;
  }
  for (const residency::TiledGraphPort port : stages.back().ports) {
    if (!residency::reads(port.access)) {
      continue;
    }
    const residency::TiledGraphResource *const resource =
        graph.resource(port.resource);
    if (resource != nullptr &&
        resource->kind == residency::GraphResourceKind::ExternalInput &&
        resource->persistence == residency::ResourcePersistence::Backing) {
      return false;
    }
  }

  // Pair admission is limited to cells with no same-batch edge between the
  // two middle stages. Planner-sealed prefix and prior-batch release edges
  // remain valid and are still enforced by Wavefront::predecessors_ready().
  std::array<residency::TiledGraphDependency,
             residency::TiledGraphDependencyCapacity>
      predecessors{};
  const std::uint64_t checked_batches =
      std::min<std::uint64_t>(2u, run.active.graph.batch_count());
  for (std::uint64_t batch = 0u; batch < checked_batches; ++batch) {
    for (std::size_t middle = 0u; middle < resources.size(); ++middle) {
      std::size_t count = 0u;
      if (!run.active.graph.predecessors(batch, middle + 1u, predecessors,
                                         count)) {
        return false;
      }
      for (std::size_t index = 0u; index < count; ++index) {
        if (predecessors[index].batch == batch &&
            (predecessors[index].stage == 1u ||
             predecessors[index].stage == 2u)) {
          return false;
        }
      }
    }
  }
  return true;
}

} // namespace rund::compute::detail::graph_reduce
