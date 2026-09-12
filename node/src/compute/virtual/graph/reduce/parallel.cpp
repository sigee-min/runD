#include "../reduce.hpp"

#include "../../backing.hpp"
#include "wavefront.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace rund::compute::detail::graph_reduce {

bool graph_wavefront_parallel_eligible(
    const VirtualPipelineState &state,
    const VirtualRunProjection &run) noexcept {
  if (!run.graph_execution() || run.graph_reduction() ||
      state.geometry.route != VirtualRoute::GraphPointwise ||
      state.pipeline == nullptr || state.pipeline->device == nullptr ||
      state.pipeline->device->backend == Backend::Cpu ||
      state.pipeline->residency == nullptr ||
      !state.pipeline->residency->graph_tiled() || run.device_vsm_required ||
      state.input_count != run.input_count || state.input_count < 2u ||
      state.input_count > VirtualPipelineState::InputCapacity ||
      state.graph_input_resource_count != state.input_count ||
      state.output == nullptr || state.output->backing == nullptr ||
      VirtualBackingAccess::resident(*state.output->backing) != nullptr ||
      !run.active.graph.valid() || run.active.graph.batch_count() == 0u) {
    return false;
  }
  for (std::size_t input = 0u; input < state.input_count; ++input) {
    const VirtualBufferState *const value = virtual_input(state, input);
    if (value == nullptr || value->backing == nullptr ||
        value->backing->max_parallel_reads() > 1u ||
        VirtualBackingAccess::resident(*value->backing) != nullptr ||
        run.inputs[input].backing == 0u) {
      return false;
    }
  }
  const residency::TiledGraphPlan &graph =
      state.pipeline->residency->tiled_graph();
  const auto stages = graph.stages();
  if (stages.size() < 4u || stages.size() > WavefrontStageCapacity) {
    return false;
  }
  static_assert(WavefrontStageCapacity <=
                std::numeric_limits<std::uint32_t>::digits);
  std::array<std::uint64_t, WavefrontStageCapacity> backings{};
  for (std::size_t stage = 0u; stage < stages.size(); ++stage) {
    for (const residency::TiledGraphPort port : stages[stage].ports) {
      if (!residency::reads(port.access)) {
        continue;
      }
      const auto *const resource = graph.resource(port.resource);
      if (resource == nullptr) {
        return false;
      }
      if (resource->kind != residency::GraphResourceKind::ExternalInput) {
        continue;
      }
      if (backings[stage] != 0u || stage + 1u == stages.size() ||
          resource->persistence != residency::ResourcePersistence::Backing) {
        return false;
      }
      const auto end = state.graph_input_resources.begin() +
                       static_cast<std::ptrdiff_t>(state.input_count);
      const auto found =
          std::find(state.graph_input_resources.begin(), end, port.resource);
      if (found == end) {
        return false;
      }
      const auto index = static_cast<std::size_t>(
          std::distance(state.graph_input_resources.begin(), found));
      backings[stage] = run.inputs[index].backing;
    }
  }
  if (backings[0u] == 0u) {
    return false;
  }

  // Eligibility comes from the sealed dependency graph, not a stage/input
  // count signature. First and recurrent batches cover the two dependency
  // forms; the live Wavefront still authenticates every actual predecessor.
  std::array<residency::TiledGraphDependency,
             residency::TiledGraphDependencyCapacity>
      predecessors{};
  for (std::uint64_t batch = 0u;
       batch < std::min<std::uint64_t>(2u, run.active.graph.batch_count());
       ++batch) {
    std::array<std::uint32_t, WavefrontStageCapacity> ancestors{};
    for (std::size_t stage = 0u; stage < stages.size(); ++stage) {
      std::size_t count = 0u;
      if (!run.active.graph.predecessors(batch, stage, predecessors, count)) {
        return false;
      }
      for (std::size_t index = 0u; index < count; ++index) {
        const auto dependency = predecessors[index];
        if (dependency.batch != batch) {
          continue;
        }
        if (dependency.stage >= stage) {
          return false;
        }
        ancestors[stage] |= ancestors[dependency.stage] |
                            (std::uint32_t{1u} << dependency.stage);
      }
    }
    bool independent = false;
    for (std::size_t later = 2u; later + 1u < stages.size(); ++later) {
      for (std::size_t earlier = 1u; earlier < later; ++earlier) {
        independent =
            independent ||
            (backings[earlier] != 0u && backings[later] != 0u &&
             backings[earlier] != backings[later] &&
             (ancestors[later] & (std::uint32_t{1u} << earlier)) == 0u);
      }
    }
    if (!independent) {
      return false;
    }
  }
  return true;
}

} // namespace rund::compute::detail::graph_reduce
