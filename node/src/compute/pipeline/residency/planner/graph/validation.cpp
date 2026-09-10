#include "internal.hpp"

#include "../../../../type.hpp"
#include "../../../plan/contract.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <utility>

namespace rund::compute::detail::residency {

Failure initialize_graph(GraphPlanningState &state) {
  // Sort only borrowed descriptors. The immutable result owns the sole
  // deep copy; comparator calls must never copy a remap vector.
  std::array<const TiledGraphResourceInput *, TiledGraphResourceCapacity> order{};
  for (std::size_t index = 0u; index < state.input.resources.size(); ++index) {
    order[index] = &state.input.resources[index];
  }
  const auto resources = std::span{order}.first(state.input.resources.size());
  std::sort(resources.begin(), resources.end(),
            [](const auto *left, const auto *right) {
              return left->resource < right->resource;
            });
  state.resources.reserve(resources.size());
  std::size_t remap_total = 0u;
  for (std::size_t index = 0u; index < resources.size(); ++index) {
    const TiledGraphResourceInput &declared = *resources[index];
    std::uint64_t capacity = 0u;
    std::uint64_t last_offset = 0u;
    if (declared.resource == 0u || !valid_type(declared.type) ||
        !valid_format(declared.type, declared.format) ||
        declared.page_bytes == 0u || declared.logical_bytes == 0u ||
        declared.page_bytes % type_bytes(declared.type) != 0u ||
        declared.logical_bytes % type_bytes(declared.type) != 0u ||
        static_cast<std::uint8_t>(declared.kind) >
            static_cast<std::uint8_t>(GraphResourceKind::ExternalOutput) ||
        (index != 0u && resources[index - 1u]->resource == declared.resource) ||
        !kernel::checked::mul(state.input.page_count, declared.page_bytes,
                              capacity) ||
        !kernel::checked::mul(state.input.page_count - 1u, declared.page_bytes,
                              last_offset) ||
        declared.logical_bytes <= last_offset ||
        declared.logical_bytes > capacity) {
      return Failure::Invalid;
    }
    if (!declared.remaps.empty()) {
      if (declared.kind != GraphResourceKind::ExternalInput ||
          declared.persistence != ResourcePersistence::Backing) {
        return Failure::Invalid;
      }
      if (state.frames == 0u || state.frames > PipelineLeafCapacity ||
          declared.remaps.size() != state.frames ||
          remap_total > GraphPageRemapCapacity - declared.remaps.size()) {
        return declared.remaps.size() > state.frames ? Failure::Capacity
                                                     : Failure::Invalid;
      }
      std::array<bool, PipelineLeafCapacity> sources{};
      std::array<bool, PipelineLeafCapacity> targets{};
      for (const GraphPageRemap remap : declared.remaps) {
        if (remap.source_local >= state.frames ||
            remap.target_local >= state.frames ||
            static_cast<std::uint8_t>(remap.source_origin) >
                static_cast<std::uint8_t>(GraphPageOrigin::End) ||
            sources[remap.source_local] || targets[remap.target_local]) {
          return Failure::Invalid;
        }
        sources[remap.source_local] = true;
        targets[remap.target_local] = true;
      }
      remap_total += declared.remaps.size();
    }
    std::vector<GraphPageRemap> remaps = declared.remaps;
    std::sort(remaps.begin(), remaps.end(),
              [](const GraphPageRemap left, const GraphPageRemap right) {
                if (left.target_local != right.target_local) {
                  return left.target_local < right.target_local;
                }
                if (left.source_local != right.source_local) {
                  return left.source_local < right.source_local;
                }
                return left.source_origin < right.source_origin;
              });
    state.resources.push_back(TiledGraphResource{
        .resource = declared.resource,
        .type = declared.type,
        .format = declared.format,
        .page_bytes = declared.page_bytes,
        .logical_bytes = declared.logical_bytes,
        .kind = declared.kind,
        .persistence = declared.persistence,
        .remaps = std::move(remaps),
    });
  }
  state.stages.reserve(state.input.stages.size());
  for (const TiledGraphStageInput &declared : state.input.stages) {
    TiledGraphStage stage{.node = declared.node, .domain = declared.domain};
    stage.ports.reserve(declared.ports.size());
    for (const TiledGraphPortInput port : declared.ports) {
      stage.ports.push_back(
          TiledGraphPort{.resource = port.resource, .access = port.access});
    }
    state.stages.push_back(std::move(stage));
  }
  std::uint32_t prior_node = 0u;
  std::size_t port_count = 0u;
  for (std::size_t index = 0u; index < state.stages.size(); ++index) {
    TiledGraphStage &stage = state.stages[index];
    bool stage_reads = false;
    bool stage_writes = false;
    std::uint16_t read_port = 0u;
    std::uint16_t write_port = 0u;
    if ((stage.domain != StageDomain::Tile &&
         stage.domain != StageDomain::TilePartial) ||
        stage.ports.empty() || stage.ports.size() > TiledGraphPortCapacity ||
        (index != 0u && stage.node <= prior_node) ||
        port_count > TiledGraphPortCapacity - stage.ports.size()) {
      return Failure::Invalid;
    }
    port_count += stage.ports.size();
    for (std::size_t port_index = 0u; port_index < stage.ports.size();
         ++port_index) {
      TiledGraphPort &port = stage.ports[port_index];
      if (find_graph_resource(state, port.resource) == nullptr ||
          static_cast<std::uint8_t>(port.access) >
              static_cast<std::uint8_t>(Access::ReadWrite) ||
          port.access == Access::ReadWrite) {
        return Failure::Invalid;
      }
      for (std::size_t prior = 0u; prior < port_index; ++prior) {
        const TiledGraphPort prior_port = stage.ports[prior];
        if (prior_port.resource == port.resource) {
          return Failure::Invalid;
        }
      }
      port.program_port = reads(port.access) ? read_port++ : write_port++;
      stage_reads = stage_reads || reads(port.access);
      stage_writes = stage_writes || writes(port.access);
    }
    if (!stage_reads || !stage_writes) {
      return Failure::Invalid;
    }
    stage.active_count_input = read_port;
    prior_node = stage.node;
  }
  return Failure::None;
}

} // namespace rund::compute::detail::residency
