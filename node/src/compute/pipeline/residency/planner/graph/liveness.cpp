#include "internal.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::compute::detail::residency {

Failure freeze_graph_liveness(GraphPlanningState &state) {
  // Freeze resource liveness and each port's next consumer in two linear
  // topology passes. Runtime projection consumes these rows directly; it
  // never rescans future stages or owns a second liveness policy.
  struct Liveness final {
    std::uint32_t next_stage{NoGraphStage};
    std::uint32_t first_stage{NoGraphStage};
    std::uint32_t last_stage{NoGraphStage};
    std::uint32_t writer_stage{NoGraphStage};
    std::size_t writer_count{};
    Access first_access{Access::Read};
  };
  std::array<Liveness, TiledGraphResourceCapacity> liveness{};
  for (std::size_t stage_index = 0u; stage_index < state.stages.size();
       ++stage_index) {
    const auto ordinal = static_cast<std::uint32_t>(stage_index);
    for (const TiledGraphPort port : state.stages[stage_index].ports) {
      const std::size_t index = index_graph_resource(state, port.resource);
      if (index >= state.resources.size()) {
        return Failure::Invalid;
      }
      Liveness &live = liveness[index];
      if (live.first_stage == NoGraphStage) {
        live.first_stage = ordinal;
        live.first_access = port.access;
      }
      live.last_stage = ordinal;
      if (writes(port.access)) {
        live.writer_stage = ordinal;
        ++live.writer_count;
      }
    }
  }
  for (std::size_t reverse = state.stages.size(); reverse != 0u; --reverse) {
    const std::size_t stage_index = reverse - 1u;
    for (TiledGraphPort &port : state.stages[stage_index].ports) {
      const std::size_t index = index_graph_resource(state, port.resource);
      if (index >= state.resources.size()) {
        return Failure::Invalid;
      }
      port.next_stage = liveness[index].next_stage;
      liveness[index].next_stage = static_cast<std::uint32_t>(stage_index);
    }
  }
  for (std::size_t ordinal = 0u; ordinal < state.resources.size(); ++ordinal) {
    TiledGraphResource &graph_resource = state.resources[ordinal];
    const Liveness live = liveness[ordinal];
    const std::uint32_t first_stage = live.first_stage;
    const std::uint32_t last_stage = live.last_stage;
    const std::uint32_t writer_stage = live.writer_stage;
    const std::size_t writer_count = live.writer_count;
    const Access first_access = live.first_access;
    if (first_stage == NoGraphStage ||
        (graph_resource.kind == GraphResourceKind::ExternalInput &&
         (graph_resource.persistence != ResourcePersistence::Backing ||
          writer_count != 0u || !reads(first_access))) ||
        (graph_resource.kind == GraphResourceKind::Internal &&
         (graph_resource.persistence != ResourcePersistence::Transient ||
          writer_count != 1u || writer_stage != first_stage ||
          first_access != Access::Write || last_stage <= writer_stage)) ||
        (graph_resource.kind == GraphResourceKind::ExternalOutput &&
         (writer_count != 1u || writer_stage != first_stage ||
          writer_stage != last_stage || first_access != Access::Write))) {
      return Failure::Invalid;
    }
    graph_resource.first_stage = first_stage;
    graph_resource.last_stage =
        graph_resource.kind == GraphResourceKind::ExternalOutput
            ? static_cast<std::uint32_t>(state.stages.size() - 1u)
            : last_stage;
    graph_resource.producer_stage =
        graph_resource.persistence == ResourcePersistence::Transient
            ? writer_stage
            : NoGraphStage;
    graph_resource.role =
        graph_resource.kind == GraphResourceKind::ExternalInput
            ? GraphResourceRole::Input
            : (graph_resource.kind == GraphResourceKind::ExternalOutput
                   ? GraphResourceRole::Output
                   : GraphResourceRole::Intermediate);
  }
  return Failure::None;
}

} // namespace rund::compute::detail::residency
