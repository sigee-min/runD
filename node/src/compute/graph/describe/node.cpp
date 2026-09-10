#include "model.hpp"

#include <utility>
#include <variant>

namespace rund::compute::detail::graph_detail::describe_detail {

Status build_nodes(const GraphState &state, Draft &draft) {
  graph::Info &graph = draft.description.info;
  graph.nodes.reserve(state.steps.size());
  draft.description.map_operations.reserve(state.steps.size());
  draft.refs.reserve(state.steps.size());
  draft.nodes.reserve(state.steps.size());
  draft.memory.reserve(state.steps.size());

  for (std::size_t index = 0u; index < state.steps.size(); ++index) {
    draft.refs.emplace_back();
    std::vector<kernel::GraphBufferRef> &refs = draft.refs.back();
    graph::Node info{.index = static_cast<std::uint32_t>(index)};
    kernel::GraphNode canonical{};
    resource_detail::MemoryNode memory{};

    const GraphStep &step = state.steps[index];
    Status status = Status::success();
    if (const auto *map = std::get_if<MapStep>(&step)) {
      status = build_map_node(state, *map, draft, info, refs, canonical, memory);
    } else if (const auto *scan = std::get_if<ScanStep>(&step)) {
      status = build_scan_node(state, *scan, draft, info, refs, canonical, memory);
    } else {
      status = build_primitive_node(state, std::get<GraphPrimitive>(step), draft,
                               info, refs, canonical, memory);
    }
    if (!status) {
      return status;
    }
    graph.nodes.push_back(std::move(info));
    draft.nodes.push_back(canonical);
    draft.memory.push_back(memory);
  }
  return Status::success();
}


} // namespace rund::compute::detail::graph_detail::describe_detail
