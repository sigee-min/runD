#pragma once

#include "../../resource/memory.hpp"
#include "../describe.hpp"

#include <kernel/program/compute/graph/schema.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace rund::compute::detail::graph_detail::describe_detail {

struct Draft final {
  Description description{};
  std::vector<std::vector<kernel::GraphBufferRef>> refs{};
  std::vector<kernel::GraphNode> nodes{};
  std::vector<resource_detail::MemoryNode> memory{};
  bool zero_work{};
};

[[nodiscard]] Status build_resources(const GraphState &state,
                                     graph::Info &info);

[[nodiscard]] Status
validate_bindings(const graph::Info &info,
                  std::span<const std::uint32_t> identity_outputs);

[[nodiscard]] Status build_nodes(const GraphState &state, Draft &draft);

void append_access(graph::Info &info, graph::Node &node,
                   std::uint32_t resource, resource::AccessMode mode);
[[nodiscard]] Status build_map_node(
    const GraphState &state, const MapStep &map, Draft &draft,
    graph::Node &info, std::vector<kernel::GraphBufferRef> &refs,
    kernel::GraphNode &canonical, resource_detail::MemoryNode &memory);
[[nodiscard]] Status build_scan_node(
    const GraphState &state, const ScanStep &scan, Draft &draft,
    graph::Node &info, std::vector<kernel::GraphBufferRef> &refs,
    kernel::GraphNode &canonical, resource_detail::MemoryNode &memory);
[[nodiscard]] Status build_primitive_node(
    const GraphState &state, const GraphPrimitive &primitive, Draft &draft,
    graph::Node &info, std::vector<kernel::GraphBufferRef> &refs,
    kernel::GraphNode &canonical, resource_detail::MemoryNode &memory);

[[nodiscard]] Status build_hazards(graph::Info &info);

[[nodiscard]] Description
finish(Draft draft, Type root, FixedFormat root_format,
       std::span<const std::uint32_t> identity_outputs,
       std::uint64_t page_bytes);

} // namespace rund::compute::detail::graph_detail::describe_detail
