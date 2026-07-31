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

[[nodiscard]] Status build_hazards(graph::Info &info);

[[nodiscard]] Description
finish(Draft draft, Type root, FixedFormat root_format,
       std::span<const std::uint32_t> identity_outputs,
       std::uint64_t page_bytes);

} // namespace rund::compute::detail::graph_detail::describe_detail
