#pragma once

#include "../state.hpp"

#include <accel/graph/node.hpp>

#include <cstdint>
#include <span>
#include <string_view>

namespace rund::compute::detail::graph_build_detail {

[[nodiscard]] bool append_map(GraphState &graph, std::string_view name,
                              std::span<const std::uint32_t> inputs,
                              std::span<const std::uint32_t> outputs,
                              std::span<const ExprRef> expressions,
                              FlowControl control = {},
                              std::span<const MapRead> reads = {});

[[nodiscard]] bool
append_primitive(GraphState &graph, std::span<const std::uint32_t> inputs,
                 std::span<const std::uint32_t> outputs, std::uint32_t output,
                 Primitive primitive, PrimitiveOptions options,
                 rund::AccelGraphNode node, FlowControl control = {});

[[nodiscard]] std::uint32_t materialize(GraphState &graph, std::uint32_t source,
                                        std::string_view name);

} // namespace rund::compute::detail::graph_build_detail
