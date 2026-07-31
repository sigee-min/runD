#pragma once

#include <rund/compute/abi/model.hpp>
#include <span>
#include <string_view>
namespace rund::compute::detail {
[[nodiscard]] std::shared_ptr<GraphState>
make_graph(const std::shared_ptr<DeviceState> &device, std::string_view name,
           std::size_t count);
[[nodiscard]] std::uint32_t
graph_input(const std::shared_ptr<GraphState> &graph, Type type);
[[nodiscard]] std::uint32_t
graph_input_count(const std::shared_ptr<GraphState> &graph, Type type,
                  std::size_t count, FixedFormat fixed_format = {});
[[nodiscard]] std::uint32_t graph_map(const std::shared_ptr<GraphState> &graph,
                                      std::span<const std::uint32_t> inputs,
                                      ExprRef expression,
                                      std::string_view name = {},
                                      std::size_t output_count = 0u);
[[nodiscard]] ValueIds graph_map_multi(const std::shared_ptr<GraphState> &graph,
                                       std::span<const std::uint32_t> inputs,
                                       std::span<const ExprRef> expressions,
                                       std::string_view name = {},
                                       std::size_t output_count = 0u);
[[nodiscard]] ValueIds
graph_map_multi_controlled(const std::shared_ptr<GraphState> &graph,
                           std::span<const std::uint32_t> inputs,
                           std::span<const ExprRef> expressions,
                           FlowControl control, std::string_view name = {},
                           std::size_t output_count = 0u);
[[nodiscard]] std::uint32_t graph_scan(const std::shared_ptr<GraphState> &graph,
                                       std::uint32_t input, Scan scan,
                                       std::uint32_t count = 0u,
                                       FlowControl control = {});
void reject_graph(const std::shared_ptr<GraphState> &graph, Reason reason);
[[nodiscard]] GraphOut
graph_primitive(const std::shared_ptr<GraphState> &graph, Primitive primitive,
                std::span<const GraphArg> inputs, PrimitiveOptions options,
                std::span<const GraphArg> recipe_outputs = {},
                FlowControl control = {});
void graph_output(const std::shared_ptr<GraphState> &graph,
                  std::uint32_t output);
void graph_outputs(const std::shared_ptr<GraphState> &graph,
                   std::span<const std::uint32_t> outputs);
void graph_identity_outputs(const std::shared_ptr<GraphState> &graph,
                            std::span<const std::uint32_t> outputs);
[[nodiscard]] Result<std::shared_ptr<ProgramState>>
compile_graph(const std::shared_ptr<GraphState> &graph,
              std::span<const Type> inputs, std::span<const Type> outputs,
              const std::shared_ptr<ProgramCacheState> &cache = {});
} // namespace rund::compute::detail
