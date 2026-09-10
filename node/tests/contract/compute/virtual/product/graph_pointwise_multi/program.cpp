#include "internal.hpp"

#include "src/compute/flow/state.hpp"
#include "src/compute/graph/compile/slice.hpp"
#include "src/compute/graph/compile/slice/semantic.hpp"

#include <algorithm>
#include <cstdio>

namespace rund_node_test_virtual::product::graph_pointwise_multi {
namespace {

template <std::uint64_t First, std::size_t Count, class Expression>
[[nodiscard]] constexpr auto add_stage(Expression value) {
  if constexpr (Count == 1u) {
    return value + First;
  } else {
    constexpr std::size_t Left = Count / 2u;
    return add_stage<First, Left>(value) +
           add_stage<First + Left, Count - Left>(value);
  }
}

} // namespace

rund::compute::Result<Program>
build_program(const rund::compute::Device &device) {
  return rund::compute::on(device)
      .input<std::uint64_t>(FrameElements)
      .zip_input<std::uint64_t>(FrameElements)
      .zip_input<std::uint64_t>(FrameElements)
      .branch([](auto first, auto second, auto third) {
        const auto prefix =
            first.map("virtual-graph-pointwise-multi-prefix", [](auto value) {
              return add_stage<1u, StageLeafCount>(value);
            });
        const auto middle =
            zip(prefix, second)
                .map("virtual-graph-pointwise-multi-middle-input",
                     [](auto prior, auto later) {
                       return prior +
                              add_stage<StageLeafCount + 1u, StageLeafCount>(
                                  later);
                     });
        return zip(middle, third)
            .map("virtual-graph-pointwise-multi-terminal-input",
                 [](auto prior, auto later) {
                   return prior +
                          add_stage<2u * StageLeafCount + 1u, StageLeafCount>(
                              later);
                 });
      })
      .compile();
}

bool validate_program(const Program &program) noexcept {
  using namespace rund::compute;
  const auto slices =
      detail::graph_compile::compile_tiled_graph_pointwise_slices(
          detail::FlowAccess::state(program));
  const auto fused = detail::graph_compile::compile_service_free_map_program(
      detail::FlowAccess::state(program));
  const bool valid =
      slices && slices->stages.size() == StageCount &&
      slices->input_resources.size() == InputCount &&
      slices->stages[0u].inputs.size() == 1u &&
      slices->stages[0u].inputs[0u] == slices->input_resources[0u] &&
      slices->stages[0u].outputs.size() == 1u &&
      slices->stages[1u].inputs.size() == 2u &&
      std::find(slices->stages[1u].inputs.begin(),
                slices->stages[1u].inputs.end(), slices->input_resources[1u]) !=
          slices->stages[1u].inputs.end() &&
      std::find(
          slices->stages[1u].inputs.begin(), slices->stages[1u].inputs.end(),
          slices->stages[0u].outputs[0u]) != slices->stages[1u].inputs.end() &&
      slices->stages[1u].outputs.size() == 1u &&
      slices->stages[2u].inputs.size() == 2u &&
      std::find(slices->stages[2u].inputs.begin(),
                slices->stages[2u].inputs.end(), slices->input_resources[2u]) !=
          slices->stages[2u].inputs.end() &&
      std::find(
          slices->stages[2u].inputs.begin(), slices->stages[2u].inputs.end(),
          slices->stages[1u].outputs[0u]) != slices->stages[2u].inputs.end() &&
      !fused && fused.reason() == Reason::ExpressionCapacity;
  if (!valid) {
    std::fprintf(stderr,
                 "Graph multi program slices=%u reason=%u stages=%zu "
                 "inputs=%zu terminal=%zu fused=%u/%u\n",
                 static_cast<unsigned>(static_cast<bool>(slices)),
                 static_cast<unsigned>(slices.reason()),
                 slices ? slices->stages.size() : 0u,
                 slices ? slices->input_resources.size() : 0u,
                 slices && slices->stages.size() > 2u
                     ? slices->stages[2u].inputs.size()
                     : 0u,
                 static_cast<unsigned>(static_cast<bool>(fused)),
                 static_cast<unsigned>(fused.reason()));
  }
  return valid;
}

} // namespace rund_node_test_virtual::product::graph_pointwise_multi
