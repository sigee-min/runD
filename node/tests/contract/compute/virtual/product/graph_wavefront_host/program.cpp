#include "internal.hpp"

#include "src/compute/flow/state.hpp"
#include "src/compute/graph/compile/slice.hpp"

namespace rund_node_test_virtual::product::graph_wavefront_host {
namespace {

template <std::uint64_t First, std::size_t Count, class Expression>
[[nodiscard]] constexpr auto add_stage(Expression value) {
  static_assert(Count != 0u);
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
            first.map("virtual-graph-host-wavefront-prefix", [](auto value) {
              return add_stage<1u, StageLeafCount>(value);
            });
        const auto left =
            second.map("virtual-graph-host-wavefront-left", [](auto value) {
              return add_stage<StageLeafCount + 1u, StageLeafCount>(value);
            });
        const auto right =
            third.map("virtual-graph-host-wavefront-right", [](auto value) {
              return add_stage<2u * StageLeafCount + 1u, StageLeafCount>(value);
            });
        return zip(prefix, left, right)
            .map("virtual-graph-host-wavefront-terminal",
                 [](auto first_value, auto left_value, auto right_value) {
                   return first_value + left_value + right_value;
                 });
      })
      .compile();
}

std::uint64_t stage_offset(const std::uint64_t first) noexcept {
  return static_cast<std::uint64_t>(StageLeafCount) *
             (static_cast<std::uint64_t>(StageLeafCount) + 1u) / 2u +
         (first - 1u) * static_cast<std::uint64_t>(StageLeafCount);
}

bool validate_program(const Program &program) noexcept {
  using namespace rund::compute;
  const auto slices =
      detail::graph_compile::compile_tiled_graph_pointwise_slices(
          detail::FlowAccess::state(program));
  return slices && slices->stages.size() == StageCount &&
         slices->input_resources.size() == InputCount &&
         slices->stages[0u].inputs.size() == 1u &&
         slices->stages[0u].inputs[0u] == slices->input_resources[0u] &&
         slices->stages[1u].inputs.size() == 1u &&
         slices->stages[1u].inputs[0u] == slices->input_resources[1u] &&
         slices->stages[2u].inputs.size() == 1u &&
         slices->stages[2u].inputs[0u] == slices->input_resources[2u] &&
         slices->stages[3u].inputs.size() == 3u;
}

} // namespace rund_node_test_virtual::product::graph_wavefront_host
