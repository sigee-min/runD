#include "internal.hpp"

#include "../graph_pointwise_shape/stage.hpp"

#include "src/compute/flow/state.hpp"
#include "src/compute/graph/compile/slice.hpp"
#include "src/compute/graph/compile/slice/semantic.hpp"

#include <cstdio>

namespace rund_node_test_virtual::product::graph_pointwise_depth_seven {

rund::compute::Result<Program>
build_program(const rund::compute::Device &device) {
  using graph_pointwise_shape::add_stage;
  return rund::compute::on(device)
      .input<std::uint64_t>(FrameElements)
      .zip_input<std::uint64_t>(FrameElements)
      .map("virtual-graph-pointwise-depth-seven-input",
           [](auto a, auto b) {
             return add_stage<1u, InputLeafCount>(a) +
                    add_stage<InputLeafCount + 1u, InputLeafCount>(b);
           })
      .map("virtual-graph-pointwise-depth-seven-a",
           [](auto value) {
             return add_stage<1u, StageLeafCount>(value) ^ 0x55ull;
           })
      .map("virtual-graph-pointwise-depth-seven-b",
           [](auto value) {
             return add_stage<StageLeafCount + 1u, StageLeafCount>(value) *
                    3ull;
           })
      .map(
          "virtual-graph-pointwise-depth-seven-c",
          [](auto value) {
            return (add_stage<2u * StageLeafCount + 1u, StageLeafCount>(value) ^
                    0xa5ull) *
                       5ull +
                   7ull;
          })
      .map("virtual-graph-pointwise-depth-seven-d",
           [](auto value) {
             return add_stage<3u * StageLeafCount + 1u, StageLeafCount>(value) |
                    0x11ull;
           })
      .map(
          "virtual-graph-pointwise-depth-seven-e",
          [](auto value) {
            return (add_stage<4u * StageLeafCount + 1u, StageLeafCount>(value) +
                    0x33ull) *
                       7ull ^
                   0x5aull;
          })
      .map(
          "virtual-graph-pointwise-depth-seven-output",
          [](auto value) {
            return (add_stage<5u * StageLeafCount + 1u, StageLeafCount>(value) ^
                    0xc3ull) *
                       9ull +
                   13ull;
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
  bool valid = slices && slices->stages.size() == StageCount &&
               slices->resources.size() == InputCount + StageCount &&
               slices->input_resources.size() == InputCount &&
               slices->stages[0u].inputs == slices->input_resources && !fused &&
               fused.reason() == Reason::ExpressionCapacity;
  for (std::size_t stage = 1u; valid && stage < StageCount; ++stage) {
    valid = slices->stages[stage].inputs.size() == 1u;
  }
  if (!valid) {
    std::fprintf(stderr,
                 "Graph depth-seven program slices=%u/%u stages=%zu "
                 "resources=%zu inputs=%zu fused=%u/%u\n",
                 static_cast<unsigned>(static_cast<bool>(slices)),
                 static_cast<unsigned>(slices.reason()),
                 slices ? slices->stages.size() : 0u,
                 slices ? slices->resources.size() : 0u,
                 slices ? slices->input_resources.size() : 0u,
                 static_cast<unsigned>(static_cast<bool>(fused)),
                 static_cast<unsigned>(fused.reason()));
  }
  return valid;
}

} // namespace rund_node_test_virtual::product::graph_pointwise_depth_seven
