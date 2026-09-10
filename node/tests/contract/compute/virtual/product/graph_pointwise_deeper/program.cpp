#include "internal.hpp"

#include "src/compute/flow/state.hpp"
#include "src/compute/graph/compile/slice.hpp"
#include "src/compute/graph/compile/slice/semantic.hpp"

#include <cstdio>

namespace rund_node_test_virtual::product::graph_pointwise_deeper {
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
      .zip_input<std::uint64_t>(FrameElements)
      .zip_input<std::uint64_t>(FrameElements)
      .map("virtual-graph-pointwise-deeper-input",
           [](auto a, auto b, auto c, auto d, auto e) {
             return add_stage<1u, InputLeafCount>(a) +
                    add_stage<InputLeafCount + 1u, InputLeafCount>(b) +
                    add_stage<2u * InputLeafCount + 1u, InputLeafCount>(c) +
                    add_stage<3u * InputLeafCount + 1u, InputLeafCount>(d) +
                    add_stage<4u * InputLeafCount + 1u, InputLeafCount>(e);
           })
      .map("virtual-graph-pointwise-deeper-middle-a",
           [](auto value) {
             return add_stage<1u, StageLeafCount>(value) ^ 0x55ull;
           })
      .map("virtual-graph-pointwise-deeper-middle-b",
           [](auto value) {
             return add_stage<StageLeafCount + 1u, StageLeafCount>(value) *
                    3ull;
           })
      .map(
          "virtual-graph-pointwise-deeper-output",
          [](auto value) {
            return (add_stage<2u * StageLeafCount + 1u, StageLeafCount>(value) ^
                    0xa5ull) *
                       5ull +
                   7ull;
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
                 "Graph deeper program slices=%u/%u stages=%zu resources=%zu "
                 "inputs=%zu fused=%u/%u\n",
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

} // namespace rund_node_test_virtual::product::graph_pointwise_deeper
