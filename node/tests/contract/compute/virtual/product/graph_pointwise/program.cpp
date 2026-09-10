#include "internal.hpp"

#include "src/compute/flow/state.hpp"
#include "src/compute/graph/compile/slice.hpp"
#include "src/compute/graph/compile/slice/semantic.hpp"

namespace rund_node_test_virtual::product::graph_pointwise {
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
      .branch([](auto values) {
        const auto left =
            values.map("virtual-graph-pointwise-left", [](auto value) {
              return add_stage<1u, StageLeafCount>(value);
            });
        const auto right =
            values.map("virtual-graph-pointwise-right", [](auto value) {
              return add_stage<StageLeafCount + 1u, StageLeafCount>(value);
            });
        return zip(left, right)
            .map("virtual-graph-pointwise-join",
                 [](auto first, auto second) { return first + second; });
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
  return slices && slices->stages.size() == StageCount &&
         slices->stages[0u].inputs == slices->stages[1u].inputs &&
         slices->stages[2u].inputs.size() == 2u && !fused &&
         fused.reason() == Reason::ExpressionCapacity;
}

} // namespace rund_node_test_virtual::product::graph_pointwise
