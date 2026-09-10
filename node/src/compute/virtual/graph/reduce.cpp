#include "reduce.hpp"

#include "reduce/internal.hpp"

namespace rund::compute::detail {

VirtualGraphResult
execute_virtual_graph_reduction(VirtualPipelineState &state,
                                const std::span<VirtualBacking *const> inputs,
                                const VirtualRunProjection &run, Stats &stats,
                                VirtualReduction &reduction) noexcept {
  return graph_reduce::execute_tiled_graph(state, inputs, nullptr, run, stats,
                                           &reduction);
}

VirtualGraphResult execute_virtual_graph_pointwise(
    VirtualPipelineState &state, const std::span<VirtualBacking *const> inputs,
    VirtualBacking &output, const VirtualRunProjection &run,
    Stats &stats) noexcept {
  return graph_reduce::execute_tiled_graph(state, inputs, &output, run, stats,
                                           nullptr);
}

} // namespace rund::compute::detail
