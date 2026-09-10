#include "local.hpp"

#include "../../graph/reduce.hpp"

namespace rund::compute::detail::virtual_run_dispatch {

VirtualRunDispatchResult
dispatch_graph(VirtualPipelineState &state,
               const std::span<VirtualBacking *const> inputs,
               VirtualBacking &output, const VirtualRunProjection &run,
               Stats &stats, VirtualRunWork &work) noexcept {
  const bool cpu = state.pipeline != nullptr &&
                   state.pipeline->device != nullptr &&
                   state.pipeline->device->backend == Backend::Cpu;
  if (cpu && !cpu_graph_ready(state)) {
    return VirtualRunDispatchResult{
        .status = Status::fail(Reason::PipelineBusy),
        .graph_execution = true,
        .certainty = VirtualRunWriteCertainty::KnownNoWrite};
  }
  const VirtualGraphResult graph =
      run.graph_reduction
          ? execute_virtual_graph_reduction(state, inputs, run, stats,
                                            work.reduction)
          : execute_virtual_graph_pointwise(state, inputs, output, run, stats);
  if (!graph.status) {
    return VirtualRunDispatchResult{
        .status = graph.status,
        .failed_page = graph.failed_page,
        .output_hash = graph.output_hash,
        .poison_pipeline = graph.poison_pipeline,
        .graph_execution = true,
        .reduction_pending = run.graph_reduction,
        .certainty = epoch_certainty(graph.poison_pipeline),
    };
  }
  return VirtualRunDispatchResult{
      .status = Status::success(),
      .output_hash = graph.output_hash,
      .direct_terminal = true,
      .graph_execution = true,
      .reduction_pending = run.graph_reduction,
  };
}

} // namespace rund::compute::detail::virtual_run_dispatch
