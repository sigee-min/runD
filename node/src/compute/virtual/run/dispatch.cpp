#include "dispatch/local.hpp"

namespace rund::compute::detail {

VirtualRunDispatchResult dispatch_virtual_route(
    VirtualPipelineState &state, const std::span<VirtualBacking *const> inputs,
    VirtualBacking &input, VirtualBacking &output, VirtualRunProjection &run,
    const VirtualDeviceVsmCandidate &candidate,
    const std::uint64_t active_count, Stats &stats, VirtualRunWork &work,
    VirtualRunTransaction &transaction, VirtualRunResources &resources,
    bool &poison_pipeline) noexcept {
  if (run.poolless_device_vsm()) {
    return virtual_run_dispatch::dispatch_poolless(
        state, inputs, output, run, candidate, active_count, stats, resources,
        poison_pipeline);
  }
  return virtual_run_dispatch::dispatch_pooled(
      state, inputs, input, output, run, candidate, active_count, stats, work,
      transaction, resources, poison_pipeline);
}

VirtualRunDispatchResult dispatch_run(
    VirtualPipelineState &state, const std::span<VirtualBacking *const> inputs,
    VirtualBacking &input, VirtualBacking &output,
    const VirtualRunProjection &run, const VirtualRunAdmission &admission,
    Stats &stats, VirtualRunWork &work,
    VirtualRunTransaction *const transaction) noexcept {
  const Status prepared = prepare_work(run, work);
  if (!prepared) {
    return VirtualRunDispatchResult{
        .status = prepared,
        .direct_terminal = true,
        .certainty = VirtualRunWriteCertainty::KnownNoWrite,
    };
  }
  if (run.graph_execution()) {
    return virtual_run_dispatch::dispatch_graph(state, inputs, output, run,
                                                stats, work);
  }
  bool handled = false;
  VirtualRunDispatchResult accelerated =
      virtual_run_dispatch::dispatch_accelerator(state, input, output, run,
                                                 admission, stats, handled);
  if (handled) {
    return accelerated;
  }
  return virtual_run_dispatch::dispatch_epochs(state, input, output, run, stats,
                                               work, transaction);
}

} // namespace rund::compute::detail
