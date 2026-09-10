#include "local.hpp"

namespace rund::compute::detail::virtual_run_dispatch {

VirtualRunDispatchResult dispatch_poolless(
    VirtualPipelineState &state, const std::span<VirtualBacking *const> inputs,
    VirtualBacking &output, const VirtualRunProjection &run,
    const VirtualDeviceVsmCandidate &candidate,
    const std::uint64_t active_count, Stats &stats,
    VirtualRunResources &resources, bool &poison_pipeline) noexcept {
  if (active_count == 0u) {
    return VirtualRunDispatchResult{.status = Status::success(),
                                    .direct_terminal = true,
                                    .empty = true,
                                    .reduction_pending = run.reduction};
  }
  if (!candidate.device_vsm()) {
    return VirtualRunDispatchResult{
        .status = Status::fail(Reason::BackendUnsupported),
        .selected = false,
        .direct_terminal = run.device_vsm_required,
        .certainty = VirtualRunWriteCertainty::KnownNoWrite,
    };
  }
  VirtualDeviceVsmPostStage post =
      prepare_virtual_device_vsm_route(state, run, candidate);
  if (!post.ready()) {
    return dispose_device_vsm_route(state, resources, post,
                                    VirtualDeviceVsmScope::Poolless,
                                    post.status, poison_pipeline);
  }
  return execute_device_vsm_stage(state, inputs, output, run, post,
                                  VirtualDeviceVsmScope::Poolless, stats,
                                  resources, poison_pipeline);
}

} // namespace rund::compute::detail::virtual_run_dispatch
