#include "local.hpp"


namespace rund::compute::detail {

VirtualRunDispatchResult dispose_device_vsm_route(
    VirtualPipelineState &state, VirtualRunResources &resources,
    VirtualDeviceVsmPostStage &post, const VirtualDeviceVsmScope scope,
    const Status failure, bool &poison_pipeline) noexcept {
  VirtualRunWriteCertainty certainty = VirtualRunWriteCertainty::KnownNoWrite;
  const Status disposed = dispose_virtual_device_vsm_route(
      state, resources, post, scope, failure, poison_pipeline, certainty);
  return VirtualRunDispatchResult{
      .status = disposed,
      .poison_pipeline = poison_pipeline,
      .selected = true,
      .direct_terminal = true,
      .certainty = certainty,
  };
}

VirtualRunDispatchResult execute_device_vsm_stage(
    VirtualPipelineState &state, const std::span<VirtualBacking *const> inputs,
    VirtualBacking &output, const VirtualRunProjection &run,
    VirtualDeviceVsmPostStage &post, const VirtualDeviceVsmScope scope,
    Stats &stats, VirtualRunResources &resources,
    bool &poison_pipeline) noexcept {
  if (scope == VirtualDeviceVsmScope::Pooled) {
    const Status closed = close_pool_stage(resources);
    if (!closed) {
      return dispose_device_vsm_route(state, resources, post, scope, closed,
                                      poison_pipeline);
    }
  }
  const Status rearmed = rearm_virtual_device_vsm_route(post);
  if (!rearmed) {
    return dispose_device_vsm_route(state, resources, post, scope, rearmed,
                                    poison_pipeline);
  }
  return dispatch_result(execute_virtual_device_vsm_route(
      state, inputs, output, run, post, stats));
}

} // namespace rund::compute::detail
