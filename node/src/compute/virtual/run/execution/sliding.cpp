#include "../execution.hpp"

#include "../../../backend.hpp"
#include "../../../pipeline/state.hpp"

namespace rund::compute::detail {

Status prepare_virtual_execution_sliding(
    VirtualPipelineState &state, const VirtualRunProjection &run,
    VirtualExecutionSlidingPrepared &prepared) noexcept {
  prepared.reset();
  const DeviceOps *const ops =
      state.pipeline == nullptr || state.pipeline->device == nullptr
          ? nullptr
          : state.pipeline->device->ops;
  return ops == nullptr ||
                 ops->virtual_execution.prepare_virtual_sliding_product ==
                     nullptr
             ? Status::fail(Reason::BackendUnsupported)
             : ops->virtual_execution.prepare_virtual_sliding_product(
                   state, run, prepared);
}

VirtualExecutionResult execute_virtual_execution_sliding(
    VirtualPipelineState &state, VirtualBacking &input, VirtualBacking &output,
    const VirtualRunProjection &run,
    const VirtualExecutionSlidingPrepared &prepared, Stats &stats) noexcept {
  const DeviceOps *const ops =
      state.pipeline == nullptr || state.pipeline->device == nullptr
          ? nullptr
          : state.pipeline->device->ops;
  return ops == nullptr ||
                 ops->virtual_execution.execute_virtual_sliding_product ==
                     nullptr
             ? VirtualExecutionResult{.status = Status::fail(
                                          Reason::BackendUnsupported)}
             : ops->virtual_execution.execute_virtual_sliding_product(
                   state, input, output, run, prepared, stats);
}

} // namespace rund::compute::detail
