#include "../route.hpp"
#include "../../../../backend.hpp"

#include "../internal.hpp"

namespace rund::compute::detail {

VirtualExecutionResult execute_virtual_device_vsm_route(
    VirtualPipelineState &state, const std::span<VirtualBacking *const> inputs,
    VirtualBacking &output, const VirtualRunProjection &run,
    VirtualDeviceVsmPostStage &prepared, Stats &stats) noexcept {
  const DeviceOps *const ops =
      state.pipeline == nullptr || state.pipeline->device == nullptr
          ? nullptr
          : state.pipeline->device->ops;
  if (ops == nullptr ||
      ops->virtual_execution.execute_virtual_device_vsm_product == nullptr) {
    return VirtualExecutionResult{
        .status = Status::fail(Reason::PipelineInvalid),
        .poison_pipeline = true,
        .certainty = VirtualRunWriteCertainty::KnownNoWrite,
    };
  }
  if (!prepared.consume()) {
    const Status status = prepared.status
                              ? Status::fail(Reason::PipelineInvalid)
                              : prepared.status;
    return VirtualExecutionResult{
        .status = status,
        .poison_pipeline = status.reason() == Reason::DeviceLost,
        .certainty = VirtualRunWriteCertainty::KnownNoWrite,
    };
  }
  return ops->virtual_execution.execute_virtual_device_vsm_product(
      state, inputs, output, run, prepared.prepared, stats);
}

} // namespace rund::compute::detail
