#pragma once

#include "execution.hpp"

namespace rund::compute::detail {

struct VirtualVsmAsyncOps;

[[nodiscard]] const VirtualVsmAsyncOps &
accel_virtual_vsm_async_ops() noexcept;

[[nodiscard]] Status prepare_accel_virtual_execution_device_vsm(
    VirtualPipelineState &, const VirtualRunProjection &,
    VirtualDeviceVsmRouteProof,
    VirtualExecutionDeviceVsmPrepared &) noexcept;

[[nodiscard]] VirtualExecutionResult execute_accel_virtual_execution_device_vsm(
    VirtualPipelineState &, std::span<VirtualBacking *const>, VirtualBacking &,
    const VirtualRunProjection &, const VirtualExecutionDeviceVsmPrepared &,
    Stats &) noexcept;

} // namespace rund::compute::detail
