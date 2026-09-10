#pragma once

#include "execution.hpp"

namespace rund::compute::detail {

// Accelerator closure entrypoints installed in DeviceOps.  Keeping their
// definitions out of the common Virtual runner preserves the CPU source/link
// hardcut while the public product route remains backend-neutral.
[[nodiscard]] Status prepare_accel_virtual_execution_sliding(
    VirtualPipelineState &, const VirtualRunProjection &,
    VirtualExecutionSlidingPrepared &) noexcept;

[[nodiscard]] Status prepare_accel_virtual_execution_sliding_locked(
    VirtualPipelineState &, const VirtualRunProjection &,
    VirtualExecutionSlidingPrepared &) noexcept;

[[nodiscard]] VirtualExecutionResult execute_accel_virtual_execution_sliding(
    VirtualPipelineState &, VirtualBacking &, VirtualBacking &,
    const VirtualRunProjection &, const VirtualExecutionSlidingPrepared &,
    Stats &) noexcept;

} // namespace rund::compute::detail
