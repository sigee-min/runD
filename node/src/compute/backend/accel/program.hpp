#pragma once

#include "../../backend.hpp"

namespace rund::compute::detail::accel_backend {

[[nodiscard]] Status compile(DeviceState &device, AccelProgram &program,
                             const rund::AccelGraph &graph);

[[nodiscard]] RangeSnapshot program_ranges(
    const AccelProgram &program, std::span<RangeInfo> rows) noexcept;

[[nodiscard]] node::accel::detail::KernelScratchPlan plan_scratch(
    const DeviceState &device, const rund::AccelKernel &kernel,
    std::uint64_t alignment, std::uint64_t page_bytes);

} // namespace rund::compute::detail::accel_backend
