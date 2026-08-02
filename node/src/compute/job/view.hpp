#pragma once

#include "state.hpp"

#include <span>

namespace rund::compute::detail {

[[nodiscard]] Result<CpuViewTransferRequirements>
plan_cpu_view_transfer_requirements(
    const std::shared_ptr<ProgramState> &program) noexcept;
[[nodiscard]] Result<CpuViewTransferLayout> plan_cpu_view_transfers(
    const std::shared_ptr<ProgramState> &program,
    std::span<const JobBufferView> input_views,
    std::span<const JobBufferView> output_views,
    const CpuViewTransferRequirements *requirements = nullptr) noexcept;
[[nodiscard]] Status
prepare_cpu_view_transfers(JobState &state,
                           const CpuViewTransferLayout *layout = nullptr);

} // namespace rund::compute::detail
