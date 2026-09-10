#pragma once

#include "../internal.hpp"

#include "../../../device/residency/pool.hpp"
#include "../../../pipeline/local.hpp"
#include "../../../pipeline/residency/planner.hpp"

#include <cstdint>
#include <memory>

namespace rund::compute::detail::virtual_prepare_detail {

[[nodiscard]] Result<std::shared_ptr<VirtualPipelineState>>
prepare_residency_pipeline(const std::shared_ptr<ProgramState> &,
                           const std::shared_ptr<VirtualBufferState> &,
                           const std::shared_ptr<VirtualBufferState> &,
                           ResidencyConfig, const VirtualGeometry &,
                           const std::shared_ptr<ProgramState> &) noexcept;

[[nodiscard]] Result<std::shared_ptr<PipelineState>> prepare_virtual_bank(
    const std::shared_ptr<ProgramState> &,
    const std::shared_ptr<ProgramState> &,
    const std::shared_ptr<residency::ResidencyPlan> &,
    const std::shared_ptr<residency::Pool> &, std::uint64_t logical_bytes,
    std::uint64_t input_frame_bytes, std::uint64_t output_frame_bytes,
    std::uint64_t resident_bytes, std::uint32_t frames,
    std::uint32_t bank) noexcept;

} // namespace rund::compute::detail::virtual_prepare_detail
