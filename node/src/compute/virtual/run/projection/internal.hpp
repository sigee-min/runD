#pragma once

#include "../projection.hpp"

namespace rund::compute::detail {

[[nodiscard]] bool
project_virtual_graph_run(VirtualPipelineState &state,
                          std::uint64_t active_count,
                          VirtualRunProjection &projection) noexcept;

[[nodiscard]] bool
project_virtual_multi_run(VirtualPipelineState &state,
                          std::uint64_t active_count,
                          VirtualRunProjection &projection) noexcept;

[[nodiscard]] bool
project_virtual_ordinary_run(VirtualPipelineState &state,
                             std::uint64_t active_count,
                             VirtualRunProjection &projection) noexcept;

[[nodiscard]] bool virtual_valid_regions(
    const residency::Pool &pool,
    const std::array<residency::FrameRegion, residency::Pool::BankCount>
        &regions,
    residency::FrameTier tier, residency::FrameRole role,
    std::uint32_t capacity) noexcept;

} // namespace rund::compute::detail
