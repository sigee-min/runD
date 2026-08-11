#pragma once

#include "../state.hpp"

#include <memory>

namespace rund::compute::detail {

void append_pipeline_residency(
    const std::shared_ptr<PipelineBuildState> &build,
    const std::shared_ptr<ProgramState> &program,
    std::shared_ptr<const residency::ResidencyPlan> pages,
    std::shared_ptr<residency::Pool> pool, std::uint64_t logical_bytes,
    std::uint64_t input_page_bytes, std::uint64_t output_page_bytes,
    std::uint64_t resident_bytes, std::uint32_t frame_count) noexcept;

[[nodiscard]] Status bind_pipeline_residency(const PipelineMemoryPlan &plan,
                                             PipelineState &state) noexcept;

} // namespace rund::compute::detail
