#pragma once

#include "../state.hpp"
#include "../state/assembly.hpp"

#include <memory>
#include <span>

namespace rund::compute::detail {

void append_pipeline_residency(
    const std::shared_ptr<PipelineBuildState> &build,
    const std::shared_ptr<ProgramState> &program,
    std::shared_ptr<const residency::ResidencyPlan> pages,
    std::shared_ptr<residency::Pool> pool, std::uint64_t logical_bytes,
    std::uint64_t input_page_bytes, std::uint64_t output_page_bytes,
    std::uint64_t resident_bytes, std::uint32_t frame_count,
    std::uint32_t bank) noexcept;

void append_pipeline_graph_residency(
    const std::shared_ptr<PipelineBuildState> &build,
    const std::shared_ptr<ProgramState> &program,
    std::shared_ptr<const residency::ResidencyPlan> pages,
    std::shared_ptr<residency::Pool> pool, std::uint64_t logical_bytes,
    std::uint64_t resident_bytes, std::uint32_t frame_count, std::uint32_t bank,
    std::uint32_t graph_stage) noexcept;

void append_pipeline_graph_semantic_residency(
    const std::shared_ptr<PipelineBuildState> &build,
    const std::shared_ptr<ProgramState> &program,
    std::shared_ptr<const residency::ResidencyPlan> pages,
    std::shared_ptr<residency::Pool> pool, std::uint64_t logical_bytes,
    std::uint64_t resident_bytes, std::uint32_t frame_count, std::uint32_t bank,
    std::uint32_t graph_stage, PipelineResidencySemantic semantic,
    std::span<const std::uint32_t> input_resources,
    std::uint32_t output_resource) noexcept;

[[nodiscard]] Status bind_pipeline_residency(const PipelineMemoryPlan &plan,
                                             PipelineState &state) noexcept;

} // namespace rund::compute::detail
