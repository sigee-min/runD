#pragma once

#include <rund/compute/abi/model.hpp>
#include <rund/compute/pipeline/memory.hpp>
#include <rund/compute/status.hpp>
#include <rund/compute/telemetry.hpp>

#include <cstdint>
#include <memory>

namespace rund::compute {
class VirtualBacking;

namespace detail {

[[nodiscard]] Result<std::shared_ptr<VirtualBufferState>>
make_virtual_buffer(std::uint64_t count, std::uint64_t element_bytes, Type type,
                    FixedFormat format,
                    std::shared_ptr<VirtualBacking> backing) noexcept;
[[nodiscard]] bool
valid_virtual_buffer(const std::shared_ptr<VirtualBufferState> &state) noexcept;
[[nodiscard]] std::uint64_t
virtual_buffer_size(const std::shared_ptr<VirtualBufferState> &state) noexcept;

[[nodiscard]] Result<std::shared_ptr<VirtualPipelineState>>
prepare_virtual_pipeline(const std::shared_ptr<ProgramState> &program,
                         const std::shared_ptr<VirtualBufferState> &input,
                         const std::shared_ptr<VirtualBufferState> &output,
                         std::uint32_t slots) noexcept;
[[nodiscard]] bool valid_virtual_pipeline(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept;
[[nodiscard]] Status run_virtual_pipeline(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept;
[[nodiscard]] Status
run_virtual_pipeline(const std::shared_ptr<VirtualPipelineState> &state,
                     std::uint64_t active_count) noexcept;
[[nodiscard]] Status begin_virtual_pipeline_samples(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept;
[[nodiscard]] Status end_virtual_pipeline_samples(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept;
[[nodiscard]] Stats virtual_pipeline_stats(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept;
[[nodiscard]] MemoryStats virtual_pipeline_memory(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept;
[[nodiscard]] PipelinePlan virtual_pipeline_plan(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept;
[[nodiscard]] Result<telemetry::Profile> virtual_pipeline_profile(
    const std::shared_ptr<VirtualPipelineState> &state) noexcept;

} // namespace detail
} // namespace rund::compute
