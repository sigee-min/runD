#pragma once

#include "state.hpp"

#include <span>

namespace rund::compute::detail {

[[nodiscard]] Status validate_virtual_pipeline_program(
    const ProgramState &program,
    std::span<const std::shared_ptr<VirtualBufferState>> inputs,
    const VirtualBufferState &output) noexcept;

[[nodiscard]] Result<std::shared_ptr<VirtualPipelineState>>
prepare_virtual_multi_device_vsm(
    const std::shared_ptr<ProgramState> &program,
    std::span<const std::shared_ptr<VirtualBufferState>> inputs,
    const std::shared_ptr<VirtualBufferState> &output, ResidencyConfig config,
    const VirtualGeometry &geometry) noexcept;

[[nodiscard]] Result<std::shared_ptr<VirtualPipelineState>>
prepare_virtual_graph_reduction(
    const std::shared_ptr<ProgramState> &program,
    std::span<const std::shared_ptr<VirtualBufferState>> inputs,
    const std::shared_ptr<VirtualBufferState> &output, ResidencyConfig config,
    GraphPageMap page_map, const VirtualGeometry &geometry) noexcept;

[[nodiscard]] Result<std::shared_ptr<VirtualPipelineState>>
prepare_virtual_graph_pointwise(
    const std::shared_ptr<ProgramState> &program,
    std::span<const std::shared_ptr<VirtualBufferState>> inputs,
    const std::shared_ptr<VirtualBufferState> &output, ResidencyConfig config,
    GraphPageMap page_map, const VirtualGeometry &geometry) noexcept;

} // namespace rund::compute::detail
