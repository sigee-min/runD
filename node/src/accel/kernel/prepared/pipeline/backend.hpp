#pragma once

#include "../run.hpp"

#include "../model.hpp"

#include <cstdint>
#include <span>

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelCheck finalize_pipeline_backend_structure(
    const rund::AccelContext &context, const BackendOps &ops,
    const PreparedKernelPipelineReservation &projection,
    std::uint64_t publication_count, std::uint64_t terminal_publication_count,
    std::uint64_t publication_command_count, std::uint64_t window_state_count,
    std::uint64_t window_descriptor_state_count,
    std::uint64_t profile_step_count, std::uint64_t profile_command_count,
    PreparedKernelPipelineReservation &result) noexcept;

[[nodiscard]] rund::AccelCheck plan_runtime_backend_structure(
    const rund::AccelContext &context,
    std::span<const PreparedKernelRun *const> runs,
    std::span<const BackendRecurrence> recurrences,
    std::span<const BackendPublish> publications, std::uint64_t profile_step_count,
    std::uint64_t profile_command_count,
    PreparedKernelPipelineReservation &structure) noexcept;

} // namespace rund::node::accel::detail
