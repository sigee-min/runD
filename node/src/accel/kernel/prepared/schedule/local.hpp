#pragma once

#include "../interface/api.hpp"
#include "../model.hpp"

#include <array>
#include <cstddef>

namespace rund::node::accel::detail::prepared {
struct PipelineState;
}

namespace rund::node::accel::detail {

[[nodiscard]] prepared::PipelineState *
PreparedScheduleState(const PreparedKernelPipeline &pipeline) noexcept;
[[nodiscard]] std::size_t PreparedScheduleSortedStates(
    const PreparedResidencyScheduleRequest &request,
    std::array<prepared::PipelineState *, ResidencyScheduleRoleCapacity>
        &states) noexcept;
[[nodiscard]] bool PreparedScheduleValidRequest(
    const rund::AccelContext &context,
    const PreparedResidencyScheduleRequest &request, const BackendOps *&ops,
    std::array<prepared::PipelineState *, ResidencyScheduleRoleCapacity>
        &states,
    std::size_t &state_count) noexcept;
void CompletePreparedScheduleRelease(
    void *raw, BackendResidencyScheduleRelease &&release) noexcept;
void CompletePreparedScheduleFinal(
    void *raw, BackendResidencyScheduleFinal &&final) noexcept;

} // namespace rund::node::accel::detail
