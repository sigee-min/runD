#pragma once

#include "../../interface/api.hpp"
#include "../../model.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::node::accel::detail::prepared_pipeline_limit {

struct State final {
  PreparedKernelPipelineReservation result{};
  const BackendOps *ops{};
  std::uint64_t entry_count{};
  std::uint64_t stream_count{};
  PreparedKernelPipelineReservation backend_projection{};
};

[[nodiscard]] bool plan_route(
    const rund::AccelContext &context,
    std::span<const PreparedKernelProgramRoute> routes,
    std::size_t route_index, PreparedKernelPipelineShape shape,
    State &state) noexcept;

[[nodiscard]] bool finalize(const rund::AccelContext &context,
                            PreparedKernelPipelineShape shape,
                            State &state) noexcept;

} // namespace rund::node::accel::detail::prepared_pipeline_limit
