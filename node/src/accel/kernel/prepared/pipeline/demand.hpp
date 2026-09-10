#pragma once

#include "../run.hpp"

#include "../model.hpp"

#include <cstdint>
#include <span>

namespace rund::node::accel::detail {

[[nodiscard]] bool backend_template_route_demand(
    std::uint64_t owner_count, std::uint64_t route_copies,
    BackendTemplateRouteDemand &demand) noexcept;

[[nodiscard]] bool plan_backend_template_route_demands(
    std::span<const PreparedKernelRun *const> runs, std::uint32_t route_copies,
    std::span<BackendTemplateRouteDemand> demands,
    std::uint64_t &unique_route_count, std::uint64_t &template_count) noexcept;

} // namespace rund::node::accel::detail
