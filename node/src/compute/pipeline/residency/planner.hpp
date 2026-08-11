#pragma once

#include "model.hpp"

namespace rund::compute::detail::residency {

[[nodiscard]] PlanResult PlanResidency(const StreamPlanInput &input) noexcept;
[[nodiscard]] PlanResult PlanResidency(const GraphPlanInput &input) noexcept;

} // namespace rund::compute::detail::residency
