#pragma once

#include "model.hpp"

namespace rund::compute::detail::residency {

[[nodiscard]] PlanResult PlanResidency(const PlanInput &input) noexcept;

} // namespace rund::compute::detail::residency
