#pragma once

#include "emit.hpp"

namespace rund::node::accel::detail::recurrence_source_detail {

using rund::kernel::LoweringArtifact;

[[nodiscard]] RecurrenceSourceRecipe
BuildRecipe(const LoweringArtifact &, std::uint64_t, std::uint64_t,
            std::span<const std::uint64_t>) noexcept;

[[nodiscard]] MapRecurrenceSourcePlan
PlanRecipe(const LoweringArtifact &, const RecurrenceSourceRecipe &) noexcept;

} // namespace rund::node::accel::detail::recurrence_source_detail
