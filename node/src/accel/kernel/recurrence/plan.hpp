#pragma once

#include "../recurrence.hpp"

namespace rund::node::accel::detail {

// Public preflight projection. `route` owns the exact pointer-free Program
// binding layout produced by Compute's sole memory planner.
[[nodiscard]] MapRecurrencePreparationPlan
PlanMapRecurrencePreparation(const KernelExecution &execution,
                             const PreparedKernelProgramRoute &route) noexcept;

// Private materialization projection. Group cardinalities come from the same
// authored recurrence marker fingerprint used by public preflight.
[[nodiscard]] MapRecurrencePreparationPlan
PlanMapRecurrencePreparation(const BackendRun &run, std::uint64_t group_count,
                             std::uint64_t history_group_count) noexcept;

// Immutable recurrence-template equivalence. Route handles, absolute buffer
// addresses, authored iteration count, and per-stream group state are not
// template identity. Source recipe, compute plan, binding specialization, and
// exact history pitch are.
[[nodiscard]] bool
SameMapRecurrenceTemplate(const MapRecurrencePreparationPlan &left,
                          const MapRecurrencePreparationPlan &right,
                          bool history) noexcept;

} // namespace rund::node::accel::detail
