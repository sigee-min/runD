#pragma once

#include <kernel/program/compute/binding/model.hpp>
#include <kernel/program/compute/metadata.hpp>
#include <kernel/program/compute/model.hpp>

namespace rund::node::accel::detail {

enum class PlanBindingInputMode {
  StagedOrResident,
  ResidentInputs,
  MapResidentViews,
};

[[nodiscard]] bool
ComputePlanHeaderValid(const rund::kernel::ComputePlan &plan,
                       rund::kernel::ComputeApi expected_api) noexcept;

[[nodiscard]] bool
FrozenCapsAdmitPlan(const rund::kernel::ComputeCaps &caps,
                    const rund::kernel::ComputePlan &plan) noexcept;

[[nodiscard]] rund::kernel::BindingValidation
BindingPlanCheck(const rund::kernel::ComputePlan &plan,
                 const rund::kernel::BindingSet &bindings,
                 const rund::kernel::ExecutionMetadata &metadata,
                 PlanBindingInputMode input_mode) noexcept;

} // namespace rund::node::accel::detail
