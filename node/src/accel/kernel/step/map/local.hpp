#pragma once

#include <accel/check.hpp>
#include <accel/context/value.hpp>

#include "../../../context/shared.hpp"
#include "../../bindings.hpp"
#include "../../plan.hpp"
#include "../../windows/planning.hpp"
#include <kernel/program/compute/binding/model.hpp>
#include <kernel/program/compute/metadata.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] const rund::kernel::ExecutionMetadata &
MapMetadata(const KernelExecutionStep &step) noexcept;

[[nodiscard]] rund::kernel::BindingSet
BindMapStep(const KernelExecutionStep &step, const PlannedStep &planned,
            const StepBinds &binds);

[[nodiscard]] rund::kernel::BindingValidation
ValidateMapStepBindings(const KernelExecutionStep &step,
                        const PlannedStep &planned,
                        const rund::kernel::BindingSet &bindings);

[[nodiscard]] rund::AccelCheck
ExecuteCpuMapStep(const rund::AccelDevice &pick, const PlannedStep &planned,
                  const DispatchWindowStorage &windows,
                  const rund::kernel::BindingSet &binds);

} // namespace rund::node::accel::detail
