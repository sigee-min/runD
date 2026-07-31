#pragma once

#include <accel/kernel/run.hpp>

#include <kernel/program/phase.hpp>

#include "../plan.hpp"
namespace rund::node::accel::detail {

[[nodiscard]] const rund::kernel::ExecutionMetadata &
StepMetadata(const KernelExecutionStep &step) noexcept;

[[nodiscard]] rund::kernel::TilePhaseDescription
PhaseFor(const KernelExecution &execution, const KernelExecutionStep &step,
         const rund::AccelRun &run, std::uint64_t step_index);

[[nodiscard]] rund::kernel::ComputePlan
PlanStep(const KernelExecution &execution, const KernelExecutionStep &step,
         const rund::AccelRun &run, std::uint64_t step_index);

void AdoptPlanStatus(PlannedStep &planned, bool ok,
                     const char *reason) noexcept;

} // namespace rund::node::accel::detail
