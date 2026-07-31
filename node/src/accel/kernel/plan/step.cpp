#include <accel/kernel/run.hpp>

#include "build.hpp"
#include "local.hpp"

namespace rund::node::accel::detail {

PlannedStep BuildPlannedStep(const KernelExecution &execution,
                             const KernelExecutionStep &step,
                             const rund::AccelRun &run,
                             const std::uint64_t step_index) {
  if (const PlanBuilder build = PlanBuilderFor(step.kind()); build != nullptr) {
    PlannedStep planned = build(execution, step, run, step_index);
    planned.domain = execution.admission.domain;
    return planned;
  }
  PlannedStep planned{};
  planned.domain = execution.admission.domain;
  planned.plan.ok = false;
  planned.plan.reason = "accel_kernel_primitive_unsupported";
  return planned;
}

} // namespace rund::node::accel::detail
