#pragma once

#include <accel/kernel/run.hpp>

#include "../plan.hpp"
#include "bindings.hpp"

#include <cstdint>
#include <utility>

namespace rund::node::accel::detail {

struct RunDispatchBuild {
  PlannedStepStorage planned_steps{};
  std::uint64_t original_dispatch_count = 0u;
  std::uint64_t final_dispatch_count = 0u;
  bool ok = false;
  const char *reason = "accel_kernel_run_invalid";
};

[[nodiscard]] inline RunDispatchBuild
BuildRunDispatch(const KernelExecution &execution, const rund::AccelRun &run) {
  RunDispatchBuild result{};

  result.planned_steps.reserve(execution.steps.size());
  DispatchCount final{};
  for (std::size_t index = 0u; index < execution.steps.size(); ++index) {
    PlannedStep planned =
        BuildPlannedStep(execution, execution.steps[index], run,
                         static_cast<std::uint64_t>(index));
    if (!planned.plan.ok) {
      result.reason = planned.plan.reason;
      return result;
    }
    if (!AddDispatchCount(final, planned.plan.dispatch_count)) {
      result.reason = final.reason;
      return result;
    }
    result.planned_steps.push_back(std::move(planned), execution.steps.size());
  }
  if (!result.planned_steps.valid()) {
    return result;
  }
  result.final_dispatch_count = final.count;
  DispatchCount original = final;
  if (!AddDispatchCount(original, execution.removed_dispatch_count)) {
    result.reason = original.reason;
    return result;
  }
  result.original_dispatch_count = original.count;
  result.ok = true;
  return result;
}

} // namespace rund::node::accel::detail
