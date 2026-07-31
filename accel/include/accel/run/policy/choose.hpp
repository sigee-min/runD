#pragma once

#include <accel/run/policy/evidence.hpp>
#include <accel/run/policy/value.hpp>
#include <kernel/program/compute/plan.hpp>

namespace rund::node::accel {

[[nodiscard]] inline RunChoice ChooseRun(const rund::kernel::ComputePlan &plan,
                                         const rund::RuntimeStats &last_stats,
                                         const RunPolicy &policy) noexcept {
  if (!rund::kernel::ComputePlanShapeValid(plan)) {
    return RunChoice{.reason = "accel_run_plan_invalid"};
  }
  if (!run_policy_detail::EvidenceOk(last_stats)) {
    return RunChoice{.reason = "accel_run_evidence_missing"};
  }
  if (!policy.staged) {
    return RunChoice{.reason = "accel_run_requires_resident"};
  }
  return RunChoice{.use_accel = true, .reason = "ok"};
}

} // namespace rund::node::accel
