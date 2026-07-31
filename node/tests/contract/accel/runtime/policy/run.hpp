#pragma once

#include <accel/runtime.hpp>

#include <accel/run/policy/choose.hpp>

#include "reject.hpp"
#include "stats.hpp"

namespace node_accel_contract {

[[nodiscard]] bool RuntimePolicyChoosesOnlyFromLocalEvidence() {
  const rund::kernel::ComputePlan plan = policy_case::RuntimePolicyPlan();
  const rund::kernel::ComputePlan cpu_plan =
      policy_case::RuntimePolicyPlanFor(rund::kernel::ComputeApi::Cpu);
  const rund::kernel::ComputePlan vulkan_plan =
      policy_case::RuntimePolicyPlanFor(rund::kernel::ComputeApi::Vulkan);
  if (!plan.ok || plan.tile_count != 8u || plan.dispatch_count == 0u ||
      plan.dispatch_window_tiles == 0u || !plan.fixed_authoritative) {
    return false;
  }
  if (!cpu_plan.ok || !vulkan_plan.ok) {
    return false;
  }

  const rund::RuntimeStats evidence{
      .dispatch_count = 1u, .ok = true, .reason = "ok"};
  const rund::RuntimeStats evidence_before = evidence;
  const rund::kernel::ComputePlan plan_before = plan;

  rund::node::accel::RunPolicy policy{};
  rund::node::accel::RunChoice choice{};
  if (!policy_case::MalformedPlansReject(plan, evidence)) {
    return false;
  }

  rund::RuntimeStats missing = evidence;
  missing.ok = false;
  choice = rund::node::accel::ChooseRun(plan, missing, policy);
  if (choice.use_accel ||
      !policy_case::ReasonIs(choice.reason, "accel_run_evidence_missing")) {
    return false;
  }

  rund::RuntimeStats empty{.ok = true, .reason = "ok"};
  choice = rund::node::accel::ChooseRun(plan, empty, policy);
  if (choice.use_accel ||
      !policy_case::ReasonIs(choice.reason, "accel_run_evidence_missing")) {
    return false;
  }

  policy.staged = false;
  choice = rund::node::accel::ChooseRun(plan, evidence, policy);
  if (choice.use_accel ||
      !policy_case::ReasonIs(choice.reason, "accel_run_requires_resident")) {
    return false;
  }

  policy = {};
  choice = rund::node::accel::ChooseRun(plan, evidence, policy);
  const rund::node::accel::RunChoice cpu_choice =
      rund::node::accel::ChooseRun(cpu_plan, evidence, policy);
  const rund::node::accel::RunChoice vulkan_choice =
      rund::node::accel::ChooseRun(vulkan_plan, evidence, policy);
  return choice.use_accel && policy_case::ReasonIs(choice.reason, "ok") &&
         cpu_choice.use_accel &&
         policy_case::ReasonIs(cpu_choice.reason, "ok") &&
         vulkan_choice.use_accel &&
         policy_case::ReasonIs(vulkan_choice.reason, "ok") &&
         policy_case::RuntimeStatsEqual(evidence, evidence_before) &&
         policy_case::RuntimeStatsEqual(
             empty, rund::RuntimeStats{.ok = true, .reason = "ok"}) &&
         plan.tile_count == plan_before.tile_count &&
         plan.dispatch_count == plan_before.dispatch_count &&
         plan.dispatch_window_tiles == plan_before.dispatch_window_tiles &&
         plan.fixed_authoritative == plan_before.fixed_authoritative &&
         plan.ok == plan_before.ok &&
         std::string_view{plan.reason} == plan_before.reason;
}

} // namespace node_accel_contract
