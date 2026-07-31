#pragma once

#include <accel/kernel/run.hpp>

#include "../kind.hpp"
#include "local.hpp"

#include <kernel/program/compute/lowering/artifact/admission.hpp>

namespace rund::node::accel::detail {

using PlanBuilder = PlannedStep (*)(const KernelExecution &,
                                    const KernelExecutionStep &,
                                    const rund::AccelRun &, std::uint64_t);

inline void AdoptPassCount(PlannedStep &planned,
                           const std::uint64_t count) noexcept {
  if (planned.plan.ok) {
    planned.plan.dispatch_count = count;
  }
}

[[nodiscard]] inline std::uint64_t
PassCount(const rund::kernel::SortPlan &plan) noexcept {
  return plan.radix_pass_count;
}

template <typename Plan>
[[nodiscard]] std::uint64_t PassCount(const Plan &plan) noexcept {
  return plan.pass_count;
}

[[nodiscard]] inline PlannedStep BuildMapPlan(const KernelExecution &execution,
                                              const KernelExecutionStep &step,
                                              const rund::AccelRun &run,
                                              const std::uint64_t step_index) {
  PlannedStep planned{};
  planned.plan = PlanStep(execution, step, run, step_index);
  if (!planned.plan.ok) {
    return planned;
  }
  planned.artifact = &step.artifact;
  planned.cpu_input = step.artifact.key.api == rund::kernel::ComputeApi::Cpu
                          ? &step.cpu_input
                          : nullptr;
  if (!planned.artifact->ok) {
    planned.plan.ok = false;
    planned.plan.reason = planned.artifact->reason;
    return planned;
  }
  const auto retained = rund::kernel::compute_lowering_detail::AdmitRetained(
      planned.plan, *planned.artifact, planned.cpu_input);
  if (!retained.ok || retained.parse_count != 0u ||
      retained.emission_count != 0u) {
    planned.plan.ok = false;
    planned.plan.reason = retained.reason;
    return planned;
  }
  planned.windows = DispatchWindows(planned.plan);
  if (!planned.windows.ok) {
    planned.plan.ok = false;
    planned.plan.reason = planned.windows.reason;
  }
  return planned;
}

template <typename Active>
[[nodiscard]] PlannedStep
BuildPrimitivePlan(const KernelExecution &, const KernelExecutionStep &step,
                   const rund::AccelRun &, std::uint64_t) {
  const auto &plan = step.operation.get<Active>().plan;
  PlannedStep planned{};
  AdoptPlanStatus(planned, plan.ok, plan.reason);
  AdoptPassCount(planned, PassCount(plan));
  return planned;
}

inline constexpr GraphKindTable<PlanBuilder> kPlanBuilder{{
    nullptr,
    BuildMapPlan,
    BuildPrimitivePlan<operation::Scan>,
    BuildPrimitivePlan<operation::Sort>,
    BuildPrimitivePlan<operation::Compact>,
    nullptr,
    nullptr,
    nullptr,
    BuildPrimitivePlan<operation::Gather>,
    BuildPrimitivePlan<operation::Reduce>,
    BuildPrimitivePlan<operation::Scatter>,
    BuildPrimitivePlan<operation::Partition>,
    BuildPrimitivePlan<operation::SegmentedScan>,
    BuildPrimitivePlan<operation::Stencil>,
    BuildPrimitivePlan<operation::Histogram>,
    BuildPrimitivePlan<operation::SegmentedReduce>,
    BuildPrimitivePlan<operation::Transform>,
    BuildPrimitivePlan<operation::Matrix>,
    BuildPrimitivePlan<operation::Factor>,
    BuildPrimitivePlan<operation::Solve>,
    BuildPrimitivePlan<operation::Spectrum>,
    BuildPrimitivePlan<operation::ScatterReduce>,
}};

[[nodiscard]] inline PlanBuilder
PlanBuilderFor(const rund::kernel::NodeKind kind) noexcept {
  const std::size_t slot = GraphKindSlot(kind);
  return slot < kPlanBuilder.size() ? kPlanBuilder[slot] : nullptr;
}

} // namespace rund::node::accel::detail
