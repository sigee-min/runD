#pragma once

#include <accel/kernel/run.hpp>

#include <node/accel/context.hpp>

#include "../context/internal.hpp"
#include "storage.hpp"
#include "windows/planning.hpp"
#include <cstddef>
#include <cstdint>
#include <kernel/core/checked.hpp>

namespace rund::node::accel::detail {

struct PlannedStep {
  rund::kernel::ComputeDomain domain = rund::kernel::ComputeDomain::Fixed;
  rund::kernel::ComputePlan plan{};
  const rund::kernel::LoweringArtifact *artifact = nullptr;
  const rund::kernel::compute_lowering_detail::ComputeInputAdmission
      *cpu_input = nullptr;
  DispatchWindowStorage windows{};
};

static_assert(sizeof(PlannedStep) <= 512u,
              "run plan exceeded its footprint budget");

struct DispatchCount {
  std::uint64_t count = 0u;
  bool ok = false;
  const char *reason = "compute_plan_invalid";
};

static constexpr std::size_t kInlinePlannedStepCapacity = 4u;
using PlannedStepStorage =
    InlineStepStorage<PlannedStep, kInlinePlannedStepCapacity>;

[[nodiscard]] inline bool AddDispatchCount(DispatchCount &result,
                                           const std::uint64_t count) noexcept {
  if (!rund::kernel::checked::add(result.count, count, result.count)) {
    result.reason = "compute_dispatch_overflow";
    return false;
  }
  return true;
}

[[nodiscard]] PlannedStep BuildPlannedStep(const KernelExecution &execution,
                                           const KernelExecutionStep &step,
                                           const rund::AccelRun &run,
                                           std::uint64_t step_index);

} // namespace rund::node::accel::detail
