#pragma once

#include <accel/context/value.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>

#include <node/accel/context.hpp>

#include "../context/internal.hpp"
#include "backend/run.hpp"
#include "bindings.hpp"
#include "plan.hpp"
#include <cstdint>

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelEvidence ExecuteKernelSteps(
    const rund::AccelContext &context, const KernelExecution &execution,
    const rund::AccelRun &run, const RunBinds &run_binds,
    const BoundResets &resets,
    const PlannedStepStorage &planned_steps,
    std::uint64_t original_dispatch_count, std::uint64_t final_dispatch_count);

} // namespace rund::node::accel::detail
