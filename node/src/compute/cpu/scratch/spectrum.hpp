#pragma once

#include "plan.hpp"

namespace rund::compute::detail {

[[nodiscard]] Result<CpuPrimitiveScratchPlan>
plan_cpu_spectrum_scratch(const CpuRuntimePrimitive &primitive) noexcept;

[[nodiscard]] Status append_cpu_spectrum_scratch_plan(
    CpuExecutionStoragePlan &arena,
    const CpuPrimitiveScratchPlan &scratch) noexcept;

[[nodiscard]] Result<CpuPrimitiveScratch>
prepare_cpu_spectrum_scratch(const CpuPrimitiveScratchPlan &request,
                             CpuPreparedArena &arena);

} // namespace rund::compute::detail
