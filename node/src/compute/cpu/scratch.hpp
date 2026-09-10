#pragma once

// Compatibility facade for the CPU primitive scratch contract. Planning and
// materialization implementations live in one direct owner per primitive
// family; this header retains only the public dispatcher surface and views.
#include "scratch/access.hpp"
#include "scratch/factor.hpp"
#include "scratch/range.hpp"
#include "scratch/scatter.hpp"
#include "scratch/solve.hpp"
#include "scratch/sort.hpp"
#include "scratch/spectrum.hpp"
#include "scratch/transform.hpp"

namespace rund::compute::detail {

[[nodiscard]] Result<CpuPrimitiveScratch>
prepare_cpu_scratch(const CpuRuntimePrimitive &primitive,
                    const CpuPrimitiveScratchPlan &plan,
                    CpuPreparedArena &arena);

[[nodiscard]] Result<CpuPrimitiveScratchPlan>
plan_cpu_scratch(const CpuRuntimePrimitive &primitive) noexcept;

[[nodiscard]] Status append_cpu_primitive_arena_plan(
    CpuExecutionStoragePlan &arena,
    const CpuPrimitiveScratchPlan &scratch) noexcept;

} // namespace rund::compute::detail
