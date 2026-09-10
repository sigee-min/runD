#include "scatter.hpp"

#include <algorithm>

namespace rund::compute::detail {

Result<CpuPrimitiveScratchPlan>
plan_cpu_scatter_reduce_scratch(const CpuRuntimePrimitive &primitive) noexcept {
  const auto *const plan =
      active_cpu_scratch_plan<kernel::ScatterReducePlan>(primitive);
  std::size_t count = 0u;
  if (plan == nullptr || !cpu_scratch_to_size(plan->element_count, count)) {
    return Result<CpuPrimitiveScratchPlan>::fail(
        Reason::CpuRuntimePlanInvalid);
  }
  return make_cpu_scratch_plan(CpuPrimitiveScratchShape::ScatterReduce,
                               sizeof(kernel::u32), {count},
                               sizeof(CpuScatterReducePrimitiveScratch));
}

Status append_cpu_scatter_reduce_scratch_plan(
    CpuExecutionStoragePlan &arena,
    const CpuPrimitiveScratchPlan &scratch) noexcept {
  CpuExecutionStoragePlan next = arena;
  next.primitive_u32_count =
      std::max(next.primitive_u32_count, scratch.counts[0]);
  const Status appended = append_cpu_scratch_object(next, scratch.host_bytes);
  if (!appended) {
    return appended;
  }
  arena = next;
  return Status::success();
}

Result<CpuPrimitiveScratch> prepare_cpu_scatter_reduce_scratch(
    const CpuPrimitiveScratchPlan &request, CpuPreparedArena &arena) {
  auto *const prepared =
      arena.claim_primitive_object<CpuScatterReducePrimitiveScratch>();
  std::size_t cursor = 0u;
  if (prepared == nullptr ||
      !bind_cpu_scratch_buffer(prepared->sorted_indices, arena.primitive_u32(),
                               cursor, request.counts[0])) {
    return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
  }
  return prepared_cpu_scratch(prepared);
}

} // namespace rund::compute::detail
