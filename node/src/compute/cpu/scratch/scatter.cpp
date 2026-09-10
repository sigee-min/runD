#include "scatter.hpp"

#include <algorithm>

namespace rund::compute::detail {

Result<CpuPrimitiveScratchPlan>
plan_cpu_scatter_scratch(const CpuRuntimePrimitive &primitive) noexcept {
  const auto *const plan =
      active_cpu_scratch_plan<kernel::ScatterPlan>(primitive);
  std::size_t capacity = 0u;
  if (plan == nullptr || !cpu_scratch_to_size(plan->scratch_slots, capacity) ||
      capacity == 0u) {
    return Result<CpuPrimitiveScratchPlan>::fail(
        Reason::ScatterTempOverflow);
  }
  return make_cpu_scratch_plan(CpuPrimitiveScratchShape::Scatter,
                               sizeof(kernel::u32), {capacity, capacity},
                               sizeof(CpuScatterPrimitiveScratch));
}

Status append_cpu_scatter_scratch_plan(
    CpuExecutionStoragePlan &arena,
    const CpuPrimitiveScratchPlan &scratch) noexcept {
  if (scratch.counts[0] == 0u || scratch.counts[0] != scratch.counts[1]) {
    return Status::fail(Reason::CpuRuntimePlanInvalid);
  }
  CpuExecutionStoragePlan next = arena;
  next.scatter_slot_count =
      std::max(next.scatter_slot_count, scratch.counts[0]);
  const Status appended = append_cpu_scratch_object(next, scratch.host_bytes);
  if (!appended) {
    return appended;
  }
  arena = next;
  return Status::success();
}

Result<CpuPrimitiveScratch> prepare_cpu_scatter_scratch(
    const CpuPrimitiveScratchPlan &request, CpuPreparedArena &arena) {
  auto *const prepared =
      arena.claim_primitive_object<CpuScatterPrimitiveScratch>();
  const std::span keys = arena.scatter_keys();
  const std::span marks = arena.scatter_marks();
  if (prepared == nullptr || keys.size() < request.counts[0] ||
      marks.size() < request.counts[1] || arena.scatter_epoch() == nullptr) {
    return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
  }
  prepared->keys = keys.first(request.counts[0]);
  prepared->marks = marks.first(request.counts[1]);
  prepared->mark_capacity = marks;
  prepared->epoch = arena.scatter_epoch();
  return prepared_cpu_scratch(prepared);
}

} // namespace rund::compute::detail
