#include "sort.hpp"

#include <algorithm>

namespace rund::compute::detail {

Result<CpuPrimitiveScratchPlan>
plan_cpu_sort_scratch(const CpuRuntimePrimitive &primitive) noexcept {
  const auto *const plan =
      active_cpu_scratch_plan<kernel::SortPlan>(primitive);
  std::size_t count = 0u;
  if (plan == nullptr || !cpu_scratch_to_size(plan->element_count, count) ||
      (plan->key != kernel::SortKey::U32 && plan->key != kernel::SortKey::U64)) {
    return Result<CpuPrimitiveScratchPlan>::fail(
        Reason::CpuRuntimePlanInvalid);
  }
  const std::uint8_t key_bytes =
      plan->key == kernel::SortKey::U64 ? sizeof(kernel::u64)
                                       : sizeof(kernel::u32);
  const std::uint64_t owner_bytes =
      plan->key == kernel::SortKey::U64
          ? sizeof(CpuSortPrimitiveScratch<kernel::u64>)
          : sizeof(CpuSortPrimitiveScratch<kernel::u32>);
  return make_cpu_scratch_plan(CpuPrimitiveScratchShape::Sort, key_bytes,
                               {count, count}, owner_bytes);
}

Status append_cpu_sort_scratch_plan(
    CpuExecutionStoragePlan &arena,
    const CpuPrimitiveScratchPlan &scratch) noexcept {
  CpuExecutionStoragePlan next = arena;
  std::size_t u32_count = 0u;
  std::size_t u64_count = 0u;
  if (scratch.element_bytes == sizeof(kernel::u64)) {
    if (!add_cpu_scratch_count(u64_count, scratch.counts[0]) ||
        !add_cpu_scratch_count(u32_count, scratch.counts[1])) {
      return Status::fail(Reason::ProgramCapacity);
    }
  } else if (scratch.element_bytes != sizeof(kernel::u32) ||
             !add_cpu_scratch_count(u32_count, scratch.counts[0]) ||
             !add_cpu_scratch_count(u32_count, scratch.counts[1])) {
    return Status::fail(Reason::ProgramCapacity);
  }
  next.primitive_u32_count = std::max(next.primitive_u32_count, u32_count);
  next.primitive_u64_count = std::max(next.primitive_u64_count, u64_count);
  const Status appended = append_cpu_scratch_object(next, scratch.host_bytes);
  if (!appended) {
    return appended;
  }
  arena = next;
  return Status::success();
}

Result<CpuPrimitiveScratch> prepare_cpu_sort_scratch(
    const CpuRuntimePrimitive &primitive,
    const CpuPrimitiveScratchPlan &request, CpuPreparedArena &arena) {
  const auto *const plan =
      active_cpu_scratch_plan<kernel::SortPlan>(primitive);
  if (plan == nullptr) {
    return Result<CpuPrimitiveScratch>::fail(Reason::CpuRuntimePlanInvalid);
  }
  std::size_t word_cursor = 0u;
  if (plan->key == kernel::SortKey::U64) {
    auto *const prepared =
        arena.claim_primitive_object<CpuSortPrimitiveScratch<kernel::u64>>();
    std::size_t key_cursor = 0u;
    if (prepared == nullptr ||
        !bind_cpu_scratch_buffer(prepared->keys, arena.primitive_u64(),
                                 key_cursor, request.counts[0]) ||
        !bind_cpu_scratch_buffer(prepared->values, arena.primitive_u32(),
                                 word_cursor, request.counts[1])) {
      return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
    }
    return prepared_cpu_scratch(prepared);
  }
  auto *const prepared =
      arena.claim_primitive_object<CpuSortPrimitiveScratch<kernel::u32>>();
  if (prepared == nullptr ||
      !bind_cpu_scratch_buffer(prepared->keys, arena.primitive_u32(),
                               word_cursor, request.counts[0]) ||
      !bind_cpu_scratch_buffer(prepared->values, arena.primitive_u32(),
                               word_cursor, request.counts[1])) {
    return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
  }
  return prepared_cpu_scratch(prepared);
}

} // namespace rund::compute::detail
