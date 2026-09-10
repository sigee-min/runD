#include "range.hpp"

#include <algorithm>

namespace rund::compute::detail {

Result<CpuPrimitiveScratchPlan>
plan_cpu_range_scratch(const CpuRuntimePrimitive &primitive) noexcept {
  const auto *const semantic =
      active_cpu_scratch_plan<kernel::WindowPlan>(primitive);
  if (semantic == nullptr || !primitive.range.has_value() ||
      !primitive.range->ok() || primitive.range->temporary_count() > 2u) {
    return Result<CpuPrimitiveScratchPlan>::fail(
        Reason::CpuRuntimePlanInvalid);
  }
  if (primitive.range->temporary_count() == 0u) {
    return empty_cpu_scratch_plan();
  }
  std::array<std::size_t, 5u> counts{};
  for (std::size_t index = 0u; index < primitive.range->temporary_count();
       ++index) {
    const auto temporary = primitive.range->temporary(index);
    if (temporary.bytes % semantic->element_bytes != 0u ||
        !cpu_scratch_to_size(temporary.bytes / semantic->element_bytes,
                             counts[index])) {
      return Result<CpuPrimitiveScratchPlan>::fail(Reason::ProgramCapacity);
    }
  }
  const bool wide = semantic->element_bytes == sizeof(kernel::u64);
  const bool unsigned_domain =
      semantic->domain == kernel::ComputeDomain::U32 ||
      semantic->domain == kernel::ComputeDomain::U64;
  const CpuPrimitiveScratchShape shape =
      wide ? (unsigned_domain ? CpuPrimitiveScratchShape::RangeU64
                              : CpuPrimitiveScratchShape::RangeI64)
           : (unsigned_domain ? CpuPrimitiveScratchShape::RangeU32
                              : CpuPrimitiveScratchShape::RangeI32);
  const std::uint64_t owner_bytes =
      wide ? (unsigned_domain ? sizeof(CpuRangeScratch<kernel::u64>)
                              : sizeof(CpuRangeScratch<kernel::i64>))
           : (unsigned_domain ? sizeof(CpuRangeScratch<kernel::u32>)
                              : sizeof(CpuRangeScratch<kernel::i32>));
  return make_cpu_scratch_plan(
      shape, static_cast<std::uint8_t>(semantic->element_bytes), counts,
      owner_bytes);
}

Status append_cpu_range_scratch_plan(
    CpuExecutionStoragePlan &arena,
    const CpuPrimitiveScratchPlan &scratch) noexcept {
  CpuExecutionStoragePlan next = arena;
  std::size_t i32_count = 0u;
  std::size_t u32_count = 0u;
  std::size_t i64_count = 0u;
  std::size_t u64_count = 0u;
  switch (scratch.shape) {
  case CpuPrimitiveScratchShape::RangeI32:
    if (!add_cpu_scratch_count(i32_count, scratch.counts[0]) ||
        !add_cpu_scratch_count(i32_count, scratch.counts[1])) {
      return Status::fail(Reason::ProgramCapacity);
    }
    break;
  case CpuPrimitiveScratchShape::RangeU32:
    if (!add_cpu_scratch_count(u32_count, scratch.counts[0]) ||
        !add_cpu_scratch_count(u32_count, scratch.counts[1])) {
      return Status::fail(Reason::ProgramCapacity);
    }
    break;
  case CpuPrimitiveScratchShape::RangeI64:
    if (!add_cpu_scratch_count(i64_count, scratch.counts[0]) ||
        !add_cpu_scratch_count(i64_count, scratch.counts[1])) {
      return Status::fail(Reason::ProgramCapacity);
    }
    break;
  case CpuPrimitiveScratchShape::RangeU64:
    if (!add_cpu_scratch_count(u64_count, scratch.counts[0]) ||
        !add_cpu_scratch_count(u64_count, scratch.counts[1])) {
      return Status::fail(Reason::ProgramCapacity);
    }
    break;
  default:
    return Status::fail(Reason::CpuRuntimePlanInvalid);
  }
  next.primitive_u32_count = std::max(next.primitive_u32_count, u32_count);
  next.primitive_u64_count = std::max(next.primitive_u64_count, u64_count);
  next.primitive_i32_count = std::max(next.primitive_i32_count, i32_count);
  next.primitive_i64_count = std::max(next.primitive_i64_count, i64_count);
  // The request carries the sealed descriptor extent. Keep this append
  // boundary independent of the concrete owner type so malformed/tampered
  // plans retain the original dispatcher semantics and use the same checked
  // byte accounting.
  const Status appended = append_cpu_scratch_object(next, scratch.host_bytes);
  if (!appended) {
    return appended;
  }
  arena = next;
  return Status::success();
}

template <class Lane>
[[nodiscard]] Result<CpuPrimitiveScratch>
prepare_cpu_range_scratch_from(const CpuPrimitiveScratchPlan &request,
                               const std::span<Lane> storage,
                               CpuPreparedArena &arena) {
  auto *const prepared = arena.claim_primitive_object<CpuRangeScratch<Lane>>();
  std::size_t cursor = 0u;
  if (prepared == nullptr ||
      !bind_cpu_scratch_buffer(prepared->first, storage, cursor,
                               request.counts[0]) ||
      !bind_cpu_scratch_buffer(prepared->second, storage, cursor,
                               request.counts[1])) {
    return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
  }
  return prepared_cpu_scratch(prepared);
}

Result<CpuPrimitiveScratch>
prepare_cpu_range_scratch(const CpuPrimitiveScratchPlan &request,
                          CpuPreparedArena &arena) {
  switch (request.shape) {
  case CpuPrimitiveScratchShape::RangeI32:
    return prepare_cpu_range_scratch_from(request, arena.primitive_i32(), arena);
  case CpuPrimitiveScratchShape::RangeU32:
    return prepare_cpu_range_scratch_from(request, arena.primitive_u32(), arena);
  case CpuPrimitiveScratchShape::RangeI64:
    return prepare_cpu_range_scratch_from(request, arena.primitive_i64(), arena);
  case CpuPrimitiveScratchShape::RangeU64:
    return prepare_cpu_range_scratch_from(request, arena.primitive_u64(), arena);
  default:
    return Result<CpuPrimitiveScratch>::fail(Reason::CpuRuntimePlanInvalid);
  }
}

} // namespace rund::compute::detail
