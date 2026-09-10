#include "transform.hpp"

#include <kernel/program/compute/transform/twiddle.hpp>

namespace rund::compute::detail {
namespace {

template <class Lane>
[[nodiscard]] Result<CpuPrimitiveScratch> prepare_transform_lane(
    const kernel::TransformPlan &plan,
    const CpuPrimitiveScratchPlan &request, CpuPreparedArena &arena) {
  auto *const prepared =
      arena.claim_primitive_object<CpuTransformScratch<Lane>>();
  if (prepared == nullptr) {
    return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
  }
  std::span<Lane> twiddle;
  if constexpr (std::is_same_v<Lane, kernel::i64>) {
    twiddle = arena.claim_transform_i64(request.counts[0]);
  } else {
    twiddle = arena.claim_transform_i32(request.counts[0]);
  }
  if (twiddle.size() != request.counts[0]) {
    return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
  }
  prepared->twiddle = twiddle;
  if (!kernel::transform_twiddle::Fill(prepared->twiddle.data(),
                                       plan.element_count, plan.direction,
                                       plan.fixed_format)) {
    return Result<CpuPrimitiveScratch>::fail(Reason::CpuRuntimePlanInvalid);
  }
  return prepared_cpu_scratch(prepared);
}

} // namespace

Result<CpuPrimitiveScratchPlan>
plan_cpu_transform_scratch(const CpuRuntimePrimitive &primitive) noexcept {
  const auto *const plan =
      active_cpu_scratch_plan<kernel::TransformPlan>(primitive);
  if (plan == nullptr || (plan->element_bytes != sizeof(kernel::i32) &&
                          plan->element_bytes != sizeof(kernel::i64))) {
    return Result<CpuPrimitiveScratchPlan>::fail(
        Reason::CpuRuntimePlanInvalid);
  }
  std::size_t count = 0u;
  if (!cpu_scratch_product_size(plan->twiddle_count, 2u, count)) {
    return Result<CpuPrimitiveScratchPlan>::fail(Reason::ProgramCapacity);
  }
  const std::uint64_t owner_bytes =
      plan->element_bytes == sizeof(kernel::i64)
          ? sizeof(CpuTransformScratch<kernel::i64>)
          : sizeof(CpuTransformScratch<kernel::i32>);
  return make_cpu_scratch_plan(CpuPrimitiveScratchShape::Transform,
                               plan->element_bytes, {count}, owner_bytes);
}

Status append_cpu_transform_scratch_plan(
    CpuExecutionStoragePlan &arena,
    const CpuPrimitiveScratchPlan &scratch) noexcept {
  CpuExecutionStoragePlan next = arena;
  if (scratch.element_bytes == sizeof(kernel::i64)) {
    if (!add_cpu_scratch_count(next.transform_i64_count, scratch.counts[0])) {
      return Status::fail(Reason::ProgramCapacity);
    }
  } else if (scratch.element_bytes == sizeof(kernel::i32)) {
    if (!add_cpu_scratch_count(next.transform_i32_count, scratch.counts[0])) {
      return Status::fail(Reason::ProgramCapacity);
    }
  } else {
    return Status::fail(Reason::CpuRuntimePlanInvalid);
  }
  const Status appended = append_cpu_scratch_object(next, scratch.host_bytes);
  if (!appended) {
    return appended;
  }
  arena = next;
  return Status::success();
}

Result<CpuPrimitiveScratch> prepare_cpu_transform_scratch(
    const CpuRuntimePrimitive &primitive,
    const CpuPrimitiveScratchPlan &request, CpuPreparedArena &arena) {
  const auto *const plan =
      active_cpu_scratch_plan<kernel::TransformPlan>(primitive);
  if (plan == nullptr) {
    return Result<CpuPrimitiveScratch>::fail(Reason::CpuRuntimePlanInvalid);
  }
  return plan->element_bytes == sizeof(kernel::i64)
             ? prepare_transform_lane<kernel::i64>(*plan, request, arena)
             : prepare_transform_lane<kernel::i32>(*plan, request, arena);
}

} // namespace rund::compute::detail
