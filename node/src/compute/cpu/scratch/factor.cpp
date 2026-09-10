#include "factor.hpp"

#include <algorithm>

namespace rund::compute::detail {
namespace {

template <class Lane>
[[nodiscard]] Result<CpuPrimitiveScratch> prepare_factor_lane(
    const CpuPrimitiveScratchPlan &request, CpuPreparedArena &arena) {
  auto *const prepared =
      arena.claim_primitive_object<CpuFactorQrScratch<Lane>>();
  if (prepared == nullptr) {
    return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
  }
  std::size_t cursor = 0u;
  const std::span<Lane> storage = [&]() noexcept {
    if constexpr (std::is_same_v<Lane, kernel::i64>) {
      return arena.primitive_i64();
    } else {
      return arena.primitive_i32();
    }
  }();
  if (!bind_cpu_scratch_buffer(prepared->values[0], storage, cursor,
                               request.counts[0]) ||
      !bind_cpu_scratch_buffer(prepared->values[1], storage, cursor,
                               request.counts[1]) ||
      !bind_cpu_scratch_buffer(prepared->values[2], storage, cursor,
                               request.counts[2])) {
    return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
  }
  return prepared_cpu_scratch(prepared);
}

} // namespace

Result<CpuPrimitiveScratchPlan>
plan_cpu_factor_scratch(const CpuRuntimePrimitive &primitive) noexcept {
  const auto *const plan =
      active_cpu_scratch_plan<kernel::FactorPlan>(primitive);
  if (plan == nullptr || (plan->element_bytes != sizeof(kernel::i32) &&
                          plan->element_bytes != sizeof(kernel::i64))) {
    return Result<CpuPrimitiveScratchPlan>::fail(
        Reason::CpuRuntimePlanInvalid);
  }
  if (plan->op != kernel::FactorOp::QR) {
    return empty_cpu_scratch_plan();
  }
  std::array<std::size_t, 5u> counts{};
  if (!cpu_scratch_product_size(plan->rows, plan->cols, counts[0]) ||
      !cpu_scratch_product_size(plan->cols, plan->cols, counts[1]) ||
      !cpu_scratch_to_size(plan->rows, counts[2])) {
    return Result<CpuPrimitiveScratchPlan>::fail(Reason::ProgramCapacity);
  }
  const std::uint64_t owner_bytes =
      plan->element_bytes == sizeof(kernel::i64)
          ? sizeof(CpuFactorQrScratch<kernel::i64>)
          : sizeof(CpuFactorQrScratch<kernel::i32>);
  return make_cpu_scratch_plan(
      CpuPrimitiveScratchShape::FactorQr,
      static_cast<std::uint8_t>(plan->element_bytes), counts, owner_bytes);
}

Status append_cpu_factor_scratch_plan(
    CpuExecutionStoragePlan &arena,
    const CpuPrimitiveScratchPlan &scratch) noexcept {
  CpuExecutionStoragePlan next = arena;
  std::size_t i32_count = 0u;
  std::size_t i64_count = 0u;
  const bool wide = scratch.element_bytes == sizeof(kernel::i64);
  const bool narrow = scratch.element_bytes == sizeof(kernel::i32);
  auto &lane_count = wide ? i64_count : i32_count;
  if (!wide && !narrow) {
    return Status::fail(Reason::CpuRuntimePlanInvalid);
  }
  if (!add_cpu_scratch_count(lane_count, scratch.counts[0]) ||
      !add_cpu_scratch_count(lane_count, scratch.counts[1]) ||
      !add_cpu_scratch_count(lane_count, scratch.counts[2])) {
    return Status::fail(Reason::ProgramCapacity);
  }
  next.primitive_i32_count = std::max(next.primitive_i32_count, i32_count);
  next.primitive_i64_count = std::max(next.primitive_i64_count, i64_count);
  const Status appended = append_cpu_scratch_object(next, scratch.host_bytes);
  if (!appended) {
    return appended;
  }
  arena = next;
  return Status::success();
}

Result<CpuPrimitiveScratch> prepare_cpu_factor_scratch(
    const CpuPrimitiveScratchPlan &request, CpuPreparedArena &arena) {
  return request.element_bytes == sizeof(kernel::i64)
             ? prepare_factor_lane<kernel::i64>(request, arena)
             : prepare_factor_lane<kernel::i32>(request, arena);
}

} // namespace rund::compute::detail
