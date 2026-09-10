#include "spectrum.hpp"

#include <algorithm>

namespace rund::compute::detail {
namespace {

template <class Lane>
[[nodiscard]] std::span<Lane>
spectrum_lane_storage(CpuPreparedArena &arena) noexcept {
  if constexpr (std::is_same_v<Lane, kernel::i64>) {
    return arena.primitive_i64();
  } else {
    return arena.primitive_i32();
  }
}

template <class Lane>
[[nodiscard]] Result<CpuPrimitiveScratch> prepare_spectrum_lane(
    const CpuPrimitiveScratchPlan &plan, CpuPreparedArena &arena) {
  std::size_t lane_cursor = 0u;
  const std::span<Lane> lanes = spectrum_lane_storage<Lane>(arena);
  if (plan.shape == CpuPrimitiveScratchShape::SpectrumEigen) {
    auto *const prepared =
        arena.claim_primitive_object<CpuSpectrumEigenScratch<Lane>>();
    if (prepared == nullptr ||
        !bind_cpu_scratch_buffer(prepared->values[0], lanes, lane_cursor,
                                 plan.counts[0]) ||
        !bind_cpu_scratch_buffer(prepared->values[1], lanes, lane_cursor,
                                 plan.counts[1]) ||
        !bind_cpu_scratch_buffer(prepared->values[2], lanes, lane_cursor,
                                 plan.counts[2])) {
      return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
    }
    return prepared_cpu_scratch(prepared);
  }
  std::size_t word_cursor = 0u;
  if (plan.shape == CpuPrimitiveScratchShape::SpectrumSvdValues) {
    auto *const prepared =
        arena.claim_primitive_object<CpuSpectrumSvdValuesScratch<Lane>>();
    if (prepared == nullptr ||
        !bind_cpu_scratch_buffer(prepared->values[0], lanes, lane_cursor,
                                 plan.counts[0]) ||
        !bind_cpu_scratch_buffer(prepared->values[1], lanes, lane_cursor,
                                 plan.counts[1]) ||
        !bind_cpu_scratch_buffer(prepared->values[2], lanes, lane_cursor,
                                 plan.counts[2]) ||
        !bind_cpu_scratch_buffer(prepared->order, arena.primitive_u64(),
                                 word_cursor, plan.counts[4])) {
      return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
    }
    return prepared_cpu_scratch(prepared);
  }
  if (plan.shape != CpuPrimitiveScratchShape::SpectrumSvdVectors) {
    return Result<CpuPrimitiveScratch>::fail(Reason::CpuRuntimePlanInvalid);
  }
  auto *const prepared =
      arena.claim_primitive_object<CpuSpectrumSvdVectorsScratch<Lane>>();
  if (prepared == nullptr ||
      !bind_cpu_scratch_buffer(prepared->values[0], lanes, lane_cursor,
                               plan.counts[0]) ||
      !bind_cpu_scratch_buffer(prepared->values[1], lanes, lane_cursor,
                               plan.counts[1]) ||
      !bind_cpu_scratch_buffer(prepared->values[2], lanes, lane_cursor,
                               plan.counts[2]) ||
      !bind_cpu_scratch_buffer(prepared->values[3], lanes, lane_cursor,
                               plan.counts[3]) ||
      !bind_cpu_scratch_buffer(prepared->order, arena.primitive_u64(),
                               word_cursor, plan.counts[4])) {
    return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
  }
  return prepared_cpu_scratch(prepared);
}

} // namespace

Result<CpuPrimitiveScratchPlan>
plan_cpu_spectrum_scratch(const CpuRuntimePrimitive &primitive) noexcept {
  const auto *const plan =
      active_cpu_scratch_plan<kernel::SpectrumPlan>(primitive);
  if (plan == nullptr || (plan->element_bytes != sizeof(kernel::i32) &&
                          plan->element_bytes != sizeof(kernel::i64))) {
    return Result<CpuPrimitiveScratchPlan>::fail(
        Reason::CpuRuntimePlanInvalid);
  }
  const kernel::u64 n =
      plan->op == kernel::SpectrumOp::Eigen ? plan->rows : plan->cols;
  std::array<std::size_t, 5u> counts{};
  if (!cpu_scratch_product_size(n, n, counts[0]) ||
      !cpu_scratch_to_size(n, counts[2])) {
    return Result<CpuPrimitiveScratchPlan>::fail(Reason::ProgramCapacity);
  }
  counts[1] = counts[0];
  const auto width = static_cast<std::uint8_t>(plan->element_bytes);
  if (plan->op == kernel::SpectrumOp::Eigen) {
    const std::uint64_t owner_bytes =
        width == sizeof(kernel::i64)
            ? sizeof(CpuSpectrumEigenScratch<kernel::i64>)
            : sizeof(CpuSpectrumEigenScratch<kernel::i32>);
    return make_cpu_scratch_plan(CpuPrimitiveScratchShape::SpectrumEigen,
                                 width, counts, owner_bytes);
  }
  if (plan->op != kernel::SpectrumOp::SVD) {
    return Result<CpuPrimitiveScratchPlan>::fail(
        Reason::CpuRuntimePlanInvalid);
  }
  counts[4] = counts[2];
  if (plan->vector_count == 0u) {
    const std::uint64_t owner_bytes =
        width == sizeof(kernel::i64)
            ? sizeof(CpuSpectrumSvdValuesScratch<kernel::i64>)
            : sizeof(CpuSpectrumSvdValuesScratch<kernel::i32>);
    return make_cpu_scratch_plan(CpuPrimitiveScratchShape::SpectrumSvdValues,
                                 width, counts, owner_bytes);
  }
  const kernel::u64 vector_cols =
      plan->vectors == kernel::SpectrumVectors::Thin
          ? std::min(plan->rows, plan->cols)
          : plan->rows;
  if (!cpu_scratch_product_size(plan->rows, vector_cols, counts[3])) {
    return Result<CpuPrimitiveScratchPlan>::fail(Reason::ProgramCapacity);
  }
  const std::uint64_t owner_bytes =
      width == sizeof(kernel::i64)
          ? sizeof(CpuSpectrumSvdVectorsScratch<kernel::i64>)
          : sizeof(CpuSpectrumSvdVectorsScratch<kernel::i32>);
  return make_cpu_scratch_plan(CpuPrimitiveScratchShape::SpectrumSvdVectors,
                               width, counts, owner_bytes);
}

Status append_cpu_spectrum_scratch_plan(
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
  if (scratch.shape == CpuPrimitiveScratchShape::SpectrumSvdValues) {
    next.primitive_u64_count =
        std::max(next.primitive_u64_count, scratch.counts[4]);
  } else if (scratch.shape == CpuPrimitiveScratchShape::SpectrumSvdVectors) {
    if (!add_cpu_scratch_count(lane_count, scratch.counts[3])) {
      return Status::fail(Reason::ProgramCapacity);
    }
    next.primitive_u64_count =
        std::max(next.primitive_u64_count, scratch.counts[4]);
  } else if (scratch.shape != CpuPrimitiveScratchShape::SpectrumEigen) {
    return Status::fail(Reason::CpuRuntimePlanInvalid);
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

Result<CpuPrimitiveScratch> prepare_cpu_spectrum_scratch(
    const CpuPrimitiveScratchPlan &request, CpuPreparedArena &arena) {
  return request.element_bytes == sizeof(kernel::i64)
             ? prepare_spectrum_lane<kernel::i64>(request, arena)
             : prepare_spectrum_lane<kernel::i32>(request, arena);
}

} // namespace rund::compute::detail
