#include "solve.hpp"

#include <algorithm>

namespace rund::compute::detail {
namespace {

template <class Lane>
[[nodiscard]] std::span<Lane>
solve_lane_storage(CpuPreparedArena &arena) noexcept {
  if constexpr (std::is_same_v<Lane, kernel::i64>) {
    return arena.primitive_i64();
  } else {
    return arena.primitive_i32();
  }
}

template <class Lane>
[[nodiscard]] Result<CpuPrimitiveScratch> prepare_solve_lane(
    const CpuPrimitiveScratchPlan &plan, CpuPreparedArena &arena) {
  std::size_t lane_cursor = 0u;
  const std::span<Lane> lanes = solve_lane_storage<Lane>(arena);
  if (plan.shape == CpuPrimitiveScratchShape::SolveQrFactor) {
    auto *const prepared =
        arena.claim_primitive_object<CpuSolveQrFactorScratch<Lane>>();
    if (prepared == nullptr ||
        !bind_cpu_scratch_buffer(prepared->y, lanes, lane_cursor,
                                 plan.counts[0])) {
      return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
    }
    return prepared_cpu_scratch(prepared);
  }
  if (plan.shape == CpuPrimitiveScratchShape::SolveLu) {
    auto *const prepared =
        arena.claim_primitive_object<CpuSolveLuScratch<Lane>>();
    std::size_t word_cursor = 0u;
    if (prepared == nullptr ||
        !bind_cpu_scratch_buffer(prepared->factor, lanes, lane_cursor,
                                 plan.counts[0]) ||
        !bind_cpu_scratch_buffer(prepared->pivots, arena.primitive_u32(),
                                 word_cursor, plan.counts[1])) {
      return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
    }
    return prepared_cpu_scratch(prepared);
  }
  if (plan.shape == CpuPrimitiveScratchShape::SolveCholesky) {
    auto *const prepared =
        arena.claim_primitive_object<CpuSolveCholeskyScratch<Lane>>();
    if (prepared == nullptr ||
        !bind_cpu_scratch_buffer(prepared->factor, lanes, lane_cursor,
                                 plan.counts[0])) {
      return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
    }
    return prepared_cpu_scratch(prepared);
  }
  if (plan.shape != CpuPrimitiveScratchShape::SolveQrMatrix) {
    return Result<CpuPrimitiveScratch>::fail(Reason::CpuRuntimePlanInvalid);
  }
  auto *const prepared =
      arena.claim_primitive_object<CpuSolveQrMatrixScratch<Lane>>();
  if (prepared == nullptr ||
      !bind_cpu_scratch_buffer(prepared->y, lanes, lane_cursor,
                               plan.counts[0]) ||
      !bind_cpu_scratch_buffer(prepared->qr[0], lanes, lane_cursor,
                               plan.counts[1]) ||
      !bind_cpu_scratch_buffer(prepared->qr[1], lanes, lane_cursor,
                               plan.counts[2]) ||
      !bind_cpu_scratch_buffer(prepared->qr[2], lanes, lane_cursor,
                               plan.counts[3])) {
    return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
  }
  return prepared_cpu_scratch(prepared);
}

} // namespace

Result<CpuPrimitiveScratchPlan>
plan_cpu_solve_scratch(const CpuRuntimePrimitive &primitive) noexcept {
  const auto *const plan =
      active_cpu_scratch_plan<kernel::SolvePlan>(primitive);
  if (plan == nullptr || (plan->element_bytes != sizeof(kernel::i32) &&
                          plan->element_bytes != sizeof(kernel::i64))) {
    return Result<CpuPrimitiveScratchPlan>::fail(
        Reason::CpuRuntimePlanInvalid);
  }
  const auto width = static_cast<std::uint8_t>(plan->element_bytes);
  CpuPrimitiveScratchShape shape{CpuPrimitiveScratchShape::None};
  std::array<std::size_t, 5u> counts{};
  std::uint64_t owner_bytes = 0u;
  if (plan->input == kernel::SolveInput::Factor) {
    if (plan->factor != kernel::FactorOp::QR) {
      return empty_cpu_scratch_plan();
    }
    if (!cpu_scratch_product_size(plan->rows, plan->rhs_cols, counts[0])) {
      return Result<CpuPrimitiveScratchPlan>::fail(Reason::ProgramCapacity);
    }
    shape = CpuPrimitiveScratchShape::SolveQrFactor;
    owner_bytes = width == sizeof(kernel::i64)
                      ? sizeof(CpuSolveQrFactorScratch<kernel::i64>)
                      : sizeof(CpuSolveQrFactorScratch<kernel::i32>);
  } else if (plan->input != kernel::SolveInput::Matrix ||
             !cpu_scratch_to_size(plan->factor_count, counts[0])) {
    return Result<CpuPrimitiveScratchPlan>::fail(
        plan->input == kernel::SolveInput::Matrix ? Reason::ProgramCapacity
                                                   : Reason::CpuRuntimePlanInvalid);
  } else if (plan->factor == kernel::FactorOp::LU) {
    if (!cpu_scratch_to_size(plan->aux_count, counts[1])) {
      return Result<CpuPrimitiveScratchPlan>::fail(Reason::ProgramCapacity);
    }
    shape = CpuPrimitiveScratchShape::SolveLu;
    owner_bytes = width == sizeof(kernel::i64)
                      ? sizeof(CpuSolveLuScratch<kernel::i64>)
                      : sizeof(CpuSolveLuScratch<kernel::i32>);
  } else if (plan->factor == kernel::FactorOp::Cholesky) {
    shape = CpuPrimitiveScratchShape::SolveCholesky;
    owner_bytes = width == sizeof(kernel::i64)
                      ? sizeof(CpuSolveCholeskyScratch<kernel::i64>)
                      : sizeof(CpuSolveCholeskyScratch<kernel::i32>);
  } else if (plan->factor == kernel::FactorOp::QR) {
    if (!cpu_scratch_product_size(plan->rows, plan->rhs_cols, counts[0]) ||
        !cpu_scratch_product_size(plan->rows, plan->rows, counts[1]) ||
        !cpu_scratch_to_size(plan->rows, counts[3])) {
      return Result<CpuPrimitiveScratchPlan>::fail(Reason::ProgramCapacity);
    }
    counts[2] = counts[1];
    shape = CpuPrimitiveScratchShape::SolveQrMatrix;
    owner_bytes = width == sizeof(kernel::i64)
                      ? sizeof(CpuSolveQrMatrixScratch<kernel::i64>)
                      : sizeof(CpuSolveQrMatrixScratch<kernel::i32>);
  } else {
    return Result<CpuPrimitiveScratchPlan>::fail(
        Reason::CpuRuntimePlanInvalid);
  }
  return make_cpu_scratch_plan(shape, width, counts, owner_bytes);
}

Status append_cpu_solve_scratch_plan(
    CpuExecutionStoragePlan &arena,
    const CpuPrimitiveScratchPlan &scratch) noexcept {
  CpuExecutionStoragePlan next = arena;
  std::size_t i32_count = 0u;
  std::size_t i64_count = 0u;
  std::size_t u32_count = 0u;
  const bool wide = scratch.element_bytes == sizeof(kernel::i64);
  const bool narrow = scratch.element_bytes == sizeof(kernel::i32);
  auto &lane_count = wide ? i64_count : i32_count;
  if (!wide && !narrow) {
    return Status::fail(Reason::CpuRuntimePlanInvalid);
  }
  switch (scratch.shape) {
  case CpuPrimitiveScratchShape::SolveQrFactor:
  case CpuPrimitiveScratchShape::SolveCholesky:
    if (!add_cpu_scratch_count(lane_count, scratch.counts[0])) {
      return Status::fail(Reason::ProgramCapacity);
    }
    break;
  case CpuPrimitiveScratchShape::SolveLu:
    if (!add_cpu_scratch_count(lane_count, scratch.counts[0])) {
      return Status::fail(Reason::ProgramCapacity);
    }
    u32_count = scratch.counts[1];
    break;
  case CpuPrimitiveScratchShape::SolveQrMatrix:
    if (!add_cpu_scratch_count(lane_count, scratch.counts[0]) ||
        !add_cpu_scratch_count(lane_count, scratch.counts[1]) ||
        !add_cpu_scratch_count(lane_count, scratch.counts[2]) ||
        !add_cpu_scratch_count(lane_count, scratch.counts[3])) {
      return Status::fail(Reason::ProgramCapacity);
    }
    break;
  default:
    return Status::fail(Reason::CpuRuntimePlanInvalid);
  }
  next.primitive_u32_count = std::max(next.primitive_u32_count, u32_count);
  next.primitive_i32_count = std::max(next.primitive_i32_count, i32_count);
  next.primitive_i64_count = std::max(next.primitive_i64_count, i64_count);
  const Status appended = append_cpu_scratch_object(next, scratch.host_bytes);
  if (!appended) {
    return appended;
  }
  arena = next;
  return Status::success();
}

Result<CpuPrimitiveScratch> prepare_cpu_solve_scratch(
    const CpuPrimitiveScratchPlan &request, CpuPreparedArena &arena) {
  return request.element_bytes == sizeof(kernel::i64)
             ? prepare_solve_lane<kernel::i64>(request, arena)
             : prepare_solve_lane<kernel::i32>(request, arena);
}

} // namespace rund::compute::detail
