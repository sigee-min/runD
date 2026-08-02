#include "scratch.hpp"

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/transform/twiddle.hpp>

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace rund::compute::detail {
namespace {

[[nodiscard]] Result<CpuPrimitiveScratch> no_cpu_scratch() {
  return Result<CpuPrimitiveScratch>::success(CpuPrimitiveScratch{});
}

template <class T>
[[nodiscard]] bool bind_buffer(std::span<T> &target, const std::span<T> storage,
                               std::size_t &cursor,
                               const std::size_t count) noexcept {
  if (cursor > storage.size() || count > storage.size() - cursor) {
    return false;
  }
  target = storage.subspan(cursor, count);
  cursor += count;
  return true;
}

template <class Scratch>
[[nodiscard]] Result<CpuPrimitiveScratch>
prepared_cpu_scratch(Scratch *const prepared) {
  if (prepared == nullptr) {
    return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
  }
  CpuPrimitiveScratch scratch{std::in_place_type<Scratch *>, prepared};
  return Result<CpuPrimitiveScratch>::success(std::move(scratch));
}

template <class Plan>
[[nodiscard]] const Plan *
active_plan(const CpuRuntimePrimitive &primitive) noexcept {
  const Plan *const plan = std::get_if<Plan>(&primitive.plan);
  return plan != nullptr && plan->ok ? plan : nullptr;
}

[[nodiscard]] bool to_size(const kernel::u64 count,
                           std::size_t &result) noexcept {
  if (count >
      static_cast<kernel::u64>(std::numeric_limits<std::size_t>::max())) {
    return false;
  }
  result = static_cast<std::size_t>(count);
  return true;
}

[[nodiscard]] bool product_size(const kernel::u64 left, const kernel::u64 right,
                                std::size_t &result) noexcept {
  kernel::u64 product = 0u;
  if (!kernel::checked::mul(left, right, product)) {
    return false;
  }
  return to_size(product, result);
}

[[nodiscard]] Result<CpuPrimitiveScratchPlan>
scratch_plan(const CpuPrimitiveScratchShape shape,
             const std::uint8_t element_bytes,
             const std::array<std::size_t, 5u> counts,
             const std::uint64_t host_bytes) noexcept {
  return Result<CpuPrimitiveScratchPlan>::success(CpuPrimitiveScratchPlan{
      .shape = shape,
      .counts = counts,
      .host_bytes = host_bytes,
      .element_bytes = element_bytes,
  });
}

[[nodiscard]] Result<CpuPrimitiveScratchPlan> empty_scratch_plan() noexcept {
  return Result<CpuPrimitiveScratchPlan>::success({});
}

[[nodiscard]] bool
scratch_shape_matches(const Primitive primitive,
                      const CpuPrimitiveScratchShape shape) noexcept {
  switch (primitive) {
  case Primitive::Sort:
  case Primitive::Argsort:
    return shape == CpuPrimitiveScratchShape::Sort;
  case Primitive::Scatter:
    return shape == CpuPrimitiveScratchShape::Scatter;
  case Primitive::ScatterReduce:
    return shape == CpuPrimitiveScratchShape::ScatterReduce;
  case Primitive::Transform:
    return shape == CpuPrimitiveScratchShape::Transform;
  case Primitive::Factor:
    return shape == CpuPrimitiveScratchShape::None ||
           shape == CpuPrimitiveScratchShape::FactorQr;
  case Primitive::Solve:
    return shape == CpuPrimitiveScratchShape::None ||
           shape == CpuPrimitiveScratchShape::SolveQrFactor ||
           shape == CpuPrimitiveScratchShape::SolveLu ||
           shape == CpuPrimitiveScratchShape::SolveCholesky ||
           shape == CpuPrimitiveScratchShape::SolveQrMatrix;
  case Primitive::Spectrum:
    return shape == CpuPrimitiveScratchShape::SpectrumEigen ||
           shape == CpuPrimitiveScratchShape::SpectrumSvdValues ||
           shape == CpuPrimitiveScratchShape::SpectrumSvdVectors;
  case Primitive::SegmentedScan:
  case Primitive::SegmentedReduce:
  case Primitive::Compact:
  case Primitive::Gather:
  case Primitive::Histogram:
  case Primitive::Partition:
  case Primitive::Reduce:
  case Primitive::Stencil:
  case Primitive::Matrix:
    return shape == CpuPrimitiveScratchShape::None;
  }
  return false;
}

template <class Lane>
[[nodiscard]] Result<CpuPrimitiveScratch>
prepare_borrowed_transform(const kernel::TransformPlan &plan,
                           const CpuPrimitiveScratchPlan &request,
                           CpuPreparedArena &arena) {
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

template <class Lane>
[[nodiscard]] Result<CpuPrimitiveScratch>
prepare_borrowed_factor(const CpuPrimitiveScratchPlan &plan,
                        const std::span<Lane> storage,
                        CpuPreparedArena &arena) {
  auto *const prepared =
      arena.claim_primitive_object<CpuFactorQrScratch<Lane>>();
  if (prepared == nullptr) {
    return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
  }
  std::size_t cursor = 0u;
  if (!bind_buffer(prepared->values[0], storage, cursor, plan.counts[0]) ||
      !bind_buffer(prepared->values[1], storage, cursor, plan.counts[1]) ||
      !bind_buffer(prepared->values[2], storage, cursor, plan.counts[2])) {
    return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
  }
  return prepared_cpu_scratch(prepared);
}

template <class Lane>
[[nodiscard]] Result<CpuPrimitiveScratch> prepare_borrowed_solve(
    const CpuPrimitiveScratchPlan &plan, const std::span<Lane> lanes,
    const std::span<kernel::u32> words, CpuPreparedArena &arena) {
  std::size_t lane_cursor = 0u;
  if (plan.shape == CpuPrimitiveScratchShape::SolveQrFactor) {
    auto *const prepared =
        arena.claim_primitive_object<CpuSolveQrFactorScratch<Lane>>();
    if (prepared == nullptr ||
        !bind_buffer(prepared->y, lanes, lane_cursor, plan.counts[0])) {
      return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
    }
    return prepared_cpu_scratch(prepared);
  }
  if (plan.shape == CpuPrimitiveScratchShape::SolveLu) {
    auto *const prepared =
        arena.claim_primitive_object<CpuSolveLuScratch<Lane>>();
    std::size_t word_cursor = 0u;
    if (prepared == nullptr ||
        !bind_buffer(prepared->factor, lanes, lane_cursor, plan.counts[0]) ||
        !bind_buffer(prepared->pivots, words, word_cursor, plan.counts[1])) {
      return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
    }
    return prepared_cpu_scratch(prepared);
  }
  if (plan.shape == CpuPrimitiveScratchShape::SolveCholesky) {
    auto *const prepared =
        arena.claim_primitive_object<CpuSolveCholeskyScratch<Lane>>();
    if (prepared == nullptr ||
        !bind_buffer(prepared->factor, lanes, lane_cursor, plan.counts[0])) {
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
      !bind_buffer(prepared->y, lanes, lane_cursor, plan.counts[0]) ||
      !bind_buffer(prepared->qr[0], lanes, lane_cursor, plan.counts[1]) ||
      !bind_buffer(prepared->qr[1], lanes, lane_cursor, plan.counts[2]) ||
      !bind_buffer(prepared->qr[2], lanes, lane_cursor, plan.counts[3])) {
    return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
  }
  return prepared_cpu_scratch(prepared);
}

template <class Lane>
[[nodiscard]] Result<CpuPrimitiveScratch> prepare_borrowed_spectrum(
    const CpuPrimitiveScratchPlan &plan, const std::span<Lane> lanes,
    const std::span<kernel::u64> words, CpuPreparedArena &arena) {
  std::size_t lane_cursor = 0u;
  if (plan.shape == CpuPrimitiveScratchShape::SpectrumEigen) {
    auto *const prepared =
        arena.claim_primitive_object<CpuSpectrumEigenScratch<Lane>>();
    if (prepared == nullptr ||
        !bind_buffer(prepared->values[0], lanes, lane_cursor, plan.counts[0]) ||
        !bind_buffer(prepared->values[1], lanes, lane_cursor, plan.counts[1]) ||
        !bind_buffer(prepared->values[2], lanes, lane_cursor, plan.counts[2])) {
      return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
    }
    return prepared_cpu_scratch(prepared);
  }
  std::size_t word_cursor = 0u;
  if (plan.shape == CpuPrimitiveScratchShape::SpectrumSvdValues) {
    auto *const prepared =
        arena.claim_primitive_object<CpuSpectrumSvdValuesScratch<Lane>>();
    if (prepared == nullptr ||
        !bind_buffer(prepared->values[0], lanes, lane_cursor, plan.counts[0]) ||
        !bind_buffer(prepared->values[1], lanes, lane_cursor, plan.counts[1]) ||
        !bind_buffer(prepared->values[2], lanes, lane_cursor, plan.counts[2]) ||
        !bind_buffer(prepared->order, words, word_cursor, plan.counts[4])) {
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
      !bind_buffer(prepared->values[0], lanes, lane_cursor, plan.counts[0]) ||
      !bind_buffer(prepared->values[1], lanes, lane_cursor, plan.counts[1]) ||
      !bind_buffer(prepared->values[2], lanes, lane_cursor, plan.counts[2]) ||
      !bind_buffer(prepared->values[3], lanes, lane_cursor, plan.counts[3]) ||
      !bind_buffer(prepared->order, words, word_cursor, plan.counts[4])) {
    return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
  }
  return prepared_cpu_scratch(prepared);
}

} // namespace

Result<CpuPrimitiveScratchPlan>
plan_cpu_scratch(const CpuRuntimePrimitive &primitive) noexcept {
  switch (primitive.kind) {
  case Primitive::SegmentedScan:
  case Primitive::SegmentedReduce:
  case Primitive::Compact:
  case Primitive::Gather:
  case Primitive::Histogram:
  case Primitive::Partition:
  case Primitive::Reduce:
  case Primitive::Stencil:
  case Primitive::Matrix:
    return empty_scratch_plan();
  case Primitive::Sort:
  case Primitive::Argsort: {
    const auto *const plan = active_plan<kernel::SortPlan>(primitive);
    std::size_t count = 0u;
    if (plan == nullptr || !to_size(plan->element_count, count) ||
        (plan->key != kernel::SortKey::U32 &&
         plan->key != kernel::SortKey::U64)) {
      return Result<CpuPrimitiveScratchPlan>::fail(
          Reason::CpuRuntimePlanInvalid);
    }
    const std::uint8_t key_bytes = plan->key == kernel::SortKey::U64
                                       ? sizeof(kernel::u64)
                                       : sizeof(kernel::u32);
    const std::uint64_t owner_bytes =
        plan->key == kernel::SortKey::U64
            ? sizeof(CpuSortPrimitiveScratch<kernel::u64>)
            : sizeof(CpuSortPrimitiveScratch<kernel::u32>);
    return scratch_plan(CpuPrimitiveScratchShape::Sort, key_bytes,
                        {count, count}, owner_bytes);
  }
  case Primitive::Scatter: {
    const auto *const plan = active_plan<kernel::ScatterPlan>(primitive);
    std::size_t capacity = 0u;
    if (plan == nullptr || !to_size(plan->scratch_slots, capacity) ||
        capacity == 0u) {
      return Result<CpuPrimitiveScratchPlan>::fail(Reason::ScatterTempOverflow);
    }
    return scratch_plan(CpuPrimitiveScratchShape::Scatter, sizeof(kernel::u32),
                        {capacity, capacity},
                        sizeof(CpuScatterPrimitiveScratch));
  }
  case Primitive::ScatterReduce: {
    const auto *const plan = active_plan<kernel::ScatterReducePlan>(primitive);
    std::size_t count = 0u;
    if (plan == nullptr || !to_size(plan->element_count, count)) {
      return Result<CpuPrimitiveScratchPlan>::fail(
          Reason::CpuRuntimePlanInvalid);
    }
    return scratch_plan(CpuPrimitiveScratchShape::ScatterReduce,
                        sizeof(kernel::u32), {count},
                        sizeof(CpuScatterReducePrimitiveScratch));
  }
  case Primitive::Transform: {
    const auto *const plan = active_plan<kernel::TransformPlan>(primitive);
    if (plan == nullptr || (plan->element_bytes != sizeof(kernel::i32) &&
                            plan->element_bytes != sizeof(kernel::i64))) {
      return Result<CpuPrimitiveScratchPlan>::fail(
          Reason::CpuRuntimePlanInvalid);
    }
    std::size_t count = 0u;
    if (!product_size(plan->twiddle_count, 2u, count)) {
      return Result<CpuPrimitiveScratchPlan>::fail(Reason::ProgramCapacity);
    }
    const std::uint64_t owner_bytes =
        plan->element_bytes == sizeof(kernel::i64)
            ? sizeof(CpuTransformScratch<kernel::i64>)
            : sizeof(CpuTransformScratch<kernel::i32>);
    return scratch_plan(CpuPrimitiveScratchShape::Transform,
                        plan->element_bytes, {count}, owner_bytes);
  }
  case Primitive::Factor: {
    const auto *const plan = active_plan<kernel::FactorPlan>(primitive);
    if (plan == nullptr || (plan->element_bytes != sizeof(kernel::i32) &&
                            plan->element_bytes != sizeof(kernel::i64))) {
      return Result<CpuPrimitiveScratchPlan>::fail(
          Reason::CpuRuntimePlanInvalid);
    }
    if (plan->op != kernel::FactorOp::QR) {
      return empty_scratch_plan();
    }
    std::array<std::size_t, 5u> counts{};
    if (!product_size(plan->rows, plan->cols, counts[0]) ||
        !product_size(plan->cols, plan->cols, counts[1]) ||
        !to_size(plan->rows, counts[2])) {
      return Result<CpuPrimitiveScratchPlan>::fail(Reason::ProgramCapacity);
    }
    const std::uint64_t owner_bytes =
        plan->element_bytes == sizeof(kernel::i64)
            ? sizeof(CpuFactorQrScratch<kernel::i64>)
            : sizeof(CpuFactorQrScratch<kernel::i32>);
    const auto width = static_cast<std::uint8_t>(plan->element_bytes);
    return scratch_plan(CpuPrimitiveScratchShape::FactorQr, width, counts,
                        owner_bytes);
  }
  case Primitive::Solve: {
    const auto *const plan = active_plan<kernel::SolvePlan>(primitive);
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
        return empty_scratch_plan();
      }
      if (!product_size(plan->rows, plan->rhs_cols, counts[0])) {
        return Result<CpuPrimitiveScratchPlan>::fail(Reason::ProgramCapacity);
      }
      shape = CpuPrimitiveScratchShape::SolveQrFactor;
      owner_bytes = width == sizeof(kernel::i64)
                        ? sizeof(CpuSolveQrFactorScratch<kernel::i64>)
                        : sizeof(CpuSolveQrFactorScratch<kernel::i32>);
    } else if (plan->input != kernel::SolveInput::Matrix ||
               !to_size(plan->factor_count, counts[0])) {
      return Result<CpuPrimitiveScratchPlan>::fail(
          plan->input == kernel::SolveInput::Matrix
              ? Reason::ProgramCapacity
              : Reason::CpuRuntimePlanInvalid);
    } else if (plan->factor == kernel::FactorOp::LU) {
      if (!to_size(plan->aux_count, counts[1])) {
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
      if (!product_size(plan->rows, plan->rhs_cols, counts[0]) ||
          !product_size(plan->rows, plan->rows, counts[1]) ||
          !to_size(plan->rows, counts[3])) {
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
    return scratch_plan(shape, width, counts, owner_bytes);
  }
  case Primitive::Spectrum: {
    const auto *const plan = active_plan<kernel::SpectrumPlan>(primitive);
    if (plan == nullptr || (plan->element_bytes != sizeof(kernel::i32) &&
                            plan->element_bytes != sizeof(kernel::i64))) {
      return Result<CpuPrimitiveScratchPlan>::fail(
          Reason::CpuRuntimePlanInvalid);
    }
    const kernel::u64 n =
        plan->op == kernel::SpectrumOp::Eigen ? plan->rows : plan->cols;
    std::array<std::size_t, 5u> counts{};
    if (!product_size(n, n, counts[0]) || !to_size(n, counts[2])) {
      return Result<CpuPrimitiveScratchPlan>::fail(Reason::ProgramCapacity);
    }
    counts[1] = counts[0];
    const auto width = static_cast<std::uint8_t>(plan->element_bytes);
    if (plan->op == kernel::SpectrumOp::Eigen) {
      const std::uint64_t owner_bytes =
          width == sizeof(kernel::i64)
              ? sizeof(CpuSpectrumEigenScratch<kernel::i64>)
              : sizeof(CpuSpectrumEigenScratch<kernel::i32>);
      return scratch_plan(CpuPrimitiveScratchShape::SpectrumEigen, width,
                          counts, owner_bytes);
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
      return scratch_plan(CpuPrimitiveScratchShape::SpectrumSvdValues, width,
                          counts, owner_bytes);
    }
    const kernel::u64 vector_cols =
        plan->vectors == kernel::SpectrumVectors::Thin
            ? std::min(plan->rows, plan->cols)
            : plan->rows;
    if (!product_size(plan->rows, vector_cols, counts[3])) {
      return Result<CpuPrimitiveScratchPlan>::fail(Reason::ProgramCapacity);
    }
    const std::uint64_t owner_bytes =
        width == sizeof(kernel::i64)
            ? sizeof(CpuSpectrumSvdVectorsScratch<kernel::i64>)
            : sizeof(CpuSpectrumSvdVectorsScratch<kernel::i32>);
    return scratch_plan(CpuPrimitiveScratchShape::SpectrumSvdVectors, width,
                        counts, owner_bytes);
  }
  }
  return Result<CpuPrimitiveScratchPlan>::fail(Reason::CpuRuntimePlanInvalid);
}

Status append_cpu_primitive_arena_plan(
    CpuExecutionStoragePlan &arena,
    const CpuPrimitiveScratchPlan &scratch) noexcept {
  if (scratch.shape == CpuPrimitiveScratchShape::None) {
    return Status::success();
  }
  std::size_t u32_count = 0u;
  std::size_t u64_count = 0u;
  std::size_t i32_count = 0u;
  std::size_t i64_count = 0u;
  CpuExecutionStoragePlan next = arena;
  const auto add_count = [](std::size_t &target,
                            const std::size_t value) noexcept {
    if (value > std::numeric_limits<std::size_t>::max() - target) {
      return false;
    }
    target += value;
    return true;
  };
  const auto finish = [&]() noexcept {
    constexpr std::uint64_t alignment = alignof(std::max_align_t);
    std::uint64_t storage_bytes = 0u;
    if (scratch.host_bytes == 0u ||
        !kernel::checked::align_up(scratch.host_bytes, alignment,
                                   storage_bytes) ||
        storage_bytes > std::numeric_limits<std::size_t>::max() ||
        scratch.host_bytes > std::numeric_limits<std::size_t>::max() ||
        !add_count(next.primitive_object_storage_bytes,
                   static_cast<std::size_t>(storage_bytes)) ||
        !add_count(next.primitive_object_payload_bytes,
                   static_cast<std::size_t>(scratch.host_bytes))) {
      return Status::fail(Reason::ProgramCapacity);
    }
    arena = next;
    return Status::success();
  };
  const auto add_lane = [&](const std::size_t value) noexcept {
    return scratch.element_bytes == sizeof(kernel::i64)
               ? add_count(i64_count, value)
               : scratch.element_bytes == sizeof(kernel::i32) &&
                     add_count(i32_count, value);
  };
  switch (scratch.shape) {
  case CpuPrimitiveScratchShape::None:
    return Status::success();
  case CpuPrimitiveScratchShape::Sort:
    if (scratch.element_bytes == sizeof(kernel::u64)) {
      if (!add_count(u64_count, scratch.counts[0]) ||
          !add_count(u32_count, scratch.counts[1])) {
        return Status::fail(Reason::ProgramCapacity);
      }
    } else if (scratch.element_bytes != sizeof(kernel::u32) ||
               !add_count(u32_count, scratch.counts[0]) ||
               !add_count(u32_count, scratch.counts[1])) {
      return Status::fail(Reason::ProgramCapacity);
    }
    break;
  case CpuPrimitiveScratchShape::Scatter:
    if (scratch.counts[0] == 0u || scratch.counts[0] != scratch.counts[1]) {
      return Status::fail(Reason::CpuRuntimePlanInvalid);
    }
    next.scatter_slot_count =
        std::max(next.scatter_slot_count, scratch.counts[0]);
    break;
  case CpuPrimitiveScratchShape::ScatterReduce:
    u32_count = scratch.counts[0];
    break;
  case CpuPrimitiveScratchShape::Transform:
    if (scratch.element_bytes == sizeof(kernel::i64)) {
      if (!add_count(next.transform_i64_count, scratch.counts[0])) {
        return Status::fail(Reason::ProgramCapacity);
      }
    } else if (scratch.element_bytes == sizeof(kernel::i32)) {
      if (!add_count(next.transform_i32_count, scratch.counts[0])) {
        return Status::fail(Reason::ProgramCapacity);
      }
    } else {
      return Status::fail(Reason::CpuRuntimePlanInvalid);
    }
    return finish();
  case CpuPrimitiveScratchShape::FactorQr:
  case CpuPrimitiveScratchShape::SpectrumEigen:
  case CpuPrimitiveScratchShape::SpectrumSvdValues:
    if (!add_lane(scratch.counts[0]) || !add_lane(scratch.counts[1]) ||
        !add_lane(scratch.counts[2])) {
      return Status::fail(Reason::ProgramCapacity);
    }
    if (scratch.shape == CpuPrimitiveScratchShape::SpectrumSvdValues) {
      u64_count = scratch.counts[4];
    }
    break;
  case CpuPrimitiveScratchShape::SolveQrFactor:
  case CpuPrimitiveScratchShape::SolveCholesky:
    if (!add_lane(scratch.counts[0])) {
      return Status::fail(Reason::ProgramCapacity);
    }
    break;
  case CpuPrimitiveScratchShape::SolveLu:
    if (!add_lane(scratch.counts[0])) {
      return Status::fail(Reason::ProgramCapacity);
    }
    u32_count = scratch.counts[1];
    break;
  case CpuPrimitiveScratchShape::SolveQrMatrix:
  case CpuPrimitiveScratchShape::SpectrumSvdVectors:
    if (!add_lane(scratch.counts[0]) || !add_lane(scratch.counts[1]) ||
        !add_lane(scratch.counts[2]) || !add_lane(scratch.counts[3])) {
      return Status::fail(Reason::ProgramCapacity);
    }
    if (scratch.shape == CpuPrimitiveScratchShape::SpectrumSvdVectors) {
      u64_count = scratch.counts[4];
    }
    break;
  }
  next.primitive_u32_count = std::max(next.primitive_u32_count, u32_count);
  next.primitive_u64_count = std::max(next.primitive_u64_count, u64_count);
  next.primitive_i32_count = std::max(next.primitive_i32_count, i32_count);
  next.primitive_i64_count = std::max(next.primitive_i64_count, i64_count);
  return finish();
}

Result<CpuPrimitiveScratch>
prepare_cpu_scratch(const CpuRuntimePrimitive &primitive,
                    const CpuPrimitiveScratchPlan &request,
                    CpuPreparedArena &arena) {
  const auto planned = plan_cpu_scratch(primitive);
  if (!planned || *planned != request ||
      !scratch_shape_matches(primitive.kind, request.shape)) {
    return Result<CpuPrimitiveScratch>::fail(Reason::CpuRuntimePlanInvalid);
  }
  if (request.shape == CpuPrimitiveScratchShape::None) {
    return no_cpu_scratch();
  }

  switch (request.shape) {
  case CpuPrimitiveScratchShape::Sort: {
    const auto *const plan = active_plan<kernel::SortPlan>(primitive);
    if (plan == nullptr) {
      return Result<CpuPrimitiveScratch>::fail(Reason::CpuRuntimePlanInvalid);
    }
    std::size_t word_cursor = 0u;
    if (plan->key == kernel::SortKey::U64) {
      auto *const prepared =
          arena.claim_primitive_object<CpuSortPrimitiveScratch<kernel::u64>>();
      std::size_t key_cursor = 0u;
      if (prepared == nullptr ||
          !bind_buffer(prepared->keys, arena.primitive_u64(), key_cursor,
                       request.counts[0]) ||
          !bind_buffer(prepared->values, arena.primitive_u32(), word_cursor,
                       request.counts[1])) {
        return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
      }
      return prepared_cpu_scratch(prepared);
    }
    auto *const prepared =
        arena.claim_primitive_object<CpuSortPrimitiveScratch<kernel::u32>>();
    if (prepared == nullptr ||
        !bind_buffer(prepared->keys, arena.primitive_u32(), word_cursor,
                     request.counts[0]) ||
        !bind_buffer(prepared->values, arena.primitive_u32(), word_cursor,
                     request.counts[1])) {
      return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
    }
    return prepared_cpu_scratch(prepared);
  }
  case CpuPrimitiveScratchShape::Scatter: {
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
  case CpuPrimitiveScratchShape::ScatterReduce: {
    auto *const prepared =
        arena.claim_primitive_object<CpuScatterReducePrimitiveScratch>();
    std::size_t cursor = 0u;
    if (prepared == nullptr ||
        !bind_buffer(prepared->sorted_indices, arena.primitive_u32(), cursor,
                     request.counts[0])) {
      return Result<CpuPrimitiveScratch>::fail(Reason::BufferCapacity);
    }
    return prepared_cpu_scratch(prepared);
  }
  case CpuPrimitiveScratchShape::Transform: {
    const auto *const plan = active_plan<kernel::TransformPlan>(primitive);
    if (plan == nullptr) {
      return Result<CpuPrimitiveScratch>::fail(Reason::CpuRuntimePlanInvalid);
    }
    return plan->element_bytes == sizeof(kernel::i64)
               ? prepare_borrowed_transform<kernel::i64>(*plan, request, arena)
               : prepare_borrowed_transform<kernel::i32>(*plan, request, arena);
  }
  case CpuPrimitiveScratchShape::FactorQr:
    return request.element_bytes == sizeof(kernel::i64)
               ? prepare_borrowed_factor<kernel::i64>(
                     request, arena.primitive_i64(), arena)
               : prepare_borrowed_factor<kernel::i32>(
                     request, arena.primitive_i32(), arena);
  case CpuPrimitiveScratchShape::SolveQrFactor:
  case CpuPrimitiveScratchShape::SolveLu:
  case CpuPrimitiveScratchShape::SolveCholesky:
  case CpuPrimitiveScratchShape::SolveQrMatrix:
    return request.element_bytes == sizeof(kernel::i64)
               ? prepare_borrowed_solve<kernel::i64>(
                     request, arena.primitive_i64(), arena.primitive_u32(),
                     arena)
               : prepare_borrowed_solve<kernel::i32>(
                     request, arena.primitive_i32(), arena.primitive_u32(),
                     arena);
  case CpuPrimitiveScratchShape::SpectrumEigen:
  case CpuPrimitiveScratchShape::SpectrumSvdValues:
  case CpuPrimitiveScratchShape::SpectrumSvdVectors:
    return request.element_bytes == sizeof(kernel::i64)
               ? prepare_borrowed_spectrum<kernel::i64>(
                     request, arena.primitive_i64(), arena.primitive_u64(),
                     arena)
               : prepare_borrowed_spectrum<kernel::i32>(
                     request, arena.primitive_i32(), arena.primitive_u64(),
                     arena);
  case CpuPrimitiveScratchShape::None:
    return no_cpu_scratch();
  }
  return Result<CpuPrimitiveScratch>::fail(Reason::CpuRuntimePlanInvalid);
}

} // namespace rund::compute::detail
