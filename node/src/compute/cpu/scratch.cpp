#include "scratch.hpp"

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/transform/twiddle.hpp>

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace rund::compute::detail {
namespace {

[[nodiscard]] Result<CpuPrimitiveScratch> no_cpu_scratch() {
  return Result<CpuPrimitiveScratch>::success(CpuPrimitiveScratch{});
}

template <class Scratch>
[[nodiscard]] Result<CpuPrimitiveScratch>
owned_cpu_scratch(std::unique_ptr<Scratch> owner) {
  CpuPrimitiveScratch scratch{std::in_place_type<std::unique_ptr<Scratch>>,
                              std::move(owner)};
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

template <class Lane>
[[nodiscard]] Result<CpuPrimitiveScratch>
prepare_factor_qr(const kernel::FactorPlan &plan) {
  std::size_t q_count = 0u;
  std::size_t r_count = 0u;
  std::size_t v_count = 0u;
  if (!product_size(plan.rows, plan.cols, q_count) ||
      !product_size(plan.cols, plan.cols, r_count) ||
      !to_size(plan.rows, v_count)) {
    return Result<CpuPrimitiveScratch>::fail(Reason::ProgramCapacity);
  }
  auto owner = std::make_unique<CpuFactorQrScratch<Lane>>();
  owner->values[0].allocate(q_count);
  owner->values[1].allocate(r_count);
  owner->values[2].allocate(v_count);
  return owned_cpu_scratch(std::move(owner));
}

template <class Lane>
[[nodiscard]] Result<CpuPrimitiveScratch>
prepare_transform(const kernel::TransformPlan &plan) {
  std::size_t count = 0u;
  if (!product_size(plan.twiddle_count, 2u, count)) {
    return Result<CpuPrimitiveScratch>::fail(Reason::ProgramCapacity);
  }
  auto owner = std::make_unique<CpuTransformScratch<Lane>>();
  owner->twiddle.allocate(count);
  if (!kernel::transform_twiddle::Fill(owner->twiddle.data(),
                                       plan.element_count, plan.direction,
                                       plan.fixed_format)) {
    return Result<CpuPrimitiveScratch>::fail(Reason::CpuRuntimePlanInvalid);
  }
  return owned_cpu_scratch(std::move(owner));
}

template <class Lane>
[[nodiscard]] Result<CpuPrimitiveScratch>
prepare_solve(const kernel::SolvePlan &plan) {
  if (plan.input == kernel::SolveInput::Factor) {
    if (plan.factor != kernel::FactorOp::QR) {
      return no_cpu_scratch();
    }
    std::size_t y_count = 0u;
    if (!product_size(plan.rows, plan.rhs_cols, y_count)) {
      return Result<CpuPrimitiveScratch>::fail(Reason::ProgramCapacity);
    }
    auto owner = std::make_unique<CpuSolveQrFactorScratch<Lane>>();
    owner->y.allocate(y_count);
    return owned_cpu_scratch(std::move(owner));
  }
  if (plan.input != kernel::SolveInput::Matrix) {
    return Result<CpuPrimitiveScratch>::fail(Reason::CpuRuntimePlanInvalid);
  }
  std::size_t factor_count = 0u;
  if (!to_size(plan.factor_count, factor_count)) {
    return Result<CpuPrimitiveScratch>::fail(Reason::ProgramCapacity);
  }
  if (plan.factor == kernel::FactorOp::LU) {
    std::size_t pivot_count = 0u;
    if (!to_size(plan.aux_count, pivot_count)) {
      return Result<CpuPrimitiveScratch>::fail(Reason::ProgramCapacity);
    }
    auto owner = std::make_unique<CpuSolveLuScratch<Lane>>();
    owner->factor.allocate(factor_count);
    owner->pivots.allocate(pivot_count);
    return owned_cpu_scratch(std::move(owner));
  }
  if (plan.factor == kernel::FactorOp::Cholesky) {
    auto owner = std::make_unique<CpuSolveCholeskyScratch<Lane>>();
    owner->factor.allocate(factor_count);
    return owned_cpu_scratch(std::move(owner));
  }
  if (plan.factor != kernel::FactorOp::QR) {
    return Result<CpuPrimitiveScratch>::fail(Reason::CpuRuntimePlanInvalid);
  }
  std::size_t y_count = 0u;
  std::size_t matrix_count = 0u;
  std::size_t vector_count = 0u;
  if (!product_size(plan.rows, plan.rhs_cols, y_count) ||
      !product_size(plan.rows, plan.rows, matrix_count) ||
      !to_size(plan.rows, vector_count)) {
    return Result<CpuPrimitiveScratch>::fail(Reason::ProgramCapacity);
  }
  auto owner = std::make_unique<CpuSolveQrMatrixScratch<Lane>>();
  owner->y.allocate(y_count);
  owner->qr[0].allocate(matrix_count);
  owner->qr[1].allocate(matrix_count);
  owner->qr[2].allocate(vector_count);
  return owned_cpu_scratch(std::move(owner));
}

template <class Lane>
[[nodiscard]] Result<CpuPrimitiveScratch>
prepare_spectrum(const kernel::SpectrumPlan &plan) {
  const kernel::u64 n =
      plan.op == kernel::SpectrumOp::Eigen ? plan.rows : plan.cols;
  std::size_t matrix_count = 0u;
  std::size_t value_count = 0u;
  if (!product_size(n, n, matrix_count) || !to_size(n, value_count)) {
    return Result<CpuPrimitiveScratch>::fail(Reason::ProgramCapacity);
  }
  if (plan.op == kernel::SpectrumOp::Eigen) {
    auto owner = std::make_unique<CpuSpectrumEigenScratch<Lane>>();
    owner->values[0].allocate(matrix_count);
    owner->values[1].allocate(matrix_count);
    owner->values[2].allocate(value_count);
    return owned_cpu_scratch(std::move(owner));
  }
  if (plan.op != kernel::SpectrumOp::SVD) {
    return Result<CpuPrimitiveScratch>::fail(Reason::CpuRuntimePlanInvalid);
  }
  if (plan.vector_count == 0u) {
    auto owner = std::make_unique<CpuSpectrumSvdValuesScratch<Lane>>();
    owner->values[0].allocate(matrix_count);
    owner->values[1].allocate(matrix_count);
    owner->values[2].allocate(value_count);
    owner->order.allocate(value_count);
    return owned_cpu_scratch(std::move(owner));
  }
  std::size_t u_count = 0u;
  const kernel::u64 vector_cols = plan.vectors == kernel::SpectrumVectors::Thin
                                      ? std::min(plan.rows, plan.cols)
                                      : plan.rows;
  if (!product_size(plan.rows, vector_cols, u_count)) {
    return Result<CpuPrimitiveScratch>::fail(Reason::ProgramCapacity);
  }
  auto owner = std::make_unique<CpuSpectrumSvdVectorsScratch<Lane>>();
  owner->values[0].allocate(matrix_count);
  owner->values[1].allocate(matrix_count);
  owner->values[2].allocate(value_count);
  owner->values[3].allocate(u_count);
  owner->order.allocate(value_count);
  return owned_cpu_scratch(std::move(owner));
}

} // namespace

Result<CpuPrimitiveScratch>
prepare_cpu_scratch(const CpuRuntimePrimitive &primitive) {
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
    return no_cpu_scratch();
  case Primitive::Sort:
  case Primitive::Argsort:
  case Primitive::Scatter:
  case Primitive::ScatterReduce:
  case Primitive::Transform:
  case Primitive::Factor:
  case Primitive::Solve:
  case Primitive::Spectrum:
    break;
  }

  try {
    switch (primitive.kind) {
    case Primitive::Sort:
    case Primitive::Argsort: {
      const auto *const plan = active_plan<kernel::SortPlan>(primitive);
      std::size_t count = 0u;
      if (plan == nullptr || !to_size(plan->element_count, count) ||
          (plan->key != kernel::SortKey::U32 &&
           plan->key != kernel::SortKey::U64)) {
        return Result<CpuPrimitiveScratch>::fail(Reason::CpuRuntimePlanInvalid);
      }
      if (plan->key == kernel::SortKey::U64) {
        auto owner = std::make_unique<CpuSortPrimitiveScratch<kernel::u64>>();
        owner->keys.allocate(count);
        owner->values.allocate(count);
        return owned_cpu_scratch(std::move(owner));
      }
      auto owner = std::make_unique<CpuSortPrimitiveScratch<kernel::u32>>();
      owner->keys.allocate(count);
      owner->values.allocate(count);
      return owned_cpu_scratch(std::move(owner));
    }
    case Primitive::Scatter: {
      const auto *const plan = active_plan<kernel::ScatterPlan>(primitive);
      std::size_t capacity = 0u;
      if (plan == nullptr || !to_size(plan->scratch_slots, capacity) ||
          capacity == 0u) {
        return Result<CpuPrimitiveScratch>::fail(Reason::ScatterTempOverflow);
      }
      auto owner = std::make_unique<CpuScatterPrimitiveScratch>();
      owner->values.keys.resize(capacity);
      owner->values.marks.resize(capacity);
      return owned_cpu_scratch(std::move(owner));
    }
    case Primitive::ScatterReduce: {
      const auto *const plan =
          active_plan<kernel::ScatterReducePlan>(primitive);
      std::size_t count = 0u;
      if (plan == nullptr || !to_size(plan->element_count, count)) {
        return Result<CpuPrimitiveScratch>::fail(Reason::CpuRuntimePlanInvalid);
      }
      auto owner = std::make_unique<CpuScatterReducePrimitiveScratch>();
      owner->sorted_indices.allocate(count);
      return owned_cpu_scratch(std::move(owner));
    }
    case Primitive::Transform: {
      const auto *const plan = active_plan<kernel::TransformPlan>(primitive);
      if (plan == nullptr || (plan->element_bytes != sizeof(kernel::i32) &&
                              plan->element_bytes != sizeof(kernel::i64))) {
        return Result<CpuPrimitiveScratch>::fail(Reason::CpuRuntimePlanInvalid);
      }
      return plan->element_bytes == sizeof(kernel::i64)
                 ? prepare_transform<kernel::i64>(*plan)
                 : prepare_transform<kernel::i32>(*plan);
    }
    case Primitive::Factor: {
      const auto *const plan = active_plan<kernel::FactorPlan>(primitive);
      if (plan == nullptr || (plan->element_bytes != sizeof(kernel::i32) &&
                              plan->element_bytes != sizeof(kernel::i64))) {
        return Result<CpuPrimitiveScratch>::fail(Reason::CpuRuntimePlanInvalid);
      }
      if (plan->op != kernel::FactorOp::QR) {
        return no_cpu_scratch();
      }
      return plan->element_bytes == sizeof(kernel::i64)
                 ? prepare_factor_qr<kernel::i64>(*plan)
                 : prepare_factor_qr<kernel::i32>(*plan);
    }
    case Primitive::Solve: {
      const auto *const plan = active_plan<kernel::SolvePlan>(primitive);
      if (plan == nullptr || (plan->element_bytes != sizeof(kernel::i32) &&
                              plan->element_bytes != sizeof(kernel::i64))) {
        return Result<CpuPrimitiveScratch>::fail(Reason::CpuRuntimePlanInvalid);
      }
      return plan->element_bytes == sizeof(kernel::i64)
                 ? prepare_solve<kernel::i64>(*plan)
                 : prepare_solve<kernel::i32>(*plan);
    }
    case Primitive::Spectrum: {
      const auto *const plan = active_plan<kernel::SpectrumPlan>(primitive);
      if (plan == nullptr || (plan->element_bytes != sizeof(kernel::i32) &&
                              plan->element_bytes != sizeof(kernel::i64))) {
        return Result<CpuPrimitiveScratch>::fail(Reason::CpuRuntimePlanInvalid);
      }
      return plan->element_bytes == sizeof(kernel::i64)
                 ? prepare_spectrum<kernel::i64>(*plan)
                 : prepare_spectrum<kernel::i32>(*plan);
    }
    case Primitive::SegmentedScan:
    case Primitive::SegmentedReduce:
    case Primitive::Compact:
    case Primitive::Gather:
    case Primitive::Histogram:
    case Primitive::Partition:
    case Primitive::Reduce:
    case Primitive::Stencil:
    case Primitive::Matrix:
      break;
    }
  } catch (const std::bad_alloc &) {
    return Result<CpuPrimitiveScratch>::fail(Reason::ProgramCapacity);
  } catch (const std::length_error &) {
    return Result<CpuPrimitiveScratch>::fail(Reason::ProgramCapacity);
  }
  return Result<CpuPrimitiveScratch>::fail(Reason::CpuRuntimePlanInvalid);
}

} // namespace rund::compute::detail
