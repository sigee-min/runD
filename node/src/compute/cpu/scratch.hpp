#pragma once

#include "graph.hpp"
#include "state.hpp"

namespace rund::compute::detail {

enum class CpuPrimitiveScratchShape : std::uint8_t {
  None,
  Sort,
  Scatter,
  ScatterReduce,
  Transform,
  FactorQr,
  SolveQrFactor,
  SolveLu,
  SolveCholesky,
  SolveQrMatrix,
  SpectrumEigen,
  SpectrumSvdValues,
  SpectrumSvdVectors,
};

// Frozen request descriptor shared by Pipeline preflight and the allocator.
// `counts` is shape-specific. CpuExecutionStoragePlan is the sole aggregate
// byte authority; the materializer may only consume these exact element
// counts.
struct CpuPrimitiveScratchPlan final {
  CpuPrimitiveScratchShape shape{CpuPrimitiveScratchShape::None};
  std::array<std::size_t, 5u> counts{};
  std::uint64_t host_bytes{};
  std::uint8_t element_bytes{};

  [[nodiscard]] constexpr bool
  operator==(const CpuPrimitiveScratchPlan &) const noexcept = default;
};

template <class Scratch>
[[nodiscard]] Scratch *
cpu_primitive_scratch(CpuPrimitiveScratch &scratch) noexcept {
  auto *const prepared = std::get_if<Scratch *>(&scratch);
  return prepared == nullptr ? nullptr : *prepared;
}

template <class Scratch>
[[nodiscard]] const Scratch *
cpu_primitive_scratch(const CpuPrimitiveScratch &scratch) noexcept {
  const auto *const prepared = std::get_if<Scratch *>(&scratch);
  return prepared == nullptr ? nullptr : *prepared;
}

[[nodiscard]] inline CpuPrimitiveScratch &
cpu_step_scratch(CpuGraphRun &run, const std::size_t step) noexcept {
  return run.storage == nullptr || run.storage->scratch.empty() ||
                 step >= run.storage->scratch.size()
             ? (run.storage == nullptr ? run.empty_scratch
                                       : run.storage->empty_scratch)
             : run.storage->scratch[step];
}

template <class Lane>
[[nodiscard]] const Lane *
cpu_transform_twiddle(CpuPrimitiveScratch &scratch) noexcept {
  auto *const owner = cpu_primitive_scratch<CpuTransformScratch<Lane>>(scratch);
  return owner == nullptr ? nullptr : owner->twiddle.data();
}

template <class Lane> struct CpuFactorScratchView final {
  Lane *q = nullptr;
  Lane *r = nullptr;
  Lane *v = nullptr;
};

template <class Lane>
[[nodiscard]] bool
cpu_factor_scratch(CpuPrimitiveScratch &scratch, const kernel::FactorPlan &plan,
                   CpuFactorScratchView<Lane> &view) noexcept {
  view = {};
  if (plan.op != kernel::FactorOp::QR) {
    return std::holds_alternative<std::monostate>(scratch);
  }
  auto *const owner = cpu_primitive_scratch<CpuFactorQrScratch<Lane>>(scratch);
  if (owner == nullptr) {
    return false;
  }
  view.q = owner->values[0].data();
  view.r = owner->values[1].data();
  view.v = owner->values[2].data();
  return true;
}

template <class Lane> struct CpuSolveScratchView final {
  Lane *factor = nullptr;
  kernel::u32 *pivots = nullptr;
  Lane *y = nullptr;
  Lane *q = nullptr;
  Lane *r = nullptr;
  Lane *v = nullptr;
};

template <class Lane>
[[nodiscard]] bool cpu_solve_scratch(CpuPrimitiveScratch &scratch,
                                     const kernel::SolvePlan &plan,
                                     CpuSolveScratchView<Lane> &view) noexcept {
  view = {};
  if (plan.input == kernel::SolveInput::Factor) {
    if (plan.factor != kernel::FactorOp::QR) {
      return std::holds_alternative<std::monostate>(scratch);
    }
    auto *const owner =
        cpu_primitive_scratch<CpuSolveQrFactorScratch<Lane>>(scratch);
    if (owner == nullptr) {
      return false;
    }
    view.y = owner->y.data();
    return true;
  }
  if (plan.input != kernel::SolveInput::Matrix) {
    return false;
  }
  if (plan.factor == kernel::FactorOp::LU) {
    auto *const owner = cpu_primitive_scratch<CpuSolveLuScratch<Lane>>(scratch);
    if (owner == nullptr) {
      return false;
    }
    view.factor = owner->factor.data();
    view.pivots = owner->pivots.data();
    return true;
  }
  if (plan.factor == kernel::FactorOp::Cholesky) {
    auto *const owner =
        cpu_primitive_scratch<CpuSolveCholeskyScratch<Lane>>(scratch);
    if (owner == nullptr) {
      return false;
    }
    view.factor = owner->factor.data();
    return true;
  }
  if (plan.factor != kernel::FactorOp::QR) {
    return false;
  }
  auto *const owner =
      cpu_primitive_scratch<CpuSolveQrMatrixScratch<Lane>>(scratch);
  if (owner == nullptr) {
    return false;
  }
  view.y = owner->y.data();
  view.q = owner->qr[0].data();
  view.r = owner->qr[1].data();
  view.v = owner->qr[2].data();
  return true;
}

template <class Lane> struct CpuSpectrumScratchView final {
  Lane *matrix = nullptr;
  Lane *vectors = nullptr;
  Lane *values = nullptr;
  kernel::u64 *order = nullptr;
  Lane *u = nullptr;
};

template <class Lane>
[[nodiscard]] bool
cpu_spectrum_scratch(CpuPrimitiveScratch &scratch,
                     const kernel::SpectrumPlan &plan,
                     CpuSpectrumScratchView<Lane> &view) noexcept {
  view = {};
  if (plan.op == kernel::SpectrumOp::Eigen) {
    auto *const owner =
        cpu_primitive_scratch<CpuSpectrumEigenScratch<Lane>>(scratch);
    if (owner == nullptr) {
      return false;
    }
    view.matrix = owner->values[0].data();
    view.vectors = owner->values[1].data();
    view.values = owner->values[2].data();
    return true;
  }
  if (plan.op != kernel::SpectrumOp::SVD) {
    return false;
  }
  if (plan.vector_count == 0u) {
    auto *const owner =
        cpu_primitive_scratch<CpuSpectrumSvdValuesScratch<Lane>>(scratch);
    if (owner == nullptr) {
      return false;
    }
    view.matrix = owner->values[0].data();
    view.vectors = owner->values[1].data();
    view.values = owner->values[2].data();
    view.order = owner->order.data();
    return true;
  }
  auto *const owner =
      cpu_primitive_scratch<CpuSpectrumSvdVectorsScratch<Lane>>(scratch);
  if (owner == nullptr) {
    return false;
  }
  view.matrix = owner->values[0].data();
  view.vectors = owner->values[1].data();
  view.values = owner->values[2].data();
  view.u = owner->values[3].data();
  view.order = owner->order.data();
  return true;
}

[[nodiscard]] Result<CpuPrimitiveScratch>
prepare_cpu_scratch(const CpuRuntimePrimitive &primitive,
                    const CpuPrimitiveScratchPlan &plan,
                    CpuPreparedArena &arena);
[[nodiscard]] Result<CpuPrimitiveScratchPlan>
plan_cpu_scratch(const CpuRuntimePrimitive &primitive) noexcept;
[[nodiscard]] Status append_cpu_primitive_arena_plan(
    CpuExecutionStoragePlan &arena,
    const CpuPrimitiveScratchPlan &scratch) noexcept;

} // namespace rund::compute::detail
