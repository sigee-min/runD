#include "support.hpp"

namespace rund_node_memory_contract::cpu_primitive {
namespace {

template <class Lane> [[nodiscard]] int CheckTypedCpuPrimitiveScratch() {
  using namespace rund::compute::detail;
  using namespace rund::kernel;
  constexpr std::uint32_t width = sizeof(Lane);
  const ComputeFixedFormat format = ScratchFixedFormat(width);

  const TransformPlan transform = PlanTransform(TransformDesc{
      .op = TransformOp::Fourier,
      .direction = TransformDir::Forward,
      .layout = TransformLayout::Split,
      .normalization = TransformNorm::None,
      .element_count = 8u,
      .fixed_format = format,
  });
  if (!transform.ok ||
      !CheckArenaCpuPrimitiveScratch<CpuTransformScratch<Lane>>(
          ScratchPrimitive(Primitive::Transform, transform),
          transform.workspace_bytes)) {
    return 12;
  }

  const FactorPlan factor_qr = PlanFactor(FactorDesc{
      .op = FactorOp::QR,
      .layout = MatrixLayout::RowMajor,
      .output = FactorOutput::Separate,
      .pivot = PivotOp::None,
      .rows = 2u,
      .cols = 2u,
      .batch_count = 3u,
      .element_bytes = width,
      .fixed_format = format,
  });
  if (!factor_qr.ok ||
      !CheckArenaCpuPrimitiveScratch<CpuFactorQrScratch<Lane>>(
          ScratchPrimitive(Primitive::Factor, factor_qr), 10u * width)) {
    return 1;
  }

  for (const FactorOp op : {FactorOp::LU, FactorOp::Cholesky}) {
    const FactorPlan plan = PlanFactor(FactorDesc{
        .op = op,
        .layout = MatrixLayout::RowMajor,
        .output = FactorOutput::Packed,
        .pivot = op == FactorOp::LU ? PivotOp::Partial : PivotOp::None,
        .rows = 2u,
        .cols = 2u,
        .batch_count = 3u,
        .element_bytes = width,
        .fixed_format = format,
    });
    if (!plan.ok || !CheckNoCpuPrimitiveScratch(
                        ScratchPrimitive(Primitive::Factor, plan))) {
      return 2;
    }
  }

  const auto solve_plan = [&](const SolveInput input, const FactorOp factor) {
    return PlanSolve(SolveDesc{
        .op = SolveOp::Linear,
        .input = input,
        .factor = factor,
        .layout = MatrixLayout::RowMajor,
        .pivot = factor == FactorOp::LU ? PivotOp::Partial : PivotOp::None,
        .rows = 2u,
        .rhs_cols = 1u,
        .batch_count = 3u,
        .element_bytes = width,
        .fixed_format = format,
    });
  };
  for (const FactorOp factor : {FactorOp::LU, FactorOp::Cholesky}) {
    const SolvePlan plan = solve_plan(SolveInput::Factor, factor);
    if (!plan.ok ||
        !CheckNoCpuPrimitiveScratch(ScratchPrimitive(Primitive::Solve, plan))) {
      return 3;
    }
  }
  const SolvePlan factor_qr_solve =
      solve_plan(SolveInput::Factor, FactorOp::QR);
  if (!factor_qr_solve.ok ||
      !CheckArenaCpuPrimitiveScratch<CpuSolveQrFactorScratch<Lane>>(
          ScratchPrimitive(Primitive::Solve, factor_qr_solve), 2u * width)) {
    return 4;
  }
  const SolvePlan matrix_lu = solve_plan(SolveInput::Matrix, FactorOp::LU);
  if (!matrix_lu.ok ||
      !CheckArenaCpuPrimitiveScratch<CpuSolveLuScratch<Lane>>(
          ScratchPrimitive(Primitive::Solve, matrix_lu),
          matrix_lu.factor_count * width +
              matrix_lu.aux_count * sizeof(rund::kernel::u32))) {
    return 5;
  }
  const SolvePlan matrix_cholesky =
      solve_plan(SolveInput::Matrix, FactorOp::Cholesky);
  if (!matrix_cholesky.ok ||
      !CheckArenaCpuPrimitiveScratch<CpuSolveCholeskyScratch<Lane>>(
          ScratchPrimitive(Primitive::Solve, matrix_cholesky),
          matrix_cholesky.factor_count * width)) {
    return 6;
  }
  const SolvePlan matrix_qr = solve_plan(SolveInput::Matrix, FactorOp::QR);
  if (!matrix_qr.ok ||
      !CheckArenaCpuPrimitiveScratch<CpuSolveQrMatrixScratch<Lane>>(
          ScratchPrimitive(Primitive::Solve, matrix_qr),
          (matrix_qr.rows * matrix_qr.rhs_cols +
           2u * matrix_qr.rows * matrix_qr.rows + matrix_qr.rows) *
              width)) {
    return 7;
  }

  const auto spectrum_plan =
      [&](const SpectrumOp op, const SpectrumVectors vectors,
          const std::uint64_t rows, const std::uint64_t cols) {
        return PlanSpectrum(SpectrumDesc{
            .op = op,
            .domain = op == SpectrumOp::Eigen ? SpectrumDomain::SymmetricReal
                                              : SpectrumDomain::GeneralReal,
            .vectors = vectors,
            .layout = MatrixLayout::RowMajor,
            .rows = rows,
            .cols = cols,
            .batch_count = 3u,
            .max_iterations = 8u,
            .element_bytes = width,
            .fixed_format = format,
        });
      };
  const SpectrumPlan eigen =
      spectrum_plan(SpectrumOp::Eigen, SpectrumVectors::ValuesOnly, 2u, 2u);
  if (!eigen.ok ||
      !CheckArenaCpuPrimitiveScratch<CpuSpectrumEigenScratch<Lane>>(
          ScratchPrimitive(Primitive::Spectrum, eigen), 10u * width)) {
    return 8;
  }
  const SpectrumPlan svd_values =
      spectrum_plan(SpectrumOp::SVD, SpectrumVectors::ValuesOnly, 3u, 2u);
  if (!svd_values.ok ||
      !CheckArenaCpuPrimitiveScratch<CpuSpectrumSvdValuesScratch<Lane>>(
          ScratchPrimitive(Primitive::Spectrum, svd_values),
          10u * width + 2u * sizeof(rund::kernel::u64))) {
    return 9;
  }
  const SpectrumPlan svd_vectors =
      spectrum_plan(SpectrumOp::SVD, SpectrumVectors::Thin, 3u, 2u);
  if (!svd_vectors.ok ||
      !CheckArenaCpuPrimitiveScratch<CpuSpectrumSvdVectorsScratch<Lane>>(
          ScratchPrimitive(Primitive::Spectrum, svd_vectors),
          16u * width + 2u * sizeof(rund::kernel::u64))) {
    return 10;
  }
  const SpectrumPlan svd_full =
      spectrum_plan(SpectrumOp::SVD, SpectrumVectors::Full, 3u, 2u);
  if (!svd_full.ok ||
      !CheckArenaCpuPrimitiveScratch<CpuSpectrumSvdVectorsScratch<Lane>>(
          ScratchPrimitive(Primitive::Spectrum, svd_full),
          19u * width + 2u * sizeof(rund::kernel::u64))) {
    return 11;
  }
  return 0;
}

} // namespace

int CheckTypedI32() {
  return CheckTypedCpuPrimitiveScratch<rund::kernel::i32>();
}

int CheckTypedI64() {
  return CheckTypedCpuPrimitiveScratch<rund::kernel::i64>();
}

} // namespace rund_node_memory_contract::cpu_primitive
