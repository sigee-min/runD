#include "model.hpp"

namespace program_compute_contract::numeric_algebra_contract {
namespace {

int test_primitive_identity_covers_every_fixed_policy_axis() {
  const rund::kernel::MatrixDesc matrix{
      .op = rund::kernel::MatrixOp::Mul,
      .arithmetic = rund::kernel::MatrixArithmetic::Fixed,
      .rows = 2u,
      .cols = 2u,
      .inner = 2u,
      .batch_count = 1u,
      .fixed_format = kFixedI1F31,
  };
  const rund::kernel::FactorDesc factor{
      .op = rund::kernel::FactorOp::LU,
      .pivot = rund::kernel::PivotOp::Partial,
      .rows = 2u,
      .cols = 2u,
      .batch_count = 1u,
      .fixed_format = kFixedI1F31,
  };
  const rund::kernel::SolveDesc solve{
      .op = rund::kernel::SolveOp::Linear,
      .input = rund::kernel::SolveInput::Matrix,
      .factor = rund::kernel::FactorOp::LU,
      .pivot = rund::kernel::PivotOp::Partial,
      .rows = 2u,
      .rhs_cols = 1u,
      .batch_count = 1u,
      .fixed_format = kFixedI1F31,
  };
  const rund::kernel::SpectrumDesc spectrum{
      .op = rund::kernel::SpectrumOp::Eigen,
      .domain = rund::kernel::SpectrumDomain::SymmetricReal,
      .vectors = rund::kernel::SpectrumVectors::ValuesOnly,
      .rows = 2u,
      .cols = 2u,
      .batch_count = 1u,
      .max_iterations = 8u,
      .fixed_format = kFixedI1F31,
  };
  const rund::kernel::TransformDesc transform{
      .op = rund::kernel::TransformOp::Fourier,
      .direction = rund::kernel::TransformDir::Forward,
      .layout = rund::kernel::TransformLayout::Split,
      .normalization = rund::kernel::TransformNorm::None,
      .element_count = 8u,
      .fixed_format = kFixedI1F31,
  };

  TEST_ASSERT(CheckFixedFormatIdentityAxes(matrix, [](const auto &value) {
                return rund::kernel::HashMatrix(value);
              }) == 0);
  TEST_ASSERT(CheckFixedFormatIdentityAxes(factor, [](const auto &value) {
                return rund::kernel::HashFactor(value);
              }) == 0);
  TEST_ASSERT(CheckFixedFormatIdentityAxes(solve, [](const auto &value) {
                return rund::kernel::HashSolve(value);
              }) == 0);
  TEST_ASSERT(CheckFixedFormatIdentityAxes(spectrum, [](const auto &value) {
                return rund::kernel::HashSpectrum(value);
              }) == 0);
  TEST_ASSERT(CheckFixedFormatIdentityAxes(transform, [](const auto &value) {
                return rund::kernel::HashTransform(value);
              }) == 0);
  return 0;
}

int test_primitive_plans_bind_fixed_format_and_policy() {
  rund::kernel::MatrixDesc matrix{
      .op = rund::kernel::MatrixOp::Mul,
      .arithmetic = rund::kernel::MatrixArithmetic::Fixed,
      .rows = 2u,
      .cols = 2u,
      .inner = 2u,
      .batch_count = 1u,
      .fixed_format = kFixedI1F31,
  };
  rund::kernel::FactorDesc factor{
      .op = rund::kernel::FactorOp::LU,
      .pivot = rund::kernel::PivotOp::Partial,
      .rows = 2u,
      .cols = 2u,
      .batch_count = 1u,
      .fixed_format = kFixedI1F31,
  };
  rund::kernel::SolveDesc solve{
      .op = rund::kernel::SolveOp::Linear,
      .input = rund::kernel::SolveInput::Matrix,
      .factor = rund::kernel::FactorOp::LU,
      .pivot = rund::kernel::PivotOp::Partial,
      .rows = 2u,
      .rhs_cols = 1u,
      .batch_count = 1u,
      .fixed_format = kFixedI1F31,
  };
  rund::kernel::SpectrumDesc spectrum{
      .op = rund::kernel::SpectrumOp::Eigen,
      .domain = rund::kernel::SpectrumDomain::SymmetricReal,
      .vectors = rund::kernel::SpectrumVectors::ValuesOnly,
      .rows = 2u,
      .cols = 2u,
      .batch_count = 1u,
      .max_iterations = 8u,
      .fixed_format = kFixedI1F31,
  };
  rund::kernel::TransformDesc transform{
      .op = rund::kernel::TransformOp::Fourier,
      .direction = rund::kernel::TransformDir::Forward,
      .layout = rund::kernel::TransformLayout::Split,
      .normalization = rund::kernel::TransformNorm::None,
      .element_count = 8u,
      .fixed_format = kFixedI1F31,
  };

  auto matrix_plan = rund::kernel::PlanMatrix(matrix);
  auto factor_plan = rund::kernel::PlanFactor(factor);
  auto solve_plan = rund::kernel::PlanSolve(solve);
  auto spectrum_plan = rund::kernel::PlanSpectrum(spectrum);
  auto transform_plan = rund::kernel::PlanTransform(transform);
  TEST_ASSERT(transform_plan.pass_count == 1u);
  TEST_ASSERT(transform_plan.element_bytes == sizeof(rund::kernel::i32));
  TEST_ASSERT(transform_plan.twiddle_count == 4u);
  TEST_ASSERT(transform_plan.workspace_bytes == 8u * sizeof(rund::kernel::i32));
  TEST_ASSERT(transform_plan.normalization_divisor == 1u);
  TEST_ASSERT(rund::kernel::MatrixPlanMatchesDesc(matrix, matrix_plan));
  TEST_ASSERT(rund::kernel::FactorPlanMatchesDesc(factor, factor_plan));
  TEST_ASSERT(rund::kernel::SolvePlanMatchesDesc(solve, solve_plan));
  TEST_ASSERT(rund::kernel::SpectrumPlanMatchesDesc(spectrum, spectrum_plan));
  TEST_ASSERT(
      rund::kernel::TransformPlanMatchesDesc(transform, transform_plan));

  auto inverse_transform = transform;
  inverse_transform.normalization = rund::kernel::TransformNorm::InverseLength;
  const auto inverse_plan = rund::kernel::PlanTransform(inverse_transform);
  TEST_ASSERT(inverse_plan.normalization_divisor == 8u);
  TEST_ASSERT(
      rund::kernel::TransformPlanMatchesDesc(inverse_transform, inverse_plan));
  auto unitary_transform = transform;
  unitary_transform.normalization = rund::kernel::TransformNorm::Unitary;
  const auto unitary_plan = rund::kernel::PlanTransform(unitary_transform);
  TEST_ASSERT(unitary_plan.normalization_divisor == 2u);
  TEST_ASSERT(
      rund::kernel::TransformPlanMatchesDesc(unitary_transform, unitary_plan));

  matrix_plan.fixed_format.fraction_bits = 30u;
  factor_plan.fixed_format.fraction_bits = 30u;
  solve_plan.fixed_format.fraction_bits = 30u;
  spectrum_plan.fixed_format.fraction_bits = 30u;
  transform_plan.fixed_format.fraction_bits = 30u;
  TEST_ASSERT(!rund::kernel::MatrixPlanMatchesDesc(matrix, matrix_plan));
  TEST_ASSERT(!rund::kernel::FactorPlanMatchesDesc(factor, factor_plan));
  TEST_ASSERT(!rund::kernel::SolvePlanMatchesDesc(solve, solve_plan));
  TEST_ASSERT(!rund::kernel::SpectrumPlanMatchesDesc(spectrum, spectrum_plan));
  TEST_ASSERT(
      !rund::kernel::TransformPlanMatchesDesc(transform, transform_plan));

  factor.fixed_format.approximation = rund::kernel::ComputeApproximation::Exact;
  solve.fixed_format.approximation = rund::kernel::ComputeApproximation::Exact;
  spectrum.fixed_format.approximation =
      rund::kernel::ComputeApproximation::Exact;
  transform.fixed_format.approximation =
      rund::kernel::ComputeApproximation::Exact;
  TEST_ASSERT(std::string_view{rund::kernel::PlanFactor(factor).reason} ==
              "compute_factor_numeric_policy_unsupported");
  TEST_ASSERT(std::string_view{rund::kernel::PlanSolve(solve).reason} ==
              "compute_solve_numeric_policy_unsupported");
  TEST_ASSERT(std::string_view{rund::kernel::PlanSpectrum(spectrum).reason} ==
              "compute_spectrum_numeric_policy_unsupported");
  TEST_ASSERT(std::string_view{rund::kernel::PlanTransform(transform).reason} ==
              "compute_transform_numeric_policy_unsupported");

  matrix.fixed_format.fraction_bits = 30u;
  TEST_ASSERT(std::string_view{rund::kernel::PlanMatrix(matrix).reason} ==
              "compute_matrix_numeric_policy_unsupported");
  return 0;
}

int test_numeric_algebra_rejects_oversize_dimensions() {
  const rund::kernel::FactorPlan factor = rund::kernel::PlanFactor(
      rund::kernel::FactorDesc{.op = rund::kernel::FactorOp::QR,
                               .pivot = rund::kernel::PivotOp::None,
                               .rows = 17u,
                               .cols = 17u,
                               .batch_count = 1u,
                               .fixed_format = kFixedI1F31});
  const rund::kernel::SolvePlan solve = rund::kernel::PlanSolve(
      rund::kernel::SolveDesc{.op = rund::kernel::SolveOp::Linear,
                              .input = rund::kernel::SolveInput::Matrix,
                              .factor = rund::kernel::FactorOp::LU,
                              .rows = 17u,
                              .rhs_cols = 1u,
                              .batch_count = 1u,
                              .fixed_format = kFixedI1F31});
  const rund::kernel::SpectrumPlan spectrum =
      rund::kernel::PlanSpectrum(rund::kernel::SpectrumDesc{
          .op = rund::kernel::SpectrumOp::Eigen,
          .domain = rund::kernel::SpectrumDomain::SymmetricReal,
          .vectors = rund::kernel::SpectrumVectors::ValuesOnly,
          .rows = 17u,
          .cols = 17u,
          .batch_count = 1u,
          .max_iterations = 32u,
          .fixed_format = kFixedI1F31});

  TEST_ASSERT(!factor.ok);
  TEST_ASSERT(std::string_view{factor.reason} ==
              "compute_factor_shape_dimension");
  TEST_ASSERT(!solve.ok);
  TEST_ASSERT(std::string_view{solve.reason} ==
              "compute_solve_shape_dimension");
  TEST_ASSERT(!spectrum.ok);
  TEST_ASSERT(std::string_view{spectrum.reason} ==
              "compute_spectrum_shape_dimension");
  return 0;
}

} // namespace

int RunIdentity() {
  if (test_primitive_identity_covers_every_fixed_policy_axis() != 0) {
    return 1;
  }
  if (test_primitive_plans_bind_fixed_format_and_policy() != 0) {
    return 1;
  }
  return test_numeric_algebra_rejects_oversize_dimensions();
}

} // namespace program_compute_contract::numeric_algebra_contract
