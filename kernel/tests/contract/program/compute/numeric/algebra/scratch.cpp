#include "model.hpp"

namespace program_compute_contract::numeric_algebra_contract {
namespace {

int test_factor_and_solve_scratch_are_prefill_independent() {
  const rund::kernel::FactorDesc factor_desc{
      .op = rund::kernel::FactorOp::QR,
      .layout = rund::kernel::MatrixLayout::RowMajor,
      .output = rund::kernel::FactorOutput::Separate,
      .pivot = rund::kernel::PivotOp::None,
      .rows = 2u,
      .cols = 2u,
      .batch_count = 1u,
      .element_bytes = sizeof(rund::kernel::i32),
      .fixed_format = kFixedI1F31,
  };
  const rund::kernel::FactorPlan factor_plan =
      rund::kernel::PlanFactor(factor_desc);
  const std::array<rund::kernel::i32, 4u> matrix{kOne, 0, 0, kOne};
  std::array<rund::kernel::i32, 8u> first_factor{};
  std::array<rund::kernel::i32, 8u> second_factor{};
  std::array<rund::kernel::u32, 1u> first_factor_status{};
  std::array<rund::kernel::u32, 1u> second_factor_status{};
  std::array<rund::kernel::i32, 4u> first_q{};
  std::array<rund::kernel::i32, 4u> second_q{};
  std::array<rund::kernel::i32, 4u> first_r{};
  std::array<rund::kernel::i32, 4u> second_r{};
  std::array<rund::kernel::i32, 2u> first_v{};
  std::array<rund::kernel::i32, 2u> second_v{};
  first_factor.fill(0x77777777);
  second_factor.fill(0x12121212);
  first_q.fill(0x11111111);
  second_q.fill(0x22222222);
  first_r.fill(0x33333333);
  second_r.fill(0x44444444);
  first_v.fill(0x55555555);
  second_v.fill(0x66666666);
  const rund::kernel::FactorResult first_factored =
      rund::kernel::ReferenceFactorScratchI32(
          matrix.data(), first_factor.data(), nullptr,
          first_factor_status.data(), factor_plan, first_q.data(),
          first_r.data(), first_v.data());
  const rund::kernel::FactorResult second_factored =
      rund::kernel::ReferenceFactorScratchI32(
          matrix.data(), second_factor.data(), nullptr,
          second_factor_status.data(), factor_plan, second_q.data(),
          second_r.data(), second_v.data());

  TEST_ASSERT(factor_plan.ok);
  TEST_ASSERT(first_factored.ok && second_factored.ok);
  TEST_ASSERT(first_factored.failed_batches == second_factored.failed_batches);
  TEST_ASSERT(first_factor_status == second_factor_status);
  TEST_ASSERT(first_factor == second_factor);
  TEST_ASSERT(first_factor[6] == 0);

  const rund::kernel::SolvePlan solve_plan =
      rund::kernel::PlanSolve(rund::kernel::SolveDesc{
          .op = rund::kernel::SolveOp::Linear,
          .input = rund::kernel::SolveInput::Factor,
          .factor = rund::kernel::FactorOp::QR,
          .layout = rund::kernel::MatrixLayout::RowMajor,
          .pivot = rund::kernel::PivotOp::None,
          .rows = 2u,
          .rhs_cols = 1u,
          .batch_count = 1u,
          .element_bytes = sizeof(rund::kernel::i32),
          .fixed_format = kFixedI1F31,
      });
  const std::array<rund::kernel::i32, 2u> rhs{kHalf, kQuarter};
  std::array<rund::kernel::i32, 2u> first_output{};
  std::array<rund::kernel::i32, 2u> second_output{};
  std::array<rund::kernel::u32, 1u> first_solve_status{};
  std::array<rund::kernel::u32, 1u> second_solve_status{};
  std::array<rund::kernel::i32, 2u> first_y{};
  std::array<rund::kernel::i32, 2u> second_y{};
  first_y.fill(0x13579bdf);
  second_y.fill(0x2468ace0);
  const rund::kernel::SolveResult first_solved =
      rund::kernel::ReferenceSolveScratchI32(
          first_factor.data(), nullptr, rhs.data(), first_output.data(),
          first_solve_status.data(), solve_plan, nullptr, nullptr,
          first_y.data(), nullptr, nullptr, nullptr);
  const rund::kernel::SolveResult second_solved =
      rund::kernel::ReferenceSolveScratchI32(
          first_factor.data(), nullptr, rhs.data(), second_output.data(),
          second_solve_status.data(), solve_plan, nullptr, nullptr,
          second_y.data(), nullptr, nullptr, nullptr);

  TEST_ASSERT(solve_plan.ok);
  TEST_ASSERT(first_solved.ok && second_solved.ok);
  TEST_ASSERT(first_solved.failed_batches == second_solved.failed_batches);
  TEST_ASSERT(first_solve_status == second_solve_status);
  TEST_ASSERT(first_output == second_output);
  return 0;
}

int test_cholesky_and_spectrum_overwrite_dead_initial_state() {
  const rund::kernel::FactorPlan cholesky =
      rund::kernel::PlanFactor(rund::kernel::FactorDesc{
          .op = rund::kernel::FactorOp::Cholesky,
          .layout = rund::kernel::MatrixLayout::RowMajor,
          .output = rund::kernel::FactorOutput::Packed,
          .pivot = rund::kernel::PivotOp::None,
          .rows = 2u,
          .cols = 2u,
          .batch_count = 1u,
          .element_bytes = sizeof(rund::kernel::i32),
          .fixed_format = kFixedI1F31,
      });
  const std::array<rund::kernel::i32, 4u> matrix{kOne, 0, 0, kOne};
  std::array<rund::kernel::i32, 4u> first_factor{};
  std::array<rund::kernel::i32, 4u> second_factor{};
  std::array<rund::kernel::u32, 1u> first_status{};
  std::array<rund::kernel::u32, 1u> second_status{};
  first_factor.fill(0x13579bdf);
  second_factor.fill(0x2468ace0);
  const rund::kernel::FactorResult first =
      rund::kernel::ReferenceFactorI32(matrix.data(), first_factor.data(),
                                       nullptr, first_status.data(), cholesky);
  const rund::kernel::FactorResult second =
      rund::kernel::ReferenceFactorI32(matrix.data(), second_factor.data(),
                                       nullptr, second_status.data(), cholesky);
  TEST_ASSERT(cholesky.ok);
  TEST_ASSERT(first.ok && second.ok);
  TEST_ASSERT(first_factor == second_factor);
  TEST_ASSERT(first_status == second_status);
  TEST_ASSERT(first_factor[1] == 0);

  const rund::kernel::SpectrumPlan eigen =
      rund::kernel::PlanSpectrum(rund::kernel::SpectrumDesc{
          .op = rund::kernel::SpectrumOp::Eigen,
          .domain = rund::kernel::SpectrumDomain::SymmetricReal,
          .vectors = rund::kernel::SpectrumVectors::ValuesOnly,
          .layout = rund::kernel::MatrixLayout::RowMajor,
          .rows = 2u,
          .cols = 2u,
          .batch_count = 1u,
          .max_iterations = 8u,
          .element_bytes = sizeof(rund::kernel::i32),
          .fixed_format = kFixedI1F31,
      });
  std::array<rund::kernel::i32, 2u> first_values{};
  std::array<rund::kernel::i32, 2u> second_values{};
  std::array<rund::kernel::u32, 1u> first_eigen_status{};
  std::array<rund::kernel::u32, 1u> second_eigen_status{};
  std::array<rund::kernel::i32, 4u> first_a{};
  std::array<rund::kernel::i32, 4u> second_a{};
  std::array<rund::kernel::i32, 4u> first_vectors{};
  std::array<rund::kernel::i32, 4u> second_vectors{};
  std::array<rund::kernel::i32, 2u> first_jacobi_values{};
  std::array<rund::kernel::i32, 2u> second_jacobi_values{};
  first_a.fill(0x11111111);
  second_a.fill(0x22222222);
  const rund::kernel::SpectrumResult first_eigen =
      rund::kernel::ReferenceSpectrumScratchI32(
          matrix.data(), first_values.data(), nullptr,
          first_eigen_status.data(), eigen, first_a.data(),
          first_vectors.data(), first_jacobi_values.data(), nullptr, nullptr);
  const rund::kernel::SpectrumResult second_eigen =
      rund::kernel::ReferenceSpectrumScratchI32(
          matrix.data(), second_values.data(), nullptr,
          second_eigen_status.data(), eigen, second_a.data(),
          second_vectors.data(), second_jacobi_values.data(), nullptr, nullptr);
  TEST_ASSERT(eigen.ok);
  TEST_ASSERT(first_eigen.ok && second_eigen.ok);
  TEST_ASSERT(first_values == second_values);
  TEST_ASSERT(first_eigen_status == second_eigen_status);
  return 0;
}

} // namespace

int RunScratch() {
  if (test_factor_and_solve_scratch_are_prefill_independent() != 0) {
    return 1;
  }
  return test_cholesky_and_spectrum_overwrite_dead_initial_state();
}

} // namespace program_compute_contract::numeric_algebra_contract
