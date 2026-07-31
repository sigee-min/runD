#include "model.hpp"

namespace program_compute_contract::numeric_algebra_contract {
namespace {

int test_solve_raw_matrix_and_factor_reuse_paths() {
  const rund::kernel::SolveDesc raw_desc{
      .op = rund::kernel::SolveOp::Linear,
      .input = rund::kernel::SolveInput::Matrix,
      .factor = rund::kernel::FactorOp::LU,
      .layout = rund::kernel::MatrixLayout::RowMajor,
      .pivot = rund::kernel::PivotOp::Partial,
      .rows = 2u,
      .rhs_cols = 1u,
      .batch_count = 1u,
      .element_bytes = sizeof(rund::kernel::i32),
      .fixed_format = kFixedI1F31,
  };
  const rund::kernel::SolvePlan raw_plan = rund::kernel::PlanSolve(raw_desc);
  std::array<rund::kernel::i32, 4u> matrix{kOne, 0, 0, kOne};
  std::array<rund::kernel::i32, 2u> rhs{kHalf, kQuarter};
  std::array<rund::kernel::i32, 2u> out{};
  std::array<rund::kernel::u32, 1u> status{};
  const rund::kernel::SolveResult raw = rund::kernel::ReferenceSolveI32(
      matrix.data(), nullptr, rhs.data(), out.data(), status.data(), raw_plan);

  std::array<rund::kernel::i32, 4u> factor{};
  std::array<rund::kernel::u32, 2u> aux{};
  std::array<rund::kernel::u32, 1u> factor_status{};
  const rund::kernel::FactorPlan factor_plan = rund::kernel::PlanFactor(
      rund::kernel::FactorDesc{.op = rund::kernel::FactorOp::LU,
                               .rows = 2u,
                               .cols = 2u,
                               .batch_count = 1u,
                               .fixed_format = kFixedI1F31});
  const rund::kernel::FactorResult factored =
      rund::kernel::ReferenceFactorI32(matrix.data(), factor.data(), aux.data(),
                                       factor_status.data(), factor_plan);
  rund::kernel::SolveDesc reuse_desc = raw_desc;
  reuse_desc.input = rund::kernel::SolveInput::Factor;
  const rund::kernel::SolvePlan reuse_plan =
      rund::kernel::PlanSolve(reuse_desc);
  std::array<rund::kernel::i32, 2u> reuse_out{};
  std::array<rund::kernel::u32, 1u> reuse_status{};
  const rund::kernel::SolveResult reuse = rund::kernel::ReferenceSolveI32(
      factor.data(), aux.data(), rhs.data(), reuse_out.data(),
      reuse_status.data(), reuse_plan);

  TEST_ASSERT(raw_plan.ok);
  TEST_ASSERT(raw.ok);
  TEST_ASSERT(raw.failed_batches == 0u);
  TEST_ASSERT(factored.ok);
  TEST_ASSERT(reuse_plan.ok);
  TEST_ASSERT(reuse.ok);
  TEST_ASSERT(reuse.failed_batches == 0u);
  TEST_ASSERT(out[0] == reuse_out[0]);
  TEST_ASSERT(out[1] == reuse_out[1]);
  return 0;
}

int test_solve_qr_raw_matrix_reference_status() {
  const rund::kernel::SolveDesc desc{
      .op = rund::kernel::SolveOp::Linear,
      .input = rund::kernel::SolveInput::Matrix,
      .factor = rund::kernel::FactorOp::QR,
      .layout = rund::kernel::MatrixLayout::RowMajor,
      .pivot = rund::kernel::PivotOp::None,
      .rows = 2u,
      .rhs_cols = 1u,
      .batch_count = 1u,
      .element_bytes = sizeof(rund::kernel::i32),
      .fixed_format = kFixedI1F31,
  };
  const rund::kernel::SolvePlan plan = rund::kernel::PlanSolve(desc);
  std::array<rund::kernel::i32, 4u> matrix{kOne, 0, 0, kOne};
  std::array<rund::kernel::i32, 2u> rhs{kHalf, kQuarter};
  std::array<rund::kernel::i32, 2u> out{};
  std::array<rund::kernel::u32, 1u> status{};
  const rund::kernel::SolveResult result = rund::kernel::ReferenceSolveI32(
      matrix.data(), nullptr, rhs.data(), out.data(), status.data(), plan);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(plan.factor_count == 8u);
  TEST_ASSERT(result.ok);
  TEST_ASSERT(result.failed_batches == 0u);
  TEST_ASSERT(status[0] ==
              static_cast<rund::kernel::u32>(rund::kernel::SolveStatus::Ok));
  return 0;
}

int check_qr_matrix_solve_matches_factor_route(
    const rund::kernel::MatrixLayout layout) {
  const std::array<rund::kernel::i32, 4u> matrix =
      layout == rund::kernel::MatrixLayout::RowMajor
          ? std::array<rund::kernel::i32, 4u>{kOne, kQuarter, 0, kHalf}
          : std::array<rund::kernel::i32, 4u>{kOne, 0, kQuarter, kHalf};
  const std::array<rund::kernel::i32, 2u> rhs{kHalf, kQuarter};
  const rund::kernel::FactorPlan factor_plan =
      rund::kernel::PlanFactor(rund::kernel::FactorDesc{
          .op = rund::kernel::FactorOp::QR,
          .layout = layout,
          .output = rund::kernel::FactorOutput::Separate,
          .pivot = rund::kernel::PivotOp::None,
          .rows = 2u,
          .cols = 2u,
          .batch_count = 1u,
          .element_bytes = sizeof(rund::kernel::i32),
          .fixed_format = kFixedI1F31,
      });
  std::array<rund::kernel::i32, 8u> factor{};
  std::array<rund::kernel::u32, 1u> factor_status{};
  const rund::kernel::FactorResult factored = rund::kernel::ReferenceFactorI32(
      matrix.data(), factor.data(), nullptr, factor_status.data(), factor_plan);

  const auto solve_plan = [&](const rund::kernel::SolveInput input) {
    return rund::kernel::PlanSolve(rund::kernel::SolveDesc{
        .op = rund::kernel::SolveOp::Linear,
        .input = input,
        .factor = rund::kernel::FactorOp::QR,
        .layout = layout,
        .pivot = rund::kernel::PivotOp::None,
        .rows = 2u,
        .rhs_cols = 1u,
        .batch_count = 1u,
        .element_bytes = sizeof(rund::kernel::i32),
        .fixed_format = kFixedI1F31,
    });
  };
  const rund::kernel::SolvePlan matrix_plan =
      solve_plan(rund::kernel::SolveInput::Matrix);
  const rund::kernel::SolvePlan reuse_plan =
      solve_plan(rund::kernel::SolveInput::Factor);
  std::array<rund::kernel::i32, 2u> direct_output{};
  std::array<rund::kernel::i32, 2u> reuse_output{};
  std::array<rund::kernel::u32, 1u> direct_status{};
  std::array<rund::kernel::u32, 1u> reuse_status{};
  const rund::kernel::SolveResult direct = rund::kernel::ReferenceSolveI32(
      matrix.data(), nullptr, rhs.data(), direct_output.data(),
      direct_status.data(), matrix_plan);
  const rund::kernel::SolveResult reuse = rund::kernel::ReferenceSolveI32(
      factor.data(), nullptr, rhs.data(), reuse_output.data(),
      reuse_status.data(), reuse_plan);

  TEST_ASSERT(factor_plan.ok && matrix_plan.ok && reuse_plan.ok);
  TEST_ASSERT(factored.ok && factored.failed_batches == 0u);
  TEST_ASSERT(direct.ok && reuse.ok);
  TEST_ASSERT(direct.failed_batches == reuse.failed_batches);
  TEST_ASSERT(direct_status == reuse_status);
  TEST_ASSERT(direct_output == reuse_output);
  const std::size_t lower = static_cast<std::size_t>(
      4u +
      rund::kernel::factor_reference_detail::Index(1u, 0u, 2u, 2u, layout));
  TEST_ASSERT(factor[lower] == 0);
  return 0;
}

int test_qr_matrix_solve_preserves_layout_and_failure_order() {
  if (check_qr_matrix_solve_matches_factor_route(
          rund::kernel::MatrixLayout::RowMajor) != 0 ||
      check_qr_matrix_solve_matches_factor_route(
          rund::kernel::MatrixLayout::ColumnMajor) != 0) {
    return 1;
  }
  const rund::kernel::SolvePlan plan =
      rund::kernel::PlanSolve(rund::kernel::SolveDesc{
          .op = rund::kernel::SolveOp::Linear,
          .input = rund::kernel::SolveInput::Matrix,
          .factor = rund::kernel::FactorOp::QR,
          .layout = rund::kernel::MatrixLayout::RowMajor,
          .pivot = rund::kernel::PivotOp::None,
          .rows = 2u,
          .rhs_cols = 1u,
          .batch_count = 2u,
          .element_bytes = sizeof(rund::kernel::i32),
          .fixed_format = kFixedI1F31,
      });
  const std::array<rund::kernel::i32, 8u> matrix{0, 0, 0, 0, kOne, 0, 0, kOne};
  const std::array<rund::kernel::i32, 4u> rhs{kHalf, kQuarter, kQuarter, kHalf};
  std::array<rund::kernel::i32, 4u> output{};
  std::array<rund::kernel::u32, 2u> status{};
  const rund::kernel::SolveResult result = rund::kernel::ReferenceSolveI32(
      matrix.data(), nullptr, rhs.data(), output.data(), status.data(), plan);
  TEST_ASSERT(plan.ok && result.ok);
  TEST_ASSERT(result.failed_batches == 1u);
  TEST_ASSERT(result.first_failed_batch == 0u);
  TEST_ASSERT(result.first_status == rund::kernel::SolveStatus::Singular);
  TEST_ASSERT(status[0] == static_cast<rund::kernel::u32>(
                               rund::kernel::SolveStatus::Singular));
  TEST_ASSERT(status[1] ==
              static_cast<rund::kernel::u32>(rund::kernel::SolveStatus::Ok));
  return 0;
}

} // namespace

int RunSolve() {
  if (test_solve_raw_matrix_and_factor_reuse_paths() != 0) {
    return 1;
  }
  if (test_solve_qr_raw_matrix_reference_status() != 0) {
    return 1;
  }
  return test_qr_matrix_solve_preserves_layout_and_failure_order();
}

} // namespace program_compute_contract::numeric_algebra_contract
