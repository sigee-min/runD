#include "model.hpp"

namespace program_compute_contract::numeric_algebra_contract {
namespace {

int test_matrix_arithmetic_law_is_identity_and_execution_input() {
  rund::kernel::MatrixDesc integer_desc{
      .op = rund::kernel::MatrixOp::Mul,
      .layout = rund::kernel::MatrixLayout::RowMajor,
      .arithmetic = rund::kernel::MatrixArithmetic::SignedWrap,
      .rows = 2u,
      .cols = 2u,
      .inner = 2u,
      .batch_count = 1u,
      .element_bytes = sizeof(rund::kernel::i32),
  };
  const auto integer_plan = rund::kernel::PlanMatrix(integer_desc);
  const std::array<rund::kernel::i32, 4> left{1, 2, 3, 4};
  const std::array<rund::kernel::i32, 4> right{5, 6, 7, 8};
  std::array<rund::kernel::i32, 4> integer_output{};
  const auto integer_result = rund::kernel::ReferenceMatrixMulI32(
      left.data(), right.data(), integer_output.data(), integer_plan);

  rund::kernel::MatrixDesc fixed_desc = integer_desc;
  fixed_desc.arithmetic = rund::kernel::MatrixArithmetic::Fixed;
  fixed_desc.fixed_format = kFixedI1F31;
  const auto fixed_plan = rund::kernel::PlanMatrix(fixed_desc);
  const std::array<rund::kernel::i32, 4> fixed_left{kOne, 0, 0, kOne};
  const std::array<rund::kernel::i32, 4> fixed_right{kOne, kOne, kOne, kOne};
  std::array<rund::kernel::i32, 4> fixed_output{};
  const auto fixed_result = rund::kernel::ReferenceMatrixMulI32(
      fixed_left.data(), fixed_right.data(), fixed_output.data(), fixed_plan);
  const auto integer_hash = rund::kernel::HashMatrix(integer_desc);
  const auto fixed_hash = rund::kernel::HashMatrix(fixed_desc);

  TEST_ASSERT(integer_plan.ok);
  TEST_ASSERT(integer_result.ok);
  TEST_ASSERT(
      (integer_output == std::array<rund::kernel::i32, 4>{19, 22, 43, 50}));
  TEST_ASSERT(fixed_plan.ok);
  TEST_ASSERT(fixed_result.ok);
  TEST_ASSERT((fixed_output ==
               std::array<rund::kernel::i32, 4>{kHalf, kHalf, kHalf, kHalf}));
  TEST_ASSERT(integer_hash.hi != fixed_hash.hi ||
              integer_hash.lo != fixed_hash.lo);

  integer_desc.arithmetic = static_cast<rund::kernel::MatrixArithmetic>(0u);
  const auto invalid = rund::kernel::PlanMatrix(integer_desc);
  TEST_ASSERT(!invalid.ok);
  TEST_ASSERT(std::string_view{invalid.reason} ==
              "compute_matrix_arithmetic_unsupported");
  return 0;
}

} // namespace

int RunMatrix() {
  return test_matrix_arithmetic_law_is_identity_and_execution_input();
}

} // namespace program_compute_contract::numeric_algebra_contract
