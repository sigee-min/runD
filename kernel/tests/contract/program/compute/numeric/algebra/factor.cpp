#include "model.hpp"

namespace program_compute_contract::numeric_algebra_contract {
namespace {

int test_factor_plan_hash_and_lu_reference() {
  const rund::kernel::FactorDesc desc{
      .op = rund::kernel::FactorOp::LU,
      .layout = rund::kernel::MatrixLayout::RowMajor,
      .output = rund::kernel::FactorOutput::Packed,
      .pivot = rund::kernel::PivotOp::Partial,
      .rows = 2u,
      .cols = 2u,
      .batch_count = 1u,
      .element_bytes = sizeof(rund::kernel::i32),
      .fixed_format = kFixedI1F31,
  };
  const rund::kernel::FactorPlan plan = rund::kernel::PlanFactor(desc);
  const rund::kernel::FactorHash first = rund::kernel::HashFactor(desc);
  rund::kernel::FactorDesc changed = desc;
  changed.pivot = rund::kernel::PivotOp::None;
  const rund::kernel::FactorHash second = rund::kernel::HashFactor(changed);
  std::array<rund::kernel::i32, 4u> input{kOne, 0, 0, kOne};
  std::array<rund::kernel::i32, 4u> factor{};
  std::array<rund::kernel::u32, 2u> aux{};
  std::array<rund::kernel::u32, 1u> status{};
  const rund::kernel::FactorResult result = rund::kernel::ReferenceFactorI32(
      input.data(), factor.data(), aux.data(), status.data(), plan);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(plan.factor_count == 4u);
  TEST_ASSERT(plan.status_count == 1u);
  TEST_ASSERT(first.hi != second.hi || first.lo != second.lo);
  TEST_ASSERT(result.ok);
  TEST_ASSERT(result.failed_batches == 0u);
  TEST_ASSERT(status[0] ==
              static_cast<rund::kernel::u32>(rund::kernel::FactorStatus::Ok));
  TEST_ASSERT(factor[0] == kOne);
  TEST_ASSERT(factor[3] == kOne);
  return 0;
}

int test_factor_cholesky_reports_non_spd_batch_status() {
  const rund::kernel::FactorDesc desc{
      .op = rund::kernel::FactorOp::Cholesky,
      .layout = rund::kernel::MatrixLayout::RowMajor,
      .output = rund::kernel::FactorOutput::Packed,
      .pivot = rund::kernel::PivotOp::None,
      .rows = 2u,
      .cols = 2u,
      .batch_count = 1u,
      .element_bytes = sizeof(rund::kernel::i32),
      .fixed_format = kFixedI1F31,
  };
  const rund::kernel::FactorPlan plan = rund::kernel::PlanFactor(desc);
  std::array<rund::kernel::i32, 4u> input{0, kOne, kOne, 0};
  std::array<rund::kernel::i32, 4u> factor{};
  std::array<rund::kernel::u32, 1u> status{};
  const rund::kernel::FactorResult result = rund::kernel::ReferenceFactorI32(
      input.data(), factor.data(), nullptr, status.data(), plan);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(result.ok);
  TEST_ASSERT(result.failed_batches == 1u);
  TEST_ASSERT(result.first_failed_batch == 0u);
  TEST_ASSERT(status[0] == static_cast<rund::kernel::u32>(
                               rund::kernel::FactorStatus::NonSpd));
  return 0;
}

} // namespace

int RunFactor() {
  if (test_factor_plan_hash_and_lu_reference() != 0) {
    return 1;
  }
  return test_factor_cholesky_reports_non_spd_batch_status();
}

} // namespace program_compute_contract::numeric_algebra_contract
