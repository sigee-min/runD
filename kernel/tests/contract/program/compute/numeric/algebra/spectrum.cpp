#include "model.hpp"

namespace program_compute_contract::numeric_algebra_contract {
namespace {

int test_spectrum_plan_hash_and_eigen_reference_status() {
  const rund::kernel::SpectrumDesc desc{
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
  };
  const rund::kernel::SpectrumPlan plan = rund::kernel::PlanSpectrum(desc);
  const rund::kernel::SpectrumHash first = rund::kernel::HashSpectrum(desc);
  rund::kernel::SpectrumDesc changed = desc;
  changed.max_iterations = 16u;
  const rund::kernel::SpectrumHash second = rund::kernel::HashSpectrum(changed);
  std::array<rund::kernel::i32, 4u> input{kOne, 0, 0, kHalf};
  std::array<rund::kernel::i32, 2u> values{};
  std::array<rund::kernel::u32, 1u> status{};
  const rund::kernel::SpectrumResult result =
      rund::kernel::ReferenceSpectrumI32(input.data(), values.data(), nullptr,
                                         status.data(), plan);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(plan.value_count == 2u);
  TEST_ASSERT(first.hi != second.hi || first.lo != second.lo);
  TEST_ASSERT(result.ok);
  TEST_ASSERT(result.failed_batches == 0u);
  TEST_ASSERT(status[0] ==
              static_cast<rund::kernel::u32>(rund::kernel::SpectrumStatus::Ok));
  TEST_ASSERT(values[0] == kOne || values[1] == kOne);
  return 0;
}

int test_spectrum_vectors_follow_input_column() {
  const rund::kernel::SpectrumDesc desc{
      .op = rund::kernel::SpectrumOp::SVD,
      .domain = rund::kernel::SpectrumDomain::GeneralReal,
      .vectors = rund::kernel::SpectrumVectors::Thin,
      .layout = rund::kernel::MatrixLayout::RowMajor,
      .rows = 2u,
      .cols = 1u,
      .batch_count = 1u,
      .max_iterations = 8u,
      .element_bytes = sizeof(rund::kernel::i32),
      .fixed_format = kFixedI1F31,
  };
  const rund::kernel::SpectrumPlan plan = rund::kernel::PlanSpectrum(desc);
  std::array<rund::kernel::i32, 2u> input{0, kOne};
  std::array<rund::kernel::i32, 1u> values{};
  std::array<rund::kernel::i32, 2u> vectors{};
  std::array<rund::kernel::u32, 1u> status{};
  const rund::kernel::SpectrumResult result =
      rund::kernel::ReferenceSpectrumI32(input.data(), values.data(),
                                         vectors.data(), status.data(), plan);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(plan.value_count == 1u);
  TEST_ASSERT(plan.vector_count == 2u);
  TEST_ASSERT(result.ok);
  TEST_ASSERT(status[0] ==
              static_cast<rund::kernel::u32>(rund::kernel::SpectrumStatus::Ok));
  TEST_ASSERT(values[0] == kOne);
  TEST_ASSERT(vectors[0] == 0);
  TEST_ASSERT(vectors[1] == kFixedMax);
  return 0;
}

int test_svd_vector_scratch_is_prefill_independent() {
  const rund::kernel::SpectrumPlan plan =
      rund::kernel::PlanSpectrum(rund::kernel::SpectrumDesc{
          .op = rund::kernel::SpectrumOp::SVD,
          .domain = rund::kernel::SpectrumDomain::GeneralReal,
          .vectors = rund::kernel::SpectrumVectors::Thin,
          .layout = rund::kernel::MatrixLayout::RowMajor,
          .rows = 2u,
          .cols = 1u,
          .batch_count = 1u,
          .max_iterations = 8u,
          .element_bytes = sizeof(rund::kernel::i32),
          .fixed_format = kFixedI1F31,
      });
  const std::array<rund::kernel::i32, 2u> input{0, 0};
  std::array<rund::kernel::i32, 1u> first_values{};
  std::array<rund::kernel::i32, 1u> second_values{};
  std::array<rund::kernel::i32, 2u> first_vectors{};
  std::array<rund::kernel::i32, 2u> second_vectors{};
  std::array<rund::kernel::u32, 1u> first_status{};
  std::array<rund::kernel::u32, 1u> second_status{};
  std::array<rund::kernel::i32, 1u> first_matrix{0x11111111};
  std::array<rund::kernel::i32, 1u> second_matrix{0x22222222};
  std::array<rund::kernel::i32, 1u> first_jacobi_vectors{0x33333333};
  std::array<rund::kernel::i32, 1u> second_jacobi_vectors{0x44444444};
  std::array<rund::kernel::i32, 1u> first_jacobi_values{0x55555555};
  std::array<rund::kernel::i32, 1u> second_jacobi_values{0x66666666};
  std::array<rund::kernel::u64, 1u> first_order{7u};
  std::array<rund::kernel::u64, 1u> second_order{9u};
  std::array<rund::kernel::i32, 2u> first_u{0x13579bdf, 0x2468ace0};
  std::array<rund::kernel::i32, 2u> second_u{0x10203040, 0x50607080};
  const rund::kernel::SpectrumResult first =
      rund::kernel::ReferenceSpectrumScratchI32(
          input.data(), first_values.data(), first_vectors.data(),
          first_status.data(), plan, first_matrix.data(),
          first_jacobi_vectors.data(), first_jacobi_values.data(),
          first_order.data(), first_u.data());
  const rund::kernel::SpectrumResult second =
      rund::kernel::ReferenceSpectrumScratchI32(
          input.data(), second_values.data(), second_vectors.data(),
          second_status.data(), plan, second_matrix.data(),
          second_jacobi_vectors.data(), second_jacobi_values.data(),
          second_order.data(), second_u.data());
  TEST_ASSERT(plan.ok && first.ok && second.ok);
  TEST_ASSERT(first.failed_batches == second.failed_batches);
  TEST_ASSERT(first_status == second_status);
  TEST_ASSERT(first_values == second_values);
  TEST_ASSERT(first_vectors == second_vectors);
  TEST_ASSERT(first_u == second_u);
  return 0;
}

int test_spectrum_eigen_vectors_need_no_svd_scratch() {
  const rund::kernel::SpectrumDesc desc{
      .op = rund::kernel::SpectrumOp::Eigen,
      .domain = rund::kernel::SpectrumDomain::SymmetricReal,
      .vectors = rund::kernel::SpectrumVectors::Full,
      .layout = rund::kernel::MatrixLayout::RowMajor,
      .rows = 2u,
      .cols = 2u,
      .batch_count = 1u,
      .max_iterations = 8u,
      .element_bytes = sizeof(rund::kernel::i32),
      .fixed_format = kFixedI1F31,
  };
  const rund::kernel::SpectrumPlan plan = rund::kernel::PlanSpectrum(desc);
  std::array<rund::kernel::i32, 4u> input{kOne, 0, 0, kHalf};
  std::array<rund::kernel::i32, 2u> values{};
  std::array<rund::kernel::i32, 4u> vectors{};
  std::array<rund::kernel::u32, 1u> status{};
  std::array<rund::kernel::i32, 4u> matrix{};
  std::array<rund::kernel::i32, 4u> jacobi_vectors{};
  std::array<rund::kernel::i32, 2u> jacobi_values{};
  const rund::kernel::SpectrumResult result =
      rund::kernel::ReferenceSpectrumScratchI32(
          input.data(), values.data(), vectors.data(), status.data(), plan,
          matrix.data(), jacobi_vectors.data(), jacobi_values.data(), nullptr,
          nullptr);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(plan.vector_count == 4u);
  TEST_ASSERT(result.ok);
  TEST_ASSERT(result.failed_batches == 0u);
  TEST_ASSERT(status[0] ==
              static_cast<rund::kernel::u32>(rund::kernel::SpectrumStatus::Ok));
  return 0;
}

} // namespace

int RunSpectrum() {
  if (test_spectrum_plan_hash_and_eigen_reference_status() != 0) {
    return 1;
  }
  if (test_spectrum_vectors_follow_input_column() != 0) {
    return 1;
  }
  if (test_svd_vector_scratch_is_prefill_independent() != 0) {
    return 1;
  }
  return test_spectrum_eigen_vectors_need_no_svd_scratch();
}

} // namespace program_compute_contract::numeric_algebra_contract
