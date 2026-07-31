#include "model.hpp"

namespace program_compute_contract::numeric_algebra_contract {
namespace {

template <typename S, std::size_t InputCount, std::size_t MatrixCount>
int CheckSymmetricAtA(const std::array<S, InputCount> &input,
                      const rund::kernel::SpectrumPlan &plan) {
  std::array<S, MatrixCount> expected{};
  std::array<S, MatrixCount> actual{};
  for (rund::kernel::u64 i = 0u; i < plan.cols; ++i) {
    for (rund::kernel::u64 j = 0u; j < plan.cols; ++j) {
      S sum = 0;
      for (rund::kernel::u64 row = 0u; row < plan.rows; ++row) {
        sum = rund::kernel::factor_reference_detail::AddSat(
            sum, rund::kernel::factor_reference_detail::MulFixed(
                     input[rund::kernel::spectrum_reference_detail::Index(
                         row, i, plan.rows, plan.cols, plan.layout)],
                     input[rund::kernel::spectrum_reference_detail::Index(
                         row, j, plan.rows, plan.cols, plan.layout)],
                     plan.fixed_format));
      }
      expected[static_cast<std::size_t>(i * plan.cols + j)] = sum;
    }
  }
  actual.fill(static_cast<S>(0x55));
  rund::kernel::spectrum_reference_detail::BuildSymmetricAtA(
      input.data(), actual.data(), plan);
  TEST_ASSERT(actual == expected);
  return 0;
}

int test_svd_symmetric_ata_is_bit_exact_for_both_lane_widths() {
  const rund::kernel::SpectrumPlan narrow =
      rund::kernel::PlanSpectrum(rund::kernel::SpectrumDesc{
          .op = rund::kernel::SpectrumOp::SVD,
          .domain = rund::kernel::SpectrumDomain::GeneralReal,
          .vectors = rund::kernel::SpectrumVectors::ValuesOnly,
          .layout = rund::kernel::MatrixLayout::RowMajor,
          .rows = 3u,
          .cols = 2u,
          .batch_count = 1u,
          .max_iterations = 8u,
          .element_bytes = sizeof(rund::kernel::i32),
          .fixed_format = kFixedI1F31,
      });
  const std::array<rund::kernel::i32, 6u> narrow_input{
      kHalf, kQuarter, -kQuarter, kHalf, kQuarter, -kHalf};
  TEST_ASSERT(narrow.ok);
  TEST_ASSERT((
      CheckSymmetricAtA<rund::kernel::i32, 6u, 4u>(narrow_input, narrow) == 0));

  const rund::kernel::SpectrumPlan wide =
      rund::kernel::PlanSpectrum(rund::kernel::SpectrumDesc{
          .op = rund::kernel::SpectrumOp::SVD,
          .domain = rund::kernel::SpectrumDomain::GeneralReal,
          .vectors = rund::kernel::SpectrumVectors::ValuesOnly,
          .layout = rund::kernel::MatrixLayout::ColumnMajor,
          .rows = 3u,
          .cols = 2u,
          .batch_count = 1u,
          .max_iterations = 8u,
          .element_bytes = sizeof(rund::kernel::i64),
          .fixed_format = kFixedI1F63,
      });
  const std::array<rund::kernel::i64, 6u> wide_input{
      kWideOne / 2, -kWideOne / 4, kWideOne / 8,
      kWideOne / 4, kWideOne / 2,  -kWideOne / 8};
  TEST_ASSERT(wide.ok);
  return CheckSymmetricAtA<rund::kernel::i64, 6u, 4u>(wide_input, wide);
}

} // namespace

int RunAccuracy() {
  return test_svd_symmetric_ata_is_bit_exact_for_both_lane_widths();
}

} // namespace program_compute_contract::numeric_algebra_contract
