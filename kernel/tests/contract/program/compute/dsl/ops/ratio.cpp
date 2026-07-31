#include "contract/program/compute/dsl/ops/local.hpp"

#include <kernel/program/compute/lowering/parse.hpp>

#include <cstddef>
#include <string>
#include <string_view>

namespace program_compute_contract {
namespace {

using namespace dsl_support;

template <typename T> [[nodiscard]] auto BuildRatioHelpers() {
  T value[4]{};
  T center[4]{};
  T scale[4]{};
  T total[4]{};
  T out[4]{};
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(i64)) {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 63>()
          .template read<"value">(value)
          .template read<"center">(center)
          .template read<"scale">(scale)
          .template read<"total">(total)
          .template write<"out">(out);
    } else {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 31>()
          .template read<"value">(value)
          .template read<"center">(center)
          .template read<"scale">(scale)
          .template read<"total">(total)
          .template write<"out">(out);
    }
  }();
  return rund::compute_dsl::def("dsl-ratio").on(body).map([](auto i, auto b) {
    auto value = b.template read<"value">();
    auto center = b.template read<"center">();
    auto scale = b.template read<"scale">();
    auto total = b.template read<"total">();
    auto out = b.template write<"out">();
    out[i] = rund::compute_dsl::ratio(value[i], scale[i]) +
             rund::compute_dsl::proportion(value[i], total[i]) +
             rund::compute_dsl::proportion(rund::compute_dsl::Axis::X, value[i],
                                      total[i]) +
             rund::compute_dsl::proportion(rund::compute_dsl::Axis::Y, value[i],
                                      total[i]) +
             rund::compute_dsl::proportion(rund::compute_dsl::Axis::X, value[i],
                                      center[i], total[i]) +
             rund::compute_dsl::proportion(rund::compute_dsl::Axis::Y, value[i],
                                      center[i], total[i]) +
             rund::compute_dsl::proportion(rund::compute_dsl::Axis::Z, value[i],
                                      center[i], total[i]) +
             rund::compute_dsl::zscore(value[i], center[i], scale[i]);
  });
}

[[nodiscard]] std::size_t CountOp(
    const rund::kernel::compute_lowering_detail::ParsedIR &parsed,
    const rund::kernel::IrOp op) {
  std::size_t count = 0u;
  for (const auto &node : parsed.nodes) {
    if (node.op == static_cast<rund::kernel::u8>(op)) {
      ++count;
    }
  }
  return count;
}

[[nodiscard]] bool HasRatioHelperOps(const std::string &source) {
  return source.find("].op=div_fixed") != std::string_view::npos &&
         source.find("].op=eq") != std::string_view::npos &&
         source.find("].op=select") != std::string_view::npos &&
         source.find("].op=clamp") != std::string_view::npos &&
         source.find("].op=sub_sat") != std::string_view::npos;
}

int test_compute_ratio_helpers_lower_through_existing_ops() {
  const auto first32 = BuildRatioHelpers<i32>();
  const auto second32 = BuildRatioHelpers<i32>();
  const auto fixed_lane64 = BuildRatioHelpers<i64>();

  TEST_ASSERT(first32.ok());
  TEST_ASSERT(second32.ok());
  TEST_ASSERT(fixed_lane64.ok());
  TEST_ASSERT(first32.ir().canonical_bytes == second32.ir().canonical_bytes);
  TEST_ASSERT(first32.ir().op_hash_hi == second32.ir().op_hash_hi);
  TEST_ASSERT(first32.ir().op_hash_lo == second32.ir().op_hash_lo);

  const auto parsed32 =
      rund::kernel::compute_lowering_detail::ParseComputeIR(first32.ir());
  TEST_ASSERT(parsed32.ok);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::DivFixed) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Eq) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Select) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Clamp) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::SubSat) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Sqrt) == 0u);

  const rund::kernel::LoweringArtifact metal32 =
      rund::kernel::LowerComputeIR(first32.ir(),
                                   rund::kernel::ComputeApi::Metal);
  const rund::kernel::LoweringArtifact vulkan64 =
      rund::kernel::LowerComputeIR(fixed_lane64.ir(),
                                   rund::kernel::ComputeApi::Vulkan);
  TEST_ASSERT(metal32.ok);
  TEST_ASSERT(vulkan64.ok);
  TEST_ASSERT(HasRatioHelperOps(metal32.source_text));
  TEST_ASSERT(HasRatioHelperOps(vulkan64.source_text));
  return 0;
}

} // namespace

int RunComputeDslOpsRatioContract() {
  return test_compute_ratio_helpers_lower_through_existing_ops();
}

} // namespace program_compute_contract
