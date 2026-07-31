#include "contract/program/compute/dsl/ops/local.hpp"

#include <kernel/program/compute/lowering/parse.hpp>

#include <cstddef>
#include <string>
#include <string_view>

namespace program_compute_contract {
namespace {

using namespace dsl_support;

template <typename T> [[nodiscard]] auto BuildRangeOp() {
  T value[4]{};
  T lo[4]{};
  T hi[4]{};
  T out[4]{};
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(i64)) {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 63>()
          .template read<"value">(value)
          .template read<"lo">(lo)
          .template read<"hi">(hi)
          .template write<"out">(out);
    } else {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 31>()
          .template read<"value">(value)
          .template read<"lo">(lo)
          .template read<"hi">(hi)
          .template write<"out">(out);
    }
  }();
  return rund::compute_dsl::def("dsl-range").on(body).map([](auto i, auto b) {
    auto value = b.template read<"value">();
    auto lo = b.template read<"lo">();
    auto hi = b.template read<"hi">();
    auto out = b.template write<"out">();

    out[i] = rund::compute_dsl::min(value[i], lo[i], hi[i]) +
             rund::compute_dsl::min(value[i], lo[i], hi[i], value[i]) +
             rund::compute_dsl::max(value[i], lo[i], hi[i]) +
             rund::compute_dsl::max(value[i], lo[i], hi[i], value[i]) +
             rund::compute_dsl::median(value[i], lo[i], hi[i]) +
             rund::compute_dsl::midrange(value[i], lo[i], hi[i]) +
             rund::compute_dsl::spread(value[i], lo[i], hi[i]) +
             rund::compute_dsl::clamp_range(value[i], hi[i], lo[i]) +
             rund::compute_dsl::bandpass(value[i], hi[i], lo[i]) +
             rund::compute_dsl::bandstop(value[i], hi[i], lo[i]) +
             rund::compute_dsl::select(
                 rund::compute_dsl::in_range(value[i], lo[i], hi[i]), value[i],
                 hi[i]) +
             rund::compute_dsl::select(
                 rund::compute_dsl::out_range(value[i], lo[i], hi[i]), lo[i],
                 value[i]);
  });
}

[[nodiscard]] std::size_t CountOp(
    const rund::kernel::compute_lowering_detail::ParsedIR& parsed,
    const rund::kernel::IrOp op) {
  std::size_t count = 0u;
  for (const auto& node : parsed.nodes) {
    if (node.op == static_cast<rund::kernel::u8>(op)) {
      ++count;
    }
  }
  return count;
}

[[nodiscard]] bool HasRangeOps(const std::string& source) {
  return source.find("].op=min") != std::string_view::npos &&
         source.find("].op=max") != std::string_view::npos &&
         source.find("].op=clamp") != std::string_view::npos &&
         source.find("].op=sub_sat") != std::string_view::npos &&
         source.find("].op=le") != std::string_view::npos &&
         source.find("].op=ge") != std::string_view::npos &&
         source.find("].op=predicate_and") != std::string_view::npos &&
         source.find("].op=predicate_not") != std::string_view::npos &&
         source.find("].op=select") != std::string_view::npos;
}

int test_compute_range_helpers_lower_through_existing_ops() {
  const auto first32 = BuildRangeOp<i32>();
  const auto second32 = BuildRangeOp<i32>();
  const auto fixed_lane64 = BuildRangeOp<i64>();

  TEST_ASSERT(first32.ok());
  TEST_ASSERT(second32.ok());
  TEST_ASSERT(fixed_lane64.ok());
  TEST_ASSERT(first32.ir().canonical_bytes == second32.ir().canonical_bytes);
  TEST_ASSERT(first32.ir().op_hash_hi == second32.ir().op_hash_hi);
  TEST_ASSERT(first32.ir().op_hash_lo == second32.ir().op_hash_lo);

  const auto parsed32 =
      rund::kernel::compute_lowering_detail::ParseComputeIR(first32.ir());
  TEST_ASSERT(parsed32.ok);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Min) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Max) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Clamp) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::MulFixed) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::SubSat) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::PredicateAnd) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::PredicateNot) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Sqrt) == 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::DivFixed) == 0u);

  const rund::kernel::LoweringArtifact metal32 =
      rund::kernel::LowerComputeIR(first32.ir(),
                                   rund::kernel::ComputeApi::Metal);
  const rund::kernel::LoweringArtifact vulkan64 =
      rund::kernel::LowerComputeIR(fixed_lane64.ir(),
                                   rund::kernel::ComputeApi::Vulkan);
  TEST_ASSERT(metal32.ok);
  TEST_ASSERT(vulkan64.ok);
  TEST_ASSERT(HasRangeOps(metal32.source_text));
  TEST_ASSERT(HasRangeOps(vulkan64.source_text));
  return 0;
}

} // namespace

int RunComputeDslOpsRangeContract() {
  return test_compute_range_helpers_lower_through_existing_ops();
}

} // namespace program_compute_contract
