#include "contract/program/compute/dsl/ops/local.hpp"

#include <kernel/program/compute/lowering/parse.hpp>

#include <cstddef>
#include <string>
#include <string_view>

namespace program_compute_contract {
namespace {

using namespace dsl_support;

template <typename T> [[nodiscard]] auto BuildStatsOp() {
  T value[4]{};
  T center[4]{};
  T extra[4]{};
  T more[4]{};
  T out[4]{};
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(i64)) {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 63>()
          .template read<"value">(value)
          .template read<"center">(center)
          .template read<"extra">(extra)
          .template read<"more">(more)
          .template write<"out">(out);
    } else {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 31>()
          .template read<"value">(value)
          .template read<"center">(center)
          .template read<"extra">(extra)
          .template read<"more">(more)
          .template write<"out">(out);
    }
  }();
  return rund::compute_dsl::def("dsl-stats").on(body).map([](auto i, auto b) {
    auto value = b.template read<"value">();
    auto center = b.template read<"center">();
    auto extra = b.template read<"extra">();
    auto more = b.template read<"more">();
    auto out = b.template write<"out">();
    out[i] = rund::compute_dsl::fixed(rund::compute_dsl::FixedOp::Half, value[i]) +
             rund::compute_dsl::fixed(rund::compute_dsl::FixedOp::Third, value[i]) +
             rund::compute_dsl::fixed(rund::compute_dsl::FixedOp::Quarter, value[i]) +
             rund::compute_dsl::mean(value[i], center[i]) +
             rund::compute_dsl::mean(value[i], center[i], extra[i]) +
             rund::compute_dsl::mean(value[i], center[i], extra[i], more[i]) +
             rund::compute_dsl::mean(rund::compute_dsl::MeanOp::Abs, value[i],
                               center[i]) +
             rund::compute_dsl::mean(rund::compute_dsl::MeanOp::Abs, value[i], center[i],
                               extra[i]) +
             rund::compute_dsl::mean(rund::compute_dsl::MeanOp::Abs, value[i], center[i],
                               extra[i], more[i]) +
             rund::compute_dsl::mean(rund::compute_dsl::MeanOp::Squared, value[i],
                               center[i]) +
             rund::compute_dsl::mean(rund::compute_dsl::MeanOp::Squared, value[i],
                               center[i], extra[i]) +
             rund::compute_dsl::mean(rund::compute_dsl::MeanOp::Squared, value[i],
                               center[i], extra[i], more[i]) +
             rund::compute_dsl::centered(value[i], center[i]) +
             rund::compute_dsl::centered(rund::compute_dsl::CenteredOp::Abs, extra[i],
                                   center[i]);
  });
}

template <typename T, rund::kernel::u8 IntegerBits,
          rund::kernel::u8 FractionBits>
[[nodiscard]] auto BuildFormatLiteralOp() {
  T value[1]{};
  T out[1]{};
  const auto body =
      rund::compute_dsl::bind(1u)
          .template fixed<IntegerBits, FractionBits,
                          rund::kernel::ComputeRounding::Down,
                          rund::kernel::ComputeOverflow::Wrap,
                          rund::kernel::ComputeApproximation::Deterministic>()
          .template read<"value">(value)
          .template write<"out">(out);
  return rund::compute_dsl::def("dsl-format-literals")
      .on(body)
      .map([](auto i, auto b) {
        auto value = b.template read<"value">();
        auto out = b.template write<"out">();
        out[i] =
            rund::compute_dsl::fixed_one(value[i]) +
            rund::compute_dsl::fixed(rund::compute_dsl::FixedOp::Half,
                                     value[i]) +
            rund::compute_dsl::fixed(rund::compute_dsl::FixedOp::Third,
                                     value[i]) +
            rund::compute_dsl::fixed(rund::compute_dsl::FixedOp::Quarter,
                                     value[i]) +
            rund::compute_dsl::fixed_max(value[i]);
      });
}

[[nodiscard]] constexpr rund::kernel::u64 NearestRatioBits(
    const unsigned fraction_bits, const rund::kernel::u64 numerator,
    const rund::kernel::u64 denominator) noexcept {
  const __uint128_t scaled =
      static_cast<__uint128_t>(numerator) << fraction_bits;
  rund::kernel::u64 quotient =
      static_cast<rund::kernel::u64>(scaled / denominator);
  const __uint128_t remainder = scaled % denominator;
  const __uint128_t twice = remainder << 1u;
  if (twice > denominator ||
      (twice == denominator && (quotient & 1u) != 0u)) {
    ++quotient;
  }
  return quotient;
}

[[nodiscard]] bool HasExactLiteral(
    const rund::kernel::compute_lowering_detail::ParsedIR &parsed,
    const rund::kernel::u64 bits, const rund::kernel::u8 integer_bits,
    const rund::kernel::u8 fraction_bits) {
  for (const auto &node : parsed.nodes) {
    const rund::kernel::u64 encoded =
        static_cast<rund::kernel::u64>(node.lhs) |
        (static_cast<rund::kernel::u64>(node.rhs) << 32u);
    if (node.op == static_cast<rund::kernel::u8>(
                       rund::kernel::IrOp::Constant) &&
        encoded == bits && node.fixed_format.integer_bits == integer_bits &&
        node.fixed_format.fraction_bits == fraction_bits &&
        node.fixed_format.rounding == rund::kernel::ComputeRounding::Down &&
        node.fixed_format.overflow == rund::kernel::ComputeOverflow::Wrap &&
        node.fixed_format.approximation ==
            rund::kernel::ComputeApproximation::Exact) {
      return true;
    }
  }
  return false;
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

[[nodiscard]] bool HasStatsOps(const std::string& source) {
  return source.find("].op=constant") != std::string_view::npos &&
         source.find("].op=mul_fixed") != std::string_view::npos &&
         source.find("].op=add_sat") != std::string_view::npos &&
         source.find("].op=sub_sat") != std::string_view::npos &&
         source.find("].op=abs") != std::string_view::npos;
}

int test_compute_stats_helpers_lower_through_existing_ops() {
  const auto first32 = BuildStatsOp<i32>();
  const auto second32 = BuildStatsOp<i32>();
  const auto fixed_lane64 = BuildStatsOp<i64>();

  TEST_ASSERT(first32.ok());
  TEST_ASSERT(second32.ok());
  TEST_ASSERT(fixed_lane64.ok());
  TEST_ASSERT(first32.ir().canonical_bytes == second32.ir().canonical_bytes);
  TEST_ASSERT(first32.ir().op_hash_hi == second32.ir().op_hash_hi);
  TEST_ASSERT(first32.ir().op_hash_lo == second32.ir().op_hash_lo);

  const auto parsed32 =
      rund::kernel::compute_lowering_detail::ParseComputeIR(first32.ir());
  TEST_ASSERT(parsed32.ok);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Constant) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::MulFixed) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::AddSat) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::SubSat) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Abs) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::DivFixed) == 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Sqrt) == 0u);

  const rund::kernel::LoweringArtifact metal32 =
      rund::kernel::LowerComputeIR(first32.ir(),
                                   rund::kernel::ComputeApi::Metal);
  const rund::kernel::LoweringArtifact vulkan64 =
      rund::kernel::LowerComputeIR(fixed_lane64.ir(),
                                   rund::kernel::ComputeApi::Vulkan);
  TEST_ASSERT(metal32.ok);
  TEST_ASSERT(vulkan64.ok);
  TEST_ASSERT(HasStatsOps(metal32.source_text));
  TEST_ASSERT(HasStatsOps(vulkan64.source_text));
  return 0;
}

int test_compute_fixed_literals_follow_declared_fraction_bits() {
  const auto fixed16 = BuildFormatLiteralOp<i32, 16u, 16u>();
  const auto fixed44 = BuildFormatLiteralOp<i64, 20u, 44u>();
  TEST_ASSERT(fixed16.ok());
  TEST_ASSERT(fixed44.ok());
  const auto parsed16 =
      rund::kernel::compute_lowering_detail::ParseComputeIR(fixed16.ir());
  const auto parsed44 =
      rund::kernel::compute_lowering_detail::ParseComputeIR(fixed44.ir());
  TEST_ASSERT(parsed16.ok);
  TEST_ASSERT(parsed44.ok);

  TEST_ASSERT(HasExactLiteral(parsed16, NearestRatioBits(16u, 1u, 1u),
                              16u, 16u));
  TEST_ASSERT(HasExactLiteral(parsed16, NearestRatioBits(16u, 1u, 2u),
                              16u, 16u));
  TEST_ASSERT(HasExactLiteral(parsed16, NearestRatioBits(16u, 1u, 3u),
                              16u, 16u));
  TEST_ASSERT(HasExactLiteral(parsed16, NearestRatioBits(16u, 1u, 4u),
                              16u, 16u));
  TEST_ASSERT(HasExactLiteral(parsed16, 0x7fffffffull, 16u, 16u));

  TEST_ASSERT(HasExactLiteral(parsed44, NearestRatioBits(44u, 1u, 1u),
                              20u, 44u));
  TEST_ASSERT(HasExactLiteral(parsed44, NearestRatioBits(44u, 1u, 2u),
                              20u, 44u));
  TEST_ASSERT(HasExactLiteral(parsed44, NearestRatioBits(44u, 1u, 3u),
                              20u, 44u));
  TEST_ASSERT(HasExactLiteral(parsed44, NearestRatioBits(44u, 1u, 4u),
                              20u, 44u));
  TEST_ASSERT(HasExactLiteral(parsed44, 0x7fffffffffffffffull, 20u, 44u));
  return 0;
}

} // namespace

int RunComputeDslOpsStatsContract() {
  if (const int result = test_compute_stats_helpers_lower_through_existing_ops();
      result != 0) {
    return result;
  }
  return test_compute_fixed_literals_follow_declared_fraction_bits();
}

} // namespace program_compute_contract
