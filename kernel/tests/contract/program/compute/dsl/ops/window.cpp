#include "contract/program/compute/dsl/ops/local.hpp"

#include <kernel/program/compute/lowering/parse.hpp>

#include <cstddef>
#include <string>
#include <string_view>

namespace program_compute_contract {
namespace {

using namespace dsl_support;

template <typename T> [[nodiscard]] auto BuildSignalWindowOp() {
  T t[4]{};
  T out[4]{};
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(i64)) {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 63>()
          .template read<"t">(t)
          .template write<"out">(out);
    } else {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 31>()
          .template read<"t">(t)
          .template write<"out">(out);
    }
  }();
  return rund::compute_dsl::def("dsl-signal-window").on(body).map(
      [](auto i, auto b) {
        auto t = b.template read<"t">();
        auto out = b.template write<"out">();
        out[i] = rund::compute_dsl::window(rund::compute_dsl::WindowOp::Hann, t[i]) +
                 rund::compute_dsl::window(rund::compute_dsl::WindowOp::Hamming, t[i]) +
                 rund::compute_dsl::window(rund::compute_dsl::WindowOp::Blackman, t[i]) +
                 rund::compute_dsl::window(rund::compute_dsl::WindowOp::Lanczos, t[i]);
      });
}

template <typename T, rund::kernel::u8 IntegerBits,
          rund::kernel::u8 FractionBits>
[[nodiscard]] auto BuildFormatWindowOp() {
  T t[1]{};
  T out[1]{};
  const auto body =
      rund::compute_dsl::bind(1u)
          .template fixed<IntegerBits, FractionBits,
                          rund::kernel::ComputeRounding::Down,
                          rund::kernel::ComputeOverflow::Wrap,
                          rund::kernel::ComputeApproximation::Deterministic>()
          .template read<"t">(t)
          .template write<"out">(out);
  return rund::compute_dsl::def("dsl-format-window")
      .on(body)
      .map([](auto i, auto b) {
        auto t = b.template read<"t">();
        auto out = b.template write<"out">();
        out[i] = rund::compute_dsl::window(
                     rund::compute_dsl::WindowOp::Hamming, t[i]) +
                 rund::compute_dsl::window(
                     rund::compute_dsl::WindowOp::Blackman, t[i]);
      });
}

[[nodiscard]] constexpr rund::kernel::u64 Q31Bits(
    const unsigned fraction_bits, const rund::kernel::u32 bits) noexcept {
  const __uint128_t scaled =
      static_cast<__uint128_t>(bits) << fraction_bits;
  constexpr rund::kernel::u64 denominator = rund::kernel::u64{1} << 31u;
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

[[nodiscard]] bool HasQ31Literal(
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

[[nodiscard]] bool HasSignalWindowOps(const std::string& source) {
  return source.find("].op=cos") != std::string_view::npos &&
         source.find("].op=sin") != std::string_view::npos &&
         source.find("].op=div_fixed") != std::string_view::npos &&
         source.find("].op=mul_fixed") != std::string_view::npos &&
         source.find("].op=clamp") != std::string_view::npos;
}

int test_compute_signal_windows_extend_window_op_surface() {
  const auto first32 = BuildSignalWindowOp<i32>();
  const auto second32 = BuildSignalWindowOp<i32>();
  const auto fixed_lane64 = BuildSignalWindowOp<i64>();

  TEST_ASSERT(first32.ok());
  TEST_ASSERT(second32.ok());
  TEST_ASSERT(fixed_lane64.ok());
  TEST_ASSERT(first32.ir().canonical_bytes == second32.ir().canonical_bytes);
  TEST_ASSERT(first32.ir().op_hash_hi == second32.ir().op_hash_hi);
  TEST_ASSERT(first32.ir().op_hash_lo == second32.ir().op_hash_lo);

  const auto parsed32 =
      rund::kernel::compute_lowering_detail::ParseComputeIR(first32.ir());
  TEST_ASSERT(parsed32.ok);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Cos) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Sin) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::DivFixed) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::MulFixed) > 0u);
  TEST_ASSERT(CountOp(parsed32, rund::kernel::IrOp::Clamp) > 0u);

  const rund::kernel::LoweringArtifact metal32 =
      rund::kernel::LowerComputeIR(first32.ir(),
                                   rund::kernel::ComputeApi::Metal);
  const rund::kernel::LoweringArtifact vulkan64 =
      rund::kernel::LowerComputeIR(fixed_lane64.ir(),
                                   rund::kernel::ComputeApi::Vulkan);
  TEST_ASSERT(metal32.ok);
  TEST_ASSERT(vulkan64.ok);
  TEST_ASSERT(HasSignalWindowOps(metal32.source_text));
  TEST_ASSERT(HasSignalWindowOps(vulkan64.source_text));
  return 0;
}

int test_compute_signal_window_coefficients_follow_declared_fraction_bits() {
  const auto fixed16 = BuildFormatWindowOp<i32, 16u, 16u>();
  const auto fixed44 = BuildFormatWindowOp<i64, 20u, 44u>();
  TEST_ASSERT(fixed16.ok());
  TEST_ASSERT(fixed44.ok());
  const auto parsed16 =
      rund::kernel::compute_lowering_detail::ParseComputeIR(fixed16.ir());
  const auto parsed44 =
      rund::kernel::compute_lowering_detail::ParseComputeIR(fixed44.ir());
  TEST_ASSERT(parsed16.ok);
  TEST_ASSERT(parsed44.ok);

  for (const rund::kernel::u32 coefficient :
       {0x451eb852u, 0x3ae147aeu, 0x35c28f5cu, 0x0a3d70a3u}) {
    TEST_ASSERT(HasQ31Literal(parsed16, Q31Bits(16u, coefficient), 16u,
                              16u));
    TEST_ASSERT(HasQ31Literal(parsed44, Q31Bits(44u, coefficient), 20u,
                              44u));
  }
  return 0;
}

} // namespace

int RunComputeDslOpsWindowSignalContract() {
  if (const int result = test_compute_signal_windows_extend_window_op_surface();
      result != 0) {
    return result;
  }
  return test_compute_signal_window_coefficients_follow_declared_fraction_bits();
}

} // namespace program_compute_contract
