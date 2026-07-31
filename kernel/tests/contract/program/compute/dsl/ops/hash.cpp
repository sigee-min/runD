#include "contract/program/compute/dsl/ops/local.hpp"

#include <kernel/program/compute/lowering/parse.hpp>

#include <string>

namespace program_compute_contract {
namespace {

using namespace dsl_support;

template <typename T> [[nodiscard]] auto BuildHashOp() {
  T value[4]{};
  T seed[4]{};
  T out[4]{};
  const auto body = [&]() {
    if constexpr (sizeof(T) == sizeof(i64)) {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 63>()
          .template read<"value">(value)
          .template read<"seed">(seed)
          .template write<"out">(out);
    } else {
      return rund::compute_dsl::bind(4u)
          .fixed<1, 31>()
          .template read<"value">(value)
          .template read<"seed">(seed)
          .template write<"out">(out);
    }
  }();
  return rund::compute_dsl::def("dsl-hash").on(body).map([](auto i, auto b) {
    auto value = b.template read<"value">();
    auto seed = b.template read<"seed">();
    auto out = b.template write<"out">();

    out[i] = rund::compute_dsl::bit_xor(
        rund::compute_dsl::hash(value[i], seed[i]),
        rund::compute_dsl::bit_xor(
            rund::compute_dsl::hash(rund::compute_dsl::HashOp::Unit, value[i]),
            rund::compute_dsl::hash(rund::compute_dsl::HashOp::Unit, value[i], seed[i])));
  });
}

template <typename T, rund::kernel::u8 IntegerBits,
          rund::kernel::u8 FractionBits>
[[nodiscard]] auto BuildFormatHashOp() {
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
  return rund::compute_dsl::def("dsl-format-hash")
      .on(body)
      .map([](auto i, auto b) {
        auto value = b.template read<"value">();
        auto out = b.template write<"out">();
        out[i] = rund::compute_dsl::hash(rund::compute_dsl::HashOp::Unit,
                                        value[i]);
      });
}

[[nodiscard]] bool HasFractionMask(
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

[[nodiscard]] bool HasHashOps(const std::string &source) {
  return source.find("].op=bit_xor") != std::string_view::npos &&
         source.find("].op=shr_logical_const") != std::string_view::npos &&
         source.find("].op=mul_wrap") != std::string_view::npos &&
         source.find("].op=bit_and") != std::string_view::npos;
}

int test_compute_hash_helpers_build_deterministic_lowerable_ir() {
  const auto first32 = BuildHashOp<i32>();
  const auto second32 = BuildHashOp<i32>();
  const auto fixed_lane64 = BuildHashOp<i64>();

  TEST_ASSERT(first32.ok());
  TEST_ASSERT(second32.ok());
  TEST_ASSERT(fixed_lane64.ok());
  TEST_ASSERT(first32.ir().canonical_bytes == second32.ir().canonical_bytes);
  TEST_ASSERT(first32.ir().op_hash_hi == second32.ir().op_hash_hi);
  TEST_ASSERT(first32.ir().op_hash_lo == second32.ir().op_hash_lo);

  const rund::kernel::LoweringArtifact metal32 =
      rund::kernel::LowerComputeIR(first32.ir(),
                                   rund::kernel::ComputeApi::Metal);
  const rund::kernel::LoweringArtifact vulkan64 =
      rund::kernel::LowerComputeIR(fixed_lane64.ir(),
                                   rund::kernel::ComputeApi::Vulkan);
  TEST_ASSERT(metal32.ok);
  TEST_ASSERT(vulkan64.ok);
  TEST_ASSERT(HasHashOps(metal32.source_text));
  TEST_ASSERT(HasHashOps(vulkan64.source_text));
  return 0;
}

int test_compute_unit_hash_masks_declared_fraction_bits() {
  const auto fixed16 = BuildFormatHashOp<i32, 16u, 16u>();
  const auto fixed44 = BuildFormatHashOp<i64, 20u, 44u>();
  TEST_ASSERT(fixed16.ok());
  TEST_ASSERT(fixed44.ok());
  const auto parsed16 =
      rund::kernel::compute_lowering_detail::ParseComputeIR(fixed16.ir());
  const auto parsed44 =
      rund::kernel::compute_lowering_detail::ParseComputeIR(fixed44.ir());
  TEST_ASSERT(parsed16.ok);
  TEST_ASSERT(parsed44.ok);
  TEST_ASSERT(HasFractionMask(parsed16, (rund::kernel::u64{1} << 16u) - 1u,
                              16u, 16u));
  TEST_ASSERT(HasFractionMask(parsed44, (rund::kernel::u64{1} << 44u) - 1u,
                              20u, 44u));
  return 0;
}

} // namespace

int RunComputeDslOpsHashContract() {
  if (const int result =
          test_compute_hash_helpers_build_deterministic_lowerable_ir();
      result != 0) {
    return result;
  }
  return test_compute_unit_hash_masks_declared_fraction_bits();
}

} // namespace program_compute_contract
