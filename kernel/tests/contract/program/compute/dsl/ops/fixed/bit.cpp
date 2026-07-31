#include "contract/program/compute/dsl/ops/local.hpp"

namespace program_compute_contract {
namespace {

using namespace dsl_support;

int test_compute_fixed_bit_ops_build_deterministic_ir() {
  const auto first32 = BuildFixedLane32BitOps();
  const auto second32 = BuildFixedLane32BitOps();
  const auto fixed_lane64 = BuildFixedLane64BitOps();

  TEST_ASSERT(first32.ok());
  TEST_ASSERT(second32.ok());
  TEST_ASSERT(fixed_lane64.ok());
  TEST_ASSERT(first32.ir().canonical_bytes == second32.ir().canonical_bytes);
  TEST_ASSERT(first32.ir().op_hash_hi == second32.ir().op_hash_hi);
  TEST_ASSERT(first32.ir().op_hash_lo == second32.ir().op_hash_lo);
  TEST_ASSERT(first32.ir().scalar == rund::kernel::ComputeScalar::Lane32);
  TEST_ASSERT(fixed_lane64.ir().scalar == rund::kernel::ComputeScalar::Lane64);

  const rund::kernel::LoweringArtifact metal32 = rund::kernel::LowerComputeIR(
      first32.ir(), rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(metal32.ok);
  TEST_ASSERT(metal32.source_text.find("].op=bit_and") !=
              std::string_view::npos);
  TEST_ASSERT(metal32.source_text.find("].op=bit_or") !=
              std::string_view::npos);
  TEST_ASSERT(metal32.source_text.find("].op=bit_xor") !=
              std::string_view::npos);
  TEST_ASSERT(metal32.source_text.find("].op=bit_not") !=
              std::string_view::npos);
  TEST_ASSERT(metal32.source_text.find("].op=shl_const") !=
              std::string_view::npos);
  TEST_ASSERT(metal32.source_text.find("].op=shr_logical_const") !=
              std::string_view::npos);
  TEST_ASSERT(metal32.source_text.find("].op=shr_arithmetic_const") !=
              std::string_view::npos);
  return 0;
}

int test_compute_fixed_bit_overshifts_reject() {
  i32 input32[4]{};
  i32 out32[4]{};
  const auto body32 = rund::compute_dsl::bind(4u)
                          .fixed<1, 31>()
                          .read<"in">(input32)
                          .write<"out">(out32);
  const auto overshift32 = rund::compute_dsl::def("dsl-fixed-bit-overshift32")
                               .on(body32)
                               .map([](auto i, auto b) {
                                 auto input = b.template read<"in">();
                                 auto out = b.template write<"out">();

                                 out[i] =
                                     rund::compute_dsl::shl_const<32>(input[i]);
                               });

  i64 input64[4]{};
  i64 out64[4]{};
  const auto body64 = rund::compute_dsl::bind(4u)
                          .fixed<1, 63>()
                          .read<"in">(input64)
                          .write<"out">(out64);
  const auto overshift64 =
      rund::compute_dsl::def("dsl-fixed-bit-overshift64")
          .on(body64)
          .map([](auto i, auto b) {
            auto input = b.template read<"in">();
            auto out = b.template write<"out">();

            out[i] = rund::compute_dsl::shr_arithmetic_const<64>(input[i]);
          });

  TEST_ASSERT(!overshift32.ok());
  TEST_ASSERT(std::string_view{overshift32.reason()} ==
              "compute_shift_count_invalid");
  TEST_ASSERT(!overshift64.ok());
  TEST_ASSERT(std::string_view{overshift64.reason()} ==
              "compute_shift_count_invalid");
  return 0;
}

} // namespace

int RunComputeDslBitOpsContract() {
  if (test_compute_fixed_bit_ops_build_deterministic_ir() != 0) {
    return 1;
  }
  return test_compute_fixed_bit_overshifts_reject();
}

} // namespace program_compute_contract
