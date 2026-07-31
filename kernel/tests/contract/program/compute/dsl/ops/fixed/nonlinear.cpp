#include "contract/program/compute/dsl/ops/local.hpp"

namespace program_compute_contract {
namespace {

using namespace dsl_support;

int test_compute_fixed_nonlinear_ops_build_deterministic_ir() {
  const auto first32 = BuildFixedLane32NonlinearOps();
  const auto second32 = BuildFixedLane32NonlinearOps();
  const auto fixed_lane64 = BuildFixedLane64NonlinearOps();

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
  TEST_ASSERT(metal32.source_text.find("].op=div_fixed") !=
              std::string_view::npos);
  TEST_ASSERT(metal32.source_text.find("].op=recip") != std::string_view::npos);
  TEST_ASSERT(metal32.source_text.find("].op=sqrt") != std::string_view::npos);
  return 0;
}

} // namespace

int RunComputeDslNonlinearOpsContract() {
  return test_compute_fixed_nonlinear_ops_build_deterministic_ir();
}

} // namespace program_compute_contract
