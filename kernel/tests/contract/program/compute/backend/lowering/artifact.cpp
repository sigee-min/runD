#include "contract/program/compute/backend/lowering/local.hpp"
#include "test/assert.hpp"

namespace program_compute_contract {
namespace {

using namespace backend_lowering_support;

int test_compute_artifact_key_includes_api_scalar_op_and_canonical_hash() {
  const auto fixed_lane32 = BuildFixedLane32Op(7);
  const auto fixed_lane64 = BuildFixedLane64Op(7);
  const rund::kernel::LoweringArtifact metal =
      rund::kernel::LowerComputeIR(fixed_lane32.ir(),
                                   rund::kernel::ComputeApi::Metal);
  const rund::kernel::LoweringArtifact vulkan =
      rund::kernel::LowerComputeIR(fixed_lane32.ir(),
                                   rund::kernel::ComputeApi::Vulkan);
  const rund::kernel::LoweringArtifact metal64 =
      rund::kernel::LowerComputeIR(fixed_lane64.ir(),
                                   rund::kernel::ComputeApi::Metal);
  const rund::kernel::compute_ir_detail::ComputeIrHash canonical =
      rund::kernel::compute_ir_detail::HashComputeIrCanonicalBytes(
          fixed_lane32.ir().canonical_bytes.data(),
          static_cast<rund::kernel::u64>(
              fixed_lane32.ir().canonical_bytes.size()));

  TEST_ASSERT(metal.ok);
  TEST_ASSERT(vulkan.ok);
  TEST_ASSERT(metal64.ok);
  TEST_ASSERT(metal.key.api == rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(vulkan.key.api == rund::kernel::ComputeApi::Vulkan);
  TEST_ASSERT(metal.key.scalar == rund::kernel::ComputeScalar::Lane32);
  TEST_ASSERT(metal64.key.scalar == rund::kernel::ComputeScalar::Lane64);
  TEST_ASSERT(metal.key.op_hash_hi == fixed_lane32.ir().op_hash_hi);
  TEST_ASSERT(metal.key.op_hash_lo == fixed_lane32.ir().op_hash_lo);
  TEST_ASSERT(metal.key.canonical_ir_hash_hi == canonical.hi);
  TEST_ASSERT(metal.key.canonical_ir_hash_lo == canonical.lo);
  TEST_ASSERT(metal.canonical_ir_bytes == fixed_lane32.ir().canonical_bytes);
  TEST_ASSERT(vulkan.canonical_ir_bytes == fixed_lane32.ir().canonical_bytes);
  TEST_ASSERT(metal64.canonical_ir_bytes == fixed_lane64.ir().canonical_bytes);
  TEST_ASSERT(!(metal.key == vulkan.key));
  TEST_ASSERT(!(metal.key == metal64.key));
  return 0;
}

}  // namespace

int RunComputeBackendLoweringArtifactContract() {
  return test_compute_artifact_key_includes_api_scalar_op_and_canonical_hash();
}

}  // namespace program_compute_contract
