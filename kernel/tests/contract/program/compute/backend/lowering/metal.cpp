#include "contract/program/compute/backend/lowering/local.hpp"
#include "test/assert.hpp"

namespace program_compute_contract {
namespace {

using namespace backend_lowering_support;

int test_compute_metal_lowering_emits_expanded_fixed_ops() {
  const auto fixed_lane32 = BuildFixedLane32ExpandedOps();
  const auto fixed_lane64 = BuildFixedLane64ExpandedOps();
  const rund::kernel::LoweringArtifact artifact32 =
      rund::kernel::LowerComputeIR(fixed_lane32.ir(),
                                   rund::kernel::ComputeApi::Metal);
  const rund::kernel::LoweringArtifact artifact64 =
      rund::kernel::LowerComputeIR(fixed_lane64.ir(),
                                   rund::kernel::ComputeApi::Metal);

  TEST_ASSERT(artifact32.ok);
  TEST_ASSERT(artifact64.ok);
  TEST_ASSERT(artifact32.source_text.find("node[") != std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=min") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=max") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=clamp") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=select") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=eq") != std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=lt") != std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=le") != std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("RundWideBool(") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("RundWideTruthy(") !=
              std::string_view::npos);
  TEST_ASSERT(artifact64.source_text.find("RundWideBool(") !=
              std::string_view::npos);
  TEST_ASSERT(artifact64.source_text.find("RundWideTruthy(") !=
              std::string_view::npos);
  TEST_ASSERT(artifact64.source_text.find("RundWideSignedLess(") !=
              std::string_view::npos);
  return 0;
}

int test_compute_metal_lowers_fixed_scalar_ops() {
  const auto fixed_lane32 = BuildFixedLane32ScalarOps();
  const auto fixed_lane64 = BuildFixedLane64ScalarOps();
  const rund::kernel::LoweringArtifact artifact32 =
      rund::kernel::LowerComputeIR(fixed_lane32.ir(),
                                   rund::kernel::ComputeApi::Metal);
  const rund::kernel::LoweringArtifact artifact64 =
      rund::kernel::LowerComputeIR(fixed_lane64.ir(),
                                   rund::kernel::ComputeApi::Metal);

  TEST_ASSERT(artifact32.ok);
  TEST_ASSERT(artifact64.ok);
  TEST_ASSERT(artifact32.source_text.find("].op=constant") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=neg") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=abs") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=abs_magnitude") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=sign") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=ne") != std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=gt") != std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=ge") != std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=predicate_not") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=predicate_and") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=predicate_or") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("RundWideNeg(") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("0x80000000u") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("0x7fffffffu") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("RundWideBool(") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("RundWideTruthy(") !=
              std::string_view::npos);
  TEST_ASSERT(artifact64.source_text.find("RundWideNeg(") !=
              std::string_view::npos);
  TEST_ASSERT(artifact64.source_text.find("0x8000000000000000ul") !=
              std::string_view::npos);
  TEST_ASSERT(artifact64.source_text.find("0x7ffffffffffffffful") !=
              std::string_view::npos);
  TEST_ASSERT(artifact64.source_text.find("RundWideBool(") !=
              std::string_view::npos);
  TEST_ASSERT(artifact64.source_text.find("RundWideTruthy(") !=
              std::string_view::npos);
  return 0;
}

int test_compute_metal_lowers_fixed_bit_ops() {
  const auto fixed_lane32 = BuildFixedLane32BitOps();
  const auto fixed_lane64 = BuildFixedLane64BitOps();
  const rund::kernel::LoweringArtifact artifact32 =
      rund::kernel::LowerComputeIR(fixed_lane32.ir(),
                                   rund::kernel::ComputeApi::Metal);
  const rund::kernel::LoweringArtifact artifact64 =
      rund::kernel::LowerComputeIR(fixed_lane64.ir(),
                                   rund::kernel::ComputeApi::Metal);

  TEST_ASSERT(artifact32.ok);
  TEST_ASSERT(artifact64.ok);
  TEST_ASSERT(artifact32.source_text.find("].op=bit_and") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=bit_or") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=bit_xor") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=bit_not") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=shl_const") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=shr_logical_const") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=shr_arithmetic_const") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("RundWideAnd(") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("RundWideShrUnsigned(") !=
              std::string_view::npos);
  TEST_ASSERT(artifact64.source_text.find("RundWideAnd(") !=
              std::string_view::npos);
  TEST_ASSERT(artifact64.source_text.find("RundWideShrUnsigned(") !=
              std::string_view::npos);
  return 0;
}

int test_compute_metal_lowers_fixed_nonlinear_ops() {
  const auto fixed_lane32 = BuildFixedLane32NonlinearOps();
  const auto fixed_lane64 = BuildFixedLane64NonlinearOps();
  const rund::kernel::LoweringArtifact artifact32 =
      rund::kernel::LowerComputeIR(fixed_lane32.ir(),
                                   rund::kernel::ComputeApi::Metal);
  const rund::kernel::LoweringArtifact artifact64 =
      rund::kernel::LowerComputeIR(fixed_lane64.ir(),
                                   rund::kernel::ComputeApi::Metal);

  TEST_ASSERT(artifact32.ok);
  TEST_ASSERT(artifact64.ok);
  TEST_ASSERT(artifact32.source_text.find("].op=div_fixed") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=recip") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=sqrt") !=
              std::string_view::npos);

  return 0;
}

int test_compute_metal_lowering_emits_integer_division() {
  const auto signed_op = BuildI32DivideOp();
  const auto unsigned_op = BuildU64DivideOp();
  const auto signed_artifact = rund::kernel::LowerComputeIR(
      signed_op.ir(), rund::kernel::ComputeApi::Metal);
  const auto unsigned_artifact = rund::kernel::LowerComputeIR(
      unsigned_op.ir(), rund::kernel::ComputeApi::Metal);

  TEST_ASSERT(signed_artifact.ok);
  TEST_ASSERT(unsigned_artifact.ok);
  TEST_ASSERT(signed_artifact.metadata.map.domain ==
              rund::kernel::ComputeDomain::I32);
  TEST_ASSERT(unsigned_artifact.metadata.map.domain ==
              rund::kernel::ComputeDomain::U64);
  TEST_ASSERT(signed_artifact.source_text.find("].op=div_signed") !=
              std::string_view::npos);
  TEST_ASSERT(unsigned_artifact.source_text.find("].op=div_unsigned") !=
              std::string_view::npos);
  TEST_ASSERT(signed_artifact.source_text.find("RundDivSigned") !=
              std::string_view::npos);
  TEST_ASSERT(unsigned_artifact.source_text.find("RundDivUnsigned") !=
              std::string_view::npos);
  return 0;
}

int test_compute_metal_lowering_emits_logical_index() {
  const auto op = BuildU64IndexOp();
  const auto artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Metal);

  TEST_ASSERT(artifact.ok);
  TEST_ASSERT(artifact.metadata.map.input_buffer_count == 0u);
  TEST_ASSERT(artifact.source_text.find("].op=index") !=
              std::string_view::npos);
  TEST_ASSERT(artifact.source_text.find("long(gid)") != std::string_view::npos);
  return 0;
}

int test_compute_metal_lowering_emits_narrow_mask_store() {
  const auto op = BuildU64MaskOp();
  const auto artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Metal);

  TEST_ASSERT(artifact.ok);
  TEST_ASSERT(artifact.metadata.map.input_bytes_per_tile == 8u);
  TEST_ASSERT(artifact.metadata.map.output_bytes_per_tile == 4u);
  TEST_ASSERT(artifact.source_text.find("inline void StoreI32") !=
              std::string_view::npos);
  TEST_ASSERT(artifact.source_text.find("StoreI32(") != std::string_view::npos);
  TEST_ASSERT(artifact.source_text.find("int(node_") != std::string_view::npos);
  return 0;
}

} // namespace

int RunComputeBackendLoweringMetalContract() {
  if (test_compute_metal_lowering_emits_expanded_fixed_ops() != 0) {
    return 1;
  }
  if (test_compute_metal_lowers_fixed_scalar_ops() != 0) {
    return 1;
  }
  if (test_compute_metal_lowers_fixed_bit_ops() != 0) {
    return 1;
  }
  if (test_compute_metal_lowers_fixed_nonlinear_ops() !=
      0) {
    return 1;
  }
  if (test_compute_metal_lowering_emits_integer_division() != 0) {
    return 1;
  }
  if (test_compute_metal_lowering_emits_logical_index() != 0) {
    return 1;
  }
  return test_compute_metal_lowering_emits_narrow_mask_store();
}

} // namespace program_compute_contract
