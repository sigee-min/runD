#include "contract/program/compute/backend/lowering/local.hpp"
#include "test/assert.hpp"

namespace program_compute_contract {
namespace {

using namespace backend_lowering_support;

int test_compute_lowering_rejects_invalid_missing_ir() {
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(rund::kernel::ComputeIR{},
                                   rund::kernel::ComputeApi::Metal);

  TEST_ASSERT(!artifact.ok);
  TEST_ASSERT(std::string_view{artifact.reason} == "compute_ir_invalid");
  TEST_ASSERT(artifact.source_text.empty());
  TEST_ASSERT(artifact.canonical_ir_bytes.empty());
  return 0;
}

int test_compute_lowering_rejects_unknown_api() {
  const auto op = BuildFixedLane32Op(7);
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(),
                                   static_cast<rund::kernel::ComputeApi>(0u));

  TEST_ASSERT(!artifact.ok);
  TEST_ASSERT(std::string_view{artifact.reason} == "compute_api_unsupported");
  return 0;
}

int test_compute_metal_lowering_is_stable_and_describes_layout() {
  const auto op = BuildFixedLane32Op(7);
  const rund::kernel::LoweringArtifact first =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Metal);
  const rund::kernel::LoweringArtifact second =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Metal);

  TEST_ASSERT(first.ok);
  TEST_ASSERT(second.ok);
  TEST_ASSERT(first.kind ==
              rund::kernel::LoweringArtifactKind::MetalSource);
  TEST_ASSERT(first.key == second.key);
  TEST_ASSERT(first.source_text == second.source_text);
  TEST_ASSERT(first.canonical_ir_bytes == op.ir().canonical_bytes);
  TEST_ASSERT(second.canonical_ir_bytes == op.ir().canonical_bytes);
  TEST_ASSERT(first.metadata.ok);
  TEST_ASSERT(first.metadata.map.api == rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(first.metadata.map.scalar == rund::kernel::ComputeScalar::Lane32);
  TEST_ASSERT(first.metadata.map.input_buffer_count == 2u);
  TEST_ASSERT(first.metadata.map.input_bytes_per_tile == 8u);
  TEST_ASSERT(first.metadata.map.output_bytes_per_tile == 4u);
  TEST_ASSERT(first.metadata.read_count == 2u);
  TEST_ASSERT(first.metadata.write_count == 1u);
  TEST_ASSERT(first.metadata.param_storage.size() == 4u);
  TEST_ASSERT(first.metadata.input_element_bytes.size() == 2u);
  TEST_ASSERT(first.metadata.input_element_bytes[0] == 4u);
  TEST_ASSERT(first.metadata.input_element_bytes[1] == 4u);
  TEST_ASSERT(!first.source_text.empty());
  TEST_ASSERT(first.source_text.find("rund.compute.metal.source") !=
              std::string_view::npos);
  TEST_ASSERT(first.source_text.find("op_hash_hi=") !=
              std::string_view::npos);
  TEST_ASSERT(first.source_text.find("canonical_ir_hash_hi=") !=
              std::string_view::npos);
  TEST_ASSERT(first.source_text.find("binding[0].kind=param") !=
              std::string_view::npos);
  TEST_ASSERT(first.source_text.find("binding[1].kind=read") !=
              std::string_view::npos);
  TEST_ASSERT(first.source_text.find("[[buffer(0)]]") !=
              std::string_view::npos);
  return 0;
}

int test_compute_metal_lowering_describes_fixed_lane64_layout() {
  const auto op = BuildFixedLane64Op(7);
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Metal);

  TEST_ASSERT(artifact.ok);
  TEST_ASSERT(artifact.kind ==
              rund::kernel::LoweringArtifactKind::MetalSource);
  TEST_ASSERT(artifact.key.scalar == rund::kernel::ComputeScalar::Lane64);
  TEST_ASSERT(artifact.metadata.ok);
  TEST_ASSERT(artifact.metadata.map.api == rund::kernel::ComputeApi::Metal);
  TEST_ASSERT(artifact.metadata.map.scalar ==
              rund::kernel::ComputeScalar::Lane64);
  TEST_ASSERT(artifact.metadata.map.input_buffer_count == 1u);
  TEST_ASSERT(artifact.metadata.map.input_bytes_per_tile == 8u);
  TEST_ASSERT(artifact.metadata.map.output_bytes_per_tile == 8u);
  TEST_ASSERT(artifact.metadata.map.param_bytes == 8u);
  TEST_ASSERT(artifact.metadata.input_element_bytes.size() == 1u);
  TEST_ASSERT(artifact.metadata.input_element_bytes[0] == 8u);
  TEST_ASSERT(artifact.metadata.param_storage.size() == 8u);
  TEST_ASSERT(!artifact.source_text.empty());
  TEST_ASSERT(artifact.source_text.find("scalar=fixed_lane64") !=
              std::string_view::npos);
  TEST_ASSERT(artifact.source_text.find("binding[0].kind=param") !=
              std::string_view::npos);
  TEST_ASSERT(artifact.source_text.find("element_bytes=8") !=
              std::string_view::npos);
  return 0;
}

}  // namespace

int RunComputeBackendLoweringBaseContract() {
  if (test_compute_lowering_rejects_invalid_missing_ir() != 0) {
    return 1;
  }
  if (test_compute_lowering_rejects_unknown_api() != 0) {
    return 1;
  }
  if (test_compute_metal_lowering_is_stable_and_describes_layout() != 0) {
    return 1;
  }
  return test_compute_metal_lowering_describes_fixed_lane64_layout();
}

}  // namespace program_compute_contract
