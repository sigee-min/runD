#include "../../local.hpp"

namespace program_compute_contract {

using namespace lowering_support;

namespace {

int FixedLane64IntegerSource() {
  const auto op = BuildFixedLane64Op(7);
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Vulkan);

  TEST_ASSERT(artifact.ok);
  TEST_ASSERT(artifact.kind ==
              rund::kernel::LoweringArtifactKind::VulkanSource);
  TEST_ASSERT(artifact.key.scalar == rund::kernel::ComputeScalar::Lane64);
  TEST_ASSERT(artifact.metadata.ok);
  TEST_ASSERT(artifact.metadata.map.api == rund::kernel::ComputeApi::Vulkan);
  TEST_ASSERT(artifact.metadata.map.scalar ==
              rund::kernel::ComputeScalar::Lane64);
  TEST_ASSERT(artifact.metadata.map.input_buffer_count == 1u);
  TEST_ASSERT(artifact.metadata.map.input_bytes_per_tile == 8u);
  TEST_ASSERT(artifact.metadata.map.output_bytes_per_tile == 8u);
  TEST_ASSERT(artifact.metadata.map.param_bytes == 8u);
  TEST_ASSERT(artifact.metadata.read_count == 1u);
  TEST_ASSERT(artifact.metadata.write_count == 1u);
  TEST_ASSERT(artifact.metadata.input_element_bytes.size() == 1u);
  TEST_ASSERT(artifact.metadata.input_element_bytes[0] == 8u);
  TEST_ASSERT(artifact.metadata.param_storage.size() == 8u);
  TEST_ASSERT(!artifact.source_text.empty());
  TEST_ASSERT(artifact.source_text.find("scalar=fixed_lane64") !=
              std::string_view::npos);
  TEST_ASSERT(artifact.source_text.find(
                  "#extension GL_EXT_shader_explicit_arithmetic_types_int64 : "
                  "require") != std::string_view::npos);
  return 0;
}

int FixedLane64RawMultiply() {
  const auto op = BuildFixedLane64MulOp(7);
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Vulkan);

  TEST_ASSERT(artifact.ok);
  TEST_ASSERT(artifact.kind ==
              rund::kernel::LoweringArtifactKind::VulkanSource);
  TEST_ASSERT(artifact.key.scalar == rund::kernel::ComputeScalar::Lane64);
  TEST_ASSERT(artifact.metadata.ok);
  TEST_ASSERT(artifact.metadata.map.scalar ==
              rund::kernel::ComputeScalar::Lane64);
  TEST_ASSERT(artifact.metadata.map.input_bytes_per_tile == 8u);
  TEST_ASSERT(artifact.metadata.map.output_bytes_per_tile == 8u);
  TEST_ASSERT(artifact.metadata.map.param_bytes == 8u);
  TEST_ASSERT(!artifact.source_text.empty());
  return 0;
}

} // namespace

int VulkanLoweringFixedLane64() {
  if (FixedLane64IntegerSource() != 0) {
    return 1;
  }
  if (FixedLane64RawMultiply() != 0) {
    return 1;
  }
  return 0;
}

} // namespace program_compute_contract
