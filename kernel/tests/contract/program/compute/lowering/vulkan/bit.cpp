#include "local.hpp"

namespace program_compute_contract {

using namespace lowering_support;

int VulkanLoweringBitOps() {
  const auto fixed_lane32 = BuildFixedLane32BitOps();
  const auto fixed_lane64 = BuildFixedLane64BitOps();
  const rund::kernel::LoweringArtifact artifact32 =
      rund::kernel::LowerComputeIR(fixed_lane32.ir(),
                                   rund::kernel::ComputeApi::Vulkan);
  const rund::kernel::LoweringArtifact artifact64 =
      rund::kernel::LowerComputeIR(fixed_lane64.ir(),
                                   rund::kernel::ComputeApi::Vulkan);

  TEST_ASSERT(artifact32.ok);
  TEST_ASSERT(artifact64.ok);
  TEST_ASSERT(artifact32.kind ==
              rund::kernel::LoweringArtifactKind::VulkanSource);
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
  TEST_ASSERT(artifact32.source_text.find("RundWideShrUnsigned(") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("RundWideAnd(") !=
              std::string_view::npos);
  TEST_ASSERT(artifact64.source_text.find("RundWideShrUnsigned(") !=
              std::string_view::npos);
  TEST_ASSERT(artifact64.source_text.find("RundWideAnd(") !=
              std::string_view::npos);
  return 0;
}

} // namespace program_compute_contract
