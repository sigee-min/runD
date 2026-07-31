#include "local.hpp"

namespace program_compute_contract {

using namespace lowering_support;

int VulkanLoweringExpanded() {
  const auto fixed_lane32 = BuildFixedLane32ExpandedOps();
  const auto fixed_lane64 = BuildFixedLane64ExpandedOps();
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
  TEST_ASSERT(artifact32.source_text.find("RundWideSignedLess(") !=
              std::string_view::npos);
  TEST_ASSERT(artifact64.source_text.find("RundWideBool(") !=
              std::string_view::npos);
  TEST_ASSERT(artifact64.source_text.find("RundWideTruthy(") !=
              std::string_view::npos);
  TEST_ASSERT(artifact64.source_text.find("RundWideSignedLess(") !=
              std::string_view::npos);
  return 0;
}

} // namespace program_compute_contract
