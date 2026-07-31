#include "local.hpp"

namespace program_compute_contract {

using namespace lowering_support;

int VulkanLoweringScalarOps() {
  const auto fixed_lane32 = BuildFixedLane32ScalarOps();
  const auto fixed_lane64 = BuildFixedLane64ScalarOps();
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
  TEST_ASSERT(artifact32.source_text.find("RundWideSignedLess(") !=
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
  TEST_ASSERT(artifact64.source_text.find("RundWideSignedLess(") !=
              std::string_view::npos);
  return 0;
}

} // namespace program_compute_contract
