#include "local.hpp"

namespace program_compute_contract {

using namespace lowering_support;
using namespace nonlinear_support;

int VulkanLoweringNonlinearOps() {
  const auto fixed_lane32 = BuildFixedLane32NonlinearOps();
  const auto fixed_lane64 = BuildFixedLane64NonlinearOps();
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
  TEST_ASSERT(artifact32.source_text.find("].op=div_fixed") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=recip") !=
              std::string_view::npos);
  TEST_ASSERT(artifact32.source_text.find("].op=sqrt") !=
              std::string_view::npos);

  return 0;
}

} // namespace program_compute_contract
