#include "internal.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include "../../../kernel.hpp"
#include "../../../scatter/local.hpp"
#include "../../pipeline/source.hpp"

namespace rund::node::accel::detail {

bool BuildVulkanScatterManifest(const KernelExecutionStep &step,
                                PreparedBackendManifest &manifest) noexcept {
  if (step.kind() != rund::kernel::NodeKind::Scatter) {
    return false;
  }
  manifest = PreparedBackendManifest{.source_build_count = 1u,
                                     .source_library_dependency_count = 1u,
                                     .pipeline_stage_count = 1u,
                                     .descriptor_set_count = 1u,
                                     .descriptor_binding_count = 5u,
                                     .descriptor_lease_count = 1u,
                                     .descriptor_dependency_count = 1u};
  std::uint64_t source_bytes = 0u;
  return VulkanScatterSourceBytes(
             step.operation.get<operation::Scatter>().plan.element,
             source_bytes) &&
         AddPreparedBackendCacheDependency(
             manifest, PreparedBackendCacheDependency{
                           .source_recipe = 0x76756c6b2e736361ull,
                           .source_upper_bytes = source_bytes,
                           .pipeline_stage_count = 1u,
                       });
}

} // namespace rund::node::accel::detail

#endif
