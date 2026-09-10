#include "internal.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include "../../../compact/local.hpp"
#include "../../../gather/local.hpp"
#include "../../../histogram/local.hpp"
#include "../../../kernel.hpp"
#include "../../pipeline/source.hpp"

namespace rund::node::accel::detail {

bool BuildVulkanCollectiveManifest(const KernelExecutionStep &step,
                                   PreparedBackendManifest &manifest) noexcept {
  switch (step.kind()) {
  case rund::kernel::NodeKind::Compact: {
    manifest = PreparedBackendManifest{.source_build_count = 3u,
                                       .source_library_dependency_count = 3u,
                                       .pipeline_stage_count = 3u,
                                       .descriptor_set_count = 3u,
                                       .descriptor_binding_count = 18u,
                                       .descriptor_lease_count = 3u,
                                       .descriptor_dependency_count = 3u};
    for (const CompactStage stage :
         {CompactStage::Classify, CompactStage::Prefix,
          CompactStage::Scatter}) {
      std::uint64_t source_bytes = 0u;
      if (!VulkanCompactSourceBytes(stage, source_bytes) ||
          !AddPreparedBackendCacheDependency(
              manifest,
              PreparedBackendCacheDependency{
                  .source_recipe = 0x76756c6b2e636d00ull +
                                   static_cast<std::uint64_t>(stage) + 1u,
                  .source_upper_bytes = source_bytes,
                  .pipeline_stage_count = 1u,
              })) {
        return false;
      }
    }
    return true;
  }
  case rund::kernel::NodeKind::Gather: {
    manifest = PreparedBackendManifest{.source_build_count = 2u,
                                       .source_library_dependency_count = 2u,
                                       .pipeline_stage_count = 2u,
                                       .descriptor_set_count = 2u,
                                       .descriptor_binding_count = 14u,
                                       .descriptor_lease_count = 2u,
                                       .descriptor_dependency_count = 2u};
    const auto element = step.operation.get<operation::Gather>().plan.element;
    for (const bool control : {true, false}) {
      std::uint64_t source_bytes = 0u;
      if (!VulkanGatherSourceBytes(element, control, source_bytes) ||
          !AddPreparedBackendCacheDependency(
              manifest, PreparedBackendCacheDependency{
                            .source_recipe = control ? 0x76756c6b2e676374ull
                                                     : 0x76756c6b2e676174ull,
                            .source_upper_bytes = source_bytes,
                            .pipeline_stage_count = 1u,
                        })) {
        return false;
      }
    }
    return true;
  }
  case rund::kernel::NodeKind::Histogram: {
    manifest = PreparedBackendManifest{.source_build_count = 2u,
                                       .source_library_dependency_count = 2u,
                                       .pipeline_stage_count = 2u,
                                       .descriptor_set_count = 2u,
                                       .descriptor_binding_count = 8u,
                                       .descriptor_lease_count = 2u,
                                       .descriptor_dependency_count = 2u};
    for (const bool clear : {true, false}) {
      std::uint64_t source_bytes = 0u;
      if (!VulkanHistogramSourceBytes(clear, source_bytes) ||
          !AddPreparedBackendCacheDependency(
              manifest, PreparedBackendCacheDependency{
                            .source_recipe = clear ? 0x76756c6b2e68636cull
                                                   : 0x76756c6b2e686374ull,
                            .source_upper_bytes = source_bytes,
                            .pipeline_stage_count = 1u,
                        })) {
        return false;
      }
    }
    return true;
  }
  default:
    return false;
  }
}

} // namespace rund::node::accel::detail

#endif
