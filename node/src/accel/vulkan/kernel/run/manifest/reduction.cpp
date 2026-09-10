#include "internal.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include "../../../kernel.hpp"
#include "../../../reduce/local.hpp"
#include "../../../scatter/reduce/model.hpp"
#include "../../../segmented/reduce/model.hpp"
#include "../../pipeline/source.hpp"

namespace rund::node::accel::detail {

bool BuildVulkanReductionManifest(const KernelExecutionStep &step,
                                  const rund::kernel::ComputePlan &plan,
                                  PreparedBackendManifest &manifest) noexcept {
  switch (step.kind()) {
  case rund::kernel::NodeKind::SegmentedReduce: {
    manifest = PreparedBackendManifest{.source_build_count = 4u,
                                       .source_library_dependency_count = 4u,
                                       .pipeline_stage_count = 4u,
                                       .descriptor_set_count = 4u,
                                       .descriptor_binding_count = 24u,
                                       .descriptor_lease_count = 4u,
                                       .descriptor_dependency_count = 4u};
    const auto &active = step.operation.get<operation::SegmentedReduce>();
    for (const VulkanSegmentedReduceStage stage :
         {VulkanSegmentedReduceStage::Classify,
          VulkanSegmentedReduceStage::Prefix,
          VulkanSegmentedReduceStage::Scatter,
          VulkanSegmentedReduceStage::Reduce}) {
      std::uint64_t source_bytes = 0u;
      if (!VulkanSegmentedReduceSourceBytes(active.plan, plan.domain, stage,
                                            source_bytes) ||
          !AddPreparedBackendCacheDependency(
              manifest, PreparedBackendCacheDependency{
                            .source_recipe = 0x76756c6b2e727300ull +
                                             static_cast<std::uint64_t>(stage),
                            .source_upper_bytes = source_bytes,
                            .pipeline_stage_count = 1u,
                        })) {
        return false;
      }
    }
    return true;
  }
  case rund::kernel::NodeKind::Reduce: {
    const auto &active = step.operation.get<operation::Reduce>();
    const std::uint64_t passes = active.plan.pass_count;
    if (!rund::kernel::checked::mul(6u, passes,
                                    manifest.descriptor_binding_count)) {
      return false;
    }
    manifest.source_build_count = 1u;
    manifest.source_library_dependency_count = 1u;
    manifest.pipeline_stage_count = 1u;
    manifest.descriptor_set_count = passes;
    manifest.descriptor_lease_count = passes;
    manifest.descriptor_dependency_count = 1u;
    std::uint64_t source_bytes = 0u;
    return VulkanReduceSourceBytes(active.desc.op, active.desc.element,
                                   active.desc.block_size, plan.domain,
                                   source_bytes) &&
           AddPreparedBackendCacheDependency(
               manifest, PreparedBackendCacheDependency{
                             .source_recipe = 0x76756c6b2e726564ull,
                             .source_upper_bytes = source_bytes,
                             .pipeline_stage_count = 1u,
                         });
  }
  case rund::kernel::NodeKind::ScatterReduce: {
    manifest = PreparedBackendManifest{.source_build_count = 3u,
                                       .source_library_dependency_count = 3u,
                                       .pipeline_stage_count = 3u,
                                       .descriptor_set_count = 3u,
                                       .descriptor_binding_count = 24u,
                                       .descriptor_lease_count = 3u,
                                       .descriptor_dependency_count = 3u};
    const auto &active = step.operation.get<operation::ScatterReduce>();
    for (const VulkanScatterReduceStage stage :
         {VulkanScatterReduceStage::Control, VulkanScatterReduceStage::Init,
          VulkanScatterReduceStage::Fold}) {
      std::uint64_t source_bytes = 0u;
      if (!VulkanScatterReduceSourceBytes(active.plan, stage, source_bytes) ||
          !AddPreparedBackendCacheDependency(
              manifest, PreparedBackendCacheDependency{
                            .source_recipe = 0x76756c6b2e737200ull +
                                             static_cast<std::uint64_t>(stage),
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
