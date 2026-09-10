#include "internal.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include "../../../kernel.hpp"
#include "../../../sort/local/api.hpp"
#include "../../../sort/local/state.hpp"
#include "../../pipeline/source.hpp"

namespace rund::node::accel::detail {

bool BuildVulkanSortManifest(const KernelExecutionStep &step,
                             const rund::kernel::ComputePlan &,
                             PreparedBackendManifest &manifest) noexcept {
  if (step.kind() != rund::kernel::NodeKind::Sort) {
    return false;
  }
  const auto &active = step.operation.get<operation::Sort>();
  const std::uint64_t passes = active.plan.radix_pass_count;
  if (!rund::kernel::checked::mul(4u, passes, manifest.descriptor_set_count) ||
      !rund::kernel::checked::add(manifest.descriptor_set_count, 1u,
                                  manifest.descriptor_set_count) ||
      !rund::kernel::checked::mul(9u, manifest.descriptor_set_count,
                                  manifest.descriptor_binding_count)) {
    return false;
  }
  manifest.source_build_count = 5u;
  manifest.source_library_dependency_count = 5u;
  manifest.pipeline_stage_count = 5u;
  manifest.descriptor_lease_count = manifest.descriptor_set_count;
  manifest.descriptor_dependency_count = 5u;
  for (const SortStage stage :
       {SortStage::Dispatch, SortStage::Classify, SortStage::Prefix,
        SortStage::Base, SortStage::Scatter}) {
    std::uint64_t source_bytes = 0u;
    if (!VulkanSortSourceBytes(active.desc.key, stage, source_bytes) ||
        !AddPreparedBackendCacheDependency(
            manifest,
            PreparedBackendCacheDependency{
                .source_recipe = 0x76756c6b2e736f00ull +
                                 static_cast<std::uint64_t>(stage) + 1u,
                .source_upper_bytes = source_bytes,
                .pipeline_stage_count = 1u,
            })) {
      return false;
    }
  }
  return true;
}

} // namespace rund::node::accel::detail

#endif
