#include "internal.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

#include "../../../kernel.hpp"
#include "../../../range/local.hpp"
#include "../../pipeline/source.hpp"

#include <kernel/core/checked.hpp>

namespace rund::node::accel::detail {

bool BuildVulkanRangeManifest(const KernelExecutionStep &step,
                              PreparedBackendManifest &manifest) noexcept {
  if (step.kind() != rund::kernel::NodeKind::Stencil &&
      step.kind() != rund::kernel::NodeKind::Window) {
    return false;
  }
  const RangePlan &range = *RangePlanFor(step.operation);
  const std::optional<RangeExec> execution = RangeExec::from(range);
  if (!execution.has_value()) {
    return false;
  }
  const std::uint64_t stage_count = range.stage_count();
  const bool controlled = step.kind() == rund::kernel::NodeKind::Window &&
                          range.shape().resident_counted();
  const std::uint32_t descriptor_count = execution->descriptor_count();
  std::uint64_t descriptor_bindings = 0u;
  std::uint64_t pipeline_stages = 0u;
  std::uint64_t descriptor_sets = 0u;
  if (!range.ok() || stage_count == 0u ||
      !rund::kernel::checked::mul(stage_count, descriptor_count,
                                  descriptor_bindings) ||
      !rund::kernel::checked::add(stage_count, controlled ? 1u : 0u,
                                  pipeline_stages) ||
      !rund::kernel::checked::add(stage_count, controlled ? 1u : 0u,
                                  descriptor_sets) ||
      (controlled && !rund::kernel::checked::add(descriptor_bindings, 4u,
                                                 descriptor_bindings))) {
    return false;
  }
  manifest = PreparedBackendManifest{
      .source_build_count = controlled ? 2u : 1u,
      .source_library_dependency_count = controlled ? 2u : 1u,
      .pipeline_stage_count = pipeline_stages,
      .descriptor_set_count = descriptor_sets,
      .descriptor_binding_count = descriptor_bindings,
      .descriptor_lease_count = descriptor_sets,
      .descriptor_dependency_count = pipeline_stages};
  std::uint64_t source_bytes = 0u;
  if (!VulkanRangeSourceBytes(*execution, source_bytes) ||
      !AddPreparedBackendCacheDependency(
          manifest, PreparedBackendCacheDependency{
                        .source_recipe = 0x76756c6b2e726e67ull,
                        .source_upper_bytes = source_bytes,
                        .pipeline_stage_count = 1u,
                    })) {
    return false;
  }
  if (controlled) {
    std::uint64_t control_source_bytes = 0u;
    if (!VulkanRangeControlSourceBytes(range, control_source_bytes) ||
        !AddPreparedBackendCacheDependency(
            manifest, PreparedBackendCacheDependency{
                          .source_recipe = 0x76756c6b2e726374ull,
                          .source_upper_bytes = control_source_bytes,
                          .pipeline_stage_count = 1u,
                      })) {
      return false;
    }
  }
  return true;
}

} // namespace rund::node::accel::detail

#endif
