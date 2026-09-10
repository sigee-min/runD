#include "../internal.hpp"

#include "../../../../../map/api.hpp"
#include "../../../../../map/local.hpp"
#include "../../../../ops/table.hpp"
#include "../../../prepare/record.hpp"

#include "../../../../../../kernel/bindings/step.hpp"

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/backend.hpp>

#include <limits>

namespace rund::node::accel::detail::vulkan_generated_indirect_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace {

[[nodiscard]] bool entry_eligible(const VulkanKernelEntry &entry,
                                  std::uint32_t &checked) noexcept {
  if (entry.ops.encode != EncodeVulkanMap ||
      entry.ops.pipeline_capture_demand !=
          DescribeVulkanMapPipelineCaptureDemand) {
    return false;
  }
  const auto *const map =
      static_cast<const VulkanMapEncodeResources *>(entry.resource.get());
  if (map == nullptr || map->prepared == nullptr || map->history_recurrence ||
      map->iterations != 1u || map->control.has_count() ||
      map->control.has_predicate() || map->windows.empty() ||
      map->descriptor_sets.count != map->windows.size() ||
      map->prepared->plan.dispatch_count != map->windows.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < map->windows.size(); ++index) {
    const auto &window = map->windows[index];
    if (window.tile_count == 0u ||
        window.begin_sequence >
            std::numeric_limits<std::uint64_t>::max() - window.tile_count ||
        map->descriptor_sets.at(index) == VK_NULL_HANDLE) {
      return false;
    }
  }
  if (!map->prepared->checks.empty()) {
    if (checked == std::numeric_limits<std::uint32_t>::max()) {
      return false;
    }
    ++checked;
  }
  return true;
}

} // namespace

bool eligible(const VulkanPipeline &pipeline) noexcept {
  if (pipeline.record == nullptr || pipeline.record->entries.empty()) {
    return false;
  }
  std::uint32_t checked = 0u;
  for (const VulkanPipelineRecordEntry &entry : pipeline.record->entries) {
    const auto *const resources =
        entry.prepared == nullptr
            ? nullptr
            : static_cast<const VulkanKernelResources *>(entry.prepared.get());
    // A generated role is deliberately a single-resource Map row.  This
    // rejects mixed fused resources before any generated descriptor is made.
    const VulkanKernelEntry *const kernel =
        resources == nullptr || resources->size() != 1u ? nullptr
                                                        : resources->entry(0u);
    if (kernel == nullptr || !entry_eligible(*kernel, checked)) {
      return false;
    }
  }
  return checked == 1u;
}

#endif

} // namespace rund::node::accel::detail::vulkan_generated_indirect_detail
