#include "evidence.hpp"

#include "../../map/api.hpp"
#include "../../map/local.hpp"
#include "../view.hpp"

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/lowering/vulkan/shape.hpp>
#include <rund/counter.hpp>

#include <limits>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

// A direct Map retains the exact windows consumed by EncodeVulkanMapWindow.
// Each window records one vkCmdDispatch with ceil(tile_count / 256)
// workgroups, while the shader's gid guard makes tile_count the exact logical
// work-item count. Device-authored indirect arguments and auxiliary View
// commands do not have that cold-known ownership and therefore keep the
// complete Program row unavailable instead of publishing a partial estimate.
bool AccumulateVulkanWork(const VulkanMapEncodeResources &map,
                          VulkanPipelineWork &work) noexcept {
  if (map.control.has_count() || map.control.has_predicate() ||
      map.windows.empty() || map.descriptor_sets.size() != map.windows.size() ||
      map.plan.dispatch_count != map.windows.size()) {
    return false;
  }
  for (const rund::kernel::ComputeDispatchWindow &window : map.windows) {
    const std::uint64_t groups =
        rund::kernel::compute_lowering_detail::VulkanMapGroupsForTiles(
            window.tile_count);
    if (groups == 0u ||
        !rund::kernel::checked::add(work.dispatch_count, 1u,
                                    work.dispatch_count) ||
        !rund::kernel::checked::add(work.workgroup_count, groups,
                                    work.workgroup_count) ||
        !rund::kernel::checked::add(work.work_item_count, window.tile_count,
                                    work.work_item_count)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool AccumulateVulkanWork(const VulkanKernelEntry &entry,
                                        VulkanPipelineWork &work) noexcept {
  if (entry.ops.encode != EncodeVulkanMap ||
      VulkanViewDispatchCount(entry.view) != 0u) {
    return false;
  }
  const auto *const map =
      static_cast<const VulkanMapEncodeResources *>(entry.resource.get());
  return map != nullptr && AccumulateVulkanWork(*map, work);
}

template <class T>
[[nodiscard]] static std::uint64_t
vector_bytes(const std::vector<T> &values) noexcept {
  constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  return values.capacity() > maximum / sizeof(T)
             ? maximum
             : static_cast<std::uint64_t>(values.capacity()) * sizeof(T);
}

[[nodiscard]] std::uint64_t
VulkanRecurrenceHostBytes(const VulkanPipeline &pipeline) noexcept {
  const auto *const map =
      static_cast<const VulkanMapEncodeResources *>(pipeline.recurrence.get());
  if (map == nullptr) {
    return 0u;
  }
  std::uint64_t bytes = sizeof(VulkanMapEncodeResources);
  for (const std::uint64_t allocation :
       {vector_bytes(map->windows), vector_bytes(map->resident.inputs),
        vector_bytes(map->resident.outputs),
        vector_bytes(map->descriptor_sets)}) {
    bytes = ::rund::detail::counter::SaturatingAdd(bytes, allocation);
  }
  return bytes;
}
#endif

} // namespace rund::node::accel::detail
