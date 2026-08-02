#include "local.hpp"
#include "../descriptor.hpp"

#include <limits>
#include <new>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
VulkanMapDescriptorArena::~VulkanMapDescriptorArena() {
  if (adapter != nullptr && pool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(adapter->device, pool, nullptr);
  }
}

bool PrepareVulkanMapDescriptorArena(
    VulkanAdapter &adapter, const VulkanCachedPipeline &pipeline,
    const rund::kernel::u64 set_count,
    std::shared_ptr<VulkanMapDescriptorArena> &arena) {
  arena.reset();
  if (set_count == 0u) {
    return false;
  }
  try {
    auto prepared = std::make_shared<VulkanMapDescriptorArena>();
    prepared->adapter = &adapter;
    if (!CreateVulkanStorageDescriptorSets(
            adapter, pipeline, set_count, prepared->pool, prepared->sets)) {
      return false;
    }
    arena = std::move(prepared);
    return true;
  } catch (const std::bad_alloc &) {
    return false;
  }
}

bool AcquireVulkanMapDescriptorSets(
    const std::shared_ptr<VulkanMapDescriptorArena> &arena,
    const rund::kernel::u64 set_count,
    VulkanMapDescriptorLease &lease) noexcept {
  lease = {};
  if (arena == nullptr || set_count == 0u ||
      set_count > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  const std::size_t count = static_cast<std::size_t>(set_count);
  std::lock_guard lock{arena->mutex};
  if (arena->next > arena->sets.size() ||
      count > arena->sets.size() - arena->next) {
    return false;
  }
  lease = VulkanMapDescriptorLease{arena, arena->next, count};
  arena->next += count;
  return true;
}
#endif

}  // namespace rund::node::accel::detail
