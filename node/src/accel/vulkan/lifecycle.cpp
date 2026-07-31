#include "local.hpp"

#include "cached/pipeline.hpp"
#include "cached/index.hpp"
#include "buffer/resident/pool.hpp"
#include "command.hpp"
#include "resident/state.hpp"

#include <memory>
namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
VulkanAdapter::VulkanAdapter()
    : pipeline_index(std::make_unique<VulkanPipelineIndex>()),
      resident(std::make_unique<VulkanResidentState>()) {}

VulkanAdapter::~VulkanAdapter() {
  StopVulkanCompletionService(*this);
  if (device != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(device);
    DestroyVulkanCommandResources(*this);
    timestamp_query_available = false;
    pipeline_index->entries.clear();
    pipeline_index->collectives.clear();
    pipelines.clear();
    collective_pipelines.clear();
    for (VulkanBuffer& buffer : reusable_buffers) {
      DestroyVulkanBuffer(*this, buffer);
    }
    reusable_buffers.clear();
    reusable_buffer_bytes = 0u;
    staging_memory.pooled = 0u;
    {
      std::lock_guard lock{resident->mutex};
      resident->buffers.clear();
      DestroyVulkanResidentStorage(*this);
    }
    vkDestroyDevice(device, nullptr);
  }
  resident.reset();
  if (instance != VK_NULL_HANDLE) {
    vkDestroyInstance(instance, nullptr);
  }
}
#endif

}  // namespace rund::node::accel::detail
