#include "state.hpp"

#include "../../command/resources.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

VulkanPipeline::~VulkanPipeline() {
  if (adapter == nullptr) {
    return;
  }
  std::scoped_lock lock{submission.mutex, adapter->mutex};
  if (command.fence != VK_NULL_HANDLE && submission.active()) {
    (void)vkWaitForFences(adapter->device, 1u, &command.fence, VK_TRUE,
                          UINT64_MAX);
  }
  if (profile != nullptr && profile->timestamps != VK_NULL_HANDLE) {
    vkDestroyQueryPool(adapter->device, profile->timestamps, nullptr);
    profile->timestamps = VK_NULL_HANDLE;
  }
  recurrence.reset();
  DestroyVulkanWindow(window);
  DestroyVulkanPipelinePublish(publish);
  DestroyVulkanPipelineControl(control);
  DestroyCommand(adapter->device, command);
}

rund::AccelCheck FailVulkanPipeline(std::shared_ptr<VulkanPipeline> &pipeline,
                                    const char *const reason) {
  DestroyVulkanWindow(pipeline->window);
  DestroyVulkanPipelinePublish(pipeline->publish);
  DestroyVulkanPipelineControl(pipeline->control);
  if (pipeline->profile != nullptr &&
      pipeline->profile->timestamps != VK_NULL_HANDLE) {
    vkDestroyQueryPool(pipeline->adapter->device, pipeline->profile->timestamps,
                       nullptr);
    pipeline->profile->timestamps = VK_NULL_HANDLE;
  }
  DestroyCommand(pipeline->adapter->device, pipeline->command);
  pipeline->adapter = nullptr;
  pipeline.reset();
  return rund::AccelCheck{false, reason};
}

bool ValidVulkanPipeline(const VulkanPipeline *const pipeline) noexcept {
  return pipeline != nullptr && pipeline->adapter != nullptr &&
         (pipeline->dispatch_count == 0u ||
          (pipeline->command.buffer != VK_NULL_HANDLE &&
           pipeline->command.fence != VK_NULL_HANDLE));
}

#endif

} // namespace rund::node::accel::detail
