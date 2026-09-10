#include "state.hpp"

#include "prepare/record.hpp"
#include "transfer.hpp"

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
  if (trace.queries != VK_NULL_HANDLE) {
    vkDestroyQueryPool(adapter->device, trace.queries, nullptr);
    trace.queries = VK_NULL_HANDLE;
  }
  DestroyCommand(adapter->device, trace.command);
  DestroyVulkanPipelineTransfer(*this);
  residency.reset();
  // The recipe may hold the last prepared-kernel owner. Its deleter takes
  // adapter->mutex independently, so let member destruction release it only
  // after this lock and the recorded native commands have been destroyed.
  recurrence.reset();
  transducers.clear();
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
  if (pipeline->trace.queries != VK_NULL_HANDLE) {
    vkDestroyQueryPool(pipeline->adapter->device, pipeline->trace.queries,
                       nullptr);
    pipeline->trace.queries = VK_NULL_HANDLE;
  }
  DestroyCommand(pipeline->adapter->device, pipeline->trace.command);
  DestroyVulkanPipelineTransfer(*pipeline);
  pipeline->residency.reset();
  DestroyCommand(pipeline->adapter->device, pipeline->command);
  pipeline->adapter = nullptr;
  // The preparation caller still holds submission.mutex and adapter->mutex.
  // Keep the failed object (including that mutex and retained kernel owners)
  // alive until its preparation scope unwinds after the lock guard.
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
