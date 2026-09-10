#include "service.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool StartVulkanCompletionService(VulkanAdapter &adapter) {
  std::unique_lock lock{adapter.completion_mutex};
  if (adapter.completion_thread.joinable()) {
    return true;
  }
  adapter.completion_stop = false;
  adapter.residency_stop = false;
  try {
    adapter.completion_thread = std::thread{RunVulkanCompletion, &adapter};
    adapter.residency_thread =
        std::thread{RunVulkanResidencyCompletion, &adapter};
  } catch (...) {
    adapter.completion_stop = true;
    adapter.residency_stop = true;
    lock.unlock();
    adapter.completion_cv.notify_all();
    adapter.residency_cv.notify_all();
    if (adapter.completion_thread.joinable()) {
      adapter.completion_thread.join();
    }
    if (adapter.residency_thread.joinable()) {
      adapter.residency_thread.join();
    }
    return false;
  }
  return true;
}

void StopVulkanCompletionService(VulkanAdapter &adapter) noexcept {
  {
    std::lock_guard lock{adapter.completion_mutex};
    adapter.completion_stop = true;
  }
  {
    std::lock_guard lock{adapter.residency_mutex};
    adapter.residency_stop = true;
  }
  adapter.completion_cv.notify_all();
  adapter.residency_cv.notify_all();
  if (adapter.completion_thread.joinable()) {
    adapter.completion_thread.join();
  }
  if (adapter.residency_thread.joinable()) {
    adapter.residency_thread.join();
  }
}
#endif

} // namespace rund::node::accel::detail
