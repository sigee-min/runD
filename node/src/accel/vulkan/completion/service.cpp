#include "service.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool StartVulkanCompletionService(VulkanAdapter &adapter) {
  std::lock_guard lock{adapter.completion_mutex};
  if (adapter.completion_thread.joinable()) {
    return true;
  }
  adapter.completion_stop = false;
  try {
    adapter.completion_thread = std::thread{RunVulkanCompletion, &adapter};
  } catch (...) {
    return false;
  }
  return true;
}

void StopVulkanCompletionService(VulkanAdapter &adapter) noexcept {
  {
    std::lock_guard lock{adapter.completion_mutex};
    adapter.completion_stop = true;
  }
  adapter.completion_cv.notify_all();
  if (adapter.completion_thread.joinable()) {
    adapter.completion_thread.join();
  }
}
#endif

} // namespace rund::node::accel::detail
