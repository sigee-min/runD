#include "local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

thread_local const char* g_vulkan_last_error = "ok";

}  // namespace

void SetVulkanLastError(VulkanAdapter& adapter, const char* const reason) {
  (void)adapter;
  const char* const stable_reason =
      reason == nullptr || reason[0] == '\0' ? "compute_backend_failed" : reason;
  g_vulkan_last_error = stable_reason;
}

const char* VulkanLastError(void* const context) {
  auto* const adapter = static_cast<VulkanAdapter*>(context);
  if (adapter == nullptr) {
    return "compute_backend_failed";
  }
  return g_vulkan_last_error;
}
#endif

}  // namespace rund::node::accel::detail
