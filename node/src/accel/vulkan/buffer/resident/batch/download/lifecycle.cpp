#include "internal.hpp"

namespace rund::node::accel::detail::batch_download_internal {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

HostReadback::HostReadback(VulkanAdapter &adapter) noexcept
    : adapter_{adapter} {
  ++adapter_.active_host_readbacks;
}

HostReadback::~HostReadback() {
  --adapter_.active_host_readbacks;
  adapter_.host_readback_cv.notify_all();
}

#endif

} // namespace rund::node::accel::detail::batch_download_internal
