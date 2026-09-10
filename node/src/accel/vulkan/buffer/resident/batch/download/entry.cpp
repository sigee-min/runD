#include "../../../../adapter/access.hpp"
#include "../../../../../backend/result.hpp"

#include "internal.hpp"
#include "../../transfer.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

BackendDownload
DownloadVulkanResidentBuffers(const rund::AccelDevice &pick,
                              const std::span<const DownloadRoute> requests,
                              const TransferAuthority authority) {
  (void)authority;
  if (!VulkanPickOwnsAdapter(pick) || requests.empty()) {
    return {};
  }
  auto *const adapter = static_cast<VulkanAdapter *>(pick.backend.context);
  std::unique_lock<std::mutex> lock{adapter->mutex};
  if (requests.size() <= kInlineTransferCapacity) {
    return batch_download_internal::DownloadVulkanResidentBuffersInline(
        *adapter, requests, lock);
  }
  return batch_download_internal::DownloadVulkanResidentBuffersLarge(
      *adapter, requests, lock);
}

#endif

} // namespace rund::node::accel::detail
