#include <accel/check.hpp>
#include <accel/device.hpp>

#include "batch/local.hpp"

#include <array>
#include <span>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
BackendDownload DownloadVulkanResidentBuffer(
    const rund::AccelDevice &pick, const rund::kernel::ResidentBufferRef &ref,
    const std::shared_ptr<void> &handle, void *const data,
    const rund::kernel::u64 bytes, const rund::kernel::u64 offset,
    const bool hash_payload) {
  std::uint64_t payload_hash = 0u;
  const std::array<DownloadRoute, 1u> request{{
      DownloadRoute{
          .resident = ref,
          .handle = handle,
          .data = data,
          .bytes = bytes,
          .offset = offset,
          .payload_hash = hash_payload ? &payload_hash : nullptr,
      },
  }};
  BackendDownload result =
      DownloadVulkanResidentBuffers(pick, std::span{request});
  result.payload_hash = hash_payload ? payload_hash : 0u;
  result.payload_hash_valid = result.check.ok && hash_payload;
  return result;
}
#endif

} // namespace rund::node::accel::detail
