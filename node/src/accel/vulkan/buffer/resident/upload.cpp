#include <accel/check.hpp>
#include <accel/device.hpp>

#include "batch/local.hpp"

#include <array>
#include <span>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
rund::AccelCheck UploadVulkanResidentBuffer(
    const rund::AccelDevice &pick, const rund::kernel::ResidentBufferRef &ref,
    const std::shared_ptr<void> &handle, const void *const data,
    const rund::kernel::u64 bytes, const rund::kernel::u64 offset) {
  const std::array<UploadRoute, 1u> request{{
      UploadRoute{
          .resident = ref,
          .handle = handle,
          .data = data,
          .bytes = bytes,
          .offset = offset,
      },
  }};
  return UploadVulkanResidentBuffers(pick, std::span{request},
                                     TransferCompletion::Queued)
      .check;
}
#endif

} // namespace rund::node::accel::detail
