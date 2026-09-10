#pragma once

#include "../../../backend/result.hpp"
#include "model.hpp"

#include <accel/device.hpp>

#include <memory>
#include <span>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] rund::AccelCheck UploadVulkanResidentBuffer(
    const rund::AccelDevice &pick, const rund::kernel::ResidentBufferRef &ref,
    const std::shared_ptr<void> &handle, const void *data,
    rund::kernel::u64 bytes, rund::kernel::u64 offset);

[[nodiscard]] BackendDownload DownloadVulkanResidentBuffer(
    const rund::AccelDevice &pick, const rund::kernel::ResidentBufferRef &ref,
    const std::shared_ptr<void> &handle, void *data, rund::kernel::u64 bytes,
    rund::kernel::u64 offset, bool hash_payload);

[[nodiscard]] BackendUpload UploadVulkanResidentBuffers(
    const rund::AccelDevice &pick, std::span<const UploadRoute> requests,
    TransferCompletion completion, TransferAuthority authority);

[[nodiscard]] BackendDownload DownloadVulkanResidentBuffers(
    const rund::AccelDevice &pick, std::span<const DownloadRoute> requests,
    TransferAuthority authority);

[[nodiscard]] BackendCopy CopyVulkanResidentBuffers(
    const rund::AccelDevice &pick, std::span<const CopyRoute> requests);

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)

} // namespace rund::node::accel::detail
