#pragma once

#include "state.hpp"

#include "../../../backend/result.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelCheck PrepareVulkanPipelineTransfer(
    const std::shared_ptr<void> &prepared, const UploadRoute &upload,
    const DownloadRoute &download, std::uint64_t exact_storage_bytes) noexcept;
[[nodiscard]] BackendUpload
UploadPreparedVulkanPipeline(const std::shared_ptr<void> &prepared,
                             const void *data, std::uint64_t bytes) noexcept;
[[nodiscard]] BackendDownload
DownloadPreparedVulkanPipeline(const std::shared_ptr<void> &prepared,
                               void *data, std::uint64_t bytes,
                               std::uint64_t *payload_hash) noexcept;

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
void DestroyVulkanPipelineTransfer(VulkanPipeline &pipeline) noexcept;

#endif

} // namespace rund::node::accel::detail
