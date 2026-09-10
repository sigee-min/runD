#pragma once

#include "../../../../../backend/result.hpp"

#include "../local.hpp"

namespace rund::node::accel::detail::batch_download_internal {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

class HostReadback final {
public:
  explicit HostReadback(VulkanAdapter &adapter) noexcept;

  HostReadback(const HostReadback &) = delete;
  HostReadback &operator=(const HostReadback &) = delete;

  ~HostReadback();

private:
  VulkanAdapter &adapter_;
};

[[nodiscard]] BackendDownload
DownloadVulkanResidentBuffersInline(VulkanAdapter &adapter,
                                    std::span<const DownloadRoute> requests,
                                    std::unique_lock<std::mutex> &lock);

[[nodiscard]] BackendDownload
DownloadVulkanResidentBuffersLarge(VulkanAdapter &adapter,
                                   std::span<const DownloadRoute> requests,
                                   std::unique_lock<std::mutex> &lock);

void MarkDownloadChunkFailure(BackendDownload &result,
                              std::span<const DownloadRoute> requests,
                              std::span<const std::uint64_t> confirmed,
                              std::span<const DownloadPlan> chunk_plans,
                              const char *reason, bool submitted) noexcept;

void MarkDownloadStagingFailure(BackendDownload &result,
                                std::span<const DownloadRoute> requests,
                                std::span<const std::uint64_t> confirmed,
                                std::size_t failed,
                                const char *reason) noexcept;

void MarkDownloadMappedFailure(
    BackendDownload &result, std::span<const DownloadRoute> requests,
    std::span<const std::uint64_t> confirmed,
    std::span<const DownloadPlan> chunk_plans) noexcept;

void MarkDownloadWaitFailure(BackendDownload &result,
                             std::span<const DownloadRoute> requests,
                             std::size_t failed, std::uint64_t failed_confirmed,
                             bool use_outcome_confirmed) noexcept;

void FinishDownloadSuccess(BackendDownload &result,
                           std::span<const DownloadRoute> requests) noexcept;

#endif

} // namespace rund::node::accel::detail::batch_download_internal
