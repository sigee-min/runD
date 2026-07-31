#pragma once

#include "../batch.hpp"

#include "../find.hpp"

#include <rund/counter.hpp>
#include "../../../command.hpp"
#include "../../../resident/access.hpp"
#include "../../../scope.hpp"
#include "../../transfer/range.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct DownloadPlan final {
  VulkanBuffer *resident = nullptr;
  void *data = nullptr;
  std::uint64_t bytes = 0u;
  VulkanTransferRange range{};
  VkDeviceSize staging_offset = 0u;
  std::size_t request = 0u;
  bool hash = false;
};

struct UploadPlan final {
  VulkanBuffer *resident = nullptr;
  std::shared_ptr<void> storage{};
  const void *data = nullptr;
  std::uint64_t bytes = 0u;
  VulkanTransferRange range{};
  VulkanUploadPreservation preservation{};
  VkDeviceSize staging_offset = 0u;
};

struct RetainedTargets final {
  std::vector<std::shared_ptr<void>> values;
};

struct BatchChunk final {
  std::size_t begin = 0u;
  std::size_t end = 0u;
  VkDeviceSize bytes = 0u;
};

struct TransferSlice final {
  std::uint64_t offset = 0u;
  std::uint64_t bytes = 0u;
};

[[nodiscard]] inline VkDeviceSize
transfer_budget(const VkDeviceSize requested) noexcept {
  return std::max<VkDeviceSize>(requested, kVulkanTransferAlignment) &
         ~(kVulkanTransferAlignment - 1u);
}

[[nodiscard]] inline bool
next_slice(const std::uint64_t offset, const std::uint64_t bytes,
           const std::uint64_t consumed, const VkDeviceSize budget,
           TransferSlice &slice) noexcept {
  if (budget < kVulkanTransferAlignment || consumed >= bytes ||
      offset > std::numeric_limits<std::uint64_t>::max() - bytes ||
      offset > std::numeric_limits<std::uint64_t>::max() - consumed) {
    return false;
  }
  const std::uint64_t start = offset + consumed;
  const std::uint64_t end = offset + bytes;
  const std::uint64_t base = start & ~(kVulkanTransferAlignment - 1u);
  if (base > std::numeric_limits<std::uint64_t>::max() - budget) {
    return false;
  }
  const std::uint64_t finish = std::min(end, base + budget);
  if (finish <= start) {
    return false;
  }
  slice = TransferSlice{.offset = start, .bytes = finish - start};
  return true;
}

template <typename Plan>
[[nodiscard]] std::vector<BatchChunk>
batch_chunks(std::vector<Plan> &plans, const VkDeviceSize staging_budget,
             const bool one_plan_per_chunk = false) {
  std::vector<BatchChunk> chunks;
  chunks.reserve(plans.size());
  std::size_t begin = 0u;
  while (begin < plans.size()) {
    VkDeviceSize bytes = 0u;
    std::size_t end = begin;
    while (end < plans.size()) {
      const VkDeviceSize plan_bytes = plans[end].range.bytes;
      if (end != begin &&
          (one_plan_per_chunk || plan_bytes > staging_budget - bytes)) {
        break;
      }
      plans[end].staging_offset = bytes;
      bytes += plan_bytes;
      ++end;
      if (one_plan_per_chunk || bytes >= staging_budget) {
        break;
      }
    }
    chunks.push_back(BatchChunk{.begin = begin, .end = end, .bytes = bytes});
    begin = end;
  }
  return chunks;
}

template <typename Transfer>
void record_staging(Transfer &result, const VkDeviceSize bytes,
                    const bool reused) noexcept {
  ::rund::detail::counter::Accumulate(result.staging_bytes, bytes);
  result.staging_peak_bytes = std::max(result.staging_peak_bytes, bytes);
  if (reused) {
    ::rund::detail::counter::Accumulate(result.staging_reused_bytes, bytes);
    ::rund::detail::counter::Accumulate(result.buffer_reuses, 1u);
  } else {
    ::rund::detail::counter::Accumulate(result.buffer_allocations, 1u);
  }
}

[[nodiscard]] std::uint64_t
latest_sequence(const VulkanAdapter &adapter) noexcept;
void wait_sequence(VulkanAdapter &adapter, std::unique_lock<std::mutex> &lock,
                   std::uint64_t sequence);
[[nodiscard]] bool overlaps(std::span<const UploadPlan> plans);
[[nodiscard]] bool encode_download(VulkanAdapter &adapter,
                                   std::span<const DownloadPlan> plans,
                                   VulkanBuffer &staging,
                                   VkDeviceSize staging_bytes);
[[nodiscard]] bool encode_preserve(VulkanAdapter &adapter,
                                   std::span<const UploadPlan> plans,
                                   VulkanBuffer &staging,
                                   VkDeviceSize staging_bytes);
[[nodiscard]] bool encode_upload(VulkanAdapter &adapter,
                                 std::span<const UploadPlan> plans,
                                 ScopedBuffer &staging,
                                 VkDeviceSize staging_bytes, bool asynchronous,
                                 std::shared_ptr<void> targets);

#endif

} // namespace rund::node::accel::detail
