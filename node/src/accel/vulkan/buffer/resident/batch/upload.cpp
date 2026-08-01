#include "local.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <utility>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

BackendUpload
UploadVulkanResidentBuffers(const rund::AccelDevice &pick,
                            const std::span<const UploadRoute> requests,
                            const TransferCompletion completion) {
  if (!VulkanPickOwnsAdapter(pick) || requests.empty()) {
    return {};
  }
  auto *const adapter = static_cast<VulkanAdapter *>(pick.backend.context);
  std::unique_lock<std::mutex> lock{adapter->mutex};
  try {
    const VkDeviceSize staging_budget =
        transfer_budget(adapter->caps.staging_bytes);
    std::vector<UploadPlan> plans;
    plans.reserve(requests.size());
    {
      VulkanResidentState &resident = VulkanResidents(*adapter);
      std::lock_guard resident_lock{resident.mutex};
      for (const UploadRoute &request : requests) {
        if (request.bytes != 0u && request.data == nullptr) {
          return BackendUpload{.check = {false, "accel_buffer_unavailable"}};
        }
        VulkanResidentBufferResult resolved = ResolveVulkanResidentBuffer(
            resident, request.resident, request.handle,
            "accel_buffer_unavailable");
        if (!resolved.check.ok || resolved.device_buffer == nullptr ||
            request.offset > resolved.ref.bytes ||
            request.bytes > resolved.ref.bytes - request.offset) {
          return BackendUpload{
              .check = {false, resolved.check.ok
                                   ? "accel_buffer_upload_overflow"
                                   : resolved.check.reason}};
        }
        if (request.bytes == 0u) {
          continue;
        }
        std::uint64_t consumed = 0u;
        while (consumed < request.bytes) {
          TransferSlice slice{};
          VulkanTransferRange range{};
          VulkanUploadPreservation preservation{};
          if (!next_slice(request.offset, request.bytes, consumed,
                          staging_budget, slice) ||
              !ResolveVulkanTransferRange(slice.offset, slice.bytes,
                                          resolved.device_buffer->bytes,
                                          range) ||
              range.bytes > staging_budget ||
              !ResolveVulkanUploadPreservation(range, slice.offset, slice.bytes,
                                               resolved.ref.bytes,
                                               preservation) ||
              slice.bytes > std::numeric_limits<std::size_t>::max() ||
              consumed > std::numeric_limits<std::size_t>::max() -
                             static_cast<std::size_t>(slice.bytes)) {
            return BackendUpload{.check = {false, "accel_buffer_unavailable"}};
          }
          plans.push_back(UploadPlan{
              .resident = resolved.device_buffer,
              .storage = resolved.storage,
              .data = static_cast<const std::byte *>(request.data) +
                      static_cast<std::size_t>(consumed),
              .bytes = slice.bytes,
              .range = range,
              .preservation = preservation,
          });
          consumed += slice.bytes;
        }
      }
    }
    if (plans.empty()) {
      return BackendUpload{.check = {true, "ok"}};
    }
    const bool overlapping = overlaps(plans);
    const std::vector<BatchChunk> chunks =
        batch_chunks(plans, staging_budget, overlapping);
    const bool asynchronous = completion == TransferCompletion::Queued &&
                              chunks.size() == 1u && !overlapping;
    BackendUpload result{.check = {true, "ok"}};
    std::uint64_t uploaded_bytes = 0u;
    for (const BatchChunk &chunk : chunks) {
      WaitForVulkanCommandSlot(*adapter, lock);
      VulkanBuffer staging_raw{};
      bool staging_reused = false;
      if (!CreateVulkanBuffer(*adapter, chunk.bytes,
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              staging_raw, &staging_reused)) {
        result.check = {false, VulkanLastError(adapter)};
        return result;
      }
      ScopedBuffer staging{*adapter, staging_raw, chunk.bytes};
      record_staging(result, chunk.bytes, staging_reused);
      const std::span<UploadPlan> chunk_plans{plans.data() + chunk.begin,
                                              chunk.end - chunk.begin};
      std::size_t preservation_regions = 0u;
      for (const UploadPlan &plan : chunk_plans) {
        preservation_regions += plan.preservation.region_count;
      }
      if (!encode_preserve(*adapter, chunk_plans, staging.buffer,
                           chunk.bytes)) {
        result.check = {false, VulkanLastError(adapter)};
        return result;
      }
      if (preservation_regions != 0u) {
        ::rund::detail::counter::Accumulate(result.command_submits, 1u);
      }
      auto *const target = static_cast<std::byte *>(staging.buffer.mapped);
      if (target == nullptr) {
        result.check = {false, "accel_vulkan_transfer_invalid"};
        return result;
      }
      for (const UploadPlan &plan : chunk_plans) {
        if (plan.preservation.padding_bytes != 0u) {
          std::memset(
              target +
                  static_cast<std::size_t>(plan.staging_offset +
                                           plan.preservation.padding_offset),
              0, static_cast<std::size_t>(plan.preservation.padding_bytes));
        }
        std::memcpy(target + static_cast<std::size_t>(plan.staging_offset +
                                                      plan.range.host_offset),
                    plan.data, static_cast<std::size_t>(plan.bytes));
        ::rund::detail::counter::Accumulate(uploaded_bytes, plan.bytes);
      }
      std::shared_ptr<void> retained_targets;
      if (asynchronous) {
        auto retained = std::make_shared<RetainedTargets>();
        retained->values.reserve(chunk_plans.size());
        for (const UploadPlan &plan : chunk_plans) {
          retained->values.push_back(plan.storage);
        }
        retained_targets = std::move(retained);
      }
      if (!encode_upload(*adapter, chunk_plans, staging, chunk.bytes,
                         asynchronous, std::move(retained_targets))) {
        result.check = {false, VulkanLastError(adapter)};
        return result;
      }
      ::rund::detail::counter::Accumulate(result.command_submits, 1u);
    }
    ::rund::detail::counter::Accumulate(adapter->host_to_device_bytes,
                                        uploaded_bytes);
    return result;
  } catch (const std::bad_alloc &) {
    return BackendUpload{.check = {false, "accel_buffer_unavailable"}};
  }
}

#endif

} // namespace rund::node::accel::detail
