#include "../../../../adapter/error.hpp"
#include "../../../create.hpp"
#include "../../../../../backend/result.hpp"

#include "internal.hpp"

#include "../../../../../../hash/fnv.hpp"
#include "../../../../../clock.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <vector>

namespace rund::node::accel::detail::batch_download_internal {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] BackendDownload DownloadVulkanResidentBuffersLarge(
    VulkanAdapter &adapter, const std::span<const DownloadRoute> requests,
    std::unique_lock<std::mutex> &lock) {
  try {
    const VkDeviceSize staging_budget =
        transfer_budget(adapter.caps.staging_bytes);
    std::vector<DownloadPlan> plans;
    plans.reserve(requests.size());
    {
      VulkanResidentState &resident = VulkanResidents(adapter);
      std::lock_guard resident_lock{resident.mutex};
      for (std::size_t request_index = 0u; request_index < requests.size();
           ++request_index) {
        const DownloadRoute &request = requests[request_index];
        ResetDownloadOutcome(request.outcome);
        if (request.bytes != 0u && request.data == nullptr) {
          MarkDownloadFailure(request.outcome,
                              DownloadRangeState::FailedNoWrite);
          return BackendDownload{.check = {false, "accel_buffer_unavailable"}};
        }
        VulkanResidentBufferResult resolved = ResolveVulkanResidentBuffer(
            resident, request.resident, request.handle,
            "accel_buffer_unavailable");
        if (!resolved.check.ok || resolved.device_buffer == nullptr ||
            request.offset > resolved.ref.bytes ||
            request.bytes > resolved.ref.bytes - request.offset) {
          MarkDownloadFailure(request.outcome,
                              DownloadRangeState::FailedNoWrite);
          return BackendDownload{
              .check = {false, resolved.check.ok
                                   ? "accel_buffer_download_overflow"
                                   : resolved.check.reason}};
        }
        if (request.bytes == 0u) {
          if (request.payload_hash != nullptr) {
            *request.payload_hash = ::rund::node::hash_detail::kFnvOffset;
          }
          MarkDownloadComplete(request.outcome, 0u,
                               ::rund::node::hash_detail::kFnvOffset,
                               request.payload_hash != nullptr);
          continue;
        }
        std::uint64_t consumed = 0u;
        while (consumed < request.bytes) {
          TransferSlice slice{};
          VulkanTransferRange range{};
          if (!next_slice(request.offset, request.bytes, consumed,
                          staging_budget, slice) ||
              !ResolveVulkanTransferRange(slice.offset, slice.bytes,
                                          resolved.device_buffer->bytes,
                                          range) ||
              range.bytes > staging_budget ||
              slice.bytes > std::numeric_limits<std::size_t>::max() ||
              consumed > std::numeric_limits<std::size_t>::max() -
                             static_cast<std::size_t>(slice.bytes)) {
            return BackendDownload{
                .check = {false, "accel_buffer_unavailable"}};
          }
          plans.push_back(DownloadPlan{
              .resident = resolved.device_buffer,
              .data = static_cast<std::byte *>(request.data) +
                      static_cast<std::size_t>(consumed),
              .bytes = slice.bytes,
              .range = range,
              .request = request_index,
              .hash = request.payload_hash != nullptr,
          });
          consumed += slice.bytes;
        }
      }
    }
    if (plans.empty()) {
      return BackendDownload{.check = {true, "ok"},
                             .ordered_prefix = requests.size(),
                             .payload_hash_valid = true};
    }
    const std::vector<BatchChunk> chunks = batch_chunks(plans, staging_budget);
    std::vector<::rund::node::hash_detail::Fnv> hashes(requests.size());
    const std::uint64_t prior_sequence = latest_sequence(adapter);
    BackendDownload result{
        .check = {true, "ok"},
        .payload_hash_valid = true,
    };
    if (adapter.active_host_readbacks ==
        std::numeric_limits<std::size_t>::max()) {
      return BackendDownload{.check = {false, "accel_vulkan_transfer_invalid"}};
    }
    HostReadback readback{adapter};
    std::uint64_t downloaded_bytes = 0u;
    std::vector<std::uint64_t> confirmed(requests.size());
    for (const BatchChunk &chunk : chunks) {
      WaitForVulkanCommandSlot(adapter, lock);
      bool staging_reused = false;
      VulkanBuffer staging_raw{};
      const std::span<DownloadPlan> chunk_plans{plans.data() + chunk.begin,
                                                chunk.end - chunk.begin};
      if (!CreateVulkanBuffer(adapter, chunk.bytes,
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              staging_raw, &staging_reused)) {
        MarkDownloadStagingFailure(result, requests, confirmed,
                                   plans[chunk.begin].request,
                                   VulkanLastError(&adapter));
        return result;
      }
      ScopedBuffer staging{adapter, staging_raw, chunk.bytes};
      record_staging(result, chunk.bytes, staging_reused);
      const std::uint64_t readback_begin = MonotonicNanoseconds();
      bool submitted = false;
      if (!encode_download(adapter, chunk_plans, staging.buffer, chunk.bytes,
                           &submitted)) {
        MarkDownloadChunkFailure(result, requests, confirmed, chunk_plans,
                                 VulkanLastError(&adapter), submitted);
        return result;
      }
      if (submitted) {
        ::rund::detail::counter::Accumulate(result.command_submits, 1u);
      }
      const auto *const source =
          static_cast<const std::byte *>(staging.buffer.mapped);
      if (source == nullptr) {
        MarkDownloadMappedFailure(result, requests, confirmed, chunk_plans);
        return result;
      }
      lock.unlock();
      for (const DownloadPlan &plan : chunk_plans) {
        const auto *const input = reinterpret_cast<const std::uint8_t *>(
            source + static_cast<std::size_t>(plan.staging_offset +
                                              plan.range.host_offset));
        auto *const output = static_cast<std::uint8_t *>(plan.data);
        const std::size_t bytes = static_cast<std::size_t>(plan.bytes);
        if (plan.hash) {
          ::rund::node::hash_detail::Fnv &hash = hashes[plan.request];
          for (std::size_t byte = 0u; byte < bytes; ++byte) {
            const std::uint8_t value = input[byte];
            output[byte] = value;
            hash.Byte(value);
          }
        } else {
          std::memcpy(output, input, bytes);
        }
        ::rund::detail::counter::Accumulate(downloaded_bytes, plan.bytes);
        ::rund::detail::counter::Accumulate(confirmed[plan.request],
                                            plan.bytes);
        if (confirmed[plan.request] == requests[plan.request].bytes) {
          const std::uint64_t hash = hashes[plan.request].Finish();
          MarkDownloadComplete(requests[plan.request].outcome,
                               confirmed[plan.request], hash,
                               requests[plan.request].payload_hash != nullptr);
        }
      }
      lock.lock();
      const std::uint64_t readback_elapsed =
          MonotonicNanoseconds() - readback_begin;
      ::rund::detail::counter::Accumulate(adapter.readback_ns,
                                          readback_elapsed);
    }
    for (std::size_t index = 0u; index < requests.size(); ++index) {
      if (requests[index].bytes != 0u &&
          requests[index].payload_hash != nullptr) {
        *requests[index].payload_hash = hashes[index].Finish();
      }
    }
    ::rund::detail::counter::Accumulate(adapter.device_to_host_bytes,
                                        downloaded_bytes);
    const rund::AccelCheck waited =
        wait_sequence(adapter, lock, prior_sequence);
    if (!waited.ok) {
      result.check = waited;
      result.payload_hash_valid = false;
      std::size_t failed = requests.size();
      for (const DownloadPlan &plan : plans) {
        failed = std::min(failed, plan.request);
      }
      if (failed < requests.size()) {
        MarkDownloadWaitFailure(result, requests, failed, confirmed[failed],
                                true);
      }
      return result;
    }
    FinishDownloadSuccess(result, requests);
    return result;
  } catch (const std::bad_alloc &) {
    return BackendDownload{.check = {false, "accel_buffer_unavailable"}};
  }
}

#endif

} // namespace rund::node::accel::detail::batch_download_internal
