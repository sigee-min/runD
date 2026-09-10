#include "../../../../adapter/error.hpp"
#include "../../../create.hpp"
#include "../../../../../backend/result.hpp"

#include "internal.hpp"

#include "../../../../../../hash/fnv.hpp"
#include "../../../../../clock.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <utility>

namespace rund::node::accel::detail::batch_download_internal {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

struct InlineDownloadRoute final {
  VulkanBuffer *resident = nullptr;
  std::shared_ptr<void> storage{};
  std::byte *data = nullptr;
  std::uint64_t bytes = 0u;
  std::uint64_t offset = 0u;
  std::uint64_t *payload_hash = nullptr;
  DownloadRangeOutcome *outcome = nullptr;
};

} // namespace

[[nodiscard]] BackendDownload DownloadVulkanResidentBuffersInline(
    VulkanAdapter &adapter, const std::span<const DownloadRoute> requests,
    std::unique_lock<std::mutex> &lock) {
  std::array<InlineDownloadRoute, kInlineTransferCapacity> routes{};
  bool has_payload = false;
  {
    VulkanResidentState &resident = VulkanResidents(adapter);
    std::lock_guard resident_lock{resident.mutex};
    for (std::size_t index = 0u; index < requests.size(); ++index) {
      const DownloadRoute &request = requests[index];
      ResetDownloadOutcome(request.outcome);
      if (request.bytes != 0u && request.data == nullptr) {
        MarkDownloadFailure(request.outcome, DownloadRangeState::FailedNoWrite);
        return BackendDownload{.check = {false, "accel_buffer_unavailable"}};
      }
      if (request.offset > request.resident.bytes ||
          request.bytes > request.resident.bytes - request.offset) {
        MarkDownloadFailure(request.outcome, DownloadRangeState::FailedNoWrite);
        return BackendDownload{
            .check = {false, "accel_buffer_download_overflow"}};
      }
      VulkanResidentBufferResult resolved = ResolveVulkanResidentBuffer(
          resident, request.resident, request.handle,
          "accel_buffer_unavailable");
      if (!resolved.check.ok || resolved.device_buffer == nullptr ||
          request.offset > resolved.ref.bytes ||
          request.bytes > resolved.ref.bytes - request.offset) {
        MarkDownloadFailure(request.outcome, DownloadRangeState::FailedNoWrite);
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
      routes[index] = InlineDownloadRoute{
          .resident = resolved.device_buffer,
          .storage = std::move(resolved.storage),
          .data = static_cast<std::byte *>(request.data),
          .bytes = request.bytes,
          .offset = request.offset,
          .payload_hash = request.payload_hash,
          .outcome = request.outcome,
      };
      has_payload = true;
    }
  }
  if (!has_payload) {
    return BackendDownload{.check = {true, "ok"}, .payload_hash_valid = true};
  }
  const VkDeviceSize staging_budget =
      transfer_budget(adapter.caps.staging_bytes);

  // Prove every slice before the first command. Streaming the bounded plan
  // rows must not turn a late range error into a partially materialized
  // download.
  for (const InlineDownloadRoute &route :
       std::span{routes}.first(requests.size())) {
    std::uint64_t consumed = 0u;
    while (consumed < route.bytes) {
      TransferSlice slice{};
      VulkanTransferRange range{};
      if (!next_slice(route.offset, route.bytes, consumed, staging_budget,
                      slice) ||
          !ResolveVulkanTransferRange(slice.offset, slice.bytes,
                                      route.resident->bytes, range) ||
          range.bytes > staging_budget ||
          slice.bytes > std::numeric_limits<std::size_t>::max() ||
          consumed > std::numeric_limits<std::size_t>::max() -
                         static_cast<std::size_t>(slice.bytes)) {
        return BackendDownload{.check = {false, "accel_buffer_unavailable"}};
      }
      consumed += slice.bytes;
    }
  }
  if (adapter.active_host_readbacks ==
      std::numeric_limits<std::size_t>::max()) {
    return BackendDownload{.check = {false, "accel_vulkan_transfer_invalid"}};
  }

  const std::uint64_t prior_sequence = latest_sequence(adapter);
  BackendDownload result{
      .check = {true, "ok"},
      .payload_hash_valid = true,
  };
  HostReadback readback{adapter};
  std::array<DownloadPlan, kInlineTransferCapacity> plans{};
  std::array<std::uint64_t, kInlineTransferCapacity> hashes{};
  hashes.fill(::rund::node::hash_detail::kFnvOffset);
  std::size_t plan_count = 0u;
  VkDeviceSize chunk_bytes = 0u;
  std::uint64_t downloaded_bytes = 0u;
  std::array<std::uint64_t, kInlineTransferCapacity> confirmed{};

  const auto flush = [&]() -> bool {
    if (plan_count == 0u) {
      return true;
    }
    WaitForVulkanCommandSlot(adapter, lock);
    bool staging_reused = false;
    VulkanBuffer staging_raw{};
    const std::span<DownloadPlan> chunk_plans =
        std::span{plans}.first(plan_count);
    if (!CreateVulkanBuffer(adapter, chunk_bytes,
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            staging_raw, &staging_reused)) {
      MarkDownloadStagingFailure(result, requests, confirmed,
                                 chunk_plans[0u].request,
                                 VulkanLastError(&adapter));
      return false;
    }
    ScopedBuffer staging{adapter, staging_raw, chunk_bytes};
    record_staging(result, chunk_bytes, staging_reused);
    const std::uint64_t readback_begin = MonotonicNanoseconds();
    bool submitted = false;
    if (!encode_download(adapter, chunk_plans, staging.buffer, chunk_bytes,
                         &submitted)) {
      MarkDownloadChunkFailure(result, requests, confirmed, chunk_plans,
                               VulkanLastError(&adapter), submitted);
      return false;
    }
    if (submitted) {
      ::rund::detail::counter::Accumulate(result.command_submits, 1u);
    }
    const auto *const source =
        static_cast<const std::byte *>(staging.buffer.mapped);
    if (source == nullptr) {
      MarkDownloadMappedFailure(result, requests, confirmed, chunk_plans);
      return false;
    }
    lock.unlock();
    for (const DownloadPlan &plan : chunk_plans) {
      const auto *const input = reinterpret_cast<const std::uint8_t *>(
          source + static_cast<std::size_t>(plan.staging_offset +
                                            plan.range.host_offset));
      auto *const output = static_cast<std::uint8_t *>(plan.data);
      const std::size_t bytes = static_cast<std::size_t>(plan.bytes);
      if (plan.hash) {
        std::uint64_t &hash = hashes[plan.request];
        for (std::size_t byte = 0u; byte < bytes; ++byte) {
          const std::uint8_t value = input[byte];
          output[byte] = value;
          hash = (hash ^ value) * ::rund::node::hash_detail::kFnvPrime;
        }
      } else {
        std::memcpy(output, input, bytes);
      }
      ::rund::detail::counter::Accumulate(downloaded_bytes, plan.bytes);
      ::rund::detail::counter::Accumulate(confirmed[plan.request], plan.bytes);
      if (confirmed[plan.request] == requests[plan.request].bytes) {
        const std::uint64_t hash = hashes[plan.request];
        MarkDownloadComplete(requests[plan.request].outcome,
                             confirmed[plan.request], hash,
                             requests[plan.request].payload_hash != nullptr);
      }
    }
    lock.lock();
    ::rund::detail::counter::Accumulate(
        adapter.readback_ns, MonotonicNanoseconds() - readback_begin);
    plan_count = 0u;
    chunk_bytes = 0u;
    return true;
  };

  for (std::size_t request_index = 0u; request_index < requests.size();
       ++request_index) {
    const InlineDownloadRoute &route = routes[request_index];
    std::uint64_t consumed = 0u;
    while (consumed < route.bytes) {
      TransferSlice slice{};
      VulkanTransferRange range{};
      if (!next_slice(route.offset, route.bytes, consumed, staging_budget,
                      slice) ||
          !ResolveVulkanTransferRange(slice.offset, slice.bytes,
                                      route.resident->bytes, range) ||
          range.bytes > staging_budget ||
          slice.bytes > std::numeric_limits<std::size_t>::max() ||
          consumed > std::numeric_limits<std::size_t>::max() -
                         static_cast<std::size_t>(slice.bytes)) {
        return BackendDownload{.check = {false, "accel_buffer_unavailable"}};
      }
      if (plan_count != 0u && range.bytes > staging_budget - chunk_bytes &&
          !flush()) {
        return result;
      }
      if (plan_count == plans.size() && !flush()) {
        return result;
      }
      plans[plan_count++] = DownloadPlan{
          .resident = route.resident,
          .data = route.data + static_cast<std::size_t>(consumed),
          .bytes = slice.bytes,
          .range = range,
          .staging_offset = chunk_bytes,
          .request = request_index,
          .hash = route.payload_hash != nullptr,
      };
      chunk_bytes += range.bytes;
      consumed += slice.bytes;
      if (chunk_bytes >= staging_budget && !flush()) {
        return result;
      }
    }
  }
  if (!flush()) {
    return result;
  }
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    if (requests[index].bytes != 0u &&
        requests[index].payload_hash != nullptr) {
      *requests[index].payload_hash = hashes[index];
    }
  }
  ::rund::detail::counter::Accumulate(adapter.device_to_host_bytes,
                                      downloaded_bytes);
  const rund::AccelCheck waited = wait_sequence(adapter, lock, prior_sequence);
  if (!waited.ok) {
    result.check = waited;
    result.payload_hash_valid = false;
    std::size_t failed = requests.size();
    for (std::size_t index = 0u; index < requests.size(); ++index) {
      if (requests[index].bytes != 0u) {
        failed = index;
        break;
      }
    }
    if (failed < requests.size()) {
      MarkDownloadWaitFailure(result, requests, failed, 0u, false);
    }
    return result;
  }
  FinishDownloadSuccess(result, requests);
  return result;
}

#endif

} // namespace rund::node::accel::detail::batch_download_internal
