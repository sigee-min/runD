#include "local.hpp"

#include "../../../../../hash/fnv.hpp"
#include "../../../../clock.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <span>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

class HostReadback final {
public:
  explicit HostReadback(VulkanAdapter &adapter) noexcept : adapter_{adapter} {
    ++adapter_.active_host_readbacks;
  }

  HostReadback(const HostReadback &) = delete;
  HostReadback &operator=(const HostReadback &) = delete;

  ~HostReadback() {
    --adapter_.active_host_readbacks;
    adapter_.host_readback_cv.notify_all();
  }

private:
  VulkanAdapter &adapter_;
};

} // namespace

BackendDownload
DownloadVulkanResidentBuffers(const rund::AccelDevice &pick,
                              const std::span<const DownloadRoute> requests) {
  if (!VulkanPickOwnsAdapter(pick) || requests.empty()) {
    return {};
  }
  auto *const adapter = static_cast<VulkanAdapter *>(pick.backend.context);
  std::unique_lock<std::mutex> lock{adapter->mutex};
  try {
    const VkDeviceSize staging_budget =
        transfer_budget(adapter->caps.staging_bytes);
    std::vector<DownloadPlan> plans;
    plans.reserve(requests.size());
    {
      VulkanResidentState &resident = VulkanResidents(*adapter);
      std::lock_guard resident_lock{resident.mutex};
      for (std::size_t request_index = 0u; request_index < requests.size();
           ++request_index) {
        const DownloadRoute &request = requests[request_index];
        if (request.bytes != 0u && request.data == nullptr) {
          return BackendDownload{.check = {false, "accel_buffer_unavailable"}};
        }
        VulkanResidentBufferResult resolved = ResolveVulkanResidentBuffer(
            resident, request.resident, request.handle,
            "accel_buffer_unavailable");
        if (!resolved.check.ok || resolved.device_buffer == nullptr ||
            request.offset > resolved.ref.bytes ||
            request.bytes > resolved.ref.bytes - request.offset) {
          return BackendDownload{
              .check = {false, resolved.check.ok
                                   ? "accel_buffer_download_overflow"
                                   : resolved.check.reason}};
        }
        if (request.bytes == 0u) {
          if (request.payload_hash != nullptr) {
            *request.payload_hash = ::rund::node::hash_detail::kFnvOffset;
          }
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
      return BackendDownload{.check = {true, "ok"}, .payload_hash_valid = true};
    }
    const std::vector<BatchChunk> chunks = batch_chunks(plans, staging_budget);
    std::vector<::rund::node::hash_detail::Fnv> hashes(requests.size());
    const std::uint64_t prior_sequence = latest_sequence(*adapter);
    BackendDownload result{
        .check = {true, "ok"},
        .payload_hash_valid = true,
    };
    if (adapter->active_host_readbacks ==
        std::numeric_limits<std::size_t>::max()) {
      return BackendDownload{.check = {false, "accel_vulkan_transfer_invalid"}};
    }
    HostReadback readback{*adapter};
    std::uint64_t downloaded_bytes = 0u;
    for (const BatchChunk &chunk : chunks) {
      WaitForVulkanCommandSlot(*adapter, lock);
      bool staging_reused = false;
      VulkanBuffer staging_raw{};
      if (!CreateVulkanBuffer(*adapter, chunk.bytes,
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              staging_raw, &staging_reused)) {
        result.check = {false, VulkanLastError(adapter)};
        result.payload_hash_valid = false;
        return result;
      }
      ScopedBuffer staging{*adapter, staging_raw, chunk.bytes};
      record_staging(result, chunk.bytes, staging_reused);
      const std::span<DownloadPlan> chunk_plans{plans.data() + chunk.begin,
                                                chunk.end - chunk.begin};
      const std::uint64_t readback_begin = MonotonicNanoseconds();
      if (!encode_download(*adapter, chunk_plans, staging.buffer,
                           chunk.bytes)) {
        result.check = {false, VulkanLastError(adapter)};
        result.payload_hash_valid = false;
        return result;
      }
      ::rund::detail::counter::Accumulate(result.command_submits, 1u);
      const auto *const source =
          static_cast<const std::byte *>(staging.buffer.mapped);
      if (source == nullptr) {
        result.check = {false, "accel_vulkan_transfer_invalid"};
        result.payload_hash_valid = false;
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
      }
      lock.lock();
      const std::uint64_t readback_elapsed =
          MonotonicNanoseconds() - readback_begin;
      ::rund::detail::counter::Accumulate(adapter->readback_ns,
                                          readback_elapsed);
    }
    for (std::size_t index = 0u; index < requests.size(); ++index) {
      if (requests[index].bytes != 0u &&
          requests[index].payload_hash != nullptr) {
        *requests[index].payload_hash = hashes[index].Finish();
      }
    }
    ::rund::detail::counter::Accumulate(adapter->device_to_host_bytes,
                                        downloaded_bytes);
    wait_sequence(*adapter, lock, prior_sequence);
    result.staging_reused = result.staging_bytes != 0u &&
                            result.staging_reused_bytes == result.staging_bytes;
    return result;
  } catch (const std::bad_alloc &) {
    return BackendDownload{.check = {false, "accel_buffer_unavailable"}};
  }
}

#endif

} // namespace rund::node::accel::detail
