#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../../../hash/fnv.hpp"
#include "../../../backend/result.hpp"
#include "../../../clock.hpp"
#include "../../resident/access.hpp"
#include "find.hpp"
#include <rund/counter.hpp>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <span>
#include <utility>
#include <vector>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

struct MetalDownloadPlan final {
  std::shared_ptr<void> owner;
  const std::byte *source = nullptr;
  void *data = nullptr;
  std::uint64_t bytes = 0u;
  std::uint64_t offset = 0u;
  std::uint64_t *payload_hash = nullptr;
};

BackendDownload DownloadMetalResidentBuffersWithScratch(
    MetalAdapter &adapter, const std::span<const DownloadRoute> requests,
    const std::span<MetalDownloadPlan> plans) {
  std::unique_lock adapter_lock{adapter.mutex};
  std::size_t plan_count = 0u;
  {
    MetalResidentState &resident = MetalResidents(adapter);
    std::lock_guard resident_lock{resident.mutex};
    for (const DownloadRoute &request : requests) {
      if (request.handle == nullptr || request.payload_hash == nullptr ||
          (request.bytes != 0u && request.data == nullptr)) {
        return {};
      }
      MetalResidentBufferResult resolved =
          ResolveMetalResidentBuffer(resident, request.resident, request.handle,
                                     "accel_buffer_unavailable");
      if (!resolved.check.ok || resolved.device_buffer == nullptr) {
        return BackendDownload{.check = {false, resolved.check.reason}};
      }
      if (request.offset > resolved.ref.bytes ||
          request.bytes > resolved.ref.bytes - request.offset) {
        return BackendDownload{
            .check = {false, "accel_buffer_download_overflow"}};
      }
      if (request.bytes == 0u) {
        *request.payload_hash = ::rund::node::hash_detail::kFnvOffset;
        continue;
      }
      if (plan_count >= plans.size()) {
        return BackendDownload{.check = {false, "accel_buffer_unavailable"}};
      }
      id<MTLBuffer> metal_buffer =
          (__bridge id<MTLBuffer>)resolved.device_buffer.get();
      const void *const contents = [metal_buffer contents];
      if (contents == nullptr) {
        return BackendDownload{.check = {false, "accel_buffer_unavailable"}};
      }
      plans[plan_count++] = MetalDownloadPlan{
          .owner = std::move(resolved.device_buffer),
          .source = static_cast<const std::byte *>(contents),
          .data = request.data,
          .bytes = request.bytes,
          .offset = request.offset,
          .payload_hash = request.payload_hash,
      };
    }
  }
  if (plan_count == 0u) {
    return BackendDownload{.check = {true, "ok"}, .payload_hash_valid = true};
  }
  if (adapter.active_host_readbacks ==
      std::numeric_limits<std::size_t>::max()) {
    return BackendDownload{.check = {false, "accel_buffer_unavailable"}};
  }
  const std::uint64_t readback_begin = MonotonicNanoseconds();
  ++adapter.active_host_readbacks;
  adapter_lock.unlock();
  std::uint64_t downloaded_bytes = 0u;
  for (const MetalDownloadPlan &plan : plans.first(plan_count)) {
    *plan.payload_hash = ::rund::node::hash_detail::CopyHash(
        plan.source + static_cast<std::size_t>(plan.offset), plan.data,
        static_cast<std::size_t>(plan.bytes));
    ::rund::detail::counter::Accumulate(downloaded_bytes, plan.bytes);
  }
  adapter_lock.lock();
  const std::uint64_t readback_elapsed =
      MonotonicNanoseconds() - readback_begin;
  ::rund::detail::counter::Accumulate(adapter.stats.readback_ns,
                                      readback_elapsed);
  ::rund::detail::counter::Accumulate(adapter.stats.device_to_host_bytes,
                                      downloaded_bytes);
  --adapter.active_host_readbacks;
  adapter.host_readback_cv.notify_all();
  return BackendDownload{.check = {true, "ok"}, .payload_hash_valid = true};
}

} // namespace

BackendDownload DownloadMetalResidentBuffer(
    const rund::AccelDevice &pick, const rund::kernel::ResidentBufferRef &ref,
    const std::shared_ptr<void> &handle, void *const data,
    const rund::kernel::u64 bytes, const rund::kernel::u64 offset,
    const bool hash_payload) {
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr || handle == nullptr ||
      (bytes != 0u && data == nullptr)) {
    return BackendDownload{.check = {false, "accel_buffer_unavailable"}};
  }
  std::unique_lock adapter_lock{adapter->mutex};
  MetalResidentBufferResult resolved{};
  {
    MetalResidentState &resident = MetalResidents(*adapter);
    std::lock_guard resident_lock{resident.mutex};
    resolved = ResolveMetalResidentBuffer(resident, ref, handle,
                                          "accel_buffer_unavailable");
  }
  if (!resolved.check.ok || resolved.device_buffer == nullptr) {
    return BackendDownload{.check = {false, resolved.check.reason}};
  }
  if (offset > resolved.ref.bytes || bytes > resolved.ref.bytes - offset) {
    return BackendDownload{.check = {false, "accel_buffer_download_overflow"}};
  }
  if (bytes == 0u) {
    return BackendDownload{
        .check = {true, "ok"},
        .payload_hash =
            hash_payload ? ::rund::node::hash_detail::kFnvOffset : 0u,
        .payload_hash_valid = hash_payload};
  }
  id<MTLBuffer> metal_buffer =
      (__bridge id<MTLBuffer>)resolved.device_buffer.get();
  const void *const contents = [metal_buffer contents];
  if (contents == nullptr) {
    return BackendDownload{.check = {false, "accel_buffer_unavailable"}};
  }
  const auto *const source = static_cast<const std::byte *>(contents);
  const std::uint64_t readback_begin = MonotonicNanoseconds();
  std::uint64_t payload_hash = 0u;
  if (hash_payload) {
    // Compute owns a read claim for the resolved resident buffer. Keep the
    // adapter lock available while the dependency-bound hash/copy loop runs;
    // raw internal transfers retain the adapter-wide serialization below.
    if (adapter->active_host_readbacks ==
        std::numeric_limits<std::size_t>::max()) {
      return BackendDownload{.check = {false, "accel_buffer_unavailable"}};
    }
    ++adapter->active_host_readbacks;
    adapter_lock.unlock();
    payload_hash = ::rund::node::hash_detail::CopyHash(
        source + static_cast<std::size_t>(offset), data,
        static_cast<std::size_t>(bytes));
    adapter_lock.lock();
  } else {
    std::memcpy(data, source + static_cast<std::size_t>(offset),
                static_cast<std::size_t>(bytes));
  }
  const std::uint64_t readback_elapsed =
      MonotonicNanoseconds() - readback_begin;
  ::rund::detail::counter::Accumulate(adapter->stats.readback_ns,
                                      readback_elapsed);
  ::rund::detail::counter::Accumulate(adapter->stats.device_to_host_bytes,
                                      bytes);
  if (hash_payload) {
    --adapter->active_host_readbacks;
    adapter->host_readback_cv.notify_all();
  }
  return BackendDownload{.check = {true, "ok"},
                         .payload_hash = payload_hash,
                         .payload_hash_valid = hash_payload};
}

BackendDownload
DownloadMetalResidentBuffers(const rund::AccelDevice &pick,
                             const std::span<const DownloadRoute> requests) {
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr || requests.empty()) {
    return {};
  }
  std::array<MetalDownloadPlan, kInlineTransferCapacity> inline_plans{};
  if (requests.size() <= inline_plans.size()) {
    return DownloadMetalResidentBuffersWithScratch(
        *adapter, requests,
        std::span<MetalDownloadPlan>{inline_plans}.first(requests.size()));
  }
  try {
    std::vector<MetalDownloadPlan> overflow_plans(requests.size());
    return DownloadMetalResidentBuffersWithScratch(*adapter, requests,
                                                   overflow_plans);
  } catch (const std::bad_alloc &) {
    return BackendDownload{.check = {false, "accel_buffer_unavailable"}};
  }
}
#else
BackendDownload
DownloadMetalResidentBuffer(const rund::AccelDevice &,
                            const rund::kernel::ResidentBufferRef &,
                            const std::shared_ptr<void> &, void *,
                            rund::kernel::u64, rund::kernel::u64, bool) {
  return BackendDownload{.check = {false, "accel_buffer_backend_unavailable"}};
}

BackendDownload
DownloadMetalResidentBuffers(const rund::AccelDevice &,
                             const std::span<const DownloadRoute>) {
  return {};
}
#endif

} // namespace rund::node::accel::detail
