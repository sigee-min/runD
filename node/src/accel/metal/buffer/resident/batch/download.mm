#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../../../../hash/fnv.hpp"
#include "../../../../backend/result.hpp"
#include "../../../../clock.hpp"
#include "../../../resident/access.hpp"
#include "../find.hpp"
#include <rund/counter.hpp>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
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
  DownloadRangeOutcome *outcome = nullptr;
};

BackendDownload DownloadMetalResidentBuffersWithScratch(
    MetalAdapter &adapter, const std::span<const DownloadRoute> requests,
    const std::span<MetalDownloadPlan> plans,
    const TransferAuthority authority) {
  std::unique_lock adapter_lock{adapter.mutex, std::defer_lock};
  if (authority == TransferAuthority::Shared) {
    adapter_lock.lock();
  }
  std::size_t plan_count = 0u;
  {
    MetalResidentState &resident = MetalResidents(adapter);
    std::unique_lock resident_lock{resident.mutex, std::defer_lock};
    if (authority == TransferAuthority::Shared) {
      resident_lock.lock();
    }
    for (const DownloadRoute &request : requests) {
      ResetDownloadOutcome(request.outcome);
      if (request.handle == nullptr || request.payload_hash == nullptr ||
          (request.bytes != 0u && request.data == nullptr)) {
        MarkDownloadFailure(request.outcome, DownloadRangeState::FailedNoWrite);
        return {};
      }
      MetalResidentBufferResult resolved =
          authority == TransferAuthority::PipelinePrivate
              ? ResolvePrivateMetalResidentBuffer(adapter, request.resident,
                                                  request.handle)
              : ResolveMetalResidentBuffer(resident, request.resident,
                                           request.handle,
                                           "accel_buffer_unavailable");
      if (!resolved.check.ok || resolved.device_buffer == nullptr) {
        MarkDownloadFailure(request.outcome, DownloadRangeState::FailedNoWrite);
        return BackendDownload{.check = {false, resolved.check.reason}};
      }
      if (request.offset > resolved.ref.bytes ||
          request.bytes > resolved.ref.bytes - request.offset) {
        MarkDownloadFailure(request.outcome, DownloadRangeState::FailedNoWrite);
        return BackendDownload{
            .check = {false, "accel_buffer_download_overflow"}};
      }
      if (request.bytes == 0u) {
        *request.payload_hash = ::rund::node::hash_detail::kFnvOffset;
        MarkDownloadComplete(request.outcome, 0u,
                             ::rund::node::hash_detail::kFnvOffset, true);
        continue;
      }
      if (plan_count >= plans.size()) {
        MarkDownloadFailure(request.outcome, DownloadRangeState::FailedNoWrite);
        return BackendDownload{.check = {false, "accel_buffer_unavailable"}};
      }
      id<MTLBuffer> metal_buffer =
          (__bridge id<MTLBuffer>)resolved.device_buffer.get();
      const void *const contents = [metal_buffer contents];
      if (contents == nullptr) {
        MarkDownloadFailure(request.outcome, DownloadRangeState::FailedNoWrite);
        return BackendDownload{.check = {false, "accel_buffer_unavailable"}};
      }
      plans[plan_count++] = MetalDownloadPlan{
          .owner = std::move(resolved.device_buffer),
          .source = static_cast<const std::byte *>(contents),
          .data = request.data,
          .bytes = request.bytes,
          .offset = request.offset,
          .payload_hash = request.payload_hash,
          .outcome = request.outcome,
      };
    }
  }
  if (plan_count == 0u) {
    return BackendDownload{.check = {true, "ok"},
                           .ordered_prefix = requests.size(),
                           .payload_hash_valid = true};
  }
  if (authority == TransferAuthority::PipelinePrivate &&
      adapter.fault_download_once.exchange(false, std::memory_order_relaxed)) {
    MarkDownloadFailure(requests[0].outcome, DownloadRangeState::FailedNoWrite);
    return BackendDownload{.check = {false, "accel_buffer_unavailable"},
                           .first_failed = 0u,
                           .first_failed_valid = true};
  }
  if (authority == TransferAuthority::Shared &&
      adapter.active_host_readbacks ==
          std::numeric_limits<std::size_t>::max()) {
    MarkDownloadFailure(requests[0].outcome, DownloadRangeState::FailedNoWrite);
    return BackendDownload{.check = {false, "accel_buffer_unavailable"},
                           .first_failed = 0u,
                           .first_failed_valid = true};
  }
  const std::uint64_t readback_begin = MonotonicNanoseconds();
  if (authority == TransferAuthority::Shared) {
    ++adapter.active_host_readbacks;
    adapter_lock.unlock();
  }
  std::uint64_t downloaded_bytes = 0u;
  for (const MetalDownloadPlan &plan : plans.first(plan_count)) {
    *plan.payload_hash = ::rund::node::hash_detail::CopyHash(
        plan.source + static_cast<std::size_t>(plan.offset), plan.data,
        static_cast<std::size_t>(plan.bytes));
    ::rund::detail::counter::Accumulate(downloaded_bytes, plan.bytes);
    MarkDownloadComplete(plan.outcome, plan.bytes, *plan.payload_hash, true);
  }
  const std::uint64_t readback_elapsed =
      MonotonicNanoseconds() - readback_begin;
  if (authority == TransferAuthority::Shared) {
    adapter_lock.lock();
    ::rund::detail::counter::Accumulate(
        adapter.stats.runtime.run.time.readback_ns, readback_elapsed);
    ::rund::detail::counter::Accumulate(
        adapter.stats.runtime.run.transfer.device_to_host_bytes,
        downloaded_bytes);
    --adapter.active_host_readbacks;
    adapter.host_readback_cv.notify_all();
  }
  return BackendDownload{.check = {true, "ok"},
                         .readback_ns = readback_elapsed,
                         .confirmed_bytes = downloaded_bytes,
                         .ordered_prefix = requests.size(),
                         .payload_hash_valid = true};
}

} // namespace

BackendDownload
DownloadMetalResidentBuffers(const rund::AccelDevice &pick,
                             const std::span<const DownloadRoute> requests,
                             const TransferAuthority authority) {
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr || requests.empty()) {
    return {};
  }
  std::array<MetalDownloadPlan, kInlineTransferCapacity> inline_plans{};
  if (requests.size() <= inline_plans.size()) {
    return DownloadMetalResidentBuffersWithScratch(
        *adapter, requests,
        std::span<MetalDownloadPlan>{inline_plans}.first(requests.size()),
        authority);
  }
  try {
    std::vector<MetalDownloadPlan> overflow_plans(requests.size());
    return DownloadMetalResidentBuffersWithScratch(*adapter, requests,
                                                   overflow_plans, authority);
  } catch (const std::bad_alloc &) {
    return BackendDownload{.check = {false, "accel_buffer_unavailable"}};
  }
}
#else
BackendDownload
DownloadMetalResidentBuffers(const rund::AccelDevice &,
                             const std::span<const DownloadRoute>,
                             const TransferAuthority) {
  return {};
}
#endif

} // namespace rund::node::accel::detail
