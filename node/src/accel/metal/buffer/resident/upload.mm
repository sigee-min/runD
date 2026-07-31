#include <accel/check.hpp>
#include <accel/device.hpp>

#include <rund/counter.hpp>
#include "../../../backend/result.hpp"
#include "../../resident/access.hpp"
#include "find.hpp"

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include <cstddef>
#include <cstring>
#include <mutex>
#include <new>
#include <span>
#include <utility>
#include <vector>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
rund::AccelCheck UploadMetalResidentBuffer(
    const rund::AccelDevice &pick, const rund::kernel::ResidentBufferRef &ref,
    const std::shared_ptr<void> &handle, const void *const data,
    const rund::kernel::u64 bytes, const rund::kernel::u64 offset) {
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr || handle == nullptr ||
      (bytes != 0u && data == nullptr)) {
    return rund::AccelCheck{false, "accel_buffer_unavailable"};
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
    return rund::AccelCheck{false, resolved.check.reason};
  }
  if (offset > resolved.ref.bytes || bytes > resolved.ref.bytes - offset) {
    return rund::AccelCheck{false, "accel_buffer_upload_overflow"};
  }
  if (bytes == 0u) {
    return rund::AccelCheck{true, "ok"};
  }
  id<MTLBuffer> metal_buffer =
      (__bridge id<MTLBuffer>)resolved.device_buffer.get();
  void *const contents = [metal_buffer contents];
  if (contents == nullptr) {
    return rund::AccelCheck{false, "accel_buffer_unavailable"};
  }
  auto *const target = static_cast<std::byte *>(contents);
  std::memcpy(target + static_cast<std::size_t>(offset), data,
              static_cast<std::size_t>(bytes));
  ::rund::detail::counter::Accumulate(adapter->stats.host_to_device_bytes,
                                      bytes);
  return rund::AccelCheck{true, "ok"};
}

BackendUpload UploadMetalResidentBuffers(
    const rund::AccelDevice &pick,
    const std::span<const UploadRoute> requests) {
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr || requests.empty()) {
    return {};
  }
  struct UploadPlan final {
    std::shared_ptr<void> owner;
    std::byte *target = nullptr;
    const void *data = nullptr;
    std::uint64_t bytes = 0u;
    std::uint64_t offset = 0u;
  };
  std::unique_lock adapter_lock{adapter->mutex};
  try {
    std::vector<UploadPlan> plans;
    plans.reserve(requests.size());
    {
      MetalResidentState &resident = MetalResidents(*adapter);
      std::lock_guard resident_lock{resident.mutex};
      for (const UploadRoute &request : requests) {
        if (request.handle == nullptr ||
            (request.bytes != 0u && request.data == nullptr)) {
          return BackendUpload{
              .check = {false, "accel_buffer_unavailable"}};
        }
        MetalResidentBufferResult resolved = ResolveMetalResidentBuffer(
            resident, request.resident, request.handle,
            "accel_buffer_unavailable");
        if (!resolved.check.ok || resolved.device_buffer == nullptr) {
          return BackendUpload{.check = resolved.check};
        }
        if (request.offset > resolved.ref.bytes ||
            request.bytes > resolved.ref.bytes - request.offset) {
          return BackendUpload{
              .check = {false, "accel_buffer_upload_overflow"}};
        }
        if (request.bytes == 0u) {
          continue;
        }
        id<MTLBuffer> metal_buffer =
            (__bridge id<MTLBuffer>)resolved.device_buffer.get();
        void *const contents = [metal_buffer contents];
        if (contents == nullptr) {
          return BackendUpload{
              .check = {false, "accel_buffer_unavailable"}};
        }
        plans.push_back(UploadPlan{
            .owner = std::move(resolved.device_buffer),
            .target = static_cast<std::byte *>(contents),
            .data = request.data,
            .bytes = request.bytes,
            .offset = request.offset,
        });
      }
    }
    std::uint64_t uploaded_bytes = 0u;
    for (const UploadPlan &plan : plans) {
      std::memcpy(plan.target + static_cast<std::size_t>(plan.offset),
                  plan.data, static_cast<std::size_t>(plan.bytes));
      ::rund::detail::counter::Accumulate(uploaded_bytes, plan.bytes);
    }
    ::rund::detail::counter::Accumulate(adapter->stats.host_to_device_bytes,
                                        uploaded_bytes);
    return BackendUpload{.check = {true, "ok"}};
  } catch (const std::bad_alloc &) {
    return BackendUpload{
        .check = {false, "accel_buffer_unavailable"}};
  }
}
#else
rund::AccelCheck
UploadMetalResidentBuffer(const rund::AccelDevice &,
                          const rund::kernel::ResidentBufferRef &,
                          const std::shared_ptr<void> &, const void *,
                          rund::kernel::u64, rund::kernel::u64) {
  return rund::AccelCheck{false, "accel_buffer_backend_unavailable"};
}

BackendUpload UploadMetalResidentBuffers(
    const rund::AccelDevice &,
    const std::span<const UploadRoute>) {
  return {};
}
#endif

} // namespace rund::node::accel::detail
