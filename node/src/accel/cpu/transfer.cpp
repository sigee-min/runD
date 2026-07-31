#include <accel/check.hpp>
#include <accel/device.hpp>

#include "buffer.hpp"
#include <rund/counter.hpp>

#include <cstring>

namespace rund::node::accel::detail {
namespace {

[[nodiscard]] rund::AccelCheck Reject(const char *const reason) noexcept {
  return rund::AccelCheck{false, reason};
}

} // namespace

rund::AccelCheck
UploadCpuResidentBuffer(const rund::AccelDevice &pick,
                        const rund::kernel::ResidentBufferRef &requested,
                        const std::shared_ptr<void> &handle,
                        const void *const data, const std::uint64_t bytes,
                        const std::uint64_t offset) {
  CpuBufferResult lookup = LookupCpuResidentBuffer(pick, requested, handle);
  CpuAdapter *const adapter = CpuAdapterFromPick(pick);
  if (!lookup.check.ok || adapter == nullptr) {
    return Reject("accel_buffer_backend_unavailable");
  }
  std::memcpy(lookup.buffer->data.data() + static_cast<std::size_t>(offset),
              data, static_cast<std::size_t>(bytes));
  std::lock_guard<std::mutex> lock{adapter->mutex};
  ::rund::detail::counter::Accumulate(adapter->host_to_device_bytes, bytes);
  return rund::AccelCheck{true, "ok"};
}

rund::AccelCheck
DownloadCpuResidentBuffer(const rund::AccelDevice &pick,
                          const rund::kernel::ResidentBufferRef &requested,
                          const std::shared_ptr<void> &handle, void *const data,
                          const std::uint64_t bytes,
                          const std::uint64_t offset) {
  CpuBufferResult lookup = LookupCpuResidentBuffer(pick, requested, handle);
  CpuAdapter *const adapter = CpuAdapterFromPick(pick);
  if (!lookup.check.ok || adapter == nullptr) {
    return Reject("accel_buffer_backend_unavailable");
  }
  std::memcpy(data,
              lookup.buffer->data.data() + static_cast<std::size_t>(offset),
              static_cast<std::size_t>(bytes));
  std::lock_guard<std::mutex> lock{adapter->mutex};
  ::rund::detail::counter::Accumulate(adapter->device_to_host_bytes, bytes);
  return rund::AccelCheck{true, "ok"};
}

} // namespace rund::node::accel::detail
