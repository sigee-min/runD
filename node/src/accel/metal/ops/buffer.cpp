#include "internal.hpp"

#include "../../backend/usage.hpp"
#include "../buffer/owner.hpp"
#include "../resident.hpp"

#include <node/accel/buffer.hpp>

#include <utility>

namespace rund::node::accel::detail::metal_ops_detail {

rund::Buffer Create(const rund::AccelDevice &pick, const rund::BufferDesc &desc,
                    const BackendBufferInitialization initialization,
                    const BackendBufferMemory,
                    const std::uint64_t exact_storage_bytes) {
  const ResidentDesc native{
      .bytes = desc.bytes,
      .element_bytes = 1u,
      .stride_bytes = 1u,
      .count = desc.bytes,
      .usage = ResidentUsage(desc.usage),
      .read_capable = desc.usage != rund::BufferUsage::WriteOnly,
      .write_capable = desc.usage != rund::BufferUsage::ReadOnly,
  };
  MetalResidentBufferResult created = CreateMetalResidentBuffer(
      pick, native, initialization == BackendBufferInitialization::Zeroed);
  if (created.check.ok && exact_storage_bytes != 0u &&
      created.storage_bytes != exact_storage_bytes) {
    return rund::Buffer{
        .check = rund::AccelCheck{false, "accel_metal_buffer_unavailable"}};
  }
  return MakeBuffer(pick, desc, created.check, created.ref,
                    std::move(created.handle), created.storage_bytes,
                    created.storage_reused);
}

std::uint64_t BufferStorageBytes(const rund::AccelDevice &pick,
                                 const std::uint64_t logical_bytes) noexcept {
  return MetalBufferStorageBytes(pick, logical_bytes);
}

rund::AccelCheck Upload(const rund::AccelDevice &pick,
                        const rund::kernel::ResidentBufferRef &ref,
                        const std::shared_ptr<void> &handle, const void *data,
                        const std::uint64_t bytes, const std::uint64_t offset) {
  return UploadMetalResidentBuffer(pick, ref, handle, data, bytes, offset);
}

BackendDownload Download(const rund::AccelDevice &pick,
                         const rund::kernel::ResidentBufferRef &ref,
                         const std::shared_ptr<void> &handle, void *data,
                         const std::uint64_t bytes, const std::uint64_t offset,
                         const bool hash_payload) {
  return DownloadMetalResidentBuffer(pick, ref, handle, data, bytes, offset,
                                     hash_payload);
}

BackendUpload UploadBatch(const rund::AccelDevice &pick,
                          const std::span<const UploadRoute> requests,
                          const TransferCompletion completion,
                          const TransferAuthority authority) {
  return UploadMetalResidentBuffers(pick, requests, completion, authority);
}

BackendDownload DownloadBatch(const rund::AccelDevice &pick,
                              const std::span<const DownloadRoute> requests,
                              const TransferAuthority authority) {
  return DownloadMetalResidentBuffers(pick, requests, authority);
}

BackendCopy CopyBatch(const rund::AccelDevice &pick,
                      const std::span<const CopyRoute> requests,
                      const TransferAuthority authority) {
  return CopyMetalResidentBuffers(pick, requests, authority);
}

BackendLookup Lookup(const rund::AccelDevice &pick,
                     const rund::kernel::ResidentBufferRef &requested,
                     const std::shared_ptr<void> &handle) {
  MetalResidentBufferResult result =
      LookupMetalResidentBuffer(pick, requested, handle);
  return BackendLookup{.check = result.check,
                       .ref = result.ref,
                       .handle = std::move(result.handle)};
}

BackendHostView HostRead(const rund::AccelDevice &pick,
                         const rund::kernel::ResidentBufferRef &requested,
                         const std::shared_ptr<void> &handle) noexcept {
  const MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  // Keep the injected download fault on the physical fallback path. This
  // makes the direct-view fast path fail closed to the ordinary copy route,
  // where the existing fault producer is consumed and cleanup is exercised.
  if (adapter == nullptr) {
    return {};
  }
  if (adapter->fault_host_read_once.exchange(false,
                                             std::memory_order_relaxed) ||
      adapter->fault_download_once.load(std::memory_order_relaxed)) {
    return {};
  }
  return ReadMetalResidentBuffer(pick, requested, handle);
}

BackendHostWriteView HostWrite(const rund::AccelDevice &pick,
                               const rund::kernel::ResidentBufferRef &requested,
                               const std::shared_ptr<void> &handle) noexcept {
  const MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr || adapter->fault_host_write_once.exchange(
                                false, std::memory_order_relaxed)) {
    return {};
  }
  return WriteMetalResidentBuffer(pick, requested, handle);
}

} // namespace rund::node::accel::detail::metal_ops_detail
