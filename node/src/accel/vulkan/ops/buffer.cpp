#include "../../backend/result.hpp"

#include "internal.hpp"

#include "../../backend/usage.hpp"
#include "../adapter/access.hpp"
#include "../buffer/resident/create.hpp"
#include "../buffer/create/telemetry.hpp"
#include "../buffer/local.hpp"
#include "../buffer/resident/model.hpp"
#include "../buffer/resident/lookup.hpp"
#include "../buffer/resident/transfer.hpp"
#include "../buffer/resident/view.hpp"

#include <node/accel/buffer.hpp>

#include <utility>

namespace rund::node::accel::detail::vulkan_ops_detail {

rund::Buffer Create(const rund::AccelDevice &pick,
                    const rund::BufferDesc &desc,
                    const BackendBufferInitialization initialization,
                    const BackendBufferMemory memory,
                    const std::uint64_t exact_storage_bytes) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const ResidentDesc native{
      .bytes = desc.bytes,
      .element_bytes = 1u,
      .stride_bytes = 1u,
      .count = desc.bytes,
      .usage = ResidentUsage(desc.usage),
      .read_capable = desc.usage != rund::BufferUsage::WriteOnly,
      .write_capable = desc.usage != rund::BufferUsage::ReadOnly,
  };
  VulkanResidentBufferResult created = CreateVulkanResidentBuffer(
      pick, native, initialization == BackendBufferInitialization::Zeroed,
      memory, exact_storage_bytes);
  return MakeBuffer(pick, desc, created.check, created.ref,
                    std::move(created.handle), created.storage_bytes,
                    created.storage_reused);
#else
  (void)pick;
  (void)desc;
  (void)initialization;
  (void)memory;
  (void)exact_storage_bytes;
  return rund::Buffer{
      .check = rund::AccelCheck{false, "accel_buffer_backend_unavailable"}};
#endif
}

std::uint64_t BufferStorageBytes(const rund::AccelDevice &pick,
                                 const std::uint64_t logical_bytes) noexcept {
  return VulkanBufferStorageBytes(pick, logical_bytes);
}

std::uint64_t
PipelineTransferStorageBytes(const rund::AccelDevice &pick,
                             const std::uint64_t logical_bytes) noexcept {
  return VulkanPipelineTransferStorageBytes(pick, logical_bytes);
}

rund::AccelCheck Upload(const rund::AccelDevice &pick,
                        const rund::kernel::ResidentBufferRef &ref,
                        const std::shared_ptr<void> &handle, const void *data,
                        const std::uint64_t bytes,
                        const std::uint64_t offset) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  return UploadVulkanResidentBuffer(pick, ref, handle, data, bytes, offset);
#else
  (void)pick;
  (void)ref;
  (void)handle;
  (void)data;
  (void)bytes;
  (void)offset;
  return rund::AccelCheck{false, "accel_buffer_backend_unavailable"};
#endif
}

BackendDownload Download(const rund::AccelDevice &pick,
                         const rund::kernel::ResidentBufferRef &ref,
                         const std::shared_ptr<void> &handle, void *data,
                         const std::uint64_t bytes,
                         const std::uint64_t offset, const bool hash_payload) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  return DownloadVulkanResidentBuffer(pick, ref, handle, data, bytes, offset,
                                      hash_payload);
#else
  (void)pick;
  (void)ref;
  (void)handle;
  (void)data;
  (void)bytes;
  (void)offset;
  (void)hash_payload;
  return {};
#endif
}

BackendUpload UploadBatch(const rund::AccelDevice &pick,
                          const std::span<const UploadRoute> requests,
                          const TransferCompletion completion,
                          const TransferAuthority authority) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  return UploadVulkanResidentBuffers(pick, requests, completion, authority);
#else
  (void)pick;
  (void)requests;
  (void)completion;
  (void)authority;
  return {};
#endif
}

BackendDownload DownloadBatch(const rund::AccelDevice &pick,
                              const std::span<const DownloadRoute> requests,
                              const TransferAuthority authority) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  return DownloadVulkanResidentBuffers(pick, requests, authority);
#else
  (void)pick;
  (void)requests;
  (void)authority;
  return {};
#endif
}

BackendCopy CopyBatch(const rund::AccelDevice &pick,
                      const std::span<const CopyRoute> requests,
                      const TransferAuthority authority) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  (void)authority;
  return CopyVulkanResidentBuffers(pick, requests);
#else
  (void)pick;
  (void)requests;
  (void)authority;
  return {};
#endif
}

BackendLookup Lookup(const rund::AccelDevice &pick,
                     const rund::kernel::ResidentBufferRef &requested,
                     const std::shared_ptr<void> &handle) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanResidentBufferResult result =
      LookupVulkanResidentBuffer(pick, requested, handle);
  return BackendLookup{.check = result.check,
                       .ref = result.ref,
                       .handle = std::move(result.handle)};
#else
  (void)pick;
  (void)requested;
  (void)handle;
  return BackendLookup{
      .check = rund::AccelCheck{false, "accel_context_buffer_invalid"}};
#endif
}

BackendHostView HostRead(const rund::AccelDevice &pick,
                         const rund::kernel::ResidentBufferRef &requested,
                         const std::shared_ptr<void> &handle) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanAdapter *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr || adapter->fault_host_read_once.exchange(
                                false, std::memory_order_relaxed)) {
    return {};
  }
  return ReadVulkanResidentBuffer(pick, requested, handle);
#else
  (void)pick;
  (void)requested;
  (void)handle;
  return {};
#endif
}

BackendHostWriteView
HostWrite(const rund::AccelDevice &pick,
          const rund::kernel::ResidentBufferRef &requested,
          const std::shared_ptr<void> &handle) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanAdapter *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr || adapter->fault_host_write_once.exchange(
                                false, std::memory_order_relaxed)) {
    return {};
  }
  return WriteVulkanResidentBuffer(pick, requested, handle);
#else
  (void)pick;
  (void)requested;
  (void)handle;
  return {};
#endif
}

} // namespace rund::node::accel::detail::vulkan_ops_detail
