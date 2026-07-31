#include "../backend/buffer.hpp"
#include "../backend/ops/table.hpp"
#include "../backend/usage.hpp"
#include "../kernel/backend/execute.hpp"

#include "adapter/api.hpp"
#include "buffer/create/telemetry.hpp"
#include "buffer/resident/model.hpp"
#include "kernel.hpp"
#include "ops.hpp"

#include <node/accel/buffer.hpp>

#include <algorithm>
#include <utility>

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelDevice PickVulkan();

namespace {

[[nodiscard]] rund::AccelDevice Pick(const bool) { return PickVulkan(); }

rund::Buffer Create(const rund::AccelDevice &pick, const rund::BufferDesc &desc,
                    const BackendBufferInitialization initialization) {
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
      pick, native, initialization == BackendBufferInitialization::Zeroed);
  return MakeBuffer(pick, desc, created.check, created.ref,
                    std::move(created.handle), created.storage_bytes,
                    created.storage_reused);
#else
  (void)pick;
  (void)desc;
  (void)initialization;
  return rund::Buffer{
      .check = rund::AccelCheck{false, "accel_buffer_backend_unavailable"}};
#endif
}

rund::AccelCheck Upload(const rund::AccelDevice &pick,
                        const rund::kernel::ResidentBufferRef &ref,
                        const std::shared_ptr<void> &handle, const void *data,
                        const std::uint64_t bytes, const std::uint64_t offset) {
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
                         const std::uint64_t bytes, const std::uint64_t offset,
                         const bool hash_payload) {
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
                          const std::span<const UploadRoute> requests) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  return UploadVulkanResidentBuffers(pick, requests);
#else
  (void)pick;
  (void)requests;
  return {};
#endif
}

BackendDownload DownloadBatch(const rund::AccelDevice &pick,
                              const std::span<const DownloadRoute> requests) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  return DownloadVulkanResidentBuffers(pick, requests);
#else
  (void)pick;
  (void)requests;
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

rund::RuntimeStats Stats(const rund::AccelDevice &pick) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  return ReadVulkanRuntimeStats(pick);
#else
  (void)pick;
  return rund::RuntimeStats{.reason = "accel_runtime_stats_unavailable"};
#endif
}

void Reset(const rund::AccelDevice &pick) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  ResetVulkanRuntimeStats(pick);
#else
  (void)pick;
#endif
}

rund::node::accel::AccelMemoryStats
Memory(const rund::AccelDevice &pick) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanAdapter *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return {};
  }
  std::lock_guard lock{adapter->mutex};
  const VulkanMemoryStats &memory = adapter->staging_memory;
  const std::uint64_t physical = VulkanPhysicalStaging(*adapter);
  return rund::node::accel::AccelMemoryStats{
      .staging =
          rund::node::accel::AccelMemoryCounter{.current = physical,
                                                .peak =
                                                    std::max(physical,
                                                             memory.peak),
                                                .cumulative = memory.cumulative,
                                                .reused = memory.reused}};
#else
  (void)pick;
  return {};
#endif
}

bool InjectDeviceLostOnce(const rund::AccelDevice &pick) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanAdapter *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return false;
  }
  adapter->fault_device_lost_once.store(true, std::memory_order_relaxed);
  return true;
#else
  (void)pick;
  return false;
#endif
}

const BackendOps Operations{
    .api = rund::AccelApi::Vulkan,
    .resident = true,
    .create = Create,
    .upload = Upload,
    .upload_batch = UploadBatch,
    .download = Download,
    .download_batch = DownloadBatch,
    .lookup = Lookup,
    .stats = Stats,
    .reset = Reset,
    .memory = Memory,
    .run = RunVulkanKernel,
    .prepare = PrepareVulkanKernel,
    .prepare_pipeline_private = PrepareVulkanPipelinePrivateKernel,
    .traffic = VulkanKernelTraffic,
    .run_batch = RunPreparedVulkanBatch,
    .prepare_pipeline = PrepareVulkanPipeline,
    .seed_prepared_pipeline_generation = SeedPreparedVulkanPipelineGeneration,
    .submit_prepared_pipeline = SubmitPreparedVulkanPipeline,
    .submit_prepared = SubmitPreparedVulkanKernel,
    .inject_device_lost_once = InjectDeviceLostOnce,
};

} // namespace

BackendEntry VulkanEntry() noexcept {
  return BackendEntry{rund::AccelApi::Vulkan, true, Pick, &Operations};
}

} // namespace rund::node::accel::detail
