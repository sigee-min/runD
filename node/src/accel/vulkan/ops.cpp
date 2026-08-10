#include "../backend/buffer.hpp"
#include "../backend/ops/table.hpp"
#include "../backend/usage.hpp"
#include "../kernel/backend/execute.hpp"

#include "adapter/api.hpp"
#include "buffer/create/telemetry.hpp"
#include "buffer/local.hpp"
#include "buffer/resident/model.hpp"
#include "kernel.hpp"
#include "kernel/manifest.hpp"
#include "kernel/pipeline/transfer.hpp"
#include "ops.hpp"
#include "range/api.hpp"

#include <node/accel/buffer.hpp>

#include <algorithm>
#include <utility>

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelDevice PickVulkan();

namespace {

[[nodiscard]] rund::AccelDevice Pick(const bool) { return PickVulkan(); }

rund::Buffer Create(const rund::AccelDevice &pick, const rund::BufferDesc &desc,
                    const BackendBufferInitialization initialization,
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
      exact_storage_bytes);
  return MakeBuffer(pick, desc, created.check, created.ref,
                    std::move(created.handle), created.storage_bytes,
                    created.storage_reused);
#else
  (void)pick;
  (void)desc;
  (void)initialization;
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
                      const std::span<const CopyRoute> requests) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  return CopyVulkanResidentBuffers(pick, requests);
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
  return rund::RuntimeStats{
      .outcome = {.reason = "accel_runtime_stats_unavailable"}};
#endif
}

void Reset(const rund::AccelDevice &pick) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  ResetVulkanRuntimeStats(pick);
#else
  (void)pick;
#endif
}

rund::AccelCheck
VirtualPipelineCapability(const rund::AccelDevice &pick) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const VulkanAdapter *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return rund::AccelCheck{false, "compute_adapter_unavailable"};
  }
  return adapter->portability_subset
             ? rund::AccelCheck{false, "compute_backend_unsupported"}
             : rund::AccelCheck{true, "ok"};
#else
  (void)pick;
  return rund::AccelCheck{false, "compute_backend_unsupported"};
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
      .staging = rund::node::accel::AccelMemoryCounter{
          .current = physical,
          .peak = std::max(physical, memory.peak),
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

bool InjectTraceUnavailableOnce(const rund::AccelDevice &pick) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  VulkanAdapter *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return false;
  }
  adapter->fault_trace_unavailable_once.store(true, std::memory_order_relaxed);
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
    .buffer_storage_bytes = BufferStorageBytes,
    .pipeline_transfer_storage_bytes = PipelineTransferStorageBytes,
    .upload = Upload,
    .upload_batch = UploadBatch,
    .download = Download,
    .download_batch = DownloadBatch,
    .copy_batch = CopyBatch,
    .lookup = Lookup,
    .stats = Stats,
    .reset = Reset,
    .memory = Memory,
    .virtual_pipeline_capability = VirtualPipelineCapability,
    .range_caps = VulkanRangeCaps,
    .run = RunVulkanKernel,
    .prepare = PrepareVulkanKernel,
    .plan_pipeline_private = PlanVulkanPipelinePrivateKernel,
    .plan_pipeline_program = PlanVulkanPipelineProgram,
    .plan_pipeline_recurrence = PlanVulkanPipelineRecurrence,
    .plan_pipeline_structure = PlanVulkanPipelineStructure,
    .build_step_manifest = BuildVulkanBackendManifest,
    .same_pipeline_program_template = SameVulkanPipelineProgramTemplate,
    .same_pipeline_template = SameVulkanPipelineTemplate,
    .observe_pipeline_template = ObserveVulkanPipelineTemplate,
    .prepare_pipeline_private = PrepareVulkanPipelinePrivateKernel,
    .traffic = VulkanKernelTraffic,
    .run_batch = RunPreparedVulkanBatch,
    .prepare_pipeline = PrepareVulkanPipeline,
    .seed_prepared_pipeline_generation = SeedPreparedVulkanPipelineGeneration,
    .prepare_pipeline_transfer = PrepareVulkanPipelineTransfer,
    .upload_prepared_pipeline = UploadPreparedVulkanPipeline,
    .download_prepared_pipeline = DownloadPreparedVulkanPipeline,
    .submit_prepared_pipeline = SubmitPreparedVulkanPipeline,
    .submit_prepared = SubmitPreparedVulkanKernel,
    .inject_device_lost_once = InjectDeviceLostOnce,
    .inject_trace_unavailable_once = InjectTraceUnavailableOnce,
};

} // namespace

BackendEntry VulkanEntry() noexcept {
  return BackendEntry{rund::AccelApi::Vulkan, true, Pick, &Operations};
}

} // namespace rund::node::accel::detail
