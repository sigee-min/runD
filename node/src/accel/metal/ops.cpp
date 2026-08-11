#include "../backend/buffer.hpp"
#include "../backend/ops/table.hpp"
#include "../backend/usage.hpp"
#include "../kernel/backend/execute.hpp"

#include "buffer/owner.hpp"
#include "kernel.hpp"
#include "kernel/manifest.hpp"
#include "kernel/template_memory.hpp"
#include "ops.hpp"
#include "range/api.hpp"
#include "resident.hpp"
#include "stats.hpp"

#include <node/accel/buffer.hpp>

#include <utility>

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelDevice PickMetal();

namespace {

[[nodiscard]] rund::AccelDevice Pick(const bool) { return PickMetal(); }

rund::Buffer Create(const rund::AccelDevice &pick, const rund::BufferDesc &desc,
                    const BackendBufferInitialization initialization,
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

rund::RuntimeStats Stats(const rund::AccelDevice &pick) {
  const MetalRuntimeStats stats = ReadMetalRuntimeStats(pick);
  return stats.runtime;
}

rund::node::accel::AccelMemoryStats
Memory(const rund::AccelDevice &pick) noexcept {
  const MetalMemoryStats memory = ReadMetalMemoryStats(pick);
  return rund::node::accel::AccelMemoryStats{
      .staging =
          rund::node::accel::AccelMemoryCounter{.current = memory.current,
                                                .peak = memory.peak,
                                                .cumulative = memory.cumulative,
                                                .reused = memory.reused}};
}

rund::AccelCheck
VirtualPipelineCapability(const rund::AccelDevice &pick) noexcept {
  return MetalPickOwnsAdapter(pick)
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, "compute_adapter_unavailable"};
}

bool InjectDeviceLostOnce(const rund::AccelDevice &pick) noexcept {
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr) {
    return false;
  }
  adapter->fault_device_lost_once.store(true, std::memory_order_relaxed);
  return true;
}

bool InjectTraceUnavailableOnce(const rund::AccelDevice &pick) noexcept {
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr) {
    return false;
  }
  adapter->fault_trace_unavailable_once.store(true, std::memory_order_relaxed);
  return true;
}

bool InjectTraceResolveDeviceLostOnce(const rund::AccelDevice &pick) noexcept {
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr) {
    return false;
  }
  adapter->fault_trace_resolve_device_lost_once.store(
      true, std::memory_order_relaxed);
  return true;
}

const BackendOps Operations{
    .api = rund::AccelApi::Metal,
    .resident = true,
    .nested_aggregate_command_count = 2u,
    .create = Create,
    .buffer_storage_bytes = BufferStorageBytes,
    .upload = Upload,
    .upload_batch = UploadBatch,
    .download = Download,
    .download_batch = DownloadBatch,
    .copy_batch = CopyBatch,
    .lookup = Lookup,
    .stats = Stats,
    .reset = ResetMetalRuntimeStats,
    .memory = Memory,
    .virtual_pipeline_capability = VirtualPipelineCapability,
    .range_caps = MetalRangeCaps,
    .run = RunMetalKernel,
    .prepare = PrepareMetalKernel,
    .plan_pipeline_private = PlanMetalPipelinePrivateKernel,
    .plan_pipeline_program = PlanMetalPipelineProgram,
    .plan_pipeline_recurrence = PlanMetalPipelineRecurrence,
    .plan_pipeline_structure = PlanMetalPipelineStructure,
    .build_step_manifest = BuildMetalBackendManifest,
    .same_pipeline_program_template = SameMetalPipelineProgramTemplate,
    .same_pipeline_template = SameMetalPipelineTemplate,
    .observe_pipeline_template = ObserveMetalPipelineTemplate,
    .prepare_pipeline_private = PrepareMetalPipelinePrivateKernel,
    .traffic = MetalKernelTraffic,
    .run_batch = RunPreparedMetalBatch,
    .prepare_pipeline = PrepareMetalPipeline,
    .seed_prepared_pipeline_generation = SeedPreparedMetalPipelineGeneration,
    .query_pipeline_residency = QueryMetalPipelineResidency,
    .stage_pipeline_residency = StageMetalPipelineResidency,
    .commit_pipeline_residency = CommitMetalPipelineResidency,
    .submit_prepared_pipeline = SubmitPreparedMetalPipeline,
    .submit_prepared = SubmitPreparedMetalKernel,
    .inject_device_lost_once = InjectDeviceLostOnce,
    .inject_trace_unavailable_once = InjectTraceUnavailableOnce,
    .inject_trace_resolve_device_lost_once = InjectTraceResolveDeviceLostOnce,
};

} // namespace

BackendEntry MetalEntry() noexcept {
  return BackendEntry{rund::AccelApi::Metal, true, Pick, &Operations};
}

} // namespace rund::node::accel::detail
