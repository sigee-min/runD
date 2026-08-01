#include "../backend/buffer.hpp"
#include "../backend/ops/table.hpp"
#include "../backend/usage.hpp"
#include "../kernel/backend/execute.hpp"

#include "buffer/owner.hpp"
#include "kernel.hpp"
#include "ops.hpp"
#include "resident.hpp"
#include "stats.hpp"

#include <node/accel/buffer.hpp>

#include <utility>

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelDevice PickMetal();

namespace {

[[nodiscard]] rund::AccelDevice Pick(const bool) { return PickMetal(); }

rund::Buffer Create(const rund::AccelDevice &pick, const rund::BufferDesc &desc,
                    const BackendBufferInitialization initialization) {
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
  return MakeBuffer(pick, desc, created.check, created.ref,
                    std::move(created.handle), created.ref.bytes, false);
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
                          const TransferCompletion completion) {
  return UploadMetalResidentBuffers(pick, requests, completion);
}

BackendDownload DownloadBatch(const rund::AccelDevice &pick,
                              const std::span<const DownloadRoute> requests) {
  return DownloadMetalResidentBuffers(pick, requests);
}

BackendCopy CopyBatch(const rund::AccelDevice &pick,
                      const std::span<const CopyRoute> requests) {
  return CopyMetalResidentBuffers(pick, requests);
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
  if (!stats.ok) {
    return rund::RuntimeStats{.reason = stats.reason};
  }
  return rund::RuntimeStats{
      .dispatch_count = stats.dispatch_count,
      .command_submit_count = stats.command_submit_count,
      .pipeline_compile_count = stats.pipeline_compile_count,
      .pipeline_cache_hit_count = stats.pipeline_cache_hit_count,
      .buffer_allocation_count = stats.buffer_allocation_count,
      .buffer_reuse_hit_count = stats.buffer_reuse_hit_count,
      .host_to_device_bytes = stats.host_to_device_bytes,
      .device_to_host_bytes = stats.device_to_host_bytes,
      .accel_kernel_ns = stats.accel_kernel_ns,
      .accel_timestamp_count = stats.accel_timestamp_count,
      .accel_timestamp_source = stats.accel_timestamp_source,
      .shader_compile_ns = stats.shader_compile_ns,
      .spirv_compile_ns = stats.spirv_compile_ns,
      .pipeline_create_ns = stats.pipeline_create_ns,
      .descriptor_setup_ns = stats.descriptor_setup_ns,
      .command_submit_wait_ns = stats.command_submit_wait_ns,
      .readback_ns = stats.readback_ns,
      .ok = true,
      .reason = "ok",
  };
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

bool InjectDeviceLostOnce(const rund::AccelDevice &pick) noexcept {
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr) {
    return false;
  }
  adapter->fault_device_lost_once.store(true, std::memory_order_relaxed);
  return true;
}

const BackendOps Operations{
    .api = rund::AccelApi::Metal,
    .resident = true,
    .nested_aggregate_command_count = 2u,
    .create = Create,
    .upload = Upload,
    .upload_batch = UploadBatch,
    .download = Download,
    .download_batch = DownloadBatch,
    .copy_batch = CopyBatch,
    .lookup = Lookup,
    .stats = Stats,
    .reset = ResetMetalRuntimeStats,
    .memory = Memory,
    .run = RunMetalKernel,
    .prepare = PrepareMetalKernel,
    .prepare_pipeline_private = PrepareMetalPipelinePrivateKernel,
    .traffic = MetalKernelTraffic,
    .run_batch = RunPreparedMetalBatch,
    .prepare_pipeline = PrepareMetalPipeline,
    .seed_prepared_pipeline_generation = SeedPreparedMetalPipelineGeneration,
    .submit_prepared_pipeline = SubmitPreparedMetalPipeline,
    .submit_prepared = SubmitPreparedMetalKernel,
    .inject_device_lost_once = InjectDeviceLostOnce,
};

} // namespace

BackendEntry MetalEntry() noexcept {
  return BackendEntry{rund::AccelApi::Metal, true, Pick, &Operations};
}

} // namespace rund::node::accel::detail
