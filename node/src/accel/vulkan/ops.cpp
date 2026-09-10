#include "../backend/ops/table.hpp"
#include "../kernel/backend/execute.hpp"

#include "kernel.hpp"
#include "kernel/manifest.hpp"
#include "kernel/pipeline/residency/local.hpp"
#include "kernel/pipeline/transfer.hpp"
#include "ops.hpp"
#include "ops/internal.hpp"
#include "range/api.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelDevice PickVulkan();

namespace {

[[nodiscard]] rund::AccelDevice Pick(const bool) { return PickVulkan(); }

const BackendOps Operations{
    .api = rund::AccelApi::Vulkan,
    .resident = true,
    .create = vulkan_ops_detail::Create,
    .buffer_storage_bytes = vulkan_ops_detail::BufferStorageBytes,
    .pipeline_transfer_storage_bytes =
        vulkan_ops_detail::PipelineTransferStorageBytes,
    .upload = vulkan_ops_detail::Upload,
    .upload_batch = vulkan_ops_detail::UploadBatch,
    .download = vulkan_ops_detail::Download,
    .download_batch = vulkan_ops_detail::DownloadBatch,
    .copy_batch = vulkan_ops_detail::CopyBatch,
    .lookup = vulkan_ops_detail::Lookup,
    .host_read = vulkan_ops_detail::HostRead,
    .host_write = vulkan_ops_detail::HostWrite,
    .stats = vulkan_ops_detail::Stats,
    .reset = vulkan_ops_detail::Reset,
    .memory = vulkan_ops_detail::Memory,
    .virtual_pipeline_capability =
        vulkan_ops_detail::VirtualPipelineCapability,
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
    .query_pipeline_residency = QueryVulkanPipelineResidency,
    .pipeline_residency_ready = VulkanPipelineResidencyReady,
    .stage_pipeline_residency = StageVulkanPipelineResidency,
    .commit_pipeline_residency = CommitVulkanPipelineResidency,
    .upload_prepared_pipeline = UploadPreparedVulkanPipeline,
    .download_prepared_pipeline = DownloadPreparedVulkanPipeline,
    .submit_prepared_pipeline = SubmitPreparedVulkanPipeline,
    .submit_prepared_window = SubmitVulkanResidencyWindow,
    .residency_window_callbacks_async = true,
    .signal_prepared_window = SignalVulkanResidencyWindow,
    .abort_prepared_window = AbortVulkanResidencyWindow,
    .prepare_prepared_schedule = PrepareVulkanResidencySchedule,
    .submit_prepared_schedule = SubmitVulkanResidencySchedule,
    .signal_prepared_schedule = SignalVulkanResidencySchedule,
    .abort_prepared_schedule = AbortVulkanResidencySchedule,
    .prepared_sliding_capability = VulkanResidencySlidingCapability,
    .submit_prepared_sliding = SubmitVulkanResidencySliding,
    .query_persistent_sliding_capability =
        QueryVulkanPreparedPersistentSlidingCapability,
    .prepare_persistent_sliding = PrepareVulkanResidencyPersistent,
    .prepare_device_vsm = PrepareVulkanDeviceVsm,
    .service_free_direct_capability =
        vulkan_ops_detail::ServiceFreeDirectCapabilityFor,
    .submit_prepared = SubmitPreparedVulkanKernel,
    .inject_device_lost_once = vulkan_ops_detail::InjectDeviceLostOnce,
    .inject_host_read_unavailable_once =
        vulkan_ops_detail::InjectHostReadUnavailableOnce,
    .inject_host_write_unavailable_once =
        vulkan_ops_detail::InjectHostWriteUnavailableOnce,
    .inject_residency_terminal_loss_once =
        vulkan_ops_detail::InjectResidencyTerminalLossOnce,
    .inject_trace_unavailable_once =
        vulkan_ops_detail::InjectTraceUnavailableOnce,
};

} // namespace

BackendEntry VulkanEntry() noexcept {
  return BackendEntry{rund::AccelApi::Vulkan, true, Pick, &Operations};
}

} // namespace rund::node::accel::detail
