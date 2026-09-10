#include "../backend/ops/table.hpp"
#include "../kernel/backend/execute.hpp"

#include "kernel.hpp"
#include "kernel/manifest.hpp"
#include "kernel/template/memory.hpp"
#include "ops.hpp"
#include "ops/internal.hpp"
#include "range/api.hpp"
#include "stats.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelDevice PickMetal();

namespace {

[[nodiscard]] rund::AccelDevice Pick(const bool) { return PickMetal(); }

const BackendOps Operations{
    .api = rund::AccelApi::Metal,
    .resident = true,
    .nested_aggregate_command_count = 2u,
    .create = metal_ops_detail::Create,
    .buffer_storage_bytes = metal_ops_detail::BufferStorageBytes,
    .upload = metal_ops_detail::Upload,
    .upload_batch = metal_ops_detail::UploadBatch,
    .download = metal_ops_detail::Download,
    .download_batch = metal_ops_detail::DownloadBatch,
    .copy_batch = metal_ops_detail::CopyBatch,
    .lookup = metal_ops_detail::Lookup,
    .host_read = metal_ops_detail::HostRead,
    .host_write = metal_ops_detail::HostWrite,
    .stats = metal_ops_detail::Stats,
    .reset = ResetMetalRuntimeStats,
    .memory = metal_ops_detail::Memory,
    .virtual_pipeline_capability = metal_ops_detail::VirtualPipelineCapability,
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
    .pipeline_residency_ready = MetalPipelineResidencyReady,
    .stage_pipeline_residency = StageMetalPipelineResidency,
    .commit_pipeline_residency = CommitMetalPipelineResidency,
    .submit_prepared_pipeline = SubmitPreparedMetalPipeline,
    .submit_prepared_window = SubmitMetalResidencyWindow,
    .residency_window_callbacks_async = true,
    .signal_prepared_window = SignalMetalResidencyWindow,
    .abort_prepared_window = AbortMetalResidencyWindow,
    .prepare_prepared_schedule = PrepareMetalResidencySchedule,
    .submit_prepared_schedule = SubmitMetalResidencySchedule,
    .signal_prepared_schedule = SignalMetalResidencySchedule,
    .abort_prepared_schedule = AbortMetalResidencySchedule,
    .prepared_sliding_capability = MetalResidencySlidingCapability,
    .submit_prepared_sliding = SubmitMetalResidencySliding,
    .query_persistent_sliding_capability =
        QueryMetalPreparedPersistentSlidingCapability,
    .prepare_persistent_sliding = PrepareMetalPersistentResidencySliding,
    .prepare_device_vsm = PrepareMetalDeviceVsm,
    .service_free_direct_capability =
        metal_ops_detail::ServiceFreeDirectCapabilityFor,
    .submit_prepared = SubmitPreparedMetalKernel,
    .inject_device_lost_once = metal_ops_detail::InjectDeviceLostOnce,
    .inject_download_failure_once = metal_ops_detail::InjectDownloadFailureOnce,
    .inject_host_read_unavailable_once =
        metal_ops_detail::InjectHostReadUnavailableOnce,
    .inject_host_write_unavailable_once =
        metal_ops_detail::InjectHostWriteUnavailableOnce,
    .inject_residency_terminal_loss_once =
        metal_ops_detail::InjectResidencyTerminalLossOnce,
    .inject_trace_unavailable_once =
        metal_ops_detail::InjectTraceUnavailableOnce,
    .inject_trace_resolve_device_lost_once =
        metal_ops_detail::InjectTraceResolveDeviceLostOnce,
};

} // namespace

BackendEntry MetalEntry() noexcept {
  return BackendEntry{rund::AccelApi::Metal, true, Pick, &Operations};
}

} // namespace rund::node::accel::detail
