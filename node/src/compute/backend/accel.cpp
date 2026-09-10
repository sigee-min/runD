#include "../backend.hpp"
#include "accel/local.hpp"
#include "accel/pipeline.hpp"
#include "accel/program.hpp"
#include "accel/transfer.hpp"

#include "../job/state.hpp"
#include "../virtual/run/device_vsm.hpp"
#include "../virtual/run/device_vsm/operations.hpp"
#include "../virtual/run/sliding.hpp"

namespace rund::compute::detail {
namespace {

const DeviceOps Operations{
    .allocate = accel_backend::allocate_buffer,
    .project_buffer_view = accel_backend::project_buffer_view,
    .buffer_storage_bytes = accel_backend::buffer_storage_bytes,
    .pipeline_transfer_storage_bytes =
        accel_backend::pipeline_transfer_storage_bytes,
    .upload = accel_backend::upload,
    .upload_batch = accel_backend::upload_batch,
    .download = accel_backend::download,
    .download_batch = accel_backend::download_batch,
    .download_prefix_capacity = accel_backend::transfer_prefix_capacity,
    .copy_batch = accel_backend::copy_batch,
    .host_read = accel_backend::host_read,
    .host_write = accel_backend::host_write,
    .compile = accel_backend::compile,
    .program_ranges = accel_backend::program_ranges,
    .plan_scratch = accel_backend::plan_scratch,
    .resolve_buffer = accel_backend::resolve_buffer,
    .prepare_job = prepare_job_accel,
    .run_job = run_job_accel,
    .submit_job = submit_job_accel,
    .finish_job = finish_job_accel,
    .plan_pipeline_preparation = accel_backend::plan_pipeline_preparation,
    .prepare_pipeline = accel_backend::prepare_pipeline,
    .run_pipeline = accel_backend::run_pipeline,
    .submit_pipeline = accel_backend::submit_pipeline,
    .seed_pipeline_generation = accel_backend::seed_pipeline_generation,
    .upload_pipeline_transfer = accel_backend::upload_pipeline_transfer,
    .download_pipeline_transfer = accel_backend::download_pipeline_transfer,
    .device_staging = accel_backend::device_staging,
    .job_staging = accel_backend::job_staging,
    .pipeline_memory = accel_backend::pipeline_memory,
    .residency =
        {
            .submit_residency_pipeline =
                accel_backend::submit_residency_pipeline,
            .submit_residency_window = accel_backend::submit_residency_window,
            .submit_residency_stream_window =
                accel_backend::submit_residency_stream_window,
            .claim_residency_stream = accel_backend::claim_residency_stream,
            .release_residency_stream = accel_backend::release_residency_stream,
            .quarantine_residency_stream =
                accel_backend::quarantine_residency_stream,
            .residency_window_capability =
                accel_backend::residency_window_capability,
            .signal_residency_window = accel_backend::signal_residency_window,
            .abort_residency_window = accel_backend::abort_residency_window,
            .prepare_residency_schedule =
                accel_backend::prepare_residency_schedule,
            .submit_residency_schedule =
                accel_backend::submit_residency_schedule,
            .signal_residency_schedule =
                accel_backend::signal_residency_schedule,
            .abort_residency_schedule = accel_backend::abort_residency_schedule,
            .prepare_residency_sliding =
                accel_backend::prepare_residency_sliding,
            .submit_residency_sliding = accel_backend::submit_residency_sliding,
            .wake_residency_sliding = accel_backend::wake_residency_sliding,
            .prepare_pipeline_residency =
                accel_backend::prepare_pipeline_residency,
            .prepare_residency_selection =
                accel_backend::prepare_residency_selection,
            .prepare_residency_execution =
                accel_backend::prepare_residency_execution,
        },
    .virtual_execution =
        {
            .prepare_virtual_sliding_product =
                prepare_accel_virtual_execution_sliding_locked,
            .execute_virtual_sliding_product =
                execute_accel_virtual_execution_sliding,
            .prepare_virtual_device_vsm_product =
                prepare_accel_virtual_execution_device_vsm,
            .execute_virtual_device_vsm_product =
                execute_accel_virtual_execution_device_vsm,
            .run_service_free_direct_product =
                accel_backend::run_service_free_direct_product,
            .virtual_pipeline_capability =
                accel_backend::virtual_pipeline_capability,
            .admit_virtual_device_vsm =
                device_vsm_product_detail::admit_virtual_device_vsm,
            .vsm_async = &accel_virtual_vsm_async_ops(),
        },
};

} // namespace

const DeviceOps &AccelDeviceOps() noexcept { return Operations; }

} // namespace rund::compute::detail
