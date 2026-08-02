#include "../backend.hpp"

#include "../../accel/context/internal/support.hpp"
#include "../../accel/context/local.hpp"
#include "../../accel/context/transfer.hpp"
#include "../../accel/graph/token.hpp"
#include "../job/state.hpp"
#include "../pipeline/state.hpp"
#include "../program/state.hpp"
#include "../stats.hpp"
#include "../status.hpp"

#include <accel/context/buffer.hpp>
#include <accel/context/buffer/descriptor.hpp>
#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>
#include <rund/counter.hpp>

#include <algorithm>
#include <array>
#include <utility>

namespace rund::compute::detail {
namespace {

constexpr std::size_t TransferCapacity = PipelineTransferCapacity;

Status allocate(DeviceState &device, BufferState &buffer,
                const std::size_t scalar_bytes, const std::size_t count,
                const bool zero_initialize) {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return Status::fail(Reason::DeviceInvalid);
  }
  rund::AccelBuffer created =
      node::accel::detail::CreateAccelBufferWithInitialization(
          accel->context,
          rund::AccelBufferDesc{
              .scalar_width_bytes = scalar_bytes,
              .count = count,
              .usage = rund::BufferUsage::ReadWrite,
          },
          zero_initialize
              ? node::accel::detail::BackendBufferInitialization::Zeroed
              : node::accel::detail::BackendBufferInitialization::
                    FullOverwrite);
  if (!created.check.ok) {
    return Status::fail(
        project_reason(created.check.reason, Reason::BufferCapacity));
  }
  buffer.storage.emplace<AccelBufferState>(
      AccelBufferState{.buffer = std::move(created)});
  const AccelBufferState *const stored = accel_buffer(buffer);
  buffer.physical_bytes =
      stored == nullptr || stored->buffer.buffer.storage_bytes == 0u
          ? buffer.bytes
          : stored->buffer.buffer.storage_bytes;
  return Status::success();
}

Status upload(DeviceState &device, BufferState &buffer, const void *const data,
              const std::size_t bytes) {
  const AccelDeviceState *const accel = accel_device(device);
  AccelBufferState *const target = accel_buffer(buffer);
  if (accel == nullptr || target == nullptr) {
    return Status::fail(Reason::TransferInvalid);
  }
  const rund::AccelCheck check = node::accel::UploadAccelBuffer(
      accel->context, target->buffer, data, bytes);
  return check.ok ? Status::success()
                  : Status::fail(
                        project_reason(check.reason, Reason::TransferInvalid));
}

Status resolve_buffer(const DeviceState &device, const BufferState &buffer,
                      std::shared_ptr<void> &handle) {
  const AccelDeviceState *const accel = accel_device(device);
  const AccelBufferState *const storage = accel_buffer(buffer);
  if (accel == nullptr || storage == nullptr) {
    handle.reset();
    return Status::fail(Reason::BindingDeviceMismatch);
  }
  const node::accel::detail::ContextAdmission admission =
      node::accel::detail::AdmitContextForSupport(accel->context);
  const rund::AccelCheck check =
      admission.check.ok ? node::accel::detail::ValidateAccelBufferForSupport(
                               admission, storage->buffer, handle)
                         : admission.check;
  if (!check.ok) {
    handle.reset();
    return Status::fail(Reason::BindingDeviceMismatch);
  }
  return Status::success();
}

UploadResult
upload_batch(DeviceState &device, const std::span<const UploadRequest> requests,
             const node::accel::detail::TransferCompletion completion) {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr || requests.empty()) {
    return UploadResult{.status = Status::fail(Reason::TransferInvalid)};
  }
  if (requests.size() > TransferCapacity) {
    return UploadResult{.status = Status::fail(Reason::PipelineCapacity)};
  }
  std::array<node::accel::detail::UploadEntry, TransferCapacity> transfers{};
  std::array<node::accel::detail::UploadRoute, TransferCapacity> routes{};
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    const UploadRequest request = requests[index];
    AccelBufferState *const target =
        request.buffer == nullptr ? nullptr : accel_buffer(*request.buffer);
    if (target == nullptr || (request.bytes != 0u && request.data == nullptr)) {
      return UploadResult{.status = Status::fail(Reason::TransferInvalid)};
    }
    transfers[index] = node::accel::detail::UploadEntry{
        .buffer = &target->buffer,
        .data = request.data,
        .bytes = request.bytes,
    };
  }
  const node::accel::detail::AccelTransfer transfer =
      node::accel::detail::UploadAccelBuffers(
          accel->context,
          std::span<const node::accel::detail::UploadEntry>{transfers.data(),
                                                            requests.size()},
          std::span<node::accel::detail::UploadRoute>{routes.data(),
                                                      requests.size()},
          completion);
  return UploadResult{
      .status = transfer.check.ok
                    ? Status::success()
                    : Status::fail(project_reason(transfer.check.reason,
                                                  Reason::TransferInvalid)),
      .staging_bytes = transfer.staging_bytes,
      .staging_peak_bytes = transfer.staging_peak_bytes,
      .staging_reused_bytes = transfer.staging_reused_bytes,
      .staging_budget = transfer.staging_budget,
      .buffer_allocations = transfer.buffer_allocations,
      .buffer_reuses = transfer.buffer_reuses,
      .command_submits = transfer.command_submits,
  };
}

DownloadResult download(DeviceState &device, const BufferState &buffer,
                        void *const data, const std::size_t bytes) {
  const AccelDeviceState *const accel = accel_device(device);
  const AccelBufferState *const resident = accel_buffer(buffer);
  if (accel == nullptr || resident == nullptr) {
    return DownloadResult{.status = Status::fail(Reason::TransferInvalid)};
  }
  const node::accel::detail::AccelTransfer transfer =
      node::accel::detail::DownloadAccelBufferMeasured(
          accel->context, resident->buffer, data, bytes, 0u, true);
  return DownloadResult{
      .status = transfer.check.ok
                    ? Status::success()
                    : Status::fail(project_reason(transfer.check.reason,
                                                  Reason::TransferInvalid)),
      .payload_hash = transfer.payload_hash,
      .staging_bytes = transfer.staging_bytes,
      .staging_peak_bytes = transfer.staging_peak_bytes,
      .staging_reused_bytes = transfer.staging_reused_bytes,
      .staging_budget = transfer.staging_budget,
      .buffer_allocations = transfer.buffer_allocations,
      .buffer_reuses = transfer.buffer_reuses,
      .command_submits = transfer.command_submits,
      .readback_ns = transfer.readback_ns,
      .staging_reused = transfer.staging_reused,
      .payload_hash_valid = transfer.check.ok && transfer.payload_hash_valid,
  };
}

DownloadResult download_batch(DeviceState &device,
                              const std::span<const DownloadRequest> requests) {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr || requests.empty()) {
    return DownloadResult{.status = Status::fail(Reason::TransferInvalid)};
  }
  if (requests.size() > TransferCapacity) {
    return DownloadResult{.status = Status::fail(Reason::PipelineCapacity)};
  }
  std::array<node::accel::detail::DownloadEntry, TransferCapacity> transfers{};
  std::array<node::accel::detail::DownloadRoute, TransferCapacity> routes{};
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    const DownloadRequest request = requests[index];
    const AccelBufferState *const source =
        request.buffer == nullptr ? nullptr : accel_buffer(*request.buffer);
    if (source == nullptr || request.payload_hash == nullptr ||
        (request.bytes != 0u && request.data == nullptr)) {
      return DownloadResult{.status = Status::fail(Reason::TransferInvalid)};
    }
    transfers[index] = node::accel::detail::DownloadEntry{
        .buffer = &source->buffer,
        .data = request.data,
        .bytes = request.bytes,
        .payload_hash = request.payload_hash,
    };
  }
  const node::accel::detail::AccelTransfer transfer =
      node::accel::detail::DownloadAccelBuffersMeasured(
          accel->context,
          std::span<const node::accel::detail::DownloadEntry>{transfers.data(),
                                                              requests.size()},
          std::span<node::accel::detail::DownloadRoute>{routes.data(),
                                                        requests.size()});
  return DownloadResult{
      .status = transfer.check.ok
                    ? Status::success()
                    : Status::fail(project_reason(transfer.check.reason,
                                                  Reason::TransferInvalid)),
      .staging_bytes = transfer.staging_bytes,
      .staging_peak_bytes = transfer.staging_peak_bytes,
      .staging_reused_bytes = transfer.staging_reused_bytes,
      .staging_budget = transfer.staging_budget,
      .buffer_allocations = transfer.buffer_allocations,
      .buffer_reuses = transfer.buffer_reuses,
      .command_submits = transfer.command_submits,
      .readback_ns = transfer.readback_ns,
      .staging_reused = transfer.staging_reused,
      .payload_hash_valid = transfer.check.ok && transfer.payload_hash_valid,
  };
}

CopyResult copy_batch(DeviceState &device,
                      const std::span<const CopyRequest> requests) {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr || requests.empty()) {
    return CopyResult{.status = Status::fail(Reason::TransferInvalid)};
  }
  if (requests.size() > TransferCapacity) {
    return CopyResult{.status = Status::fail(Reason::PipelineCapacity)};
  }
  std::array<node::accel::detail::CopyEntry, TransferCapacity> transfers{};
  std::array<node::accel::detail::CopyRoute, TransferCapacity> routes{};
  for (std::size_t index = 0u; index < requests.size(); ++index) {
    const CopyRequest request = requests[index];
    const AccelBufferState *const source =
        request.source == nullptr ? nullptr : accel_buffer(*request.source);
    AccelBufferState *const target =
        request.target == nullptr ? nullptr : accel_buffer(*request.target);
    if (source == nullptr || target == nullptr) {
      return CopyResult{.status = Status::fail(Reason::TransferInvalid)};
    }
    transfers[index] = node::accel::detail::CopyEntry{
        .source = &source->buffer,
        .target = &target->buffer,
        .bytes = request.bytes,
    };
  }
  const node::accel::detail::AccelCopy copied =
      node::accel::detail::CopyAccelBuffers(
          accel->context,
          std::span<const node::accel::detail::CopyEntry>{transfers.data(),
                                                          requests.size()},
          std::span<node::accel::detail::CopyRoute>{routes.data(),
                                                    requests.size()});
  return CopyResult{
      .status = copied.check.ok
                    ? Status::success()
                    : Status::fail(project_reason(copied.check.reason,
                                                  Reason::TransferInvalid)),
      .command_submits = copied.command_submits,
  };
}

Status compile(DeviceState &device, AccelProgram &program,
               const rund::AccelGraph &graph) {
  AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return Status::fail(Reason::AccelProgramInvalid);
  }
  program.kernel = node::accel::CompileAccelKernel(accel->context, graph);
  if (!program.kernel.check.ok) {
    return Status::fail(
        project_reason(program.kernel.check.reason, Reason::LoweringInvalid));
  }
  const node::accel::detail::KernelTokenRetainedMemory token_memory =
      node::accel::detail::MeasureKernelTokenRetainedMemory(program.kernel);
  if (!token_memory.exact) {
    return Status::fail(Reason::AccelProgramInvalid);
  }
  program.kernel_token_host_bytes = token_memory.host_bytes;
  return Status::success();
}

node::accel::detail::KernelScratchPlan
plan_scratch(const DeviceState &device, const rund::AccelKernel &kernel,
             const std::uint64_t alignment, const std::uint64_t page_bytes) {
  const AccelDeviceState *const accel = accel_device(device);
  return accel == nullptr ? node::accel::detail::KernelScratchPlan{}
                          : node::accel::detail::PlanKernelScratch(
                                accel->context, kernel, alignment, page_bytes);
}

MemoryCounter device_staging(const DeviceState &device) noexcept {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return {};
  }
  const node::accel::AccelMemoryCounter memory =
      node::accel::ReadAccelMemoryStats(accel->pick).staging;
  return MemoryCounter{.current = memory.current,
                       .peak = memory.peak,
                       .cumulative = memory.cumulative,
                       .reused = memory.reused,
                       .budget = memory.budget};
}

MemoryCounter job_staging(const JobState &job) noexcept {
  node::accel::detail::PreparedMemory memory =
      node::accel::detail::ReadPreparedKernelMemory(job.prepared);
  const node::accel::detail::PreparedMemory pending =
      node::accel::detail::ReadPreparedKernelMemory(job.write_prepared);
  node::accel::detail::accumulate_serial_memory(memory, pending);
  return MemoryCounter{.current = memory.current,
                       .peak = memory.peak,
                       .cumulative = memory.cumulative,
                       .reused = memory.reused,
                       .budget = memory.budget};
}

node::accel::detail::PreparedPipelineMemory
pipeline_memory(const PipelineState &pipeline) noexcept {
  node::accel::detail::PreparedPipelineMemory memory =
      node::accel::detail::ReadPreparedKernelPipelineMemory(pipeline.prepared);
  const node::accel::detail::PreparedPipelineMemory alternate =
      node::accel::detail::ReadPreparedKernelPipelineMemory(
          pipeline.alternate_prepared);
  node::accel::detail::accumulate_serial_memory(memory.host, alternate.host);
  node::accel::detail::accumulate_serial_memory(memory.device,
                                                alternate.device);
  node::accel::detail::accumulate_serial_memory(memory.staging,
                                                alternate.staging);
  // Primary and transactional alternate share one immutable template
  // registry. Report that owner once after combining the two stream-local
  // pipelines so retained host memory is neither omitted nor doubled.
  node::accel::detail::accumulate_serial_memory(
      memory.host,
      node::accel::detail::ReadPreparedKernelTemplateRegistryMemory(
          pipeline.accel_templates));
  return memory;
}

node::accel::detail::PreparedKernelPipelineReservation
plan_pipeline_preparation(
    const DeviceState &device,
    const std::span<const node::accel::detail::PreparedKernelProgramRoute>
        routes,
    const node::accel::detail::PreparedKernelPipelineShape shape,
    node::accel::detail::PreparedKernelTemplateRegistry &templates) noexcept {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return {};
  }
  return node::accel::detail::PlanPreparedKernelPipelineLimit(
      accel->context, routes, shape, templates);
}

node::accel::detail::PreparedPipelineEvidence
run_pipeline(const DeviceState &device,
             const node::accel::detail::PreparedKernelPipeline &pipeline) {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return node::accel::detail::PreparedPipelineEvidence{
        .check = {false, "accel_device_invalid"}};
  }
  return node::accel::detail::RunPreparedKernelPipeline(accel->context,
                                                        pipeline);
}

node::accel::detail::PreparedKernelPipeline prepare_pipeline(
    const DeviceState &device,
    const std::span<const node::accel::detail::PreparedKernelRun *const>
        prepared,
    const std::span<const std::uint8_t> barriers,
    const std::span<const std::uint32_t> declared_steps,
    const std::span<const node::accel::detail::BackendRecurrence> recurrences,
    const std::span<const node::accel::detail::BackendPublish> publications,
    const std::uint32_t declared_step_count,
    const std::uint32_t generation_stride, const bool profile_steps,
    node::accel::detail::PreparedKernelTemplateRegistry *const templates) {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return {};
  }
  return node::accel::detail::PrepareKernelPipeline(
      accel->context, prepared, barriers, declared_steps, recurrences,
      publications, declared_step_count, generation_stride, profile_steps,
      templates);
}

rund::AccelCheck submit_pipeline(
    const DeviceState &device,
    const node::accel::detail::PreparedKernelPipeline &pipeline,
    std::shared_ptr<void> lifetime,
    const node::accel::detail::PreparedPipelineCompletion completion,
    void *const user) noexcept {
  const AccelDeviceState *const accel = accel_device(device);
  return accel == nullptr ? rund::AccelCheck{false, "accel_device_invalid"}
                          : node::accel::detail::SubmitPreparedKernelPipeline(
                                accel->context, pipeline, std::move(lifetime),
                                completion, user);
}

rund::AccelCheck seed_pipeline_generation(
    const node::accel::detail::PreparedKernelPipeline &pipeline,
    const std::uint32_t generation) noexcept {
  return node::accel::detail::SeedPreparedKernelPipelineGeneration(pipeline,
                                                                   generation);
}

const DeviceOps Operations{
    .allocate = allocate,
    .upload = upload,
    .upload_batch = upload_batch,
    .download = download,
    .download_batch = download_batch,
    .copy_batch = copy_batch,
    .compile = compile,
    .plan_scratch = plan_scratch,
    .resolve_buffer = resolve_buffer,
    .prepare_job = prepare_job_accel,
    .run_job = run_job_accel,
    .submit_job = submit_job_accel,
    .finish_job = finish_job_accel,
    .plan_pipeline_preparation = plan_pipeline_preparation,
    .prepare_pipeline = prepare_pipeline,
    .run_pipeline = run_pipeline,
    .submit_pipeline = submit_pipeline,
    .seed_pipeline_generation = seed_pipeline_generation,
    .device_staging = device_staging,
    .job_staging = job_staging,
    .pipeline_memory = pipeline_memory,
};

} // namespace

const DeviceOps &AccelDeviceOps() noexcept { return Operations; }

} // namespace rund::compute::detail
