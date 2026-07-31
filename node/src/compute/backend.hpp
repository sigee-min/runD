#pragma once

#include "../accel/kernel/prepared.hpp"
#include "device/state.hpp"

#include <accel/graph/value.hpp>
#include <accel/kernel/run/binding.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace rund::compute::detail {

struct AccelProgram;
struct JobState;
struct PipelineState;
struct ProgramState;
using JobDone = void (*)(void *, Result<RunState>) noexcept;
using PipelineDone = node::accel::detail::PreparedPipelineCompletion;

struct DownloadResult final {
  Status status{Status::success()};
  std::uint64_t payload_hash{};
  std::uint64_t staging_bytes{};
  std::uint64_t staging_peak_bytes{};
  std::uint64_t staging_reused_bytes{};
  std::uint64_t staging_budget{};
  std::uint64_t buffer_allocations{};
  std::uint64_t buffer_reuses{};
  std::uint64_t command_submits{};
  std::uint64_t readback_ns{};
  bool staging_reused{};
  bool payload_hash_valid{};
};

struct UploadResult final {
  Status status{Status::success()};
  std::uint64_t staging_bytes{};
  std::uint64_t staging_peak_bytes{};
  std::uint64_t staging_reused_bytes{};
  std::uint64_t staging_budget{};
  std::uint64_t buffer_allocations{};
  std::uint64_t buffer_reuses{};
  std::uint64_t command_submits{};
};

struct UploadRequest final {
  BufferState *buffer = nullptr;
  const void *data = nullptr;
  std::size_t bytes = 0u;
};

struct DownloadRequest final {
  const BufferState *buffer = nullptr;
  void *data = nullptr;
  std::size_t bytes = 0u;
  std::uint64_t *payload_hash = nullptr;
};

struct DeviceOps final {
  Status (*allocate)(DeviceState &, BufferState &, std::size_t, std::size_t,
                     bool) = nullptr;
  Status (*upload)(DeviceState &, BufferState &, const void *,
                   std::size_t) = nullptr;
  UploadResult (*upload_batch)(DeviceState &,
                               std::span<const UploadRequest>) = nullptr;
  DownloadResult (*download)(DeviceState &, const BufferState &, void *,
                             std::size_t) = nullptr;
  DownloadResult (*download_batch)(DeviceState &,
                                   std::span<const DownloadRequest>) = nullptr;
  Status (*compile)(DeviceState &, AccelProgram &,
                    const rund::AccelGraph &) = nullptr;
  node::accel::detail::KernelScratchPlan (*plan_scratch)(
      const DeviceState &, const rund::AccelKernel &, std::uint64_t,
      std::uint64_t) = nullptr;
  Status (*resolve_buffer)(const DeviceState &, const BufferState &,
                           std::shared_ptr<void> &) = nullptr;
  Status (*prepare_job)(const std::shared_ptr<JobState> &,
                        std::span<rund::AccelRunBinding>) = nullptr;
  Result<RunState> (*run_job)(const std::shared_ptr<JobState> &) = nullptr;
  Status (*submit_job)(const std::shared_ptr<JobState> &, std::shared_ptr<void>,
                       JobDone, void *) noexcept = nullptr;
  Result<RunState> (*finish_job)(const std::shared_ptr<JobState> &,
                                 const rund::AccelEvidence &) = nullptr;
  node::accel::detail::PreparedKernelPipeline (*prepare_pipeline)(
      const DeviceState &,
      std::span<const node::accel::detail::PreparedKernelRun *const>,
      std::span<const std::uint8_t>, std::span<const std::uint32_t>,
      std::span<const node::accel::detail::BackendRecurrence>,
      std::span<const node::accel::detail::BackendPublish>, std::uint32_t,
      std::uint32_t, bool) = nullptr;
  node::accel::detail::PreparedPipelineEvidence (*run_pipeline)(
      const DeviceState &,
      const node::accel::detail::PreparedKernelPipeline &) = nullptr;
  rund::AccelCheck (*submit_pipeline)(
      const DeviceState &, const node::accel::detail::PreparedKernelPipeline &,
      std::shared_ptr<void>, PipelineDone, void *) noexcept = nullptr;
  rund::AccelCheck (*seed_pipeline_generation)(
      const node::accel::detail::PreparedKernelPipeline &,
      std::uint32_t) noexcept = nullptr;
  MemoryCounter (*device_staging)(const DeviceState &) noexcept = nullptr;
  MemoryCounter (*job_staging)(const JobState &) noexcept = nullptr;
  node::accel::detail::PreparedPipelineMemory (*pipeline_memory)(
      const PipelineState &) noexcept = nullptr;
};

[[nodiscard]] const DeviceOps &AccelDeviceOps() noexcept;

} // namespace rund::compute::detail
