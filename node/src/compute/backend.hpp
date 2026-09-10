#pragma once

#include "../../include/rund/compute/abi/state.hpp"

#include "../accel/backend/result.hpp"
#include "../accel/kernel/callback.hpp"
#include "../accel/kernel/prepared/callback.hpp"
#include "backend/residency.hpp"
#include "backend/transfer/model.hpp"
#include "backend/virtual.hpp"

#include <accel/graph/value.hpp>
#include <accel/kernel/run/binding.hpp>
#include <rund/compute/program/range.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace rund {
struct AccelKernel;
}

namespace rund::node::accel::detail {
struct KernelScratchPlan;
struct PreparedKernelPipelineReservation;
struct PreparedKernelProgramRoute;
struct PreparedKernelPipelineShape;
struct PreparedKernelRun;
struct PreparedKernelPipeline;
struct BackendRecurrence;
struct BackendPublish;
struct PreparedPipelineMemory;
struct PreparedKernelTemplateRegistry;
} // namespace rund::node::accel::detail

namespace rund::compute::detail {

struct AccelProgram;
struct JobState;
struct PipelineState;

using JobDone = void (*)(void *, Result<RunState>) noexcept;
using PipelineDone = node::accel::detail::PreparedPipelineCompletion;

struct DeviceOps final {
  Status (*allocate)(DeviceState &, BufferState &, std::size_t, std::size_t,
                     bool, node::accel::detail::BackendBufferMemory,
                     std::uint64_t exact_storage_bytes) = nullptr;
  // Re-seals a typed capability over the same physical allocation; no native
  // allocation or transfer is permitted at this boundary.
  Status (*project_buffer_view)(const BufferState &, BufferState &) = nullptr;
  std::uint64_t (*buffer_storage_bytes)(const DeviceState &,
                                        std::uint64_t) noexcept = nullptr;
  std::uint64_t (*pipeline_transfer_storage_bytes)(
      const DeviceState &, std::uint64_t) noexcept = nullptr;
  UploadResult (*upload)(DeviceState &, BufferState &, const void *,
                         std::size_t) = nullptr;
  UploadResult (*upload_batch)(DeviceState &, std::span<const UploadRequest>,
                               node::accel::detail::TransferCompletion,
                               node::accel::detail::TransferAuthority) =
      nullptr;
  DownloadResult (*download)(DeviceState &, const BufferState &, void *,
                             std::size_t, std::size_t) = nullptr;
  DownloadResult (*download_batch)(
      DeviceState &, std::span<const DownloadRequest>,
      node::accel::detail::TransferAuthority) = nullptr;
  // Maximum ordered prefix batch accepted by the backend-neutral resident
  // transfer owner. Zero requires callers to use the scalar operation.
  std::size_t download_prefix_capacity = 0u;
  CopyResult (*copy_batch)(DeviceState &, std::span<const CopyRequest>,
                           node::accel::detail::TransferAuthority) = nullptr;
  // Backend-owned immutable capability. A nonempty result proves a stable,
  // coherent, read-only Host view for the complete Buffer lifetime. Compute
  // may read it only after the exact native execution terminal.
  BufferReadView (*host_read)(const DeviceState &,
                              const BufferState &) noexcept = nullptr;
  // Authenticates a stable coherent writable view of the complete Buffer.
  // This capability never substitutes for an Authority lease over the exact
  // Input range that a caller intends to mutate.
  BufferWriteView (*host_write)(const DeviceState &,
                                const BufferState &) noexcept = nullptr;
  Status (*compile)(DeviceState &, AccelProgram &,
                    const rund::AccelGraph &) = nullptr;
  RangeSnapshot (*program_ranges)(const AccelProgram &,
                                  std::span<RangeInfo>) noexcept = nullptr;
  node::accel::detail::KernelScratchPlan (*plan_scratch)(
      const DeviceState &, const rund::AccelKernel &, std::uint64_t,
      std::uint64_t) = nullptr;
  Status (*resolve_buffer)(const DeviceState &, const BufferState &,
                           std::shared_ptr<void> &) = nullptr;
  Status (*prepare_job)(const std::shared_ptr<JobState> &,
                        std::span<rund::AccelRunBinding>) = nullptr;
  Result<RunState> (*run_job)(const std::shared_ptr<JobState> &) = nullptr;
  Status (*submit_job)(const std::shared_ptr<JobState> &, std::shared_ptr<void>,
                       JobDone, void *,
                       node::accel::detail::KernelTiming) noexcept = nullptr;
  Result<RunState> (*finish_job)(const std::shared_ptr<JobState> &,
                                 const rund::AccelEvidence &) = nullptr;
  node::accel::detail::PreparedKernelPipelineReservation (
      *plan_pipeline_preparation)(
      const DeviceState &,
      std::span<const node::accel::detail::PreparedKernelProgramRoute>,
      node::accel::detail::PreparedKernelPipelineShape,
      node::accel::detail::PreparedKernelTemplateRegistry &) noexcept = nullptr;
  node::accel::detail::PreparedKernelPipeline (*prepare_pipeline)(
      const DeviceState &,
      std::span<const node::accel::detail::PreparedKernelRun *const>,
      std::span<const std::uint8_t>, std::span<const std::uint32_t>,
      std::span<const node::accel::detail::BackendRecurrence>,
      std::span<const node::accel::detail::BackendPublish>, std::uint32_t,
      std::uint32_t, bool,
      node::accel::detail::PreparedKernelTemplateRegistry *) = nullptr;
  node::accel::detail::PreparedPipelineEvidence (*run_pipeline)(
      const DeviceState &, const node::accel::detail::PreparedKernelPipeline &,
      node::accel::detail::PipelineSubmitMode) = nullptr;
  rund::AccelCheck (*submit_pipeline)(
      const DeviceState &, const node::accel::detail::PreparedKernelPipeline &,
      std::shared_ptr<void>, PipelineDone, void *,
      node::accel::detail::KernelTiming,
      node::accel::detail::PipelineSubmitMode) noexcept = nullptr;
  rund::AccelCheck (*seed_pipeline_generation)(
      const node::accel::detail::PreparedKernelPipeline &,
      std::uint32_t) noexcept = nullptr;
  UploadResult (*upload_pipeline_transfer)(PipelineState &, const void *,
                                           std::size_t) noexcept = nullptr;
  DownloadResult (*download_pipeline_transfer)(
      PipelineState &, void *, std::size_t, std::uint64_t *) noexcept = nullptr;
  MemoryCounter (*device_staging)(const DeviceState &) noexcept = nullptr;
  MemoryCounter (*job_staging)(const JobState &) noexcept = nullptr;
  node::accel::detail::PreparedPipelineMemory (*pipeline_memory)(
      const PipelineState &) noexcept = nullptr;
  DeviceResidencyOps residency{};
  VirtualDeviceOps virtual_execution{};
};

[[nodiscard]] const DeviceOps &AccelDeviceOps() noexcept;

} // namespace rund::compute::detail
