#pragma once

#include "../../accel/kernel/prepared/callback.hpp"
#include <accel/check.hpp>
#include <rund/compute/status.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace rund::node::accel::detail {
struct BackendResidencySchedulePreparation;
struct BackendResidencyWindowAbort;
struct BackendResidencyWindowSignal;
struct PreparedKernelPipeline;
struct PreparedResidencyScheduleControl;
struct PreparedResidencyScheduleRequest;
struct PreparedResidencyScheduleRole;
struct PreparedResidencySlidingControl;
struct PreparedResidencySlidingRequest;
struct PreparedResidencySlidingRole;
struct PreparedResidencyStreamControl;
struct PreparedResidencyWindowControl;
struct PreparedResidencyWindowRequest;
enum class ResidencySlidingMemory : std::uint8_t;
} // namespace rund::node::accel::detail
namespace rund::compute::detail::residency::execution {
struct Owner;
}

namespace rund::compute {
struct Stats;
}

namespace rund::compute::detail {
struct DeviceState;

struct PipelineState;

struct DeviceResidencyOps final {
  rund::AccelCheck (*submit_residency_pipeline)(
      const DeviceState &, const node::accel::detail::PreparedKernelPipeline &,
      std::span<const std::uint32_t>, std::shared_ptr<void>,
      node::accel::detail::PreparedPipelineCompletion,
      void *) noexcept = nullptr;
  // One backend-neutral public handoff for a bounded W<=4 selected-Pipeline
  // window.  The request contains copied locals and strong prepared owners;
  // native adapters own only command/timeline execution.  Compute remains the
  // sole Pipeline-attempt and Host-service authority.
  Status (*submit_residency_window)(
      const DeviceState &,
      const node::accel::detail::PreparedResidencyWindowRequest &,
      node::accel::detail::PreparedResidencyWindowControl &) noexcept = nullptr;
  Status (*submit_residency_stream_window)(
      const DeviceState &,
      const node::accel::detail::PreparedResidencyWindowRequest &,
      node::accel::detail::PreparedResidencyWindowControl &,
      node::accel::detail::PreparedResidencyStreamControl &) noexcept = nullptr;
  Status (*claim_residency_stream)(
      const DeviceState &,
      std::span<const node::accel::detail::PreparedKernelPipeline>,
      std::uint64_t plan_identity, std::uint64_t token,
      std::uint64_t generation,
      node::accel::detail::PreparedResidencyStreamControl &) noexcept = nullptr;
  Status (*release_residency_stream)(
      const DeviceState &,
      node::accel::detail::PreparedResidencyStreamControl &,
      std::uint64_t plan_identity, std::uint64_t token,
      std::uint64_t generation, bool quarantine) noexcept = nullptr;
  Status (*quarantine_residency_stream)(
      const DeviceState &,
      node::accel::detail::PreparedResidencyStreamControl &,
      std::uint64_t plan_identity, std::uint64_t token,
      std::uint64_t generation) noexcept = nullptr;
  Status (*residency_window_capability)(
      const DeviceState &,
      std::span<const node::accel::detail::PreparedKernelPipeline>,
      bool &) noexcept = nullptr;
  Status (*signal_residency_window)(
      const DeviceState &, const node::accel::detail::PreparedKernelPipeline &,
      const node::accel::detail::BackendResidencyWindowSignal &) noexcept =
      nullptr;
  Status (*abort_residency_window)(
      const DeviceState &,
      node::accel::detail::PreparedResidencyWindowControl &,
      const node::accel::detail::BackendResidencyWindowAbort &) noexcept =
      nullptr;
  node::accel::detail::BackendResidencySchedulePreparation (
      *prepare_residency_schedule)(
      const DeviceState &,
      std::span<const node::accel::detail::PreparedResidencyScheduleRole>,
      std::uint64_t epoch_count,
      std::size_t tail_local_count) noexcept = nullptr;
  Status (*submit_residency_schedule)(
      const DeviceState &,
      const node::accel::detail::PreparedResidencyScheduleRequest &,
      node::accel::detail::PreparedResidencyScheduleControl &,
      node::accel::detail::PreparedResidencyStreamControl &) noexcept = nullptr;
  Status (*signal_residency_schedule)(
      const DeviceState &, const node::accel::detail::PreparedKernelPipeline &,
      const node::accel::detail::BackendResidencyWindowSignal &) noexcept =
      nullptr;
  Status (*abort_residency_schedule)(
      const DeviceState &,
      node::accel::detail::PreparedResidencyScheduleControl &,
      const node::accel::detail::BackendResidencyWindowAbort &) noexcept =
      nullptr;
  Status (*prepare_residency_sliding)(
      const DeviceState &,
      std::span<const node::accel::detail::PreparedResidencySlidingRole>,
      node::accel::detail::ResidencySlidingMemory,
      node::accel::detail::PreparedResidencySlidingControl &) noexcept =
      nullptr;
  Status (*submit_residency_sliding)(
      const DeviceState &,
      const node::accel::detail::PreparedResidencySlidingRequest &,
      node::accel::detail::PreparedResidencySlidingControl &) noexcept =
      nullptr;
  Status (*wake_residency_sliding)(
      const DeviceState &,
      node::accel::detail::PreparedResidencySlidingControl &) noexcept =
      nullptr;
  // Optional backend-neutral cold staging for a prepared pipeline's native
  // residency owner. A backend may report that this owner is unavailable;
  // callers preserve their existing higher-level fallback in that case.
  Status (*prepare_pipeline_residency)(PipelineState &) noexcept = nullptr;
  // Strict cold preparation for an Authority-selected execution. The
  // accelerator must prove a retained selected-command path; a whole-command
  // transfer owner is not an acceptable fallback for Authority-selected
  // locals, and Compute never switches on the native API.
  Status (*prepare_residency_selection)(PipelineState &) noexcept = nullptr;
  // Cold, policy-free native topology seam. It aliases already-accounted
  // prepared resources and deliberately takes no invocation Plan. Unsupported
  // native adapters must leave Owner empty and fail BackendUnsupported.
  Status (*prepare_residency_execution)(
      const DeviceState &,
      std::span<const node::accel::detail::PreparedKernelPipeline *const>,
      residency::execution::Owner &) noexcept = nullptr;
};

} // namespace rund::compute::detail
