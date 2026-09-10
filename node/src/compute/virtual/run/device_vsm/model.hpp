#pragma once

#include "../../../../accel/kernel/residency/device_vsm.hpp"
#include "../../../device/residency/execution/device_vsm/registration.hpp"
#include "../../../pipeline/execution/attempt.hpp"
#include "../execution.hpp"

#include <accel/context/buffer.hpp>
#include <rund/storage.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

namespace rund::compute::detail::device_vsm_product_detail {

enum class DeviceVsmTestPhase : std::uint8_t {
  Armed,
  Accepted,
  Callback,
  Released,
  Failed,
};

struct DeviceVsmTestBarrier final {
  std::atomic<DeviceVsmTestPhase> phase{DeviceVsmTestPhase::Armed};

  void publish(const DeviceVsmTestPhase value) noexcept {
    phase.store(value, std::memory_order_release);
    phase.notify_all();
  }

  [[nodiscard]] DeviceVsmTestPhase load() const noexcept {
    return phase.load(std::memory_order_acquire);
  }

  void wait_released() const noexcept {
    DeviceVsmTestPhase value = load();
    while (value != DeviceVsmTestPhase::Released) {
      phase.wait(value, std::memory_order_acquire);
      value = load();
    }
  }
};

inline constexpr std::size_t DeviceVsmPipelineCapacity =
    node::accel::detail::DeviceVsmGraphStageCapacity;
// Pipeline ownership is bounded by graph stages. Planner resources are a
// separate authority: a stage graph can own external inputs, intermediates,
// and its public output without making each resource a native pipeline.

struct DeviceVsmPipelineSnapshot final {
  std::array<std::uint64_t, DeviceVsmPipelineCapacity> generation{};
  std::array<std::uint8_t, DeviceVsmPipelineCapacity> parity{};
  std::size_t count{};
};

struct DeviceVsmProductEvidence final {
  node::accel::detail::DeviceVsmEvidence native{};
  std::uint64_t cold_prepare_count{};
  std::uint64_t warm_rearm_count{};
  std::uint64_t public_handoff_count{};
  std::uint64_t authority_accept_count{};
  std::uint64_t pipeline_terminal_count{};
  std::uint64_t backing_publication_count{};
  std::uint64_t output_hash_observation_count{};
  std::uint64_t output_hash_reuse_count{};
  node::accel::detail::DeviceVsmTerminal final_terminal{
      node::accel::detail::DeviceVsmTerminal::Known};
  Reason final_reason{Reason::CompletionInvalid};
  Code final_code{Code::Execution};
  std::uint32_t public_resident_input_count{};
  std::uint32_t whole_run_staged_input_count{};
  bool final_received{};
  bool quarantined{};
  bool public_resident_output{};
  bool whole_run_staged_output{};
  // The current DeviceVsm request has no running-GPU/Host page-service ABI.
  // Keep this distinct from native bounded resident-to-ring page access.
  bool bounded_external_page_service{};
};

struct DeviceVsmProductOwner final {
  using ReleaseRegistration = residency::RegistrationResult (*)(
      residency::DeviceVsmRegistration *) noexcept;

  // UnknownMayWrite must outlive the public VirtualPipeline and Device
  // wrappers. The self-cycle intentionally retains the exact prepared
  // buffers, Pipeline states, Pool, memory reservation, and Authority proof;
  // adapter-wide quarantine prevents any future native reuse.
  std::shared_ptr<DeviceVsmProductOwner> quarantine{};
  storage::Reservation capacity{};
  std::shared_ptr<storage::Reservation> memory{};
  std::array<std::shared_ptr<PipelineState>, DeviceVsmPipelineCapacity>
      pipelines{};
  std::array<std::uint32_t, DeviceVsmPipelineCapacity> pipeline_stages{};
  std::size_t pipeline_count{};
  std::shared_ptr<const node::accel::detail::DeviceVsmProof> proof{};
  node::accel::detail::DeviceVsmPreparation preparation{};
  // The callback retains this owner through notification and wake. A caller
  // may destroy its transient run as soon as it observes completion.
  std::atomic_bool done{false};
  bool submitted{};
  std::array<rund::AccelBuffer, VirtualPipelineState::InputCapacity> inputs{};
  std::size_t input_count{};
  rund::AccelBuffer output{};
  // Non-null rows are exact runD-minted VirtualBacking resident owners. They
  // prevent a cached DeviceVsm proof from being rebound to another physical
  // Buffer and distinguish direct GPU addressability from Host staging.
  std::array<std::shared_ptr<BufferState>, VirtualPipelineState::InputCapacity>
      resident_inputs{};
  std::shared_ptr<BufferState> resident_output{};
  std::array<std::vector<std::byte>, VirtualPipelineState::InputCapacity>
      input_staging{};
  std::vector<std::byte> output_staging{};
  // StagedLoop owns host-visible fallback buffers and performs backing I/O
  // through their mapped views.  An empty staging vector is intentional.
  // This copy is the immutable route and endpoint authority consumed by warm
  // authentication and all staging workers.
  VirtualDeviceVsmRouteProof route_proof{};
  std::shared_ptr<residency::DeviceVsmRegistration> registration{};
  ReleaseRegistration release_registration{};
  std::shared_ptr<DeviceVsmProductEvidence> evidence{};
  std::uint64_t cold_prepare_count{};
  std::uint64_t warm_rearm_count{};
  std::array<std::uint64_t, VirtualPipelineState::InputCapacity>
      output_hash_input_versions{};
  std::uint64_t output_hash{};
  std::uint64_t output_hash_observation_count{};
  std::uint64_t output_hash_reuse_count{};
  bool output_hash_valid{};
};

struct DeviceVsmProductRun final {
  std::shared_ptr<DeviceVsmProductOwner> owner{};
  std::shared_ptr<residency::DeviceVsmRegistration> registration{};
  std::optional<residency::DirectRecurrenceLease> lease{};
  VirtualPipelineState *state{};
  std::array<VirtualBacking *, VirtualPipelineState::InputCapacity> inputs{};
  std::array<std::span<std::byte>, VirtualPipelineState::InputCapacity>
      input_staging{};
  std::span<std::byte> output_staging{};
  std::size_t input_count{};
  VirtualBacking *output{};
  const VirtualRunProjection *projection{};
  Stats *stats{};
  DeviceVsmPipelineSnapshot snapshot{};
  node::accel::detail::DeviceVsmFinal final{};
  node::accel::detail::DeviceVsmSubmissionControl submission_control{};
  node::accel::detail::DeviceVsmRequest request{};
  std::atomic_uint64_t callback_count{};
  std::array<bool, DeviceVsmPipelineCapacity> pipeline_started{};
  bool backing_recovery{};
  bool backing_may_write{};
  bool publication_success{};
  bool output_hash_observed{};
  bool wait_for_callback{true};
  bool poison_pipeline{};
  void *wake_user{};
  VirtualWake wake{};
  // Source-private cohort evidence; UINT64_MAX means no failed member.
  std::uint64_t failed_input{ResidencyStats::no_failed_page};
  std::uint64_t failed_page{ResidencyStats::no_failed_page};
  std::uint64_t output_hash{};
};

struct DeviceVsmPublication final {
  DeviceVsmProductRun *run{};
  Status status{Status::fail(Reason::CompletionInvalid)};
  std::array<std::unique_lock<std::mutex>, DeviceVsmPipelineCapacity>
      pipeline_locks{};
  std::size_t pipeline_lock_count{};
};

void test_phase(DeviceVsmProductRun &, DeviceVsmTestPhase) noexcept;

} // namespace rund::compute::detail::device_vsm_product_detail
