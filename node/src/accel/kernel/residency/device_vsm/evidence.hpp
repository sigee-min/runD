#pragma once

#include "identity.hpp"

#include <accel/check.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

inline constexpr std::size_t DeviceVsmResultWordCount = 13u;
inline constexpr std::size_t DeviceVsmOverflowWord = 6u;
inline constexpr std::size_t DeviceVsmFailedPageWord = 7u;
inline constexpr std::size_t DeviceVsmGraphWavefrontStepsWord = 8u;
inline constexpr std::size_t DeviceVsmGraphWavefrontTraceWord = 9u;
inline constexpr std::size_t DeviceVsmWindowBoundaryTransitionsWord = 8u;
inline constexpr std::size_t DeviceVsmWindowFootprintChecksumWord = 9u;
inline constexpr std::size_t DeviceVsmRingReuseTransitionsWord = 10u;
inline constexpr std::size_t DeviceVsmRingScheduleChecksumWord = 11u;
inline constexpr std::size_t DeviceVsmRingRoundTripsWord = 12u;
// The result buffer stores this raw sentinel in a 32-bit word.  Once a
// terminal is projected into private evidence, no-page is the public 64-bit
// sentinel so it cannot alias a real page coordinate after widening.
inline constexpr std::uint32_t DeviceVsmNoFailedPage =
    std::numeric_limits<std::uint32_t>::max();

enum class DeviceVsmTerminal : std::uint8_t { Known, UnknownMayWrite };

struct DeviceVsmEvidence final {
  DeviceVsmIdentity proof{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t nonce{};
  std::uint64_t page_count{};
  std::uint64_t failed_page{std::numeric_limits<std::uint64_t>::max()};
  std::uint64_t generated_epochs{};
  std::uint64_t completed_epochs{};
  std::uint64_t graph_wavefront_steps{};
  std::uint64_t graph_wavefront_trace{};
  std::uint64_t canonical_boundary_transitions{};
  std::uint64_t canonical_footprint_checksum{};
  std::uint64_t ring_reuse_transitions{};
  std::uint64_t ring_schedule_checksum{};
  std::uint64_t ring_round_trips{};
  std::uint64_t ring_state_bytes{};
  std::uint64_t ring_scratch_bytes{};
  std::uint64_t window_seed_dispatches{};
  std::uint64_t window_compute_dispatches{};
  std::uint64_t window_internal_dispatches{};
  std::uint64_t forecasted_pages{};
  std::uint64_t promoted_pages{};
  std::uint64_t drained_pages{};
  std::uint64_t persisted_pages{};
  std::uint64_t overlap_reused_bytes{};
  std::uint64_t gpu_backing_read_bytes{};
  std::uint64_t gpu_backing_write_bytes{};
  std::uint64_t native_submit_count{};
  std::uint64_t epoch_native_submit_count{};
  std::uint64_t payload_dispatch_count{};
  std::uint64_t tile_dispatch_count{};
  std::uint64_t host_service_turn_count{};
  std::uint64_t host_epoch_callback_count{};
  std::uint64_t final_callback_count{};
  std::uint64_t completed_ns{};
  std::uint64_t kernel_ns{};
  std::uint64_t kernel_samples{};
  std::uint64_t submit_wait_ns{};
  std::uint8_t max_live_frames{};
  std::uint64_t native_check_reason{};
  std::uint32_t native_check_code{};
  bool result_mapped{};
  bool native_check_ok{};
  bool result_acquired{};
  bool may_write{};
  // Host-side lifecycle marks use the same steady clock as submit telemetry.
  // They are retained for bounded phase diagnostics; no absolute mark is a
  // public duration or admission fact.
  std::uint64_t submit_ns{};
  std::uint64_t callback_ns{};
  std::uint64_t final_ns{};
};

struct DeviceVsmFinal final {
  rund::AccelCheck check{false, "accel_kernel_pipeline_invalid"};
  DeviceVsmTerminal terminal{DeviceVsmTerminal::Known};
  DeviceVsmEvidence evidence{};
};

using DeviceVsmFinalCompletion = void (*)(void *, DeviceVsmFinal &&) noexcept;

} // namespace rund::node::accel::detail
