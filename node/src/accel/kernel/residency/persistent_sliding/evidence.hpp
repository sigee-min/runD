#pragma once

#include "../sliding.hpp"

#include <accel/check.hpp>

#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

inline constexpr std::uint64_t PersistentSlidingNoCoordinate =
    std::numeric_limits<std::uint64_t>::max();

// One aggregate GPU-produced observation for the complete recurrence. No
// per-coordinate receipt or callback journal crosses the product seam.
struct PersistentResidencySlidingEvidence final {
  std::uint64_t plan_identity{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t owner_nonce{};
  std::uint64_t cell_id{};
  std::uint64_t cell_domain{};
  std::uint64_t coordinate_count{};
  std::uint64_t accepted_coordinates{};
  std::uint64_t gpu_completed_coordinates{};
  std::uint64_t completed_prefix{};
  std::uint64_t suppressed_first{PersistentSlidingNoCoordinate};
  std::uint64_t suppressed_count{};
  std::uint64_t native_submit_count{};
  std::uint64_t epoch_native_submit_count{};
  std::uint64_t backend_epoch_callback_count{};
  std::uint64_t host_epoch_callback_count{};
  std::uint64_t backing_wait_count{};
  std::uint64_t backing_signal_count{};
  std::uint64_t backing_acknowledgement_count{};
  std::uint64_t backing_service_failure_count{};
  std::uint64_t failed_admission_count{};
  std::uint64_t first_failed_admission_coordinate{
      PersistentSlidingNoCoordinate};
  std::uint64_t first_service_failure_coordinate{PersistentSlidingNoCoordinate};
  std::uint64_t final_callback_count{};
  std::uint64_t queue_calls{};
  std::uint64_t accepted_end{};
  std::uint64_t chunk_submit_count{};
  std::uint64_t first_failure_coordinate{PersistentSlidingNoCoordinate};
  std::uint64_t completed_ns{};
  std::uint8_t width{};
};

struct PersistentResidencySlidingFinal final {
  rund::AccelCheck check{false, "accel_kernel_pipeline_invalid"};
  NativeTerminal terminal{NativeTerminal::Known};
  PersistentResidencySlidingEvidence evidence{};
};

using PersistentResidencySlidingFinalCompletion =
    void (*)(void *, PersistentResidencySlidingFinal &&) noexcept;

} // namespace rund::node::accel::detail
