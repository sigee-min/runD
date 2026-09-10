#pragma once

#include "request.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>

namespace rund::node::accel::detail {

// Caller-owned whole-run service binding. Only Host-initiated service
// operations mutate its monotonic frontiers; a backend may deliver one Final
// but never calls a coordinate callback through this control.
struct PersistentResidencySlidingControl final {
  mutable std::mutex gate{};
  std::shared_ptr<void> native{};
  PersistentResidencySlidingCapability capability{};
  std::array<PersistentResidencySlidingRole, PersistentResidencySlidingCapacity>
      roles{};
  std::uint64_t plan_identity{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t owner_nonce{};
  std::uint64_t coordinate_count{};
  PersistentResidencySlidingMode mode{
      PersistentResidencySlidingMode::OneSubmit};
  std::uint64_t chunk_first{};
  std::uint64_t chunk_span{};
  // Raw accepted prefix. Zero means that no physical submit was accepted;
  // it is never interpreted as the complete coordinate count.
  std::uint64_t accepted_end{};
  std::uint64_t chunk_submit_count{};
  std::uint64_t next_ready_coordinate{};
  std::uint64_t next_wait_coordinate{};
  std::uint64_t next_observation_coordinate{};
  std::uint64_t next_acknowledgement_coordinate{};
  std::uint64_t native_submit_count{};
  std::uint64_t queue_calls{};
  std::uint64_t epoch_native_submit_count{};
  std::uint64_t backend_epoch_callback_count{};
  std::uint64_t backing_wait_count{};
  std::uint64_t backing_signal_count{};
  std::uint64_t backing_acknowledgement_count{};
  std::uint64_t backing_service_failure_count{};
  std::uint64_t failed_admission_count{};
  std::uint64_t first_failed_admission_coordinate{
      PersistentSlidingNoCoordinate};
  std::uint64_t first_service_failure_coordinate{PersistentSlidingNoCoordinate};
  // Semantic completion and failure evidence is owned by Control. Backend
  // owners retain only physical handles and diagnostic traces.
  std::uint64_t gpu_completed_coordinates{};
  std::uint64_t completed_prefix{};
  std::uint64_t suppressed_first{PersistentSlidingNoCoordinate};
  std::uint64_t suppressed_count{};
  std::uint64_t first_failure_coordinate{PersistentSlidingNoCoordinate};
  rund::AccelCheck first_failure{true, "ok"};
  std::uint64_t final_callback_count{};
  std::uint8_t width{};
  bool active{};
  bool service_failed{};
  bool quarantined{};
};

} // namespace rund::node::accel::detail
