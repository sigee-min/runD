#pragma once

#include "capability.hpp"
#include "evidence.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace rund::node::accel::detail {

struct PersistentResidencySlidingRequest;
struct PersistentResidencySlidingRequestCell;
struct PersistentResidencySlidingTicket;

struct PersistentResidencySlidingRole final {
  std::shared_ptr<void> prepared{};
  std::array<std::uint32_t, ResidencyWindowLocalCapacity> locals{};
  std::size_t local_count{};
  std::uint32_t first_control_generation{};
  std::uint32_t control_generation_stride{};
  std::uint64_t first_descriptor_generation{};
  std::uint64_t descriptor_generation_stride{};
  std::uint8_t slot{};
};

// One immutable whole-run product handoff. Project, Release, and Returned are
// deliberately absent: adding any of them would turn this API back into the
// Host-driven PreparedResidencySliding fallback.
struct PersistentResidencySlidingRequest final {
  std::uint64_t plan_identity{};
  std::uint64_t token{};
  std::uint64_t generation{};
  // Private Authority/Sliding owner credential. It is never encoded into a
  // device payload and is required for every stage, commit, and Final.
  std::uint64_t owner_nonce{};
  std::uint64_t cell_id{};
  std::uint64_t cell_domain{};
  std::uint64_t coordinate_count{};
  // The common service owns the half-open physical chunk selected for this
  // submit. Backends only validate it and report acceptance.
  std::uint64_t first_coordinate{};
  std::uint64_t chunk_count{};
  std::size_t tail_local_count{};
  std::array<PersistentResidencySlidingRole, PersistentResidencySlidingCapacity>
      roles{};
  std::shared_ptr<void> lowering{};
  std::shared_ptr<const void> admission{};
  PersistentResidencySlidingFinalCompletion final{};
  void *user{};
  // The ticket is a cold-owned shared bridge. It contains only pending state;
  // the committed request lives in its RequestCell and is never back-pointed.
  std::shared_ptr<PersistentResidencySlidingTicket> ticket{};
  ResidencySlidingMemory memory{ResidencySlidingMemory::HostCoherent};
  std::uint8_t width{};
  // Mode is the immutable admission authority.  chunk_count describes only
  // the span selected for this submit and must never be used to rediscover
  // the mode.
  PersistentResidencySlidingMode mode{
      PersistentResidencySlidingMode::OneSubmit};
};

struct PersistentResidencySlidingRequestCell final {
  PersistentResidencySlidingRequest committed{};
  std::uint64_t id{};
  std::uint64_t domain{};
};

struct PersistentResidencySlidingTicket final {
  std::shared_ptr<PersistentResidencySlidingRequestCell> cell{};
  std::optional<PersistentResidencySlidingRequest> pending{};
  bool active{};
  bool quarantined{};
};

} // namespace rund::node::accel::detail
