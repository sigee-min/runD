#pragma once

#include "../../../../pipeline/residency/model.hpp"
#include "../credentials/cpu.hpp"
#include "../credentials/epoch.hpp"
#include "../graph.hpp"
#include "../transfer.hpp"
#include "frame.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rund::compute::detail::residency::registry_model {

// Lease lifecycle is the single state machine shared by ordinary, Graph,
// writeback, and retry rows.  The physical slot remains composed by
// Authority; no model object owns a second lease table.
enum class LeaseState : std::uint8_t {
  Free,
  Prepared,
  Computing,
  Drain,
  RetryReady,
  CpuQuarantined
};

struct LeaseSlot final {
  std::vector<std::uint32_t> undo_frames;
  std::vector<Frame> undo;
  std::vector<CacheBinding> bindings;
  std::vector<CacheTransition> transitions;
  std::vector<GraphLeasePort> ports;
  std::vector<GraphPageRemap> remaps;
  std::vector<GraphRelocation> relocations;
  std::vector<std::uint32_t> relocation_frames;
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t coordinate{};
  std::uint64_t book_domain{};
  FrameRegion retry_region{};
  CpuReservationKey cpu_key{};
  Identity plan{};
  GraphPersistIdentity identity{};
  std::uint64_t cycle{};
  LeaseState state{LeaseState::Free};
  bool cpu_bound{};
  bool terminal{};
};

} // namespace rund::compute::detail::residency::registry_model
