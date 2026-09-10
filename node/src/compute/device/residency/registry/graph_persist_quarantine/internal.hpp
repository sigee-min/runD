#pragma once

#include "../model/cpu.hpp"
#include "../../../../virtual/graph/reduce/receipts.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::compute::detail::residency::registry_model {

// Bounded read-only copies of the CPU receipt book.  These records are
// created before taking Authority's gate and are consumed only by the
// quarantine preflight; they never outlive one discard transaction.
struct CpuQuarantineOwner::BookSnapshot final {
  struct Epoch final {
    graph_reduce::CpuReceiptBook::SlotState state{};
    Authority *authority{};
    CpuReservationKey key{};
    std::uint64_t token{};
    std::uint64_t generation{};
    const void *permit{};
    const void *handle{};
    std::size_t bank{};
    std::uint8_t role{};
  };

  struct Unknown final {
    Authority *authority{};
    Identity plan{};
    GraphPersistIdentity identity{};
    FrameRegion region{};
    std::size_t page_count{};
    std::uint64_t domain{};
    std::uint64_t token{};
    std::uint64_t generation{};
    std::uint64_t coordinate{};
    bool terminal{};
    bool quarantined{};
    bool live{};
  };

  std::uint64_t domain{};
  std::size_t live_receipts{};
  std::size_t live_permits{};
  bool valid{};
  std::array<Epoch, graph_reduce::CpuReceiptBook::SlotCount> epochs{};
  std::array<Unknown, graph_reduce::CpuReceiptBook::UnknownCapacity> unknowns{};
};

struct CpuQuarantineOwner::Plan final {
  std::array<std::size_t, 4u> epochs{};
  std::array<std::size_t, execution::GraphPersistSlotCapacity> unknowns{};
  std::array<std::size_t, execution::GraphPersistSlotCapacity> retries{};
  std::array<std::uint32_t, execution::GraphPersistAggregateCapacity> claims{};
  std::size_t epoch_count{};
  std::size_t unknown_count{};
  std::size_t retry_count{};
  std::size_t claim_count{};
};

} // namespace rund::compute::detail::residency::registry_model
