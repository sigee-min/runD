#pragma once

#include "local.hpp"

#include "../allocation.hpp"

#include <array>
#include <cstdint>

namespace rund_node_memory_contract {

[[nodiscard]] inline bool
ValidCounter(const rund::compute::MemoryCounter counter) noexcept {
  return counter.peak >= counter.current &&
         counter.cumulative >= counter.current &&
         counter.reused <= counter.cumulative &&
         (counter.budget == 0u || counter.budget >= counter.current);
}

struct PhysicalInternal final {
  bool complete = false;
  std::size_t count{};
  std::uint64_t bytes{};
};

template <class Owner>
[[nodiscard]] PhysicalInternal PhysicalInternalMemory(const Owner &owner) {
  std::array<rund::compute::MemoryEntry, 64u> entries{};
  const rund::compute::MemorySnapshot snapshot = owner.memory_snapshot(entries);
  PhysicalInternal result{.complete = !snapshot.truncated()};
  for (std::size_t index = 0u; index < snapshot.written; ++index) {
    const rund::compute::MemoryEntry &entry = entries[index];
    if (entry.use != rund::compute::MemoryUse::Internal ||
        (entry.category != rund::compute::MemoryCategory::Host &&
         entry.category != rund::compute::MemoryCategory::Device)) {
      continue;
    }
    ++result.count;
    result.bytes += entry.bytes.current;
  }
  return result;
}

[[nodiscard]] inline bool
SameCounter(const rund::compute::MemoryCounter &first,
            const rund::compute::MemoryCounter &second) {
  return first.current == second.current && first.peak == second.peak &&
         first.cumulative == second.cumulative &&
         first.reused == second.reused && first.budget == second.budget;
}

[[nodiscard]] inline bool SameStats(const rund::compute::MemoryStats &first,
                                    const rund::compute::MemoryStats &second) {
  return first.backend == second.backend && first.scope == second.scope &&
         SameCounter(first.host, second.host) &&
         SameCounter(first.frame, second.frame) &&
         SameCounter(first.tile, second.tile) &&
         SameCounter(first.resident, second.resident) &&
         SameCounter(first.staging, second.staging) &&
         SameCounter(first.device, second.device) &&
         SameCounter(first.transfer, second.transfer);
}

struct SnapshotAccounting final {
  std::array<rund::compute::MemoryCounter, 7u> categories{};
  std::size_t metadata_entries{};
  std::size_t tile_entries{};
  std::size_t internal_entries{};
  bool complete = false;
  bool valid = true;
  bool allocation_free = false;
};

inline void AddCounter(rund::compute::MemoryCounter &total,
                       const rund::compute::MemoryCounter value) noexcept {
  total.current += value.current;
  total.peak += value.peak;
  total.cumulative += value.cumulative;
  total.reused += value.reused;
  total.budget += value.budget;
}

template <class Owner>
[[nodiscard]] SnapshotAccounting
SnapshotMemory(const Owner &owner, rund::compute::MemoryStats &sum) {
  using namespace rund::compute;
  std::array<MemoryEntry, 128u> entries{};
  node_compute_allocation::Start();
  const MemorySnapshot snapshot = owner.memory_snapshot(entries);
  node_compute_allocation::Stop();
  SnapshotAccounting accounting{
      .complete = !snapshot.truncated() && snapshot.written == snapshot.total,
      .allocation_free = node_compute_allocation::Count() == 0u,
  };
  sum = snapshot.summary;
  for (std::size_t index = 0u; index < snapshot.written; ++index) {
    const MemoryEntry &entry = entries[index];
    const std::size_t category = static_cast<std::size_t>(entry.category);
    if (category >= accounting.categories.size()) {
      accounting.valid = false;
      continue;
    }
    accounting.valid = accounting.valid && ValidCounter(entry.bytes);
    AddCounter(accounting.categories[category], entry.bytes);
    accounting.metadata_entries += entry.use == MemoryUse::Metadata ? 1u : 0u;
    accounting.tile_entries += entry.category == MemoryCategory::Tile &&
                                       entry.use == MemoryUse::Scratch
                                   ? 1u
                                   : 0u;
    accounting.internal_entries +=
        entry.use == MemoryUse::Internal &&
                (entry.category == MemoryCategory::Host ||
                 entry.category == MemoryCategory::Device)
            ? 1u
            : 0u;
  }
  const std::array<MemoryCounter, 7u> expected{
      sum.host,    sum.frame,  sum.tile,    sum.resident,
      sum.staging, sum.device, sum.transfer};
  for (std::size_t index = 0u; index < expected.size(); ++index) {
    accounting.valid =
        accounting.valid &&
        SameCounter(accounting.categories[index], expected[index]);
  }
  return accounting;
}

template <class Owner>
[[nodiscard]] bool ReadMemory(const Owner &owner,
                              rund::compute::MemoryStats &memory) {
  node_compute_allocation::Start();
  memory = owner.memory();
  node_compute_allocation::Stop();
  return node_compute_allocation::Count() == 0u;
}

} // namespace rund_node_memory_contract
