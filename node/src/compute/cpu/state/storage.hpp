#pragma once

#include "arena.hpp"
#include "collective.hpp"
#include "map.hpp"
#include "primitive.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rund::compute::detail {

struct CpuGraphProgram;
struct CpuRuntimeGraph;

// Mutable Map/collective/primitive execution storage. A standalone Job owns a
// private instance. A Pipeline owns one instance per Program and shares it
// only across routes proved serial by the Pipeline gate.
struct CpuGraphStorage final {
  const CpuGraphProgram *program = nullptr;
  // Lifetime edge to the one arena whose execution spans are bound below.
  // Route and Job binding slices share this same physical authority.
  std::shared_ptr<CpuPreparedArena> prepared_arena;
  // Program steps address compact dense run arrays through immutable indices.
  // Exact reserve before emplacement removes one heap owner and one pointer
  // chase per Map/collective while preserving O(1) step lookup.
  std::vector<CpuMapRun> maps;
  std::vector<CpuCollectiveRun> collectives;
  std::vector<std::size_t> map_by_step;
  std::vector<std::size_t> collective_by_step;
  std::vector<CpuPrimitiveScratch> scratch;
  CpuPrimitiveScratch empty_scratch{};
};

inline constexpr std::size_t NoCpuGraphStorageIndex = ~std::size_t{0};

[[nodiscard]] inline CpuMapRun *cpu_map_run(CpuGraphStorage &storage,
                                            const std::size_t step) noexcept {
  if (step >= storage.map_by_step.size()) {
    return nullptr;
  }
  const std::size_t index = storage.map_by_step[step];
  return index < storage.maps.size() ? &storage.maps[index] : nullptr;
}

[[nodiscard]] inline const CpuMapRun *
cpu_map_run(const CpuGraphStorage &storage, const std::size_t step) noexcept {
  if (step >= storage.map_by_step.size()) {
    return nullptr;
  }
  const std::size_t index = storage.map_by_step[step];
  return index < storage.maps.size() ? &storage.maps[index] : nullptr;
}

[[nodiscard]] inline CpuCollectiveRun *
cpu_collective_run(CpuGraphStorage &storage, const std::size_t step) noexcept {
  if (step >= storage.collective_by_step.size()) {
    return nullptr;
  }
  const std::size_t index = storage.collective_by_step[step];
  return index < storage.collectives.size() ? &storage.collectives[index]
                                            : nullptr;
}

[[nodiscard]] inline const CpuCollectiveRun *cpu_collective_run(
    const CpuGraphStorage &storage, const std::size_t step) noexcept {
  if (step >= storage.collective_by_step.size()) {
    return nullptr;
  }
  const std::size_t index = storage.collective_by_step[step];
  return index < storage.collectives.size() ? &storage.collectives[index]
                                            : nullptr;
}

struct CpuStorageBytes final {
  std::uint64_t host{};
  std::uint64_t tile{};

  [[nodiscard]] constexpr bool
  operator==(const CpuStorageBytes &) const noexcept = default;
};

// Allocation-free sealed view over one immutable CpuGraphProgram. Per-step
// plans stay with that Program; this descriptor freezes their checked retained
// extent and the exact top-level allocation counts without copying a second
// plan table.
struct CpuGraphStoragePlan final {
  const CpuGraphProgram *program = nullptr;
  const CpuRuntimeGraph *runtime = nullptr;
  std::uint64_t graph_hash{};
  std::size_t step_count{};
  std::size_t map_count{};
  std::size_t collective_count{};
  std::size_t scratch_count{};
  std::size_t scratch_slots{};
  CpuStorageBytes containers{};
  CpuStorageBytes maps{};
  CpuStorageBytes collectives{};
  // Per-Program retained state excludes the Pipeline/Job-owned prepared arena.
  CpuStorageBytes private_total{};
  CpuExecutionStoragePlan execution{};

  [[nodiscard]] constexpr bool
  operator==(const CpuGraphStoragePlan &) const noexcept = default;
};

} // namespace rund::compute::detail
