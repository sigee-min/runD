#pragma once

#include "../state.hpp"

#include <cstddef>
#include <cstdint>

namespace rund::compute::detail {

// These descriptors are private to storage planning/materialization.  They
// carry the exact immutable Program owner and the arena slice requirements;
// they are not another runtime storage table.
struct CpuMapRunPlan final {
  const CpuProgram *program = nullptr;
  std::size_t workers{};
  std::size_t scratch_words{};
  std::size_t scratch_word_count{};
  kernel::ComputeTileRunStoragePlan tiles{};
  CpuStorageBytes bytes{};
};

struct CpuCollectiveRunPlan final {
  const CpuCollective *program = nullptr;
  std::size_t tile_count{};
  bool needs_prefixes{};
  kernel::ComputeTileRunStoragePlan tiles{};
  CpuStorageBytes bytes{};
};

[[nodiscard]] bool add_cpu_storage_bytes(CpuStorageBytes &total,
                                         const CpuStorageBytes value) noexcept;

[[nodiscard]] Result<CpuMapRunPlan>
plan_cpu_map_run(const CpuProgram &program) noexcept;

[[nodiscard]] Result<CpuCollectiveRunPlan>
plan_cpu_collective_run(const CpuCollective &program) noexcept;

[[nodiscard]] Result<CpuStorageBytes>
plan_cpu_graph_containers(std::size_t steps,
                          std::size_t scratch_slots) noexcept;

} // namespace rund::compute::detail
