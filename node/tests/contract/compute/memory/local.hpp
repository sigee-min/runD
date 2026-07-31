#pragma once

#include <rund/compute.hpp>

#include <cstdint>
#include <limits>

namespace rund_node_memory_contract {

inline constexpr std::uint64_t kCounterMaximum =
    std::numeric_limits<std::uint64_t>::max();

[[nodiscard]] bool CheckCounterSaturation();
[[nodiscard]] bool CheckPreparedMemorySnapshot();
[[nodiscard]] bool ValidStats(const rund::compute::MemoryStats &) noexcept;
[[nodiscard]] int CheckCpuProgramOwnerDeltas();
[[nodiscard]] int CheckCpuPrimitiveScratchOwnership();
[[nodiscard]] int CheckCpuCollectiveScratchOwnership();
[[nodiscard]] int CheckValueRouteArena();
[[nodiscard]] int CheckCpuGraphStorageFormula();
[[nodiscard]] int CheckAccelMemory(rund::compute::Backend);
[[nodiscard]] int CheckAccelProgramHostAccounting(rund::compute::Backend);
[[nodiscard]] int CheckVulkanMemoryModel();
[[nodiscard]] int CheckRetainedJobMemory(rund::compute::Backend);
[[nodiscard]] int CheckSortRunMemory(rund::compute::Backend);

} // namespace rund_node_memory_contract
