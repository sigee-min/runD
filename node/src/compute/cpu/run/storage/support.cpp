#include "local.hpp"

#include "../../scratch.hpp"

#include <kernel/core/checked.hpp>

#include <limits>

namespace rund::compute::detail {
namespace {

[[nodiscard]] bool to_u64(const std::size_t value,
                          std::uint64_t &result) noexcept {
  if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
    if (value > std::numeric_limits<std::uint64_t>::max()) {
      return false;
    }
  }
  result = static_cast<std::uint64_t>(value);
  return true;
}

[[nodiscard]] bool extent_bytes(const std::size_t count,
                                const std::uint64_t width,
                                std::uint64_t &result) noexcept {
  std::uint64_t count64 = 0u;
  return to_u64(count, count64) && kernel::checked::mul(count64, width, result);
}

[[nodiscard]] bool add_host(CpuStorageBytes &total,
                            const std::uint64_t bytes) noexcept {
  return kernel::checked::add(total.host, bytes, total.host);
}

} // namespace

bool add_cpu_storage_bytes(CpuStorageBytes &total,
                           const CpuStorageBytes value) noexcept {
  return kernel::checked::add(total.host, value.host, total.host) &&
         kernel::checked::add(total.tile, value.tile, total.tile);
}

Result<CpuMapRunPlan> plan_cpu_map_run(const CpuProgram &program) noexcept {
  const kernel::ComputeTileRunStoragePlan tiles =
      program.tile_plan.run_plan().storage_plan();
  if (!tiles) {
    return Result<CpuMapRunPlan>::fail(Reason::TileRunCapacity);
  }
  CpuMapRunPlan plan{
      .program = &program,
      .workers = static_cast<std::size_t>(program.workers),
      .scratch_words = program.scratch_words,
      .tiles = tiles,
      .bytes = CpuStorageBytes{.host = sizeof(CpuMapRun)},
  };
  if (plan.scratch_words != 0u &&
      plan.workers >
          std::numeric_limits<std::size_t>::max() / plan.scratch_words) {
    return Result<CpuMapRunPlan>::fail(Reason::ProgramCapacity);
  }
  plan.scratch_word_count = plan.workers * plan.scratch_words;
  return Result<CpuMapRunPlan>::success(plan);
}

Result<CpuCollectiveRunPlan>
plan_cpu_collective_run(const CpuCollective &program) noexcept {
  const kernel::ComputeTileRunStoragePlan tiles =
      program.tile_plan.run_plan().storage_plan();
  if (!tiles) {
    return Result<CpuCollectiveRunPlan>::fail(Reason::TileRunCapacity);
  }
  CpuCollectiveRunPlan plan{
      .program = &program,
      .tile_count = static_cast<std::size_t>(program.tile_count),
      .needs_prefixes = program.needs_prefixes,
      .tiles = tiles,
      .bytes = CpuStorageBytes{.host = sizeof(CpuCollectiveRun)},
  };
  return Result<CpuCollectiveRunPlan>::success(plan);
}

Result<CpuStorageBytes>
plan_cpu_graph_containers(const std::size_t steps,
                          const std::size_t scratch_slots) noexcept {
  CpuStorageBytes bytes{.host = sizeof(CpuGraphStorage)};
  std::uint64_t maps = 0u;
  std::uint64_t collectives = 0u;
  std::uint64_t scratch = 0u;
  if (!extent_bytes(steps, sizeof(std::size_t), maps) ||
      !extent_bytes(steps, sizeof(std::size_t), collectives) ||
      !extent_bytes(scratch_slots, sizeof(CpuPrimitiveScratch), scratch) ||
      !add_host(bytes, maps) || !add_host(bytes, collectives) ||
      !add_host(bytes, scratch)) {
    return Result<CpuStorageBytes>::fail(Reason::ProgramCapacity);
  }
  return Result<CpuStorageBytes>::success(bytes);
}

} // namespace rund::compute::detail
