#include "local.hpp"

#include "../model.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/job/state.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace rund_node_memory_contract::graph_detail {

int CheckResidentJobMemory(
    rund::compute::Program<std::uint32_t(std::uint32_t)> &program,
    rund::compute::Device &device, const std::uint64_t baseline) {
  using namespace rund::compute;
  constexpr std::uint64_t bytes = FirstInput.size() * sizeof(std::uint32_t);
  constexpr std::uint64_t internal_values = 1u;
  constexpr std::uint64_t external_values = 2u;
  constexpr std::uint64_t input_values = 1u;
  constexpr std::uint64_t output_values = external_values - input_values;
  constexpr std::uint64_t concurrent_jobs = 2u;
  constexpr std::uint64_t cached_run =
      (input_values + output_values + internal_values) * bytes;
  constexpr std::uint64_t per_job =
      (2u * input_values + output_values + internal_values) * bytes;
  constexpr std::uint64_t total_physical =
      cached_run + concurrent_jobs * per_job;

  auto first = program.resident(FirstInput);
  auto second = program.resident(SecondInput);
  if (!first || !second) {
    return 4;
  }
  const std::shared_ptr<detail::JobState> first_state =
      detail::JobAccess::state(*first);
  const std::shared_ptr<detail::JobState> second_state =
      detail::JobAccess::state(*second);
  if (first_state == nullptr || second_state == nullptr) {
    return 14;
  }
  MemoryStats first_memory{};
  MemoryStats second_memory{};
  const bool first_memory_allocation_free = ReadMemory(*first, first_memory);
  const bool second_memory_allocation_free = ReadMemory(*second, second_memory);
  MemoryStats first_snapshot_stats{};
  MemoryStats second_snapshot_stats{};
  const SnapshotAccounting first_snapshot =
      SnapshotMemory(*first, first_snapshot_stats);
  const SnapshotAccounting second_snapshot =
      SnapshotMemory(*second, second_snapshot_stats);
  if (first_state->cpu == nullptr || second_state->cpu == nullptr ||
      !first_memory_allocation_free || !second_memory_allocation_free ||
      !first_snapshot.complete || !second_snapshot.complete ||
      !first_snapshot.valid || !second_snapshot.valid ||
      !first_snapshot.allocation_free || !second_snapshot.allocation_free ||
      first_snapshot.metadata_entries != 1u ||
      second_snapshot.metadata_entries != 1u ||
      first_snapshot.tile_entries != 1u || second_snapshot.tile_entries != 1u ||
      first_snapshot.internal_entries != 1u ||
      second_snapshot.internal_entries != 1u ||
      !SameStats(first_memory, first_snapshot_stats) ||
      !SameStats(second_memory, second_snapshot_stats) ||
      first_memory.tile.current == 0u || second_memory.tile.current == 0u) {
    return 15;
  }
  const PhysicalInternal first_internal = PhysicalInternalMemory(*first);
  const PhysicalInternal second_internal = PhysicalInternalMemory(*second);
  if (!first_internal.complete || !second_internal.complete ||
      first_internal.count != 1u || second_internal.count != 1u ||
      first_internal.bytes != internal_values * bytes ||
      second_internal.bytes != internal_values * bytes ||
      first_memory.resident.current != per_job ||
      second_memory.resident.current != per_job ||
      device.memory().host.current != baseline + total_physical) {
    return 5;
  }
  if (!first->run() || !second->run()) {
    return 6;
  }
  const MemoryStats first_warm_memory = first->memory();
  const MemoryStats second_warm_memory = second->memory();
  if (first_warm_memory.host.current != first_memory.host.current ||
      second_warm_memory.host.current != second_memory.host.current ||
      first_warm_memory.tile.current != first_memory.tile.current ||
      second_warm_memory.tile.current != second_memory.tile.current) {
    return 16;
  }
  const auto first_output = first->read();
  const auto second_output = second->read();
  const Stats first_stats = first->stats();
  const Stats second_stats = second->stats();
  return first_output && second_output &&
                 *first_output == std::vector<std::uint32_t>{2u, 5u, 9u, 14u} &&
                 *second_output ==
                     std::vector<std::uint32_t>{5u, 9u, 12u, 14u} &&
                 first_stats.graph_hash != 0u &&
                 second_stats.graph_hash != 0u &&
                 first_stats.graph_hash == second_stats.graph_hash &&
                 first_stats.output_hash != 0u && second_stats.output_hash != 0u
             ? 0
             : 7;
}

} // namespace rund_node_memory_contract::graph_detail
