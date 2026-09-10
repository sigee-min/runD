#include "plan.hpp"

#include "format.hpp"

#include <string_view>

namespace rund::measure::compute::preparation_memory::report {

void PrintPlanFields(const PreparationMemoryObservation &observed) {
  const auto &plan = observed.plan;
  PrintUnsigned(observed.contract ? 1u : 0u);
  PrintSize(Maximum);
  PrintSize(Outer);
  PrintSize(Tile);
  PrintSize(Inner);
  PrintSize(SecondInner);
  PrintSize(OrdinaryIterations);
  PrintSize(SeedScanMapPairs);
  PrintUnsigned(observed.plan_contract ? 1u : 0u);
  PrintUnsigned(plan.logical_bytes);
  PrintUnsigned(plan.live_bytes);
  PrintUnsigned(plan.physical_bytes);
  PrintUnsigned(plan.persistent_bytes);
  PrintUnsigned(plan.state_bytes);
  PrintUnsigned(plan.transient_bytes);
  PrintUnsigned(plan.prepared_bytes);
  PrintUnsigned(plan.prepared_buffer_bytes);
  PrintUnsigned(plan.prepared_host_bytes);
  PrintUnsigned(plan.prepared_tile_bytes);
  PrintUnsigned(plan.prepared_native_bytes);
  PrintUnsigned(plan.scratch_bytes);
  PrintUnsigned(plan.peak_bytes);
  PrintUnsigned(plan.arena_extent_bytes);
  PrintUnsigned(plan.committed_peak_bytes);
  PrintUnsigned(plan.total_bytes);
  PrintUnsigned(plan.allocation_count);
  PrintUnsigned(plan.reuse_count);
  PrintUnsigned(plan.prepared_template_count);
  PrintUnsigned(plan.prepared_command_count);
  PrintUnsigned(plan.node_count);
  PrintUnsigned(plan.resource_count);
  PrintUnsigned(plan.barrier_count);
  PrintUnsigned(plan.largest_bytes);
  PrintSize(plan.largest_step);
  PrintSize(plan.largest_iteration);
  PrintSize(plan.largest_outer_window);
  PrintSize(plan.largest_inner_iteration);
  PrintUnsigned(static_cast<std::uint64_t>(plan.largest_nested_phase));
  PrintSize(plan.largest_chunk);
  PrintUnsigned(observed.retained_group_available ? 1u : 0u);
  PrintCsvField(observed.retained_group_available
                    ? std::string_view{MemoryCategoryName(
                          observed.largest_retained_group.category)}
                    : std::string_view{});
  PrintCsvField(
      observed.retained_group_available
          ? std::string_view{MemoryUseName(observed.largest_retained_group.use)}
          : std::string_view{});
  PrintUnsigned(observed.retained_group_available
                    ? observed.largest_retained_group.index
                    : 0u);
  PrintUnsigned(observed.retained_group_available
                    ? observed.largest_retained_group.bytes.current
                    : 0u);
  PrintCsvField(observed.retained_group_available
                    ? std::string_view{"pipeline"}
                    : std::string_view{"not_materialized"});
  PrintSize(plan.peak_step);
  PrintSize(plan.peak_iteration);
  PrintSize(plan.peak_outer_window);
  PrintSize(plan.peak_inner_iteration);
  PrintUnsigned(static_cast<std::uint64_t>(plan.peak_nested_phase));
}

} // namespace rund::measure::compute::preparation_memory::report
