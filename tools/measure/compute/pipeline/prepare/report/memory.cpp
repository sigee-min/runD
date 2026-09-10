#include "memory.hpp"

#include "format.hpp"

namespace rund::measure::compute::preparation_memory::report {

void PrintMemoryFields(const PreparationMemoryObservation &observed) {
  const auto &memory = observed.memory;
  PrintDecimal(observed.plan_wall_us);
  PrintUnsigned(observed.plan_current_rss_before);
  PrintUnsigned(observed.plan_current_rss_after);
  PrintUnsigned(observed.plan_rss_before);
  PrintUnsigned(observed.plan_rss_after);
  PrintUnsigned(::rund::detail::counter::Delta(observed.plan_rss_before,
                                               observed.plan_rss_after));
  PrintUnsigned(observed.short_budget_checked ? 1u : 0u);
  PrintUnsigned(observed.short_budget_rejected ? 1u : 0u);
  PrintUnsigned(observed.short_budget_no_allocation ? 1u : 0u);
  PrintUnsigned(static_cast<std::uint64_t>(observed.short_budget_reason));
  PrintUnsigned(observed.prepare_attempted ? 1u : 0u);
  PrintUnsigned(observed.prepare_ok ? 1u : 0u);
  PrintUnsigned(observed.plan_frozen ? 1u : 0u);
  PrintUnsigned(observed.memory_contract ? 1u : 0u);
  PrintUnsigned(observed.backend_observed ? 1u : 0u);
  PrintUnsigned(observed.backend_reservation_contract ? 1u : 0u);
  PrintUnsigned(observed.backend_telemetry_contract ? 1u : 0u);
  PrintUnsigned(observed.precise_failure ? 1u : 0u);
  PrintDecimal(observed.prepare_wall_us);
  PrintUnsigned(observed.prepare_current_rss_before);
  PrintUnsigned(observed.prepare_current_rss_after);
  PrintUnsigned(observed.prepare_rss_before);
  PrintUnsigned(observed.prepare_rss_after);
  PrintUnsigned(::rund::detail::counter::Delta(observed.prepare_rss_before,
                                               observed.prepare_rss_after));
  PrintUnsigned(observed.prepare_host_allocation_count);
  PrintUnsigned(observed.prepare_host_allocation_bytes);
  PrintUnsigned(observed.prepare_rss_within_committed_peak ? 1u : 0u);
  PrintUnsigned(memory.host.current);
  PrintUnsigned(memory.host.peak);
  PrintUnsigned(memory.tile.current);
  PrintUnsigned(memory.tile.peak);
  PrintUnsigned(memory.resident.current);
  PrintUnsigned(memory.resident.peak);
  PrintUnsigned(memory.staging.current);
  PrintUnsigned(memory.staging.peak);
  PrintUnsigned(memory.device.current);
  PrintUnsigned(memory.device.peak);
  PrintUnsigned(observed.backend_memory.host.current);
  PrintUnsigned(observed.backend_memory.host.peak);
  PrintUnsigned(observed.backend_memory.device.current);
  PrintUnsigned(observed.backend_memory.device.peak);
  PrintUnsigned(observed.backend_memory.staging.current);
  PrintUnsigned(observed.backend_memory.staging.peak);
}

} // namespace rund::measure::compute::preparation_memory::report
