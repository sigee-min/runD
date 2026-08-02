#include "model.hpp"

#include "../process.hpp"
#include "../suite/core.hpp"

#include "allocation.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/backend.hpp"
#include "src/compute/pipeline/state.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <utility>

namespace rund::measure::compute {
namespace {

constexpr std::size_t Maximum = 516096u;
constexpr std::size_t Tile = 8192u;
constexpr std::size_t Inner = 64u;
constexpr std::size_t SecondInner = 1u;
constexpr std::size_t OrdinaryIterations = 64u;
constexpr std::size_t Outer = Maximum / Tile;
constexpr std::size_t Domain = 64u;
constexpr std::size_t SeedScanMapPairs = 500u;
constexpr std::size_t PreparedTemplates =
    (Outer + 2u + 3u) + 1u + 3u + (Outer + SecondInner + 3u) + 1u;
constexpr std::size_t PreparedCommands = Outer * (Inner + 2u) + 1u +
                                         OrdinaryIterations +
                                         Outer * (SecondInner + 2u) + 1u;

static_assert(Maximum % Tile == 0u);
static_assert(Outer == 63u);
static_assert(PreparedTemplates == 140u);
static_assert(PreparedCommands == 4413u);

struct PreparationMemoryObservation final {
  ::rund::compute::PipelinePlan plan{};
  ::rund::compute::MemoryStats memory{};
  ::rund::compute::MemoryEntry largest_retained_group{};
  ::rund::node::accel::detail::PreparedPipelineMemory backend_memory{};
  ::rund::node::accel::detail::PreparedKernelPipelineReservation
      backend_limit{};
  ::rund::node::accel::detail::PreparedKernelPipelineReservation
      backend_consumed{};
  ::rund::compute::Code code{::rund::compute::Code::Ok};
  ::rund::compute::Reason reason{::rund::compute::Reason::Ok};
  ::rund::compute::Location location{};
  std::string_view error{};
  const char *status{"invalid"};
  double plan_wall_us{};
  std::uint64_t plan_current_rss_before{};
  std::uint64_t plan_current_rss_after{};
  std::uint64_t plan_rss_before{};
  std::uint64_t plan_rss_after{};
  double prepare_wall_us{};
  std::uint64_t prepare_current_rss_before{};
  std::uint64_t prepare_current_rss_after{};
  std::uint64_t prepare_rss_before{};
  std::uint64_t prepare_rss_after{};
  std::uint64_t prepare_host_allocation_count{};
  std::uint64_t prepare_host_allocation_bytes{};
  bool prepare_rss_within_committed_peak{};
  bool plan_contract{};
  bool short_budget_checked{};
  bool short_budget_rejected{};
  bool short_budget_no_allocation{};
  ::rund::compute::Reason short_budget_reason{::rund::compute::Reason::Ok};
  bool prepare_attempted{};
  bool prepare_ok{};
  bool plan_frozen{};
  bool memory_contract{};
  bool backend_observed{};
  bool backend_reservation_contract{};
  bool backend_telemetry_contract{};
  bool retained_group_available{};
  bool precise_failure{};
  bool contract{};
};

template <class T>
void CaptureFailure(PreparationMemoryObservation &observed,
                    const ::rund::compute::Result<T> &failure) noexcept {
  observed.code = failure.code();
  observed.reason = failure.reason();
  observed.location = failure.location();
  observed.error = failure.error();
}

[[nodiscard]] bool PlanContract(const ::rund::compute::PipelinePlan &plan,
                                const Backend backend) noexcept {
  const std::uint64_t prepared_components =
      plan.prepared_buffer_bytes + plan.prepared_host_bytes +
      plan.prepared_tile_bytes + plan.prepared_native_bytes;
  const bool backend_components = backend == Backend::Cpu
                                      ? plan.prepared_host_bytes != 0u &&
                                            plan.prepared_tile_bytes != 0u &&
                                            plan.prepared_native_bytes == 0u
                                      : plan.prepared_native_bytes != 0u;
  return plan.outer_window_count == Outer && plan.tile_capacity == Tile &&
         plan.inner_iteration_count == Inner &&
         plan.prepared_template_count == PreparedTemplates &&
         plan.prepared_command_count == PreparedCommands &&
         plan.node_count >= 2u * SeedScanMapPairs &&
         plan.resource_count != 0u && plan.barrier_count != 0u &&
         plan.allocation_count != 0u && plan.reuse_count != 0u &&
         plan.prepared_bytes == prepared_components &&
         plan.prepared_bytes != 0u &&
         plan.peak_bytes ==
             plan.state_bytes + plan.transient_bytes + plan.prepared_bytes &&
         plan.arena_extent_bytes <= plan.peak_bytes &&
         plan.committed_peak_bytes >= plan.peak_bytes &&
         (backend == Backend::Cpu
              ? plan.arena_extent_bytes != 0u
              : plan.arena_extent_bytes == 0u &&
                    plan.committed_peak_bytes == plan.peak_bytes) &&
         plan.total_bytes == plan.persistent_bytes + plan.peak_bytes &&
         plan.physical_bytes == plan.peak_bytes &&
         plan.logical_bytes > plan.live_bytes &&
         plan.live_bytes <= plan.physical_bytes && backend_components;
}

[[nodiscard]] bool
PreciseFailure(const PreparationMemoryObservation &observed) noexcept {
  return observed.location.native_reason_key != nullptr &&
         observed.location.step != ::rund::compute::Location::none &&
         observed.location.template_index != ::rund::compute::Location::none &&
         ::rund::compute::valid_pipeline_nested_coordinate(
             observed.location.nested_phase,
             observed.location.outer_iteration !=
                 ::rund::compute::Location::none,
             observed.location.inner_iteration !=
                 ::rund::compute::Location::none);
}

[[nodiscard]] constexpr bool
NoAllocation(const ::rund::compute::MemoryCounter before,
             const ::rund::compute::MemoryCounter after) noexcept {
  return after.current <= before.current && after.peak == before.peak &&
         after.cumulative == before.cumulative &&
         after.reused == before.reused && after.budget == before.budget;
}

[[nodiscard]] constexpr bool
NoAllocation(const ::rund::compute::MemoryStats &before,
             const ::rund::compute::MemoryStats &after) noexcept {
  return before.backend == after.backend && before.scope == after.scope &&
         NoAllocation(before.host, after.host) &&
         NoAllocation(before.frame, after.frame) &&
         NoAllocation(before.tile, after.tile) &&
         NoAllocation(before.resident, after.resident) &&
         NoAllocation(before.staging, after.staging) &&
         NoAllocation(before.device, after.device) &&
         NoAllocation(before.transfer, after.transfer);
}

[[nodiscard]] bool
PlanObservationContract(const PreparationMemoryObservation &observed) noexcept {
  return observed.plan_contract && observed.plan_wall_us > 0.0 &&
         observed.plan_current_rss_before != 0u &&
         observed.plan_current_rss_after != 0u &&
         observed.plan_rss_before != 0u &&
         observed.plan_rss_after >= observed.plan_rss_before;
}

[[nodiscard]] bool
PreparedMemoryContract(const ::rund::compute::PipelinePlan &plan,
                       const ::rund::compute::MemoryStats &memory,
                       const Backend backend) noexcept {
  using namespace ::rund::compute;
  if (!memory.available() || memory.backend != backend ||
      memory.scope != MemoryScope::Pipeline) {
    return false;
  }

  const std::uint64_t planned_resident =
      plan.state_bytes + plan.transient_bytes + plan.prepared_buffer_bytes;
  const std::uint64_t planned_host =
      planned_resident + plan.prepared_host_bytes;
  if (memory.resident.current > plan.peak_bytes) {
    return false;
  }
  if (backend == Backend::Cpu) {
    return plan.prepared_native_bytes == 0u &&
           memory.resident.current == planned_resident &&
           memory.host.current <= planned_host &&
           memory.tile.current == plan.prepared_tile_bytes &&
           memory.host.current <= plan.peak_bytes &&
           memory.tile.current <= plan.peak_bytes - memory.host.current;
  }

  // Accelerator native reservations cover runD-owned command/descriptor
  // storage. Driver allocation granularity is deliberately telemetry-only,
  // so require a frozen nonzero reservation and observable native storage,
  // without claiming byte equality the backend cannot guarantee.
  return plan.prepared_native_bytes != 0u && memory.host.current != 0u &&
         (memory.device.current != 0u || memory.staging.current != 0u);
}

[[nodiscard]] constexpr const char *
MemoryCategoryName(const ::rund::compute::MemoryCategory category) noexcept {
  using ::rund::compute::MemoryCategory;
  switch (category) {
  case MemoryCategory::Host:
    return "host";
  case MemoryCategory::Frame:
    return "frame";
  case MemoryCategory::Tile:
    return "tile";
  case MemoryCategory::Resident:
    return "resident";
  case MemoryCategory::Staging:
    return "staging";
  case MemoryCategory::Device:
    return "device";
  case MemoryCategory::Transfer:
    return "transfer";
  }
  return "unknown";
}

[[nodiscard]] constexpr const char *
MemoryUseName(const ::rund::compute::MemoryUse use) noexcept {
  using ::rund::compute::MemoryUse;
  switch (use) {
  case MemoryUse::Metadata:
    return "metadata";
  case MemoryUse::Input:
    return "input";
  case MemoryUse::PendingInput:
    return "pending_input";
  case MemoryUse::Output:
    return "output";
  case MemoryUse::Internal:
    return "internal";
  case MemoryUse::Scratch:
    return "scratch";
  case MemoryUse::Coordinator:
    return "coordinator";
  case MemoryUse::Traffic:
    return "traffic";
  }
  return "unknown";
}

[[nodiscard]] bool CaptureLargestRetainedGroup(
    PreparationMemoryObservation &observed,
    const ::rund::compute::Pipeline &pipeline) noexcept {
  using namespace ::rund::compute;
  std::array<MemoryEntry, 16u> entries{};
  const MemorySnapshot snapshot = pipeline.memory_snapshot(entries);
  if (snapshot.summary.scope != MemoryScope::Pipeline || snapshot.truncated()) {
    return false;
  }
  const MemoryEntry *largest = nullptr;
  for (std::size_t index = 0u; index < snapshot.written; ++index) {
    const MemoryEntry &entry = entries[index];
    if (entry.bytes.current != 0u &&
        (largest == nullptr || entry.bytes.current > largest->bytes.current)) {
      largest = &entry;
    }
  }
  if (largest == nullptr) {
    return false;
  }
  observed.largest_retained_group = *largest;
  observed.retained_group_available = true;
  return true;
}

[[nodiscard]] constexpr bool
Contains(const ::rund::compute::MemoryCounter &total,
         const ::rund::node::accel::detail::PreparedMemory part) noexcept {
  return part.current <= total.current && part.peak <= total.peak &&
         part.cumulative <= total.cumulative && part.reused <= total.reused &&
         part.budget <= total.budget;
}

[[nodiscard]] bool
CaptureBackendPreparation(PreparationMemoryObservation &observed,
                          const ::rund::compute::Pipeline &pipeline,
                          const Backend backend) noexcept {
  using namespace ::rund::compute;
  if (backend == Backend::Cpu) {
    observed.backend_reservation_contract =
        observed.plan.prepared_native_bytes == 0u;
    observed.backend_telemetry_contract = true;
    return observed.backend_reservation_contract;
  }

  const std::shared_ptr<::rund::compute::detail::PipelineState> &state =
      ::rund::compute::detail::PipelineStateAccess::state(pipeline);
  if (state == nullptr || state->device == nullptr ||
      state->device->ops == nullptr ||
      state->device->ops->pipeline_memory == nullptr) {
    return false;
  }

  observed.backend_observed = true;
  observed.backend_memory = state->device->ops->pipeline_memory(*state);
  observed.backend_limit = state->accel_templates.limit;
  observed.backend_consumed = state->accel_templates.reservation;
  const auto &recurrence_limit = observed.backend_limit.map_recurrence;
  const auto &recurrence_consumed = observed.backend_consumed.map_recurrence;
  const bool recurrence_reservation_contract =
      recurrence_limit.terminal_template_group_capacity <=
          recurrence_limit.group_count &&
      recurrence_limit.history_template_group_capacity ==
          recurrence_limit.group_count -
              recurrence_limit.terminal_template_group_capacity &&
      recurrence_limit.group_count == recurrence_consumed.group_count &&
      recurrence_limit.history_group_count ==
          recurrence_consumed.history_group_count &&
      recurrence_limit.template_count == recurrence_consumed.template_count &&
      recurrence_limit.terminal_template_group_capacity ==
          recurrence_consumed.terminal_template_group_capacity &&
      recurrence_limit.history_template_group_capacity ==
          recurrence_consumed.history_template_group_capacity &&
      recurrence_limit.descriptor_set_count ==
          recurrence_consumed.descriptor_set_count &&
      recurrence_limit.descriptor_count ==
          recurrence_consumed.descriptor_count &&
      recurrence_limit.template_native_allocation_count ==
          recurrence_consumed.template_native_allocation_count;
  const bool backend_structure_contract =
      observed.backend_limit.occurrence_count ==
          observed.backend_consumed.occurrence_count &&
      observed.backend_limit.backend_dispatch_count ==
          observed.backend_consumed.backend_dispatch_count &&
      observed.backend_limit.backend_reset_dispatch_count ==
          observed.backend_consumed.backend_reset_dispatch_count &&
      observed.backend_limit.backend_window_dispatch_count ==
          observed.backend_consumed.backend_window_dispatch_count &&
      observed.backend_limit.backend_indirect_dispatch_count ==
          observed.backend_consumed.backend_indirect_dispatch_count &&
      observed.backend_limit.backend_step_occurrence_count ==
          observed.backend_consumed.backend_step_occurrence_count &&
      observed.backend_limit.backend_status_source_count ==
          observed.backend_consumed.backend_status_source_count &&
      observed.backend_limit.backend_status_entry_count ==
          observed.backend_consumed.backend_status_entry_count &&
      observed.backend_limit.backend_telemetry_count ==
          observed.backend_consumed.backend_telemetry_count &&
      observed.backend_limit.backend_status_command_count ==
          observed.backend_consumed.backend_status_command_count &&
      observed.backend_limit.backend_telemetry_command_count ==
          observed.backend_consumed.backend_telemetry_command_count &&
      observed.backend_limit.backend_window_control_command_count ==
          observed.backend_consumed.backend_window_control_command_count &&
      observed.backend_limit.backend_publication_command_count ==
          observed.backend_consumed.backend_publication_command_count &&
      observed.backend_consumed.backend_command_chunk_count <=
          observed.backend_limit.backend_command_chunk_count &&
      observed.backend_consumed.backend_command_native_bytes <=
          observed.backend_limit.backend_command_native_bytes &&
      observed.backend_limit.backend_parameter_bytes ==
          observed.backend_consumed.backend_parameter_bytes &&
      observed.backend_limit.backend_step_description_count ==
          observed.backend_consumed.backend_step_description_count;
  observed.backend_reservation_contract =
      observed.backend_limit.ok && observed.backend_consumed.ok &&
      observed.backend_limit.fingerprint_hi != 0u &&
      observed.backend_limit.fingerprint_lo != 0u &&
      observed.backend_limit.host_bytes <= observed.plan.prepared_host_bytes &&
      observed.backend_limit.native_bytes ==
          observed.plan.prepared_native_bytes &&
      recurrence_reservation_contract && backend_structure_contract &&
      ::rund::node::accel::detail::PreparedKernelPipelineReservationWithin(
          observed.backend_consumed, observed.backend_limit);

  // Pipeline-private Run resources and opaque driver allocation granularity
  // are reported by the complete public counters. The backend bridge is the
  // exact primary/alternate/registry subset, so it must reconcile into those
  // counters without pretending physical Device bytes equal logical native
  // requests.
  observed.backend_telemetry_contract =
      observed.backend_memory.host.current <=
          observed.backend_limit.host_bytes &&
      observed.backend_memory.host.peak <= observed.backend_limit.host_bytes &&
      Contains(observed.memory.host, observed.backend_memory.host) &&
      Contains(observed.memory.device, observed.backend_memory.device) &&
      Contains(observed.memory.staging, observed.backend_memory.staging) &&
      (observed.backend_memory.device.current != 0u ||
       observed.backend_memory.staging.current != 0u);
  return observed.backend_reservation_contract &&
         observed.backend_telemetry_contract;
}

void PrintUnsigned(const std::uint64_t value) {
  std::printf(",%llu", static_cast<unsigned long long>(value));
}

void PrintSize(const std::size_t value) {
  PrintUnsigned(static_cast<std::uint64_t>(value));
}

void PrintObservation(const Backend backend, const bool materialize,
                      const PreparationMemoryObservation &observed) {
  const auto &plan = observed.plan;
  const auto &memory = observed.memory;
  const auto &location = observed.location;
  std::printf("prepare_memory,%s,%s,%s", Name(backend),
              materialize ? "materialize" : "plan_only", observed.status);
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
  std::putchar(',');
  PrintCsv(observed.retained_group_available
               ? std::string_view{MemoryCategoryName(
                     observed.largest_retained_group.category)}
               : std::string_view{});
  std::putchar(',');
  PrintCsv(
      observed.retained_group_available
          ? std::string_view{MemoryUseName(observed.largest_retained_group.use)}
          : std::string_view{});
  PrintUnsigned(observed.retained_group_available
                    ? observed.largest_retained_group.index
                    : 0u);
  PrintUnsigned(observed.retained_group_available
                    ? observed.largest_retained_group.bytes.current
                    : 0u);
  std::putchar(',');
  PrintCsv(observed.retained_group_available
               ? std::string_view{"pipeline"}
               : std::string_view{"not_materialized"});
  PrintSize(plan.peak_step);
  PrintSize(plan.peak_iteration);
  PrintSize(plan.peak_outer_window);
  PrintSize(plan.peak_inner_iteration);
  PrintUnsigned(static_cast<std::uint64_t>(plan.peak_nested_phase));
  std::printf(",%.3f", observed.plan_wall_us);
  PrintUnsigned(observed.plan_current_rss_before);
  PrintUnsigned(observed.plan_current_rss_after);
  PrintUnsigned(observed.plan_rss_before);
  PrintUnsigned(observed.plan_rss_after);
  PrintUnsigned(
      NonnegativeDelta(observed.plan_rss_after, observed.plan_rss_before));
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
  std::printf(",%.3f", observed.prepare_wall_us);
  PrintUnsigned(observed.prepare_current_rss_before);
  PrintUnsigned(observed.prepare_current_rss_after);
  PrintUnsigned(observed.prepare_rss_before);
  PrintUnsigned(observed.prepare_rss_after);
  PrintUnsigned(NonnegativeDelta(observed.prepare_rss_after,
                                 observed.prepare_rss_before));
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
  PrintUnsigned(observed.backend_limit.fingerprint_hi);
  PrintUnsigned(observed.backend_limit.fingerprint_lo);
  PrintUnsigned(observed.backend_limit.host_bytes);
  PrintUnsigned(observed.backend_consumed.host_bytes);
  PrintUnsigned(observed.backend_limit.native_bytes);
  PrintUnsigned(observed.backend_consumed.native_bytes);
  PrintUnsigned(observed.backend_limit.template_source_bytes);
  PrintUnsigned(observed.backend_consumed.template_source_bytes);
  PrintUnsigned(observed.backend_limit.source_transient_bytes);
  PrintUnsigned(observed.backend_consumed.source_transient_bytes);
  PrintUnsigned(observed.backend_limit.host_transient_bytes);
  PrintUnsigned(observed.backend_consumed.host_transient_bytes);
  PrintUnsigned(observed.backend_limit.map_recurrence.group_count);
  PrintUnsigned(observed.backend_consumed.map_recurrence.group_count);
  PrintUnsigned(observed.backend_limit.map_recurrence.history_group_count);
  PrintUnsigned(observed.backend_consumed.map_recurrence.history_group_count);
  PrintUnsigned(observed.backend_limit.map_recurrence.template_count);
  PrintUnsigned(observed.backend_consumed.map_recurrence.template_count);
  PrintUnsigned(
      observed.backend_limit.map_recurrence.terminal_template_group_capacity);
  PrintUnsigned(observed.backend_consumed.map_recurrence
                    .terminal_template_group_capacity);
  PrintUnsigned(
      observed.backend_limit.map_recurrence.history_template_group_capacity);
  PrintUnsigned(
      observed.backend_consumed.map_recurrence.history_template_group_capacity);
  PrintUnsigned(observed.backend_limit.map_recurrence.route_host_bytes);
  PrintUnsigned(observed.backend_consumed.map_recurrence.route_host_bytes);
  PrintUnsigned(observed.backend_limit.map_recurrence.template_host_bytes);
  PrintUnsigned(observed.backend_consumed.map_recurrence.template_host_bytes);
  PrintUnsigned(observed.backend_limit.map_recurrence.template_source_bytes);
  PrintUnsigned(observed.backend_consumed.map_recurrence.template_source_bytes);
  PrintUnsigned(observed.backend_limit.map_recurrence.descriptor_set_count);
  PrintUnsigned(observed.backend_consumed.map_recurrence.descriptor_set_count);
  PrintUnsigned(observed.backend_limit.map_recurrence.descriptor_count);
  PrintUnsigned(observed.backend_consumed.map_recurrence.descriptor_count);
  PrintUnsigned(
      observed.backend_limit.map_recurrence.route_native_allocation_count);
  PrintUnsigned(
      observed.backend_consumed.map_recurrence.route_native_allocation_count);
  PrintUnsigned(
      observed.backend_limit.map_recurrence.template_native_allocation_count);
  PrintUnsigned(observed.backend_consumed.map_recurrence
                    .template_native_allocation_count);
  PrintUnsigned(observed.backend_limit.route_count);
  PrintUnsigned(observed.backend_consumed.route_count);
  PrintUnsigned(observed.backend_limit.template_count);
  PrintUnsigned(observed.backend_consumed.template_count);
  PrintUnsigned(observed.backend_limit.occurrence_count);
  PrintUnsigned(observed.backend_consumed.occurrence_count);
  PrintUnsigned(observed.backend_limit.backend_dispatch_count);
  PrintUnsigned(observed.backend_consumed.backend_dispatch_count);
  PrintUnsigned(observed.backend_limit.backend_reset_dispatch_count);
  PrintUnsigned(observed.backend_consumed.backend_reset_dispatch_count);
  PrintUnsigned(observed.backend_limit.backend_window_dispatch_count);
  PrintUnsigned(observed.backend_consumed.backend_window_dispatch_count);
  PrintUnsigned(observed.backend_limit.backend_indirect_dispatch_count);
  PrintUnsigned(observed.backend_consumed.backend_indirect_dispatch_count);
  PrintUnsigned(observed.backend_limit.backend_step_occurrence_count);
  PrintUnsigned(observed.backend_consumed.backend_step_occurrence_count);
  PrintUnsigned(observed.backend_limit.backend_command_count);
  PrintUnsigned(observed.backend_consumed.backend_command_count);
  PrintUnsigned(observed.backend_limit.backend_command_chunk_count);
  PrintUnsigned(observed.backend_consumed.backend_command_chunk_count);
  PrintUnsigned(observed.backend_limit.backend_command_native_bytes);
  PrintUnsigned(observed.backend_consumed.backend_command_native_bytes);
  PrintUnsigned(observed.backend_limit.backend_step_description_count);
  PrintUnsigned(observed.backend_consumed.backend_step_description_count);
  PrintUnsigned(observed.backend_limit.backend_status_source_count);
  PrintUnsigned(observed.backend_consumed.backend_status_source_count);
  PrintUnsigned(observed.backend_limit.backend_status_entry_count);
  PrintUnsigned(observed.backend_consumed.backend_status_entry_count);
  PrintUnsigned(observed.backend_limit.backend_telemetry_count);
  PrintUnsigned(observed.backend_consumed.backend_telemetry_count);
  PrintUnsigned(observed.backend_limit.backend_status_command_count);
  PrintUnsigned(observed.backend_consumed.backend_status_command_count);
  PrintUnsigned(observed.backend_limit.backend_telemetry_command_count);
  PrintUnsigned(observed.backend_consumed.backend_telemetry_command_count);
  PrintUnsigned(observed.backend_limit.backend_window_control_command_count);
  PrintUnsigned(observed.backend_consumed.backend_window_control_command_count);
  PrintUnsigned(observed.backend_limit.backend_publication_command_count);
  PrintUnsigned(observed.backend_consumed.backend_publication_command_count);
  PrintUnsigned(observed.backend_limit.backend_parameter_bytes);
  PrintUnsigned(observed.backend_consumed.backend_parameter_bytes);
  PrintUnsigned(observed.backend_limit.descriptor_set_count);
  PrintUnsigned(observed.backend_consumed.descriptor_set_count);
  PrintUnsigned(observed.backend_limit.descriptor_count);
  PrintUnsigned(observed.backend_consumed.descriptor_count);
  PrintUnsigned(observed.backend_limit.native_allocation_count);
  PrintUnsigned(observed.backend_consumed.native_allocation_count);
  PrintUnsigned(observed.backend_limit.backend_native_object_count);
  PrintUnsigned(observed.backend_consumed.backend_native_object_count);
  PrintUnsigned(static_cast<std::uint64_t>(observed.code));
  PrintUnsigned(static_cast<std::uint64_t>(observed.reason));
  std::putchar(',');
  PrintCsv(observed.error);
  PrintUnsigned(location.known() ? 1u : 0u);
  PrintUnsigned(location.step);
  PrintUnsigned(location.iteration);
  PrintUnsigned(location.node);
  PrintUnsigned(location.template_index);
  PrintUnsigned(location.occurrence_index);
  PrintUnsigned(location.outer_iteration);
  PrintUnsigned(location.inner_iteration);
  PrintUnsigned(static_cast<std::uint64_t>(location.nested_phase));
  std::putchar(',');
  PrintCsv(location.native_reason_key == nullptr
               ? std::string_view{}
               : std::string_view{location.native_reason_key});
  std::putchar('\n');
}

template <std::size_t Max, std::size_t Width>
[[nodiscard]] auto LargeSeedProgram(::rund::compute::Device &device) {
  using namespace ::rund::compute;
  return on(device)
      .input<std::uint32_t>(Max)
      .zip_input<std::uint32_t>(Domain)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto queue, auto domain, auto total, auto ordinal) {
        auto current = resident<Max, Width>(total, ordinal);
        auto active = queue.gather(current.items());
        auto values = domain.gather(active);
        for (std::size_t pass = 0u; pass < SeedScanMapPairs; ++pass) {
          values = values.scan(pass % 2u == 0u ? Scan::InclusiveSum
                                               : Scan::ExclusiveSum);
          values = values.map("measure-prepare-memory-seed-map",
                              [](auto value) { return value + 1u; });
        }
        return outputs(values.reduce(Reduce::Sum), values);
      })
      .compile();
}

[[nodiscard]] auto ActionProgram(::rund::compute::Device &device) {
  using namespace ::rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(Tile)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto value, auto window, auto window_count) {
        auto next = value.map("measure-prepare-memory-action",
                              [](auto item) { return item + 1u; });
        auto retained = window.map("measure-prepare-memory-window-retain",
                                   [](auto item) { return item; });
        auto retained_count =
            window_count.map("measure-prepare-memory-window-count",
                             [](auto item) { return item; });
        return outputs(next, retained, retained_count);
      })
      .compile();
}

[[nodiscard]] auto FoldProgram(::rund::compute::Device &device) {
  using namespace ::rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(Tile)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto outer, auto tile, auto window, auto window_count) {
        (void)window_count;
        auto next =
            outer.combine("measure-prepare-memory-fold", tile,
                          [](auto left, auto right) { return left + right; });
        auto published = window.map("measure-prepare-memory-window-publish",
                                    [](auto item) { return item; });
        return outputs(next, published);
      })
      .compile();
}

template <std::size_t Max, std::size_t Width>
[[nodiscard]] auto SecondSeedProgram(::rund::compute::Device &device) {
  using namespace ::rund::compute;
  return on(device)
      .input<std::uint32_t>(Max)
      .zip_input<std::uint32_t>(Domain)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto queue, auto domain, auto total, auto ordinal) {
        auto current = resident<Max, Width>(total, ordinal);
        auto active = queue.gather(current.items());
        auto values = domain.gather(active);
        for (std::size_t pass = 0u; pass < SeedScanMapPairs; ++pass) {
          values = values.scan(pass % 2u == 0u ? Scan::InclusiveSum
                                               : Scan::ExclusiveSum);
          values = values.map("measure-prepare-memory-second-seed-map",
                              [](auto value) { return value + 1u; });
        }
        return values.reduce(Reduce::Sum);
      })
      .compile();
}

[[nodiscard]] auto SecondActionProgram(::rund::compute::Device &device) {
  using namespace ::rund::compute;
  return on(device)
      .map<std::uint32_t>("measure-prepare-memory-second-action", 1u,
                          [](auto value) { return value + 1u; })
      .compile();
}

[[nodiscard]] auto SecondFoldProgram(::rund::compute::Device &device) {
  using namespace ::rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto outer, auto tile) {
        return outer.combine(
            "measure-prepare-memory-second-fold", tile,
            [](auto left, auto right) { return left + right; });
      })
      .compile();
}

[[nodiscard]] auto ConsumeProgram(::rund::compute::Device &device) {
  using namespace ::rund::compute;
  return on(device)
      .map<std::uint32_t>("measure-prepare-memory-consume", Maximum,
                          [](auto value) { return value; })
      .compile();
}

[[nodiscard]] auto OrdinaryRecurrenceProgram(::rund::compute::Device &device) {
  using namespace ::rund::compute;
  return on(device)
      .map<std::uint32_t>("measure-prepare-memory-recurrence", Maximum,
                          [](auto value) { return value + 1u; })
      .compile();
}

[[nodiscard]] auto PublishProgram(::rund::compute::Device &device) {
  using namespace ::rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto value, auto published) {
        return value.combine("measure-prepare-memory-publish", published,
                             [](auto next, auto) { return next; });
      })
      .compile();
}

} // namespace

void PrintPreparationMemoryColumns() {
  std::fputs(
      "prepare_memory_columns,backend,mode,status,contract_ok,maximum,"
      "outer_windows,tile,inner_iterations,second_inner_iterations,"
      "ordinary_iterations,seed_scan_map_pairs,plan_contract,"
      "logical_bytes,live_bytes,physical_bytes,persistent_bytes,state_bytes,"
      "transient_bytes,prepared_bytes,prepared_buffer_bytes,"
      "prepared_host_bytes,prepared_tile_bytes,prepared_native_bytes,"
      "scratch_bytes,peak_bytes,arena_extent_bytes,committed_peak_bytes,"
      "total_bytes,allocation_count,reuse_count,"
      "template_count,command_count,node_count,resource_count,barrier_count,"
      "largest_bytes,largest_step,largest_iteration,largest_outer_window,"
      "largest_inner_iteration,largest_phase,largest_chunk,"
      "largest_retained_group_available,largest_retained_group_category,"
      "largest_retained_group_use,largest_retained_group_index,"
      "largest_retained_group_current_bytes,largest_retained_group_lifetime,"
      "peak_step,peak_iteration,peak_outer_window,peak_inner_iteration,"
      "peak_phase,"
      "plan_wall_us,process_current_rss_plan_before_bytes,"
      "process_current_rss_plan_after_bytes,"
      "process_max_rss_plan_before_bytes,"
      "process_max_rss_plan_after_bytes,process_max_rss_plan_delta_bytes,"
      "short_budget_checked,short_budget_rejected,"
      "short_budget_no_allocation,short_budget_reason,prepare_attempted,"
      "prepare_ok,plan_frozen,memory_contract,backend_observed,"
      "backend_reservation_contract,backend_telemetry_contract,"
      "precise_failure,"
      "prepare_wall_us,process_current_rss_prepare_before_bytes,"
      "process_current_rss_prepare_after_bytes,"
      "process_max_rss_prepare_before_bytes,"
      "process_max_rss_prepare_after_bytes,"
      "process_max_rss_prepare_delta_bytes,"
      "prepare_host_allocation_count,prepare_host_allocation_bytes,"
      "process_max_rss_prepare_within_committed_peak,"
      "pipeline_host_current_bytes,pipeline_host_peak_bytes,"
      "pipeline_tile_current_bytes,pipeline_tile_peak_bytes,"
      "pipeline_resident_current_bytes,pipeline_resident_peak_bytes,"
      "pipeline_staging_current_bytes,pipeline_staging_peak_bytes,"
      "pipeline_device_current_bytes,pipeline_device_peak_bytes,"
      "backend_host_current_bytes,backend_host_peak_bytes,"
      "backend_device_current_bytes,backend_device_peak_bytes,"
      "backend_staging_current_bytes,backend_staging_peak_bytes,"
      "backend_limit_fingerprint_hi,backend_limit_fingerprint_lo,"
      "backend_limit_host_bytes,backend_consumed_host_bytes,"
      "backend_limit_native_bytes,backend_consumed_native_bytes,"
      "backend_limit_source_bytes,backend_consumed_source_bytes,"
      "backend_limit_source_transient_bytes,"
      "backend_consumed_source_transient_bytes,"
      "backend_limit_host_transient_bytes,"
      "backend_consumed_host_transient_bytes,"
      "backend_limit_recurrence_groups,backend_consumed_recurrence_groups,"
      "backend_limit_recurrence_history_groups,"
      "backend_consumed_recurrence_history_groups,"
      "backend_limit_recurrence_templates,"
      "backend_consumed_recurrence_templates,"
      "backend_limit_recurrence_terminal_template_group_capacity,"
      "backend_consumed_recurrence_terminal_template_group_capacity,"
      "backend_limit_recurrence_history_template_group_capacity,"
      "backend_consumed_recurrence_history_template_group_capacity,"
      "backend_limit_recurrence_route_host_bytes,"
      "backend_consumed_recurrence_route_host_bytes,"
      "backend_limit_recurrence_template_host_bytes,"
      "backend_consumed_recurrence_template_host_bytes,"
      "backend_limit_recurrence_source_bytes,"
      "backend_consumed_recurrence_source_bytes,"
      "backend_limit_recurrence_descriptor_sets,"
      "backend_consumed_recurrence_descriptor_sets,"
      "backend_limit_recurrence_descriptors,"
      "backend_consumed_recurrence_descriptors,"
      "backend_limit_recurrence_route_native_allocations,"
      "backend_consumed_recurrence_route_native_allocations,"
      "backend_limit_recurrence_template_native_allocations,"
      "backend_consumed_recurrence_template_native_allocations,"
      "backend_limit_routes,"
      "backend_consumed_routes,backend_limit_templates,"
      "backend_consumed_templates,backend_limit_occurrences,"
      "backend_consumed_occurrences,backend_limit_dispatches,"
      "backend_consumed_dispatches,backend_limit_reset_dispatches,"
      "backend_consumed_reset_dispatches,backend_limit_window_dispatches,"
      "backend_consumed_window_dispatches,backend_limit_indirect_dispatches,"
      "backend_consumed_indirect_dispatches,backend_limit_step_occurrences,"
      "backend_consumed_step_occurrences,backend_limit_commands,"
      "backend_consumed_commands,backend_limit_command_chunks,"
      "backend_consumed_command_chunks,backend_limit_command_native_bytes,"
      "backend_consumed_command_native_bytes,"
      "backend_limit_description_steps,"
      "backend_consumed_description_steps,backend_limit_status_sources,"
      "backend_consumed_status_sources,backend_limit_status_entries,"
      "backend_consumed_status_entries,backend_limit_telemetry_descriptions,"
      "backend_consumed_telemetry_descriptions,"
      "backend_limit_status_commands,backend_consumed_status_commands,"
      "backend_limit_telemetry_commands,"
      "backend_consumed_telemetry_commands,"
      "backend_limit_window_control_commands,"
      "backend_consumed_window_control_commands,"
      "backend_limit_publication_commands,"
      "backend_consumed_publication_commands,backend_limit_parameter_bytes,"
      "backend_consumed_parameter_bytes,backend_limit_descriptor_sets,"
      "backend_consumed_descriptor_sets,backend_limit_descriptors,"
      "backend_consumed_descriptors,backend_limit_native_allocations,"
      "backend_consumed_native_allocations,backend_limit_native_objects,"
      "backend_consumed_native_objects,prepare_code,"
      "prepare_reason,prepare_error,location_known,location_step,"
      "location_iteration,location_node,location_template,"
      "location_occurrence,location_outer,location_inner,location_phase,"
      "native_reason_key\n",
      stdout);
}

bool MeasurePreparationMemory(const Backend backend, const bool materialize) {
  using namespace ::rund::compute;
  PreparationMemoryObservation observed{};
  auto device = open(TargetFor(backend));
  if (!device) {
    CaptureFailure(observed, device);
    observed.status =
        observed.code == Code::Unavailable ? "unavailable" : "open_failed";
    observed.contract = observed.code == Code::Unavailable;
    PrintObservation(backend, materialize, observed);
    return observed.contract;
  }

  auto seed = LargeSeedProgram<Maximum, Tile>(*device);
  auto action = ActionProgram(*device);
  auto fold = FoldProgram(*device);
  auto second_seed = SecondSeedProgram<Maximum, Tile>(*device);
  auto second_action = SecondActionProgram(*device);
  auto second_fold = SecondFoldProgram(*device);
  auto consume = ConsumeProgram(*device);
  auto recurrence = OrdinaryRecurrenceProgram(*device);
  auto publish = PublishProgram(*device);
  if (!seed || !action || !fold || !second_seed || !second_action ||
      !second_fold || !consume || !recurrence || !publish) {
    if (!seed) {
      CaptureFailure(observed, seed);
      observed.status = "seed_compile_failed";
    } else if (!action) {
      CaptureFailure(observed, action);
      observed.status = "action_compile_failed";
    } else if (!fold) {
      CaptureFailure(observed, fold);
      observed.status = "fold_compile_failed";
    } else if (!second_seed) {
      CaptureFailure(observed, second_seed);
      observed.status = "second_seed_compile_failed";
    } else if (!second_action) {
      CaptureFailure(observed, second_action);
      observed.status = "second_action_compile_failed";
    } else if (!second_fold) {
      CaptureFailure(observed, second_fold);
      observed.status = "second_fold_compile_failed";
    } else if (!consume) {
      CaptureFailure(observed, consume);
      observed.status = "consume_compile_failed";
    } else if (!recurrence) {
      CaptureFailure(observed, recurrence);
      observed.status = "recurrence_compile_failed";
    } else {
      CaptureFailure(observed, publish);
      observed.status = "publish_compile_failed";
    }
    observed.contract = observed.code == Code::Unavailable;
    PrintObservation(backend, materialize, observed);
    return observed.contract;
  }

  auto outer = device->buffer<std::uint32_t>(1u);
  auto queue = device->buffer<std::uint32_t>(Maximum);
  auto domain = device->buffer<std::uint32_t>(Domain);
  auto count = device->buffer<std::uint32_t>(1u);
  auto first_output = device->buffer<std::uint32_t>(1u);
  auto first_window = device->buffer<std::uint32_t>(Maximum);
  auto consumed = device->buffer<std::uint32_t>(Maximum);
  auto recurrence_output = device->buffer<std::uint32_t>(Maximum);
  auto second_output = device->buffer<std::uint32_t>(1u);
  auto published_result = device->buffer<std::uint32_t>(1u);
  auto pending_result = device->buffer<std::uint32_t>(1u);
  if (!outer || !queue || !domain || !count || !first_output || !first_window ||
      !consumed || !recurrence_output || !second_output || !published_result ||
      !pending_result) {
    if (!outer) {
      CaptureFailure(observed, outer);
    } else if (!queue) {
      CaptureFailure(observed, queue);
    } else if (!domain) {
      CaptureFailure(observed, domain);
    } else if (!count) {
      CaptureFailure(observed, count);
    } else if (!first_output) {
      CaptureFailure(observed, first_output);
    } else if (!first_window) {
      CaptureFailure(observed, first_window);
    } else if (!consumed) {
      CaptureFailure(observed, consumed);
    } else if (!recurrence_output) {
      CaptureFailure(observed, recurrence_output);
    } else if (!second_output) {
      CaptureFailure(observed, second_output);
    } else if (!published_result) {
      CaptureFailure(observed, published_result);
    } else {
      CaptureFailure(observed, pending_result);
    }
    observed.status = observed.code == Code::Unavailable
                          ? "unavailable"
                          : "buffer_setup_failed";
    observed.contract = observed.code == Code::Unavailable;
    PrintObservation(backend, materialize, observed);
    return observed.contract;
  }

  const auto first_body = tile_repeat<Inner>(*seed, *action, *fold);
  const auto second_body =
      tile_repeat<SecondInner>(*second_seed, *second_action, *second_fold);
  const auto make_builder = [&]() {
    auto candidate = pipeline(*device);
    candidate.state(*published_result, *pending_result)
        .windows<Maximum, Tile>(
            first_body, window(*count), read(*outer, *queue, *domain),
            write_final(*first_output), write_window(*first_window))
        .then(*consume, read(*first_window), write(*consumed))
        .repeat<OrdinaryIterations>(*recurrence, read(*consumed),
                                    write_final(*recurrence_output))
        .windows<Maximum, Tile>(
            second_body, window(*count),
            read(*first_output, *recurrence_output, *domain),
            write_final(*second_output))
        .then(*publish, read(*second_output, *published_result),
              write(*pending_result))
        .commit();
    return candidate;
  };
  auto builder = make_builder();
  observed.plan_current_rss_before = ProcessCurrentResidentBytes();
  observed.plan_rss_before = ProcessMaximumResidentBytes();
  const auto plan_begin = Clock::now();
  const auto planned = builder.plan();
  const auto plan_end = Clock::now();
  observed.plan_current_rss_after = ProcessCurrentResidentBytes();
  observed.plan_rss_after = ProcessMaximumResidentBytes();
  observed.plan_wall_us =
      std::chrono::duration<double, std::micro>(plan_end - plan_begin).count();
  if (!planned) {
    CaptureFailure(observed, planned);
    observed.status = "plan_failed";
    observed.contract = observed.code == Code::Unavailable;
    PrintObservation(backend, materialize, observed);
    return observed.contract;
  }
  observed.plan = *planned;
  observed.plan_contract = PlanContract(observed.plan, backend);
  if (!observed.plan_contract) {
    observed.status = "plan_contract_failed";
    PrintObservation(backend, materialize, observed);
    return false;
  }

  const auto check_short_budget = [&]() {
    observed.short_budget_checked = true;
    auto short_builder = make_builder();
    const auto short_plan = short_builder.plan();
    if (!short_plan || *short_plan != observed.plan ||
        observed.plan.peak_bytes == 0u) {
      return false;
    }
    const MemoryStats before = device->memory();
    auto rejected =
        std::move(short_builder)
            .budget(MemoryBudget{.bytes = observed.plan.peak_bytes - 1u})
            .prepare();
    const MemoryStats after = device->memory();
    observed.short_budget_reason = rejected.reason();
    observed.short_budget_rejected =
        !rejected && rejected.reason() == Reason::PipelineMemoryBudget;
    observed.short_budget_no_allocation = NoAllocation(before, after);
    return observed.short_budget_rejected &&
           observed.short_budget_no_allocation;
  };

  if (!materialize) {
    observed.contract =
        PlanObservationContract(observed) && check_short_budget();
    observed.status = observed.contract ? "plan_ok" : "plan_contract_failed";
    PrintObservation(backend, materialize, observed);
    return observed.contract;
  }

  observed.prepare_attempted = true;
  observed.prepare_current_rss_before = ProcessCurrentResidentBytes();
  observed.prepare_rss_before = ProcessMaximumResidentBytes();
  node_compute_allocation::Start();
  const auto begin = Clock::now();
  auto prepared = std::move(builder)
                      .budget(MemoryBudget{.bytes = observed.plan.peak_bytes})
                      .prepare();
  const auto end = Clock::now();
  node_compute_allocation::Stop();
  observed.prepare_host_allocation_count = node_compute_allocation::Count();
  observed.prepare_host_allocation_bytes = node_compute_allocation::Bytes();
  observed.prepare_current_rss_after = ProcessCurrentResidentBytes();
  observed.prepare_rss_after = ProcessMaximumResidentBytes();
  observed.prepare_wall_us =
      std::chrono::duration<double, std::micro>(end - begin).count();
  observed.prepare_rss_within_committed_peak = WithinAdditionalResidentBytes(
      observed.prepare_current_rss_before, observed.prepare_current_rss_after,
      observed.prepare_rss_before, observed.prepare_rss_after,
      observed.plan.committed_peak_bytes);
  if (!prepared) {
    CaptureFailure(observed, prepared);
    const bool short_budget_contract = check_short_budget();
    if (observed.code == Code::Unavailable) {
      observed.status = "unavailable";
      observed.contract = PlanObservationContract(observed) &&
                          short_budget_contract &&
                          observed.prepare_rss_within_committed_peak;
      PrintObservation(backend, materialize, observed);
      return observed.contract;
    }
    observed.precise_failure = PreciseFailure(observed);
    observed.status =
        observed.precise_failure ? "precise_failure" : "imprecise_failure";
    observed.contract = PlanObservationContract(observed) &&
                        short_budget_contract && observed.precise_failure &&
                        observed.prepare_rss_within_committed_peak;
    PrintObservation(backend, materialize, observed);
    return observed.contract;
  }

  observed.prepare_ok = true;
  observed.plan_frozen = prepared->plan() == observed.plan;
  observed.memory = prepared->memory();
  observed.memory_contract =
      CaptureLargestRetainedGroup(observed, *prepared) &&
      PreparedMemoryContract(observed.plan, observed.memory, backend) &&
      CaptureBackendPreparation(observed, *prepared, backend);
  const bool short_budget_contract = check_short_budget();
  const bool prepared_contract =
      PlanObservationContract(observed) && short_budget_contract &&
      observed.plan_frozen && observed.memory_contract &&
      observed.prepare_wall_us > 0.0 && observed.prepare_rss_before != 0u &&
      observed.prepare_rss_after >= observed.prepare_rss_before;
  observed.contract =
      prepared_contract && observed.prepare_rss_within_committed_peak;
  observed.status =
      observed.contract
          ? "prepare_ok"
          : (prepared_contract && !observed.prepare_rss_within_committed_peak
                 ? "process_rss_contract_failed"
                 : "prepare_contract_failed");
  PrintObservation(backend, materialize, observed);
  return observed.contract;
}

} // namespace rund::measure::compute
